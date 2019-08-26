use super::gl;
use super::shader;
use super::shader::traits::*;
use super::ContextKey;
use super::{inner_gl_unsafe, inner_gl_unsafe_static, GlResource};
use crate::tuple::TupleIndex;
use crate::tuple::{AttachFront, RemoveFront};
use gl::types::*;
use gl::Gl;
use shader::api::*;
use std::marker::PhantomData;

use std::mem;
use std::ptr;

pub struct ArrayMesh<'a, T: ShaderArgs> {
    bindings: Box<[MeshBufferBinding]>,
    num_indices: u32,
    num_instances: u32,
    id: usize,
    // the mesh might contain its buffers
    // or it might reference them
    _to_drop: Box<[Buffer]>,
    phantom: PhantomData<std::rc::Rc<&'a T>>,
}

impl<'a, T: ShaderArgsClass<TransparentArgs>> ArrayMesh<'a, T> {
    pub fn new<S: IntoMesh<Args = T> + 'a>(s: S) -> ArrayMesh<'a, T> {
        let mut bindings = Vec::with_capacity(T::NARGS);
        let mut to_drop = Vec::new();
        let (indices, instances) = s.add_bindings(&mut bindings, &mut to_drop);
        if instances == 0 {
            panic!("The number of instances cannot be 0.");
        }
        ArrayMesh {
            bindings: bindings.into_boxed_slice(),
            num_indices: indices,
            num_instances: instances,
            // note: types that implement IntoMesh should not be
            // send or sync, and should only be able to be constructed
            // on the thread owning the gl_draw
            id: unsafe { get_mesh_id() },
            _to_drop: to_drop.into_boxed_slice(),
            phantom: PhantomData,
        }
    }
}

pub struct IndexMesh<'a, T: ShaderArgs> {
    bindings: Box<[MeshBufferBinding]>,
    index_buffer: u32,
    index_type: GLenum,
    num_indices: u32,
    num_elements: u32,
    num_instances: u32,
    id: usize,
    // the mesh might contain its buffers
    // or it might reference them
    _to_drop: Box<[Buffer]>,
    phantom: PhantomData<std::rc::Rc<&'a T>>,
}

impl<'a, T: ShaderArgsClass<TransparentArgs>> IndexMesh<'a, T> {
    pub fn new<S: IntoMesh<Args = T> + 'a, I: IndexType>(
        s: S,
        i: &'a ElementBuffer<I>,
    ) -> IndexMesh<'a, T> {
        let mut bindings = Vec::with_capacity(T::NARGS);
        let mut to_drop = Vec::new();
        let (indices, instances) = s.add_bindings(&mut bindings, &mut to_drop);
        if instances == 0 {
            panic!("The number of instances cannot be 0.");
        }
        IndexMesh {
            bindings: bindings.into_boxed_slice(),
            index_buffer: i.buffer.get_id(),
            index_type: I::TYPE,
            num_indices: (i.buffer.buffer_len / std::mem::size_of::<I>()) as u32,
            num_elements: indices,
            num_instances: instances,
            // note: types that implement IntoMesh should not be
            // send or sync, and should only be able to be constructed
            // on the thread owning the gl_draw
            id: unsafe { get_mesh_id() },
            _to_drop: to_drop.into_boxed_slice(),
            phantom: PhantomData,
        }
    }
}

pub mod unsafe_api {
    use super::*;
    use fnv::FnvHashMap;
    use glfw::ffi as glfw_raw;
    use Gl;

    static mut MESH_NUM: usize = 1;

    pub unsafe fn create_array_mesh<'a, T: ShaderArgsClass<TransparentArgs>>(
        buffers: Box<[Buffer]>,
        bindings: Box<[MeshBufferBinding]>,
        num_indices: u32,
        num_instances: u32,
    ) -> ArrayMesh<'a, T> {
        ArrayMesh {
            bindings: bindings,
            num_indices: num_indices,
            num_instances: num_instances,
            id: get_mesh_id(),
            _to_drop: buffers,
            phantom: PhantomData,
        }
    }

    /// This function must be called on the thread that owns the GlDraw
    pub unsafe fn get_mesh_id() -> usize {
        let n = MESH_NUM;
        MESH_NUM += 1;
        n
    }

    #[derive(Clone, Copy)]
    pub struct MeshBufferBinding {
        pub(crate) buffer: u32,
        pub(crate) size: u8,
        pub(crate) instance_divisor: u16,
        /// the 1s 2s, and 4s bits represent the data type,
        /// 0 = U8
        /// 1 = U16
        /// 2 = U32
        /// 3 = I8
        /// 4 = I16
        /// 5 = I32
        /// 6 = Float
        /// the 8s and 16s bits represent whether to normalize the data,
        /// and whether to use VertexAttribPointer or VertexAttribIPointer
        /// 0 = float, unnormed
        /// 8 = float, normed
        /// 16 = int
        pub(crate) btype: u8,
        pub(crate) offset: u32,
        pub(crate) stride: u32,
    }

    pub enum BindingFn {
        NormFloat,
        Float,
        Int,
    }

    impl MeshBufferBinding {
        pub fn new<T: GlDataType>(
            buffer: u32,
            size: u8,
            instance_divisor: u16,
            ty: BindingFn,
            offset: u32,
            stride: u32,
        ) -> MeshBufferBinding {
            if size > 4 {
                panic!("Size must be either 1, 2, 3, or 4.");
            }
            let btype = match T::TYPE {
                gl::UNSIGNED_BYTE => 0,
                gl::UNSIGNED_SHORT => 1,
                gl::UNSIGNED_INT => 2,
                gl::BYTE => 3,
                gl::SHORT => 4,
                gl::INT => 5,
                _ => 6,
            };

            let fn_ty = match ty {
                BindingFn::NormFloat => 8,
                BindingFn::Float => 0,
                BindingFn::Int => 16,
            };
            let btype = if T::TYPE == gl::FLOAT {
                btype
            } else {
                btype + fn_ty
            };
            MeshBufferBinding {
                buffer: buffer,
                size: size,
                instance_divisor: instance_divisor,
                btype: btype,
                offset: offset,
                stride: stride,
            }
        }
    }

    static mut VAO_MAP: *mut FnvHashMap<(usize, usize), GLuint> = 0 as *mut _;

    /// Add the vao to the map. This vao will be bound to the current window.
    #[inline]
    pub unsafe fn add_vao(id: usize, vao: GLuint) {
        if VAO_MAP.is_null() {
            let b = Box::new(
                FnvHashMap::<(usize, usize), GLuint>::with_capacity_and_hasher(
                    16,
                    Default::default(),
                ),
            );
            VAO_MAP = Box::into_raw(b);
        }
        (*VAO_MAP).insert((id, crate::opengl::CURRENT_WINDOW as usize), vao);
    }

    /// Search for a vao corresponding to the current window and
    /// the id given.
    #[inline]
    pub unsafe fn get_vao(id: usize) -> Option<GLuint> {
        if VAO_MAP.is_null() {
            return None;
        }
        (*VAO_MAP)
            .get(&(id, crate::opengl::CURRENT_WINDOW as usize))
            .map(|vao| *vao)
    }

    /// Clear all the vaos owned by a certain mesh. This should be
    /// called when the mesh is dropped.
    #[inline]
    pub unsafe fn clear_vaos(id: usize) {
        if !VAO_MAP.is_null() {
            (*VAO_MAP).retain(|key, vao| {
                // but what if a mesh is dropped when there aren't any
                // active windows? windows clear vaos on destruction,
                // so the gl will never be used in an orphan state.
                // this is why we can't have one call to gl::with_current
                // wrapping the call to `retain`, because it would cause
                // the creation of an invalid reference, which is illegal
                // just by existing
                if key.0 != id {
                    true
                } else {
                    gl::with_current(|gl| gl.DeleteVertexArrays(1, vao));
                    false
                }
            });
        }
    }

    #[inline]
    pub(crate) unsafe fn clear_window_vaos(window_ptr: *mut glfw_raw::GLFWwindow) {
        if !VAO_MAP.is_null() {
            (*VAO_MAP).retain(|key, _| key.1 != window_ptr as usize);
        }
    }

    #[inline]
    pub fn get_mesh_size(binding: MeshBufferBinding, blen: usize) -> (u32, u32) {
        let type_size = match binding.btype & 0b111 {
            0 => 1,
            1 => 2,
            2 => 4,
            3 => 1,
            4 => 2,
            5 => 4,
            _ => 4,
        };
        let size = type_size * binding.size as u32;
        let stride;
        if binding.stride == 0 {
            stride = size;
        } else {
            stride = binding.stride;
        }
        let mut sizes = (
            ((blen - binding.offset as usize - size as usize) / stride as usize) + 1,
            std::u32::MAX as usize,
        );
        if binding.instance_divisor != 0 {
            sizes = (
                std::u32::MAX as usize,
                (((blen - binding.offset as usize - size as usize) / stride as usize) + 1)
                    * binding.instance_divisor as usize,
            );
        }
        if sizes.0 > std::u32::MAX as usize {
            sizes.0 = std::u32::MAX as usize;
        }
        if sizes.1 > std::u32::MAX as usize {
            sizes.1 = std::u32::MAX as usize;
        }
        (sizes.0 as u32, sizes.1 as u32)
    }

    #[inline]
    pub unsafe fn bind_mesh_buffer(gl: &Gl, i: usize, binding: MeshBufferBinding) {
        let gl_draw = super::inner_gl_unsafe_static();
        gl.BindBuffer(
            gl::ARRAY_BUFFER,
            gl_draw.resource_list[binding.buffer as usize],
        );
        let data_type = match binding.btype & 0b111 {
            0 => gl::UNSIGNED_BYTE,
            1 => gl::UNSIGNED_SHORT,
            2 => gl::UNSIGNED_INT,
            3 => gl::BYTE,
            4 => gl::SHORT,
            5 => gl::INT,
            _ => gl::FLOAT,
        };

        match binding.btype & 0b11000 {
            0 => {
                gl.VertexAttribPointer(
                    i as u32,
                    binding.size as i32,
                    data_type,
                    gl::FALSE,
                    binding.stride as i32,
                    binding.offset as usize as *const _,
                );
            }
            8 => {
                gl.VertexAttribPointer(
                    i as u32,
                    binding.size as i32,
                    data_type,
                    gl::TRUE,
                    binding.stride as i32,
                    binding.offset as usize as *const _,
                );
            }
            _ => {
                gl.VertexAttribIPointer(
                    i as u32,
                    binding.size as i32,
                    data_type,
                    binding.stride as i32,
                    binding.offset as usize as *const _,
                );
            }
        }
        if binding.instance_divisor != 0 {
            gl.VertexAttribDivisor(i as u32, binding.instance_divisor as u32);
        }
        gl.EnableVertexAttribArray(i as u32);
    }
}

use self::unsafe_api::*;

pub unsafe trait Mesh<In: ShaderArgs> {
    type Drawer;

    /// Drawers are allowed to assume that the context they exist in
    /// has all the necessary objects bound correctly for a draw call to
    /// work. Calling this function violates.
    unsafe fn create_drawer(&self, mode: super::DrawMode) -> Self::Drawer;

    unsafe fn bind(&self, gl: &Gl);
}

#[allow(missing_copy_implementations)]
pub struct ArrayDrawer {
    mode: GLenum,
    num_indices: u32,
    num_instances: u32,
}

impl ArrayDrawer {
    pub fn draw_all(&self) {
        unsafe {
            gl::with_current(|gl| gl.DrawArrays(self.mode, 0, self.num_indices as i32));
        }
    }

    pub fn draw_arrays(&self, start: u32, count: u32) {
        if cfg!(feature = "draw_call_bounds_checks") {
            if (start + count) > self.num_indices {
                panic!(
                    "The smallest array in the mesh has {} elements, but the draw call requires {}.",
                    self.num_indices, start + count
                );
            }
        }
        unsafe {
            gl::with_current(|gl| gl.DrawArrays(self.mode, start as i32, count as i32));
        }
    }
}

unsafe impl<'a, T: ShaderArgs> Mesh<T> for ArrayMesh<'a, T> {
    type Drawer = ArrayDrawer;

    unsafe fn create_drawer(&self, mode: super::DrawMode) -> ArrayDrawer {
        ArrayDrawer {
            mode: mode.mode,
            num_indices: self.num_indices,
            num_instances: self.num_instances,
        }
    }

    unsafe fn bind(&self, gl: &Gl) {
        if let Some(vao) = get_vao(self.id) {
            gl.BindVertexArray(vao);
        } else {
            let mut vao = 0;
            gl.GenVertexArrays(1, &mut vao);
            gl.BindVertexArray(vao);
            for i in 0..self.bindings.len() {
                bind_mesh_buffer(gl, i, self.bindings[i]);
            }
            gl.BindBuffer(gl::ARRAY_BUFFER, 0);
            add_vao(self.id, vao);
        }
    }
}

impl<'a, T: ShaderArgs> Drop for ArrayMesh<'a, T> {
    fn drop(&mut self) {
        unsafe {
            clear_vaos(self.id);
        }
    }
}

unsafe impl<'a, T: ShaderArgs> Mesh<T> for IndexMesh<'a, T> {
    type Drawer = ();

    unsafe fn create_drawer(&self, mode: super::DrawMode) -> () {
        unimplemented!();
    }

    unsafe fn bind(&self, gl: &Gl) {
        if let Some(vao) = get_vao(self.id) {
            gl.BindVertexArray(vao);
        } else {
            let mut vao = 0;
            gl.GenVertexArrays(1, &mut vao);
            gl.BindVertexArray(vao);
            for i in 0..self.bindings.len() {
                bind_mesh_buffer(gl, i, self.bindings[i]);
            }
            let gl_draw = inner_gl_unsafe_static();
            gl.BindBuffer(
                gl::ELEMENT_ARRAY_BUFFER,
                gl_draw.resource_list[self.index_buffer as usize],
            );
            gl.BindBuffer(gl::ARRAY_BUFFER, 0);
            add_vao(self.id, vao);
        }
    }
}

impl<'a, T: ShaderArgs> Drop for IndexMesh<'a, T> {
    fn drop(&mut self) {
        unsafe {
            clear_vaos(self.id);
        }
    }
}

#[repr(C)]
pub struct VertexBuffer<T: GlDataType> {
    buffer: Buffer,
    phantom: PhantomData<T>,
}

impl<T: GlDataType> VertexBuffer<T> {
    pub fn new(_context: ContextKey, data: &[T]) -> VertexBuffer<T> {
        unsafe {
            // note: we could take the gl from the window, but that window is
            // not necessarily the active window. If some window exists, then there
            // is an active window
            gl::with_current(|gl| {
                let mut buffer = 0;
                gl.GenBuffers(1, &mut buffer);
                gl.BindBuffer(gl::ARRAY_BUFFER, buffer);
                gl.BufferData(
                    gl::ARRAY_BUFFER,
                    (data.len() * mem::size_of::<T>()) as isize,
                    data.as_ptr() as *const _,
                    gl::STATIC_DRAW,
                );
                gl.BindBuffer(gl::ARRAY_BUFFER, 0);
                let hints = ((gl::STATIC_DRAW as usize) << 32) + gl::ARRAY_BUFFER as usize;

                let id = inner_gl_unsafe()
                    .get_resource_generic::<Buffer>(buffer, Some(hints as *mut ()));
                let b = Buffer {
                    buffer_id: id,
                    buffer_len: data.len() * mem::size_of::<T>(),
                    phantom: PhantomData,
                };
                VertexBuffer {
                    buffer: b,
                    phantom: PhantomData,
                }
            })
        }
    }

    pub fn len(&self) -> usize {
        self.buffer.buffer_len / mem::size_of::<T>()
    }

    pub fn num_bytes(&self) -> usize {
        self.buffer.buffer_len
    }

    pub fn retype<'a, S: GlDataType>(&'a self) -> &'a VertexBuffer<S> {
        // a vertex buffer always has the same layout regardless of type
        unsafe { mem::transmute::<&'a VertexBuffer<T>, &'a VertexBuffer<S>>(self) }
    }

    pub fn retype_mut<'a, S: GlDataType>(&'a mut self) -> &'a mut VertexBuffer<S> {
        // a vertex buffer always has the same layout regardless of type
        // the borrow check makes the first reference unusable, avoiding aliased mutables
        unsafe { mem::transmute::<&'a mut VertexBuffer<T>, &'a mut VertexBuffer<S>>(self) }
    }

    pub fn from_buffer(buffer: Buffer) -> VertexBuffer<T> {
        VertexBuffer {
            buffer: buffer,
            phantom: PhantomData,
        }
    }

    pub fn into_inner(self) -> Buffer {
        self.buffer
    }
}

pub struct BufferBinding<'a, T: ArgParameter<TransparentArgs>> {
    binding: MeshBufferBinding,
    len: usize,
    phantom: PhantomData<std::rc::Rc<&'a T>>,
}

impl<'a, T: ArgType + ArgParameter<TransparentArgs>> Clone for BufferBinding<'a, T> {
    fn clone(&self) -> BufferBinding<'a, T> {
        BufferBinding {
            binding: self.binding,
            len: self.len,
            phantom: PhantomData,
        }
    }
}

impl<'a, T: ArgType + ArgParameter<TransparentArgs>> Copy for BufferBinding<'a, T> {}

pub struct OwnedBufferBinding<T: ArgParameter<TransparentArgs>> {
    binding: MeshBufferBinding,
    buffer: Buffer,
    phantom: PhantomData<T>,
}

pub unsafe trait IntoMesh {
    type Args: ShaderArgsClass<TransparentArgs>;

    fn add_bindings(
        self,
        bindings: &mut Vec<MeshBufferBinding>,
        to_drop: &mut Vec<Buffer>,
    ) -> (u32, u32);
}

unsafe impl IntoMesh for () {
    type Args = ();

    #[allow(unused)]
    #[inline(always)]
    fn add_bindings(
        self,
        _bindings: &mut Vec<MeshBufferBinding>,
        _to_drop: &mut Vec<Buffer>,
    ) -> (u32, u32) {
        (std::u32::MAX, std::u32::MAX)
    }
}

unsafe impl<'a, T: ArgParameter<TransparentArgs>> IntoMesh for BufferBinding<'a, T> {
    type Args = (T,);

    #[inline]
    fn add_bindings(
        self,
        bindings: &mut Vec<MeshBufferBinding>,
        _to_drop: &mut Vec<Buffer>,
    ) -> (u32, u32) {
        bindings.push(self.binding);
        get_mesh_size(self.binding, self.len)
    }
}

unsafe impl<T: ArgParameter<TransparentArgs>> IntoMesh for OwnedBufferBinding<T> {
    type Args = (T,);

    #[inline]
    fn add_bindings(
        self,
        bindings: &mut Vec<MeshBufferBinding>,
        to_drop: &mut Vec<Buffer>,
    ) -> (u32, u32) {
        bindings.push(self.binding);
        let len = self.buffer.buffer_len;
        to_drop.push(self.buffer);
        get_mesh_size(self.binding, len)
    }
}

unsafe impl<T: RemoveFront> IntoMesh for T
where
    T::Remaining: IntoMesh,
    T::Front: IntoMesh,
    <T::Front as IntoMesh>::Args: RemoveFront<Remaining = ()>,
    <T::Remaining as IntoMesh>::Args:
        AttachFront<<<T::Front as IntoMesh>::Args as RemoveFront>::Front>,
    <<T::Remaining as IntoMesh>::Args as AttachFront<
        <<T::Front as IntoMesh>::Args as RemoveFront>::Front,
    >>::AttachFront: ShaderArgsClass<TransparentArgs>,
{
    type Args = <<T::Remaining as IntoMesh>::Args as AttachFront<
        <<T::Front as IntoMesh>::Args as RemoveFront>::Front,
    >>::AttachFront;

    #[inline]
    fn add_bindings(
        self,
        bindings: &mut Vec<MeshBufferBinding>,
        to_drop: &mut Vec<Buffer>,
    ) -> (u32, u32) {
        let (front, remaining) = self.remove_front();
        let size1 = front.add_bindings(bindings, to_drop);
        let size2 = remaining.add_bindings(bindings, to_drop);
        (
            std::cmp::min(size1.0, size2.0),
            std::cmp::min(size1.1, size2.1),
        )
    }
}

fn wrap<'a, T: ArgType + ArgParameter<TransparentArgs>>(
    binding: MeshBufferBinding,
    len: usize,
) -> BufferBinding<'a, T> {
    BufferBinding {
        binding: binding,
        len: len,
        phantom: PhantomData,
    }
}

fn wrap_owned<T: ArgType + ArgParameter<TransparentArgs>>(
    binding: MeshBufferBinding,
    buffer: Buffer,
) -> OwnedBufferBinding<T> {
    OwnedBufferBinding {
        binding: binding,
        buffer: buffer,
        phantom: PhantomData,
    }
}

impl<T: IntType> VertexBuffer<T> {
    pub fn int_binding_ref<
        'a,
        C: TupleIndex<(T::Binding1, T::Binding2, T::Binding3, T::Binding4)>,
    >(
        &'a self,
        _components: C,
        offset: Option<u32>,
        stride: Option<u32>,
        instance_divisor: Option<u16>,
    ) -> BufferBinding<'a, C::I>
    where
        C::I: ArgType + ArgParameter<TransparentArgs>,
    {
        let bid = self.buffer.buffer_id;
        let off = if let Some(o) = offset { o } else { 0 };
        let stride = if let Some(s) = stride { s } else { 0 };
        let div = if let Some(d) = instance_divisor { d } else { 0 };
        let b = MeshBufferBinding {
            buffer: bid,
            size: (C::N + 1) as u8,
            instance_divisor: div,
            // 16 = int
            btype: 16 + T::ENUM,
            offset: off,
            stride: stride,
        };

        wrap(b, self.buffer.buffer_len)
    }

    pub fn int_binding<'a, C: TupleIndex<(T::Binding1, T::Binding2, T::Binding3, T::Binding4)>>(
        self,
        _components: C,
        offset: Option<u32>,
        stride: Option<u32>,
        instance_divisor: Option<u16>,
    ) -> OwnedBufferBinding<C::I>
    where
        C::I: ArgType + ArgParameter<TransparentArgs>,
    {
        let bid = self.buffer.buffer_id;
        let off = if let Some(o) = offset { o } else { 0 };
        let stride = if let Some(s) = stride { s } else { 0 };
        let div = if let Some(d) = instance_divisor { d } else { 0 };
        let b = MeshBufferBinding {
            buffer: bid,
            size: (C::N + 1) as u8,
            instance_divisor: div,
            // 16 = int
            btype: 16 + T::ENUM,
            offset: off,
            stride: stride,
        };

        wrap_owned(b, self.buffer)
    }

    pub fn float_binding_ref<'a, C: TupleIndex<(Float, Float2, Float3, Float4)>>(
        &'a self,
        _components: C,
        offset: Option<u32>,
        stride: Option<u32>,
        instance_divisor: Option<u16>,
    ) -> BufferBinding<'a, C::I>
    where
        C::I: ArgType + ArgParameter<TransparentArgs>,
    {
        let bid = self.buffer.buffer_id;
        let off = if let Some(o) = offset { o } else { 0 };
        let stride = if let Some(s) = stride { s } else { 0 };
        let div = if let Some(d) = instance_divisor { d } else { 0 };
        let b = MeshBufferBinding {
            buffer: bid,
            size: (C::N + 1) as u8,
            instance_divisor: div,
            // 0 = float, unnormed
            btype: 0 + T::ENUM,
            offset: off,
            stride: stride,
        };

        wrap(b, self.buffer.buffer_len)
    }

    pub fn float_binding<C: TupleIndex<(Float, Float2, Float3, Float4)>>(
        self,
        _components: C,
        offset: Option<u32>,
        stride: Option<u32>,
        instance_divisor: Option<u16>,
    ) -> OwnedBufferBinding<C::I>
    where
        C::I: ArgType + ArgParameter<TransparentArgs>,
    {
        let bid = self.buffer.buffer_id;
        let off = if let Some(o) = offset { o } else { 0 };
        let stride = if let Some(s) = stride { s } else { 0 };
        let div = if let Some(d) = instance_divisor { d } else { 0 };
        let b = MeshBufferBinding {
            buffer: bid,
            size: (C::N + 1) as u8,
            instance_divisor: div,
            // 0 = float, unnormed
            btype: 0 + T::ENUM,
            offset: off,
            stride: stride,
        };

        wrap_owned(b, self.buffer)
    }

    pub fn norm_float_binding_ref<'a, C: TupleIndex<(Float, Float2, Float3, Float4)>>(
        &'a self,
        _components: C,
        offset: Option<u32>,
        stride: Option<u32>,
        instance_divisor: Option<u16>,
    ) -> BufferBinding<'a, C::I>
    where
        C::I: ArgType + ArgParameter<TransparentArgs>,
    {
        let bid = self.buffer.buffer_id;
        let off = if let Some(o) = offset { o } else { 0 };
        let stride = if let Some(s) = stride { s } else { 0 };
        let div = if let Some(d) = instance_divisor { d } else { 0 };
        let b = MeshBufferBinding {
            buffer: bid,
            size: (C::N + 1) as u8,
            instance_divisor: div,
            // 8 = float, normed
            btype: 8 + T::ENUM,
            offset: off,
            stride: stride,
        };

        wrap(b, self.buffer.buffer_len)
    }

    pub fn norm_float_binding<C: TupleIndex<(Float, Float2, Float3, Float4)>>(
        self,
        _components: C,
        offset: Option<u32>,
        stride: Option<u32>,
        instance_divisor: Option<u16>,
    ) -> OwnedBufferBinding<C::I>
    where
        C::I: ArgType + ArgParameter<TransparentArgs>,
    {
        let bid = self.buffer.buffer_id;
        let off = if let Some(o) = offset { o } else { 0 };
        let stride = if let Some(s) = stride { s } else { 0 };
        let div = if let Some(d) = instance_divisor { d } else { 0 };
        let b = MeshBufferBinding {
            buffer: bid,
            size: (C::N + 1) as u8,
            instance_divisor: div,
            // 8 = float, normed
            btype: 8 + T::ENUM,
            offset: off,
            stride: stride,
        };

        wrap_owned(b, self.buffer)
    }
}

impl VertexBuffer<f32> {
    pub fn binding_ref<'a, C: TupleIndex<(Float, Float2, Float3, Float4)>>(
        &'a self,
        _components: C,
        offset: Option<u32>,
        stride: Option<u32>,
        instance_divisor: Option<u16>,
    ) -> BufferBinding<'a, C::I>
    where
        C::I: ArgType + ArgParameter<TransparentArgs>,
    {
        let bid = self.buffer.buffer_id;
        let off = if let Some(o) = offset { o } else { 0 };
        let stride = if let Some(s) = stride { s } else { 0 };
        let div = if let Some(d) = instance_divisor { d } else { 0 };
        let b = MeshBufferBinding {
            buffer: bid,
            // a f32 is 4 byte
            size: (C::N + 1) as u8,
            instance_divisor: div,
            // 6 = Float
            btype: 6,
            offset: off,
            stride: stride,
        };

        wrap(b, self.buffer.buffer_len)
    }

    pub fn binding<C: TupleIndex<(Float, Float2, Float3, Float4)>>(
        self,
        _components: C,
        offset: Option<u32>,
        stride: Option<u32>,
        instance_divisor: Option<u16>,
    ) -> OwnedBufferBinding<C::I>
    where
        C::I: ArgType + ArgParameter<TransparentArgs>,
    {
        let bid = self.buffer.buffer_id;
        let off = if let Some(o) = offset { o } else { 0 };
        let stride = if let Some(s) = stride { s } else { 0 };
        let div = if let Some(d) = instance_divisor { d } else { 0 };
        let b = MeshBufferBinding {
            buffer: bid,
            size: (C::N + 1) as u8,
            instance_divisor: div,
            // 6 = Float
            btype: 6,
            offset: off,
            stride: stride,
        };

        wrap_owned(b, self.buffer)
    }
}

use crate::tuple;

pub const ONE: tuple::T0 = tuple::I0;

pub const TWO: tuple::T1 = tuple::I1;

pub const THREE: tuple::T2 = tuple::I2;

pub const FOUR: tuple::T3 = tuple::I3;

pub unsafe trait IntType: GlDataType {
    type Binding1;
    type Binding2;
    type Binding3;
    type Binding4;
    const ENUM: u8;
}

unsafe impl IntType for u8 {
    type Binding1 = UInt;
    type Binding2 = UInt2;
    type Binding3 = UInt3;
    type Binding4 = UInt4;
    const ENUM: u8 = 0;
}

unsafe impl IntType for u16 {
    type Binding1 = UInt;
    type Binding2 = UInt2;
    type Binding3 = UInt3;
    type Binding4 = UInt4;
    const ENUM: u8 = 1;
}

unsafe impl IntType for u32 {
    type Binding1 = UInt;
    type Binding2 = UInt2;
    type Binding3 = UInt3;
    type Binding4 = UInt4;
    const ENUM: u8 = 2;
}

unsafe impl IntType for i8 {
    type Binding1 = Int;
    type Binding2 = Int2;
    type Binding3 = Int3;
    type Binding4 = Int4;
    const ENUM: u8 = 3;
}

unsafe impl IntType for i16 {
    type Binding1 = Int;
    type Binding2 = Int2;
    type Binding3 = Int3;
    type Binding4 = Int4;
    const ENUM: u8 = 4;
}

unsafe impl IntType for i32 {
    type Binding1 = Int;
    type Binding2 = Int2;
    type Binding3 = Int3;
    type Binding4 = Int4;
    const ENUM: u8 = 5;
}

pub unsafe trait IndexType: GlDataType {}

unsafe impl IndexType for u8 {}

unsafe impl IndexType for u16 {}

unsafe impl IndexType for u32 {}

pub struct ElementBuffer<T: IndexType> {
    buffer: Buffer,
    phantom: PhantomData<T>,
}

impl<T: IndexType> ElementBuffer<T> {
    pub fn new(buffer: Buffer) -> ElementBuffer<T> {
        ElementBuffer {
            buffer: buffer,
            phantom: PhantomData,
        }
    }

    pub fn into_inner(self) -> Buffer {
        self.buffer
    }

    pub fn buffer_ref(&self) -> &Buffer {
        &self.buffer
    }
}

/// A generic buffer that can hold any type of data.
///
/// Buffers in opengl fundamentally don't care about what type of data is in them (this is
/// different from textures, which have a specific, implementation defined, type and format).
pub struct Buffer {
    pub(crate) buffer_id: u32,
    // the size in bytes of the buffer.
    buffer_len: usize,
    // like all gl resources, a buffer cannot be used on different threads
    phantom: PhantomData<std::rc::Rc<()>>,
}

impl Buffer {
    #[inline]
    pub fn get_id(&self) -> u32 {
        self.buffer_id
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.buffer_len
    }
}

impl GlResource for Buffer {
    unsafe fn adopt(ptr: *mut (), id: u32) -> Option<*mut ()> {
        // it isn't really worth aligning the pointer
        let len = ptr::read_unaligned(ptr as *const u64);
        let hints = ptr::read_unaligned((ptr as *const u64).offset(1));
        let buffer_type_hint = (hints & 0xFF) as u32;
        let buffer_use_hint = (hints >> 32) as u32;
        let mut buffer = 0;
        gl::with_current(|gl| {
            gl.GenBuffers(1, &mut buffer);
            // in theory, implementations might use the target to make optimizations
            gl.BindBuffer(buffer_type_hint, buffer);
            gl.BufferData(
                buffer_type_hint,
                len as isize,
                (ptr as *const u64).offset(2) as *const _,
                buffer_use_hint,
            );
            gl.BindBuffer(buffer_type_hint, 0);
        });

        inner_gl_unsafe().resource_list[id as usize] = buffer;

        let _drop_vec = Vec::from_raw_parts(ptr as *mut u8, len as usize + 8, len as usize + 8);

        // note: the pointer no longer actually points to data, instead it stores two u32s
        // that represent the buffer type and usage.
        if cfg!(target_pointer_width = "64") {
            Some(hints as *mut ())
        } else if cfg!(target_pointer_width = "32") {
            Some(buffer_type_hint as *mut ())
        } else {
            panic!("Supported pointer widths are 32 and 64 bits.");
        }
    }

    unsafe fn drop_while_orphaned(ptr: *mut (), _id: u32) {
        let len = ptr::read_unaligned(ptr as *const u64);
        let _drop_vec = Vec::from_raw_parts(ptr as *mut u8, len as usize + 8, len as usize + 8);
    }

    unsafe fn cleanup(_ptr: *mut (), _id: u32) {
        // no cleanup neccessary since no heap allocated memory is used when not orphaned
    }

    unsafe fn orphan(id: u32, ptr: *mut ()) -> *mut () {
        gl::with_current(|gl| {
            let buffer = inner_gl_unsafe_static().resource_list[id as usize];
            gl.BindBuffer(gl::ARRAY_BUFFER, buffer);
            let hints = if cfg!(target_pointer_width = "64") {
                ptr as u64
            } else if cfg!(target_pointer_width = "32") {
                let mut h = 0;
                gl.GetBufferParameteriv(gl::ARRAY_BUFFER, gl::BUFFER_USAGE, &mut h);
                (h << 32) as u64 + ptr as u64
            } else {
                panic!("Supported pointer widths are 32 and 64 bits.");
            };
            let mut len = 0;
            gl.GetBufferParameteri64v(gl::ARRAY_BUFFER, gl::BUFFER_SIZE, &mut len);

            let mut data_vec = Vec::<u8>::with_capacity(len as usize + 8);
            let ptr = data_vec.as_mut_ptr();
            mem::forget(data_vec);

            ptr::write_unaligned(ptr as *mut u64, len as u64);
            ptr::write_unaligned((ptr as *mut u64).offset(1), hints);

            gl.GetBufferSubData(
                gl::ARRAY_BUFFER,
                0,
                len as isize,
                (ptr as *mut u64).offset(2) as *mut _,
            );
            gl.BindBuffer(gl::ARRAY_BUFFER, 0);

            ptr as *mut ()
        })
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        let gl_draw = unsafe { inner_gl_unsafe() };
        unsafe {
            let b = gl_draw.resource_list[self.buffer_id as usize];
            gl::with_current(|gl| gl.DeleteBuffers(1, &b));
        }
        gl_draw.remove_resource(self.buffer_id);
    }
}

pub mod uniform {
    use super::Gl;
    use crate::opengl::shader::{api::*, traits::*};
    use crate::opengl::ContextKey;
    use crate::tuple::{RemoveFront, TupleIndex};
    use nalgebra as na;
    use std::marker::PhantomData;

    // note: 0 is a reserved value
    static mut NUM_UNIFORMS: u64 = 1;

    pub struct Uniforms<T: ShaderArgs> {
        data: Box<[u32]>,
        id: u64,
        phantom: PhantomData<T>,
    }

    impl<T: ShaderArgs + ShaderArgsClass<UniformArgs>> Uniforms<T> {
        pub(crate) fn new_inner<S: SetUniforms<T>>(uniforms: S) -> Uniforms<T> {
            let mut i = 0;
            let mut len = 0;
            while i < T::NARGS {
                len += T::get_param(i).num_elements;
                i += 1;
            }
            let id;
            unsafe {
                id = NUM_UNIFORMS;
                NUM_UNIFORMS += 1;
            }
            let mut data = Vec::with_capacity(len as usize);
            // `copy_to_slice` will fill in the entire vec,
            // so the uninitialized memory is not a problem
            unsafe {
                data.set_len(len as usize);
            }
            uniforms.copy_to_slice(&mut data[..]);
            Uniforms {
                data: data.into_boxed_slice(),
                id: id,
                phantom: PhantomData,
            }
        }

        #[inline]
        pub fn new<S: SetUniforms<T>>(_window: ContextKey, uniforms: S) -> Uniforms<T> {
            Self::new_inner(uniforms)
        }

        pub(crate) fn default_inner() -> Uniforms<T> {
            let mut i = 0;
            let mut len = 0;
            while i < T::NARGS {
                len += T::get_param(i).num_elements;
                i += 1;
            }
            let id;
            unsafe {
                id = NUM_UNIFORMS;
                NUM_UNIFORMS += 1;
            }
            let v = vec![0u32; len as usize];
            Uniforms {
                data: v.into_boxed_slice(),
                id: id,
                phantom: PhantomData,
            }
        }

        // Initializes the uniforms to all be 0
        pub fn default(_context: ContextKey) -> Uniforms<T> {
            Self::default_inner()
        }

        pub fn set_val<S: TupleIndex<T>, U: SetUniforms<(S::I,)>>(&mut self, u: U)
        where
            (S::I,): ShaderArgs,
        {
            let mut i = 0;
            let mut len = 0;
            // the compiler can likely optimize this, since the length is
            // a constant with respect to the type parameters
            while i < S::N {
                len += T::get_param(i).num_elements;
                i += 1;
            }

            let slice =
                &mut self.data[len as usize..len as usize + T::get_param(i).num_elements as usize];

            u.copy_to_slice(slice);
        }

        #[inline]
        pub(crate) unsafe fn set_uniform(&self, gl: &Gl, n: u32, data_point: usize, location: u32) {
            let f = T::get_param(n as usize);
            let fn_ptr = gl.fn_ptrs[f.func];
            if f.is_mat {
                (std::mem::transmute::<_, extern "system" fn(i32, i32, u8, *const u32)>(fn_ptr))(
                    location as i32,
                    f.array_count as i32,
                    super::gl::FALSE,
                    self.data[data_point..].as_ptr(),
                );
            } else {
                (std::mem::transmute::<_, extern "system" fn(i32, i32, *const u32)>(fn_ptr))(
                    location as i32,
                    f.array_count as i32,
                    self.data[data_point..].as_ptr(),
                );
            }
        }

        pub(crate) unsafe fn set_uniforms<F: Iterator<Item = u32>>(
            &self,
            gl: &Gl,
            mut locations: F,
        ) {
            let mut n = 0;
            let mut data_point = 0;
            while n < T::NARGS as u32 {
                self.set_uniform(gl, n, data_point, locations.next().unwrap());
                data_point += T::get_param(n as usize).num_elements as usize;
                n += 1;
            }
        }
    }

    pub unsafe trait SetUniforms<T: ShaderArgs>: Copy {
        fn copy_to_slice(self, slice: &mut [u32]);
    }

    unsafe impl SetUniforms<()> for () {
        fn copy_to_slice(self, _slice: &mut [u32]) {
            // don't do anything
        }
    }

    unsafe impl<T: RemoveFront + Copy, U: ShaderArgs + RemoveFront> SetUniforms<U> for T
    where
        T::Front: SetUniforms<(U::Front,)>,
        T::Remaining: SetUniforms<U::Remaining>,
        U::Front: ArgParameter<UniformArgs> + ArgType,
        U::Remaining: ShaderArgs,
    {
        fn copy_to_slice(self, slice: &mut [u32]) {
            let (front, remaining) = self.remove_front();
            front.copy_to_slice(slice);
            let new_front =
                <U::Front as ArgParameter<UniformArgs>>::get_param().num_elements as usize;
            remaining.copy_to_slice(&mut slice[new_front..]);
        }
    }

    macro_rules! set_uniform {
        ($t:ty, $arg:ident) => {
            unsafe impl SetUniforms<($arg,)> for $t {
                fn copy_to_slice(self, slice: &mut [u32]) {
                    // this is safe because the types used will only be
                    // f32, i32, and u32
                    slice[0] = unsafe { std::mem::transmute(self) };
                }
            }
        };
        (;$t:ty, $arg:ident) => {
            unsafe impl SetUniforms<($arg,)> for $t {
                fn copy_to_slice(self, slice: &mut [u32]) {
                    // this is safe because the types used will only be
                    // f32, i32, and u32
                    slice.copy_from_slice(unsafe { std::mem::transmute::<_, &[u32]>(&self[..]) });
                }
            }
        };
        (;;$t:ty, $arg:ident) => {
            unsafe impl SetUniforms<($arg,)> for $t {
                fn copy_to_slice(self, slice: &mut [u32]) {
                    // this is safe because the types used will only be
                    // f32, i32, and u32
                    slice.copy_from_slice(unsafe {
                        std::mem::transmute::<_, &[u32]>(self.as_slice())
                    });
                }
            }
        };
    }

    set_uniform!(f32, Float);
    set_uniform!(;[f32; 2], Float2);
    set_uniform!(;[f32; 3], Float3);
    set_uniform!(;[f32; 4], Float4);
    set_uniform!(u32, UInt);
    set_uniform!(;[u32; 2], UInt2);
    set_uniform!(;[u32; 3], UInt3);
    set_uniform!(;[u32; 4], UInt4);
    set_uniform!(i32, Int);
    set_uniform!(;[i32; 2], Int2);
    set_uniform!(;[i32; 3], Int3);
    set_uniform!(;[i32; 4], Int4);

    set_uniform!(;[f32; 4], Float2x2);
    set_uniform!(;[f32; 6], Float2x3);
    set_uniform!(;[f32; 8], Float2x4);
    set_uniform!(;[f32; 6], Float3x2);
    set_uniform!(;[f32; 9], Float3x3);
    set_uniform!(;[f32; 12], Float3x4);
    set_uniform!(;[f32; 8], Float4x2);
    set_uniform!(;[f32; 12], Float4x3);
    set_uniform!(;[f32; 16], Float4x4);

    set_uniform!(;;na::Matrix2<f32>, Float2x2);
    set_uniform!(;;na::Matrix2x3<f32>, Float2x3);
    set_uniform!(;;na::Matrix2x4<f32>, Float2x4);
    set_uniform!(;;na::Matrix3x2<f32>, Float3x2);
    set_uniform!(;;na::Matrix3<f32>, Float3x3);
    set_uniform!(;;na::Matrix3x4<f32>, Float3x4);
    set_uniform!(;;na::Matrix4x2<f32>, Float4x2);
    set_uniform!(;;na::Matrix4x3<f32>, Float4x3);
    set_uniform!(;;na::Matrix4<f32>, Float4x4);
}
