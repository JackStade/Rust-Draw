use super::gl;
use super::shader;
use super::shader::traits::*;
use super::{inner_gl_unsafe, inner_gl_unsafe_static, GlResource};
use crate::tuple::TupleIndex;
use crate::tuple::{AttachFront, RemoveFront};
use gl::types::*;
use gl::Gl;
use shader::{Float, Float2, Float3, Float4, Int, Int2, Int3, Int4, UInt, UInt2, UInt3, UInt4};
use std::marker::PhantomData;

use std::mem;
use std::ptr;

pub struct ArrayMesh<T: ShaderArgs> {
    // the mesh contains buffers, which prevent it from being send or sync
    buffers: Box<[Buffer]>,
    bindings: Box<[MeshBufferBinding]>,
    num_indices: u32,
    num_instances: u32,
    id: usize,
    phantom: PhantomData<T>,
}

pub struct IndexMesh<T: ShaderArgs> {
    buffers: Box<[Buffer]>,
    bindings: Box<[MeshBufferBinding]>,
    index_buffer: Buffer,
    index_type: GLenum,
    num_indices: u32,
    num_instances: u32,
    id: usize,
    phantom: PhantomData<T>,
}

enum MeshDrawTypeEnum<'a> {
    Single(u32, u32),
    Multiple(&'a [u32], &'a [u32]),
    Instanced(u32, u32, u32),
}

pub mod unsafe_api {
    use super::*;
    use fnv::FnvHashMap;
    use glfw::ffi as glfw_raw;
    use Gl;

    static mut MESH_NUM: usize = 1;

    pub unsafe fn create_array_mesh<T: ShaderArgsClass<TransparentArgs>>(
        buffers: Box<[Buffer]>,
        bindings: Box<[MeshBufferBinding]>,
        num_indices: u32,
        num_instances: u32,
    ) -> ArrayMesh<T> {
        ArrayMesh {
            buffers: buffers,
            bindings: bindings,
            num_indices: num_indices,
            num_instances: num_instances,
            id: get_mesh_id(),
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
        /// the 1s 2s, and 3s bits represent the data type,
        /// 0 = U8
        /// 1 = U16
        /// 2 = U32
        /// 3 = I8
        /// 4 = I16
        /// 5 = I32
        /// 6 = Float
        /// the 4s and 5s bits represent whether to normalize the data,
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
            (*VAO_MAP).retain(|key, _| key.0 != id);
        }
    }

    #[inline]
    pub(crate) unsafe fn clear_window_vaos(window_ptr: *mut glfw_raw::GLFWwindow) {
        if !VAO_MAP.is_null() {
            (*VAO_MAP).retain(|key, _| key.1 != window_ptr as usize);
        }
    }

    #[inline]
    pub fn get_mesh_size<B: std::ops::Index<usize, Output = Buffer>>(
        i: usize,
        buffers: &B,
        binding: MeshBufferBinding,
    ) -> (u32, u32) {
        let type_size = match (binding.btype & 0b111) {
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
        let blen = buffers[binding.buffer as usize].buffer_len;
        let mut sizes = (
            ((blen as u32 - binding.offset - size) / stride) + 1,
            std::u32::MAX,
        );
        if (binding.instance_divisor != 0) {
            sizes = (
                std::u32::MAX,
                (((blen as u32 - binding.offset - size) / stride) + 1)
                    * binding.instance_divisor as u32,
            );
        }
        sizes
    }

    #[inline]
    pub unsafe fn bind_mesh_buffer<B: std::ops::Index<usize, Output = Buffer> + ?Sized>(
        gl: &Gl,
        i: usize,
        buffers: &B,
        binding: MeshBufferBinding,
    ) {
        let gl_draw = super::inner_gl_unsafe_static();
        gl.BindBuffer(
            gl::ARRAY_BUFFER,
            gl_draw.resource_list[buffers[binding.buffer as usize].buffer_id as usize],
        );
        let data_type = match (binding.btype & 0b111) {
            0 => gl::UNSIGNED_BYTE,
            1 => gl::UNSIGNED_SHORT,
            2 => gl::UNSIGNED_INT,
            3 => gl::BYTE,
            4 => gl::SHORT,
            5 => gl::INT,
            _ => gl::FLOAT,
        };

        match (binding.btype & 0b11000) {
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
        if (binding.instance_divisor != 0) {
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
    pub fn draw_arrays(self, start: u32, count: u32) {
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

unsafe impl<T: ShaderArgs> Mesh<T> for ArrayMesh<T> {
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
            for i in 0..self.buffers.len() {
                bind_mesh_buffer(gl, i, &self.buffers[..], self.bindings[i]);
            }
            gl.BindBuffer(gl::ARRAY_BUFFER, 0);
            add_vao(self.id, vao);
        }
    }
}

impl<T: ShaderArgs> Drop for ArrayMesh<T> {
    fn drop(&mut self) {
        unsafe {
            clear_vaos(self.id);
        }
    }
}

unsafe impl<T: ShaderArgs> Mesh<T> for IndexMesh<T> {
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
            for i in 0..self.buffers.len() {
                bind_mesh_buffer(gl, i, &self.buffers[..], self.bindings[i]);
            }
            let gl_draw = inner_gl_unsafe_static();
            gl.BindBuffer(
                gl::ELEMENT_ARRAY_BUFFER,
                gl_draw.resource_list[self.index_buffer.buffer_id as usize],
            );
            gl.BindBuffer(gl::ARRAY_BUFFER, 0);
            add_vao(self.id, vao);
        }
    }
}

impl<T: ShaderArgs> Drop for IndexMesh<T> {
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
    pub fn new(_window: &super::GlWindow, data: &[T]) -> VertexBuffer<T> {
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

pub unsafe trait InterfaceBinding {
    type Bind: ShaderArgs;

    unsafe fn bind_all_to_vao(self, gl: &Gl, location: u32);
}

unsafe impl<T: ArgType + ArgParameter<TransparentArgs>, B: BindBuffer<T>> InterfaceBinding
    for BufferBinding<T, B>
where
    (T,): ShaderArgs,
{
    type Bind = (T,);

    unsafe fn bind_all_to_vao(self, gl: &Gl, location: u32) {
        self.binding.bind_to_vao(gl, location);
    }
}

unsafe impl InterfaceBinding for () {
    type Bind = ();

    unsafe fn bind_all_to_vao(self, gl: &Gl, location: u32) {
        // don't do anything
    }
}

unsafe impl<
        S: ArgType + ArgParameter<TransparentArgs>,
        B: BindBuffer<S>,
        R: InterfaceBinding,
        T: RemoveFront<Front = BufferBinding<S, B>, Remaining = R>,
    > InterfaceBinding for T
where
    R::Bind: AttachFront<S>,
    <R::Bind as AttachFront<S>>::AttachFront: ShaderArgs,
{
    type Bind = <R::Bind as AttachFront<S>>::AttachFront;

    unsafe fn bind_all_to_vao(self, gl: &Gl, location: u32) {
        let (front, remaining) = self.remove_front();
        front.binding.bind_to_vao(gl, location);
        remaining.bind_all_to_vao(gl, location + S::get_param().num_input_locations);
    }
}

pub mod uniform {
    use super::Gl;
    use crate::opengl::shader::{
        traits::*, Float, Float2, Float2x2, Float2x3, Float2x4, Float3, Float3x2, Float3x3,
        Float3x4, Float4, Float4x2, Float4x3, Float4x4, Int, Int2, Int3, Int4, UInt, UInt2, UInt3,
        UInt4,
    };
    use crate::opengl::GlWindow;
    use crate::tuple::{AttachFront, RemoveFront, TupleIndex};
    use nalgebra as na;
    use std::marker::PhantomData;
    use std::rc::Rc;

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
            uniforms.push_to_vec(&mut data);
            Uniforms {
                data: data.into_boxed_slice(),
                id: id,
                phantom: PhantomData,
            }
        }

        pub fn new<S: SetUniforms<T>>(_window: &GlWindow, uniforms: S) -> Uniforms<T> {
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
        pub fn default(_window: &GlWindow) -> Uniforms<T> {
            Self::default_inner()
        }

        pub fn set_val<S: TupleIndex<T>, U: SetUniform<S::I>>(&mut self, u: U) {
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

        pub(crate) unsafe fn set_uniforms<F: FnMut() -> u32>(&self, gl: &Gl, mut locations: F) {
            let mut n = 0;
            let mut data_point = 0;
            while n < T::NARGS as u32 {
                self.set_uniform(gl, n, data_point, locations());
                data_point += T::get_param(n as usize).num_elements as usize;
                n += 1;
            }
        }
    }

    pub unsafe trait SetUniforms<T: ShaderArgs>: Copy {
        fn push_to_vec(self, vec: &mut Vec<u32>);
    }

    unsafe impl SetUniforms<()> for () {
        fn push_to_vec(self, vec: &mut Vec<u32>) {
            // don't do anything
        }
    }

    unsafe impl<T: RemoveFront + Copy, A, U: ShaderArgs + RemoveFront<Front = A>> SetUniforms<U> for T
    where
        T::Front: SetUniform<A>,
        T::Remaining: SetUniforms<U::Remaining>,
        U::Front: ArgParameter<UniformArgs>,
        U::Remaining: ShaderArgs,
    {
        fn push_to_vec(self, vec: &mut Vec<u32>) {
            let (front, remaining) = self.remove_front();
            let l = vec.len();
            unsafe {
                let new_len = l + U::Front::get_param().num_elements as usize;
                assert!(vec.capacity() >= new_len);
                vec.set_len(new_len);
            }
            front.copy_to_slice(&mut vec[l..]);
            remaining.push_to_vec(vec);
        }
    }

    pub unsafe trait SetUniform<T>: Copy {
        fn copy_to_slice(&self, vec: &mut [u32]);
    }

    macro_rules! set_uniform {
        ($t:ty, $arg:ident) => {
            unsafe impl SetUniform<$arg> for $t {
                fn copy_to_slice(&self, slice: &mut [u32]) {
                    // this is safe because the types used will only be
                    // f32, i32, and u32
                    slice[0] = unsafe { std::mem::transmute(*self) };
                }
            }
        };
        (;$t:ty, $arg:ident) => {
            unsafe impl SetUniform<$arg> for $t {
                fn copy_to_slice(&self, slice: &mut [u32]) {
                    // this is safe because the types used will only be
                    // f32, i32, and u32
                    slice.copy_from_slice(unsafe { std::mem::transmute::<_, &[u32]>(&self[..]) });
                }
            }
        };
        (;;$t:ty, $arg:ident) => {
            unsafe impl SetUniform<$arg> for $t {
                fn copy_to_slice(&self, slice: &mut [u32]) {
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

#[derive(Clone, Copy)]
pub struct BufferBinding<T: ArgType + ArgParameter<TransparentArgs>, B: BindBuffer<T>> {
    binding: B,
    phantom: PhantomData<T>,
}

pub unsafe trait BindBuffer<T: ArgType + ArgParameter<TransparentArgs>> {
    /// Binds the buffer (or in the case of matrix types, buffers) to the
    /// currently active VAO, setting the vertex attribute pointer.

    /// This function should not enable that attribute. This function should bind
    /// a buffer to gl::ARRAY_BUFFER, and the caller of this function is responsible
    /// for binding 0 to gl::ARRAY_BUFFER after it is done calling it.
    unsafe fn bind_to_vao(&self, gl: &Gl, location: u32);
}

fn wrap<T: ArgType + ArgParameter<TransparentArgs>, B: BindBuffer<T>>(
    binding: B,
) -> BufferBinding<T, B> {
    BufferBinding {
        binding: binding,
        phantom: PhantomData,
    }
}

pub struct VecBufferBinding<'a, T: ArgType + ArgParameter<TransparentArgs>> {
    buffer: GLuint,
    // 0 - normed float
    // 1 - unnormed float
    // 2 - int
    int_norm: u8,
    comps: u8,

    ty: GLenum,

    offset: u32,
    stride: u32,
    // the number of indices in the bound buffer
    len: usize,
    phantom: PhantomData<&'a T>,
}

impl<'a, T: ArgType + ArgParameter<TransparentArgs>> Copy for VecBufferBinding<'a, T> {}

impl<'a, T: ArgType + ArgParameter<TransparentArgs>> Clone for VecBufferBinding<'a, T> {
    fn clone(&self) -> VecBufferBinding<'a, T> {
        *self
    }
}

unsafe impl<'a, T: ArgType + ArgParameter<TransparentArgs>> BindBuffer<T>
    for VecBufferBinding<'a, T>
{
    unsafe fn bind_to_vao(&self, gl: &Gl, location: u32) {
        gl.BindBuffer(gl::ARRAY_BUFFER, self.buffer);
        if self.int_norm == 2 {
            gl.VertexAttribIPointer(
                location,
                self.comps as i32,
                self.ty,
                self.stride as i32,
                self.offset as *mut _,
            );
        } else {
            let norm = if self.int_norm < 1 {
                gl::TRUE
            } else {
                gl::FALSE
            };
            gl.VertexAttribPointer(
                location,
                self.comps as i32,
                self.ty,
                norm,
                self.stride as i32,
                self.offset as *mut _,
            );
        };
    }
}

impl<'a, T: ArgType + ArgParameter<TransparentArgs>> VecBufferBinding<'a, T> {
    fn new<S: GlDataType>(
        buffer: GLuint,
        int_norm: u8,
        offset: Option<u32>,
        stride: Option<u32>,
        comps: usize,
        len: usize,
    ) -> VecBufferBinding<'a, T> {
        let s = mem::size_of::<S>();
        let offset = if let Some(off) = offset {
            if off as usize > len {
                panic!("Offset {} is longer than length {}.", off, len);
            }
            off
        } else {
            0
        };
        let stride = if let Some(st) = stride {
            st
        } else {
            // note: this can never be larger than u32::MAX
            (comps * s) as u32
        };

        let num_elements = if len - offset as usize >= comps * s {
            (len - offset as usize - comps * s) / stride as usize + 1
        } else {
            0
        };

        VecBufferBinding {
            buffer: buffer,
            int_norm: int_norm,
            comps: comps as u8,

            ty: S::TYPE,

            offset: offset,
            stride: stride,

            len: num_elements,

            phantom: PhantomData,
        }
    }
}

impl<T: IntType> VertexBuffer<T> {
    pub fn int_binding<'a, C: TupleIndex<(T::Binding1, T::Binding2, T::Binding3, T::Binding4)>>(
        &'a self,
        _components: C,
        offset: Option<u32>,
        stride: Option<u32>,
    ) -> BufferBinding<C::I, VecBufferBinding<'a, C::I>>
    where
        C::I: ArgType + ArgParameter<TransparentArgs>,
    {
        wrap(VecBufferBinding::new::<T>(
            unsafe { inner_gl_unsafe_static().resource_list[self.buffer.buffer_id as usize] },
            2,
            offset,
            stride,
            C::N,
            self.buffer.buffer_len,
        ))
    }

    pub fn float_binding<'a, C: TupleIndex<(Float, Float2, Float3, Float4)>>(
        &'a self,
        _components: C,
        offset: Option<u32>,
        stride: Option<u32>,
    ) -> BufferBinding<C::I, VecBufferBinding<'a, C::I>>
    where
        C::I: ArgType + ArgParameter<TransparentArgs>,
    {
        wrap(VecBufferBinding::new::<T>(
            unsafe { inner_gl_unsafe_static().resource_list[self.buffer.buffer_id as usize] },
            1,
            offset,
            stride,
            C::N,
            self.buffer.buffer_len,
        ))
    }

    pub fn norm_float_binding<'a, C: TupleIndex<(Float, Float2, Float3, Float4)>>(
        &'a self,
        _components: C,
        offset: Option<u32>,
        stride: Option<u32>,
    ) -> BufferBinding<C::I, VecBufferBinding<'a, C::I>>
    where
        C::I: ArgType + ArgParameter<TransparentArgs>,
    {
        wrap(VecBufferBinding::new::<T>(
            unsafe { inner_gl_unsafe_static().resource_list[self.buffer.buffer_id as usize] },
            0,
            offset,
            stride,
            C::N,
            self.buffer.buffer_len,
        ))
    }
}

impl VertexBuffer<f32> {
    pub fn binding<'a, C: TupleIndex<(Float, Float2, Float3, Float4)>>(
        &'a self,
        _components: C,
        offset: Option<u32>,
        stride: Option<u32>,
    ) -> BufferBinding<C::I, VecBufferBinding<'a, C::I>>
    where
        C::I: ArgType + ArgParameter<TransparentArgs>,
    {
        wrap(VecBufferBinding::new::<f32>(
            unsafe { inner_gl_unsafe_static().resource_list[self.buffer.buffer_id as usize] },
            1,
            offset,
            stride,
            C::N,
            self.buffer.buffer_len,
        ))
    }
}

pub unsafe trait IntType: GlDataType {
    type Binding1;
    type Binding2;
    type Binding3;
    type Binding4;
}

unsafe impl IntType for u8 {
    type Binding1 = UInt;
    type Binding2 = UInt2;
    type Binding3 = UInt3;
    type Binding4 = UInt4;
}

unsafe impl IntType for u16 {
    type Binding1 = UInt;
    type Binding2 = UInt2;
    type Binding3 = UInt3;
    type Binding4 = UInt4;
}

unsafe impl IntType for u32 {
    type Binding1 = UInt;
    type Binding2 = UInt2;
    type Binding3 = UInt3;
    type Binding4 = UInt4;
}

unsafe impl IntType for i8 {
    type Binding1 = Int;
    type Binding2 = Int2;
    type Binding3 = Int3;
    type Binding4 = Int4;
}

unsafe impl IntType for i16 {
    type Binding1 = Int;
    type Binding2 = Int2;
    type Binding3 = Int3;
    type Binding4 = Int4;
}

unsafe impl IntType for i32 {
    type Binding1 = Int;
    type Binding2 = Int2;
    type Binding3 = Int3;
    type Binding4 = Int4;
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
        gl_draw.remove_resource(self.buffer_id);
    }
}
