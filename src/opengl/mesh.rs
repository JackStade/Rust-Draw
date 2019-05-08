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

#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum BindingType {
    Float,
    NormU8,
    NormU16,
    NormU32,
    NormI8,
    NormI16,
    NormI32,
    FloatU8,
    FloatU16,
    FloatU32,
    FloatI8,
    FloatI16,
    FloatI32,
    U8,
    U16,
    U32,
    I8,
    I16,
    I32,
}

#[derive(Clone, Copy)]
struct MeshBufferBinding {
    buffer: u32,
    size: u8,
    instance_divisor: u8,
    btype: BindingType,
    offset: u32,
    stride: u32,
}

pub struct Mesh<T: ShaderArgs> {
    buffers: Box<[Buffer]>,
    bindings: Box<[MeshBufferBinding]>,
    num_verts: usize,
    phantom: PhantomData<T>,
}

pub struct IndexMesh<T: ShaderArgs> {
    buffers: Box<[Buffer]>,
    bindings: Box<[MeshBufferBinding]>,
    index_buffer: Buffer,
    num_indices: usize,
    phantom: PhantomData<T>,
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
                let hints = (gl::STATIC_DRAW as usize) << 32 + gl::ARRAY_BUFFER as usize;

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
        traits::*, Float, Float2, Float3, Float4, Int, Int2, Int3, Int4, UInt, UInt2, UInt3, UInt4,
    };
    use crate::opengl::GlWindow;
    use crate::tuple::{AttachFront, RemoveFront, TupleIndex};
    use std::marker::PhantomData;

    static mut NUM_UNIFORMS: u64 = 0;

    pub struct Uniforms<T: ShaderArgs> {
        data: Box<[u32]>,
        id: u64,
        phantom: PhantomData<T>,
    }

    struct UniformFn {
        index: u32,
        func: *const std::os::raw::c_void,
        is_mat: bool,
    }

    impl<T: ShaderArgs + ShaderArgsClass<UniformArgs>> Uniforms<T> {
        // note: new_inner must only be called on the main thread.
        pub(crate) fn new_inner<S: SetUniforms<T>>(uniforms: S) -> Uniforms<T> {
            let mut i = 0;
            let mut len = 0;
            while i < T::NARGS {
                len += T::get_param(i).num_elements;
                i += 1;
            }
            let mut data = Vec::with_capacity(len as usize);
            uniforms.push_to_vec(&mut data);
            let id;
            // new_innder() will only be called on the thread owning the GlDraw
            unsafe {
                id = NUM_UNIFORMS;
                NUM_UNIFORMS += 1;
            }
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
            let v = vec![0u32; len as usize];
            let id;
            unsafe {
                id = NUM_UNIFORMS;
                NUM_UNIFORMS += 1;
            }
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

        pub fn set_val<S: TupleIndex<T>, U: SetUniform<Arg = S::I>>(&self, u: U) {
            let mut i = 0;
            let mut len = 0;
            while i < S::N {
                len += T::get_param(i).num_elements;
                i += 1;
            }
        }

        #[inline]
        pub(crate) unsafe fn set_uniform(&self, gl: &Gl, n: u32, data_point: usize, location: u32) {
            let f = T::get_param(n as usize);
            unimplemented!();
            /*if f.is_mat {
                crate::opengl::uniform_functions::call_mat(
                    location as i32,
                    f.array_count as i32,
                    &self.data[data_point..],
                    ,
                );
            } else {
                crate::opengl::uniform_functions::call(
                    location as i32,
                    f.array_count as i32,
                    &self.data[data_point..],
                    ,
                );
            }*/
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

    unsafe impl<
            T: RemoveFront + Copy,
            U: ShaderArgs + RemoveFront<Front = <T::Front as SetUniform>::Arg>,
        > SetUniforms<U> for T
    where
        T::Front: SetUniform + ArgParameter<UniformArgs>,
        T::Remaining: SetUniforms<U::Remaining>,
        U::Remaining: ShaderArgs,
    {
        fn push_to_vec(self, vec: &mut Vec<u32>) {
            let (front, remaining) = self.remove_front();
            let l = vec.len();
            unsafe {
                let new_len = l + T::Front::get_param().num_elements as usize;
                assert!(vec.capacity() >= new_len);
                vec.set_len(new_len);
            }
            front.copy_to_slice(&mut vec[l..]);
            remaining.push_to_vec(vec);
        }
    }

    pub unsafe trait SetUniform: Copy {
        type Arg;

        fn copy_to_slice(&self, vec: &mut [u32]);
    }

    macro_rules! set_uniform {
        ($t:ty, $arg:ident) => {
            unsafe impl SetUniform for $t {
                type Arg = $arg;

                fn copy_to_slice(&self, slice: &mut [u32]) {
                    // this is safe because the types used will only be
                    // f32, i32, and u32
                    slice[0] = unsafe { std::mem::transmute(*self) };
                }
            }
        };
        (;$t:ty, $arg:ident) => {
            unsafe impl SetUniform for $t {
                type Arg = $arg;

                fn copy_to_slice(&self, slice: &mut [u32]) {
                    // this is safe because the types used will only be
                    // f32, i32, and u32
                    slice.copy_from_slice(unsafe { std::mem::transmute::<_, &[u32]>(&self[..]) });
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
            (len - offset as usize - comps * s) / (stride + 1) as usize
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
