use super::shader;
use super::shader::traits::*;
use super::{inner_gl_unsafe, inner_gl_unsafe_static, GlResource};
use crate::swizzle;
use gl;
use gl::types::*;
use shader::{Float, Float2, Float3, Float4, Int, Int2, Int3, Int4, UInt, UInt2, UInt3, UInt4};
use std::marker::PhantomData;
use swizzle::Swizzle4;
use swizzle::{AttachFront, RemoveFront};

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

struct MeshBufferBinding {
    buffer: u32,
    size: u8,
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
            let mut buffer = 0;
            gl::GenBuffers(1, &mut buffer);
            gl::BindBuffer(gl::ARRAY_BUFFER, buffer);
            gl::BufferData(
                gl::ARRAY_BUFFER,
                (data.len() * mem::size_of::<T>()) as isize,
                data.as_ptr() as *const _,
                gl::STATIC_DRAW,
            );
            gl::BindBuffer(gl::ARRAY_BUFFER, 0);
            let hints = (gl::STATIC_DRAW as usize) << 32 + gl::ARRAY_BUFFER as usize;

            let id =
                inner_gl_unsafe().get_resource_generic::<Buffer>(buffer, Some(hints as *mut ()));
            let b = Buffer {
                buffer_id: id,
                buffer_len: data.len() * mem::size_of::<T>(),
                phantom: PhantomData,
            };
            VertexBuffer {
                buffer: b,
                phantom: PhantomData,
            }
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

    unsafe fn bind_all_to_vao<F: FnMut() -> u32>(self, locations: F);
}

unsafe impl<T: ArgType + ArgParameter<InterfaceArgs>, B: BindBuffer<T>> InterfaceBinding
    for BufferBinding<T, B>
where
    (T,): ShaderArgs,
{
    type Bind = (T,);

    unsafe fn bind_all_to_vao<F: FnMut() -> u32>(self, mut locations: F) {
        self.binding.bind_to_vao(locations());
    }
}

unsafe impl InterfaceBinding for ()
{
    type Bind = ();

    unsafe fn bind_all_to_vao<F: FnMut() -> u32>(self, mut locations: F) {
    	// don't do anything
    }
}

unsafe impl<
        S: ArgType + ArgParameter<InterfaceArgs>,
        B: BindBuffer<S>,
        R: InterfaceBinding,
        T: RemoveFront<Front = BufferBinding<S, B>, Remaining = R>,
    > InterfaceBinding for T
where
    R::Bind: AttachFront<S>,
    <R::Bind as AttachFront<S>>::AttachFront: ShaderArgs,
{
    type Bind = <R::Bind as AttachFront<S>>::AttachFront;

    unsafe fn bind_all_to_vao<F: FnMut() -> u32>(self, mut locations: F) {
        let (front, remaining) = self.remove_front();
        front.binding.bind_to_vao(locations());
        remaining.bind_all_to_vao(locations);
    }
}

#[derive(Clone, Copy)]
pub struct BufferBinding<T: ArgType + ArgParameter<InterfaceArgs>, B: BindBuffer<T>> {
    binding: B,
    phantom: PhantomData<T>,
}

pub unsafe trait BindBuffer<T: ArgType + ArgParameter<InterfaceArgs>> {
    /// Binds the buffer (or in the case of matrix types, buffers) to the
    /// currently active VAO, setting the vertex attribute pointer.

    /// This function should not enable that attribute. This function should bind
    /// a buffer to gl::ARRAY_BUFFER, and the caller of this function is responsible
    /// for binding 0 to gl::ARRAY_BUFFER after it is done calling it.
    unsafe fn bind_to_vao(&self, location: u32);
}

fn wrap<T: ArgType + ArgParameter<InterfaceArgs>, B: BindBuffer<T>>(
    binding: B,
) -> BufferBinding<T, B> {
    BufferBinding { 
    	binding: binding,
    	phantom: PhantomData,
    }
}

pub struct VecBufferBinding<'a, T: ArgType + ArgParameter<InterfaceArgs>> {
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

impl<'a, T: ArgType + ArgParameter<InterfaceArgs>> Copy for VecBufferBinding<'a, T> {}

impl<'a, T: ArgType + ArgParameter<InterfaceArgs>> Clone for VecBufferBinding<'a, T> {
    fn clone(&self) -> VecBufferBinding<'a, T> {
        *self
    }
}

unsafe impl<'a, T: ArgType + ArgParameter<InterfaceArgs>> BindBuffer<T>
    for VecBufferBinding<'a, T>
{
    unsafe fn bind_to_vao(&self, location: u32) {
        gl::BindBuffer(gl::ARRAY_BUFFER, self.buffer);
        if self.int_norm == 2 {
            gl::VertexAttribIPointer(
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
            gl::VertexAttribPointer(
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

impl<'a, T: ArgType + ArgParameter<InterfaceArgs>> VecBufferBinding<'a, T> {
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
    pub fn int_binding<'a, C: Swizzle4<T::Binding1, T::Binding2, T::Binding3, T::Binding4>>(
        &'a self,
        _components: C,
        offset: Option<u32>,
        stride: Option<u32>,
    ) -> BufferBinding<C::S, VecBufferBinding<'a, C::S>>
    where
        C::S: ArgType + ArgParameter<InterfaceArgs>,
    {
        wrap(VecBufferBinding::new::<T>(
            unsafe { inner_gl_unsafe_static().resource_list[self.buffer.buffer_id as usize] },
            2,
            offset,
            stride,
            <C as swizzle::SZ>::N,
            self.buffer.buffer_len,
        ))
    }

    pub fn float_binding<'a, C: Swizzle4<Float, Float2, Float3, Float4>>(
        &'a self,
        _components: C,
        offset: Option<u32>,
        stride: Option<u32>,
    ) -> BufferBinding<C::S, VecBufferBinding<'a, C::S>>
    where
        C::S: ArgType + ArgParameter<InterfaceArgs>,
    {
        wrap(VecBufferBinding::new::<T>(
            unsafe { inner_gl_unsafe_static().resource_list[self.buffer.buffer_id as usize] },
            1,
            offset,
            stride,
            <C as swizzle::SZ>::N,
            self.buffer.buffer_len,
        ))
    }

    pub fn norm_float_binding<'a, C: Swizzle4<Float, Float2, Float3, Float4>>(
        &'a self,
        _components: C,
        offset: Option<u32>,
        stride: Option<u32>,
    ) -> BufferBinding<C::S, VecBufferBinding<'a, C::S>>
    where
        C::S: ArgType + ArgParameter<InterfaceArgs>,
    {
        wrap(VecBufferBinding::new::<T>(
            unsafe { inner_gl_unsafe_static().resource_list[self.buffer.buffer_id as usize] },
            0,
            offset,
            stride,
            <C as swizzle::SZ>::N,
            self.buffer.buffer_len,
        ))
    }
}

impl VertexBuffer<f32> {
    pub fn binding<'a, C: Swizzle4<Float, Float2, Float3, Float4>>(
        &'a self,
        _components: C,
        offset: Option<u32>,
        stride: Option<u32>,
    ) -> BufferBinding<C::S, VecBufferBinding<'a, C::S>>
    where
        C::S: ArgType + ArgParameter<InterfaceArgs>,
    {
        wrap(VecBufferBinding::new::<f32>(
            unsafe { inner_gl_unsafe_static().resource_list[self.buffer.buffer_id as usize] },
            1,
            offset,
            stride,
            <C as swizzle::SZ>::N,
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
}

/// A generic buffer that can hold any type of data.
///
/// Buffers in opengl fundamentally don't care about what type of data is in them (this is
/// different from textures, which have a specific, implementation defined, type and format).
pub struct Buffer {
    buffer_id: u32,
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
        gl::GenBuffers(1, &mut buffer);
        // in theory, implementations might use the target to make optimizations
        gl::BindBuffer(buffer_type_hint, buffer);
        gl::BufferData(
            buffer_type_hint,
            len as isize,
            (ptr as *const u64).offset(2) as *const _,
            buffer_use_hint,
        );
        gl::BindBuffer(buffer_type_hint, 0);

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
        let buffer = inner_gl_unsafe_static().resource_list[id as usize];
        gl::BindBuffer(gl::ARRAY_BUFFER, buffer);
        let hints = if cfg!(target_pointer_width = "64") {
            ptr as u64
        } else if cfg!(target_pointer_width = "32") {
            let mut h = 0;
            gl::GetBufferParameteriv(gl::ARRAY_BUFFER, gl::BUFFER_USAGE, &mut h);
            (h << 32) as u64 + ptr as u64
        } else {
            panic!("Supported point widths are 32 and 64 bits.");
        };
        let mut len = 0;
        gl::GetBufferParameteri64v(gl::ARRAY_BUFFER, gl::BUFFER_SIZE, &mut len);

        let mut data_vec = Vec::<u8>::with_capacity(len as usize + 8);
        let ptr = data_vec.as_mut_ptr();
        mem::forget(data_vec);

        ptr::write_unaligned(ptr as *mut u64, len as u64);
        ptr::write_unaligned((ptr as *mut u64).offset(1), hints);

        gl::GetBufferSubData(
            gl::ARRAY_BUFFER,
            0,
            len as isize,
            (ptr as *mut u64).offset(2) as *mut _,
        );
        gl::BindBuffer(gl::ARRAY_BUFFER, 0);

        ptr as *mut ()
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        let gl_draw = unsafe { inner_gl_unsafe() };
        gl_draw.remove_resource(self.buffer_id);
    }
}
