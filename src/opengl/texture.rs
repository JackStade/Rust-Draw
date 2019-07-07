use super::gl;
use super::shader;
use super::{inner_gl, inner_gl_unsafe, inner_gl_unsafe_static, GlDraw, GlDrawCore};
use crate::tuple::{RemoveFront, TypeIterator};
use gl::types::*;
use gl::Gl;
use shader::traits::*;
use shader::ProgramBuilderItem;
use std::marker::PhantomData;
use std::{mem, ptr};

pub unsafe trait TextureComponents {
    const FORMAT: GLuint;
    const COMPONENTS: u32;
}

pub struct RGBA {}

pub struct RGB {}

pub struct RG {}

pub struct R {}

pub struct Depth {}

/// Stencil only textures are only supported in opengl 4.4 and higher. In opengl 4.3
/// they can be used for renderbuffers
#[cfg(feature = "opengl43")]
pub struct Stencil {}

pub struct DepthStencil {}

unsafe impl TextureComponents for RGBA {
    const FORMAT: GLuint = gl::RGBA;
    const COMPONENTS: u32 = 4;
}

unsafe impl TextureComponents for RGB {
    const FORMAT: GLuint = gl::RGB;
    const COMPONENTS: u32 = 3;
}

unsafe impl TextureComponents for RG {
    const FORMAT: GLuint = gl::RG;
    const COMPONENTS: u32 = 2;
}

unsafe impl TextureComponents for R {
    const FORMAT: GLuint = gl::RED;
    const COMPONENTS: u32 = 1;
}

unsafe impl TextureComponents for Depth {
    const FORMAT: GLuint = gl::DEPTH_COMPONENT;
    const COMPONENTS: u32 = 1;
}

#[cfg(feature = "opengl44")]
unsafe impl TextureComponents for Stencil {
    const FORMAT: GLuint = gl::STENCIL_INDEX;
    const COMPONENTS: u32 = 1;
}

unsafe impl TextureComponents for DepthStencil {
    const FORMAT: GLuint = gl::DEPTH_STENCIL;
    // opengl would tell you that this format has 2 components,
    // but since we are wrapping the packed depth/stencil formats
    // into a single struct, this is the best way to do this
    const COMPONENTS: u32 = 1;
}

pub unsafe trait TextureBufferType {
    type Data: GlDataType;
    type Components: TextureComponents;
}

pub unsafe trait TextureFormat: TextureBufferType {
    type Texture2D: ArgType;

    const INTERNAL_FORMAT: GLuint;
}

pub unsafe trait TargetTexture: TextureFormat {
    type Target: ArgParameter<OutputArgs>;
}

/// A floating point texture.
///
/// If the data type is an integer, then it is converted to a float when accessed by the GPU. The
/// data will be mapped to [0, 1] for unsigned types and [-1, 1] for signed types.
///
/// `Data` can be one of:
/// * `u8`
/// * `i8`
/// * `u16`
/// * `i16`
/// * `f32`
/// Note that integer types longer than 16 bits are not supported by OpenGL (this is likely because
/// 32 bit normalized types cannot be represented losslessly with an f32)
pub struct TextureData<C: TextureComponents, Data> {
    phantom: PhantomData<(C, Data)>,
}

unsafe impl<C: TextureComponents, Data: GlDataType> TextureBufferType for TextureData<C, Data> {
    type Data = Data;
    type Components = C;
}

macro_rules! impl_format {
	($x:ident, $comp:ty, $t2d:ty, $($data:ty, $i_f:expr,)*) => (
		$(
			unsafe impl TextureFormat for $x<$comp, $data> {
				type Texture2D = $t2d;

				const INTERNAL_FORMAT: GLuint = $i_f;
			}
		)*
	)
}

macro_rules! impl_target {
    ($x:ident, $comp:ty, $t:ty, $($data:ty,)*) => (
        $(
            unsafe impl TargetTexture for $x<$comp, $data> {
                type Target = $t;
            }
        )*
    )
}

impl_format!(
    TextureData,
    RGBA,
    Sampler2D,
    u8,
    gl::RGBA8,
    i8,
    gl::RGBA8_SNORM,
    u16,
    gl::RGBA16,
    i16,
    gl::RGBA16_SNORM,
    f32,
    gl::RGBA32F,
);

impl_format!(
    TextureData,
    RGB,
    Sampler2D,
    u8,
    gl::RGB8,
    i8,
    gl::RGB8_SNORM,
    u16,
    gl::RGB16,
    i16,
    gl::RGB16_SNORM,
    f32,
    gl::RGB32F,
);

impl_format!(
    TextureData,
    RG,
    Sampler2D,
    u8,
    gl::RG8,
    i8,
    gl::RG8_SNORM,
    u16,
    gl::RG16,
    i16,
    gl::RG16_SNORM,
    f32,
    gl::RG32F,
);

impl_format!(
    TextureData,
    R,
    Sampler2D,
    u8,
    gl::R8,
    i8,
    gl::R8_SNORM,
    u16,
    gl::R16,
    i16,
    gl::R16_SNORM,
    f32,
    gl::R32F,
);

impl_target!(TextureData, RGBA, Float4, u8, u16, f32,);
impl_target!(TextureData, RG, Float2, u8, u16, f32,);
impl_target!(TextureData, R, Float, u8, u16, f32,);

/// An integer texture format, for when it is useful to be able to read integer values directly
/// in shaders.
///
/// `Data` can be one of:
/// * `u8`
/// * `i8`
/// * `u16`
/// * `i16`
/// * `u32`
/// * `i32`
/// Note that integer types longer than 32 bits are not supported, and that there is significant
/// precision loss when normalizing integers longer than 32 bits.
///
/// # Note
///
/// This type actually represents two different types of glsl samplers. When `Data` is a signed
/// type, it can be bound to an `IntSampler`, when `Data` is unsigned it can be bound to a `UIntSampler`
pub struct IntTextureData<C: TextureComponents, Data> {
    phantom: PhantomData<(C, Data)>,
}

unsafe impl<C: TextureComponents, Data: GlDataType> TextureBufferType for IntTextureData<C, Data> {
    type Data = Data;
    type Components = C;
}

impl_format!(
    IntTextureData,
    RGBA,
    UIntSampler2D,
    u8,
    gl::RGBA8UI,
    u16,
    gl::RGBA16UI,
    u32,
    gl::RGBA32UI,
);

impl_format!(
    IntTextureData,
    RGB,
    UIntSampler2D,
    u8,
    gl::RGB8UI,
    u16,
    gl::RGB16UI,
    u32,
    gl::RGB32UI,
);

impl_format!(
    IntTextureData,
    RG,
    UIntSampler2D,
    u8,
    gl::RG8UI,
    u16,
    gl::RG16UI,
    u32,
    gl::RG32UI,
);

impl_format!(
    IntTextureData,
    R,
    UIntSampler2D,
    u8,
    gl::R8UI,
    u16,
    gl::R16UI,
    u32,
    gl::R32UI,
);

impl_format!(
    IntTextureData,
    RGBA,
    IntSampler2D,
    i8,
    gl::RGBA8I,
    i16,
    gl::RGBA16I,
    i32,
    gl::RGBA32I,
);

impl_format!(
    IntTextureData,
    RGB,
    IntSampler2D,
    i8,
    gl::RGB8I,
    i16,
    gl::RGB16I,
    i32,
    gl::RGB32I,
);

impl_format!(
    IntTextureData,
    RG,
    IntSampler2D,
    i8,
    gl::RG8I,
    i16,
    gl::RG16I,
    i32,
    gl::RG32I,
);

impl_format!(
    IntTextureData,
    R,
    IntSampler2D,
    i8,
    gl::R8I,
    i16,
    gl::R16I,
    i32,
    gl::R32I,
);

impl_target!(IntTextureData, RGBA, UInt4, u8, u16, u32,);
impl_target!(IntTextureData, RG, UInt2, u8, u16, u32,);
impl_target!(IntTextureData, R, UInt, u8, u16, u32,);

impl_target!(IntTextureData, RGBA, Int4, i8, i16, i32,);
impl_target!(IntTextureData, RG, Int2, i8, i16, i32,);
impl_target!(IntTextureData, R, Int, i8, i16, i32,);

pub struct Depth16TextureData {}

pub struct Depth24TextureData {}

pub struct Depth32FloatTextureData {}

pub struct Stencil8TextureData {}

pub struct Depth24Stencil8TextureData {}

pub struct Depth32FStencil8TextureData {}

macro_rules! ds_format {
    ($t:ty, $data:ty, $c:ty, $t2d:ty, $if:expr) => {
        unsafe impl TextureBufferType for $t {
            type Data = $data;
            type Components = $c;
        }

        unsafe impl TextureFormat for $t {
            type Texture2D = $t2d;

            const INTERNAL_FORMAT: GLuint = $if;
        }
    };
}

ds_format!(
    Depth16TextureData,
    u16,
    Depth,
    UIntSampler2D,
    gl::DEPTH_COMPONENT16
);
ds_format!(
    Depth24TextureData,
    Depth24,
    Depth,
    UIntSampler2D,
    gl::DEPTH_COMPONENT24
);
ds_format!(
    Depth32FloatTextureData,
    f32,
    Depth,
    Sampler2D,
    gl::DEPTH_COMPONENT32F
);
ds_format!(
    Depth24Stencil8TextureData,
    Depth24Stencil8,
    DepthStencil,
    UIntSampler2D,
    gl::DEPTH24_STENCIL8
);
ds_format!(
    Depth32FStencil8TextureData,
    Depth32FStencil8,
    DepthStencil,
    Sampler2D,
    gl::DEPTH32F_STENCIL8
);

#[cfg(feature = "opengl44")]
ds_format!(
    Stencil8TextureData,
    u8,
    Stencil,
    UIntSampler2D,
    gl::STENCIL_INDEX8
);

#[derive(Clone, Copy)]
#[repr(C)]
/// Only the low 24 bits are relevant, the high 8 bits are padding
pub struct Depth24 {
    pub data: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
/// the low 8 bits are stencil, the high
/// 24 bits are depth
pub struct Depth24Stencil8 {
    pub data: u32,
}

impl Depth24Stencil8 {
    pub fn depth(self) -> u32 {
        self.data >> 8
    }

    pub fn stencil(self) -> u8 {
        (self.data & 0xFF) as u8
    }
}

#[derive(Clone, Copy)]
#[repr(C)]
/// Only the last 8 bits of stencil are relevant,
/// the first 24 bits are padding
pub struct Depth32FStencil8 {
    pub depth: f32,
    pub stencil: u32,
}

impl Depth32FStencil8 {
    pub fn depth(self) -> f32 {
        self.depth
    }

    pub fn stencil(self) -> u8 {
        (self.stencil & 0xFF) as u8
    }
}

macro_rules! format_data {
    ($($t:ty, $ty:expr,)*) => (
        $(
            unsafe impl GlDataType for $t {
                const TYPE: gl::types::GLenum = $ty;
            }
        )*
    )
}

format_data!(
    Depth24,
    gl::UNSIGNED_INT,
    Depth24Stencil8,
    gl::UNSIGNED_INT_24_8,
    Depth32FStencil8,
    gl::FLOAT_32_UNSIGNED_INT_24_8_REV,
);

pub unsafe trait BindTexture<T> {
    unsafe fn bind(&self, gl: &Gl);
}

use super::GlResource;

pub struct Texture2D<F: TextureFormat> {
    pub(crate) image_id: u32,

    pub(crate) width: u32,
    pub(crate) height: u32,
    // images cannot be send or sync because they should not be dropped
    // on a different thread
    phantom: PhantomData<std::rc::Rc<F>>,
}

impl<F: TextureFormat> Drop for Texture2D<F> {
    fn drop(&mut self) {
        let gl_draw = unsafe { inner_gl_unsafe() };
        gl_draw.remove_resource(self.image_id);
    }
}

impl<F: TextureFormat> GlResource for Texture2D<F> {
    unsafe fn adopt(ptr: *mut (), id: u32) -> Option<*mut ()> {
        // `ptr` will be 4-aligned
        let ptr = ptr as *mut u32;
        let [width, height, drop_len] = ptr::read(ptr as *const [u32; 3]);
        let mut tex = 0;
        gl::with_current(|gl| {
            gl.GenTextures(1, &mut tex);
            Texture2D::<F>::load_image(
                inner_gl_unsafe(),
                gl,
                tex,
                width,
                height,
                ptr.offset(3) as *const _,
                4,
            );
        });
        // ptr is of type *const u32, and drop_len is the number of u32s to drop
        let _drop_vec = Vec::from_raw_parts(ptr, drop_len as usize, drop_len as usize);
        let gl_draw = inner_gl_unsafe();
        // the image id now points to the new texture, so that the texture can be draw
        gl_draw.resource_list[id as usize] = tex;
        // the buffer is freed here
        None
    }

    unsafe fn drop_while_orphaned(ptr: *mut (), _id: u32) {
        let ptr = ptr as *mut u32;
        let [_, _, drop_len] = ptr::read(ptr as *const [u32; 3]);
        let _drop_vec = Vec::from_raw_parts(ptr, drop_len as usize, drop_len as usize);
    }

    unsafe fn cleanup(_ptr: *mut (), _id: u32) {
        // no cleanup is neccesary since textures do not store additional data when
        // not in an orphan state
    }

    unsafe fn orphan(id: u32, _ptr: *mut ()) -> *mut () {
        gl::with_current(|gl| {
            let data_size = mem::size_of::<F::Data>();
            let num_comps = F::Components::COMPONENTS;
            let mut w = 0u32;
            let mut h = 0u32;
            let tex = inner_gl_unsafe().resource_list[id as usize];

            gl.BindTexture(gl::TEXTURE_2D, tex);
            gl.GetTexLevelParameteriv(
                gl::TEXTURE_2D,
                0,
                gl::TEXTURE_WIDTH,
                &mut w as *mut _ as *mut _,
            );
            gl.GetTexLevelParameteriv(
                gl::TEXTURE_2D,
                0,
                gl::TEXTURE_WIDTH,
                &mut h as *mut _ as *mut _,
            );
            // we need to determine the size of each row (in bytes) so that the buffer has room for padding bytes
            let row_size = data_size as u32 * num_comps * w;
            // adding 3 will round the size up if it is not already a multiple of 4
            let buffer_row_size = (row_size + 3) >> 2;
            let mut buff = Vec::<u32>::with_capacity(3 + buffer_row_size as usize * h as usize);
            // cannot push here because rust could in theory reallocate
            buff.set_len(3);
            buff[0] = 0;
            buff[1] = h;
            buff[2] = buff.capacity() as u32;
            // into_orphan_data will only be called on the main thread with a context loaded
            gl.PixelStorei(gl::PACK_ALIGNMENT, 4);
            gl.GetTexImage(
                gl::TEXTURE_2D,
                0,
                F::Components::FORMAT,
                F::Data::TYPE,
                // the first 3 bytes are used for other data
                buff.as_mut_ptr().offset(3) as *mut _,
            );
            let ptr = buff.as_mut_ptr() as *mut ();
            // can't drop the vec
            mem::forget(buff);
            ptr
        })
    }
}

impl<F: TextureFormat> Texture2D<F> {
    pub fn new(
        _context: super::ContextKey,
        width: u32,
        height: u32,
        data: &[F::Data],
    ) -> Texture2D<F> {
        let gl_draw = unsafe { inner_gl_unsafe() };
        let num_comps = F::Components::COMPONENTS;
        if data.len() < (width as usize * height as usize * num_comps as usize) {
            panic!("Slice length of {} is too small to create a texture of size {}x{} with {} components per pixel.", data.len(), width, height, num_comps);
        }
        let id;
        let mut tex = 0;
        unsafe {
            gl::with_current(|gl| {
                gl.GenTextures(1, &mut tex);
                Self::load_image(
                    gl_draw,
                    gl,
                    tex,
                    width,
                    height,
                    data.as_ptr(),
                    mem::align_of::<F::Data>(),
                );
            })
        }
        id = gl_draw.get_resource_generic::<Self>(tex, None);
        Texture2D {
            image_id: id,

            width: width,
            height: height,

            phantom: PhantomData,
        }
    }

    /// It can be useful when rendering to textures to generate an emtpy texture. The
    /// contents of the texture are undefined. It could just be 0, or it could be random
    /// data.
    pub fn uninitialized(_context: super::ContextKey, width: u32, height: u32) -> Texture2D<F> {
        let gl_draw = unsafe { inner_gl_unsafe() };
        let id;
        let mut tex = 0;
        unsafe {
            gl::with_current(|gl| {
                gl.GenTextures(1, &mut tex);
                Self::load_image(
                    gl_draw,
                    gl,
                    tex,
                    width,
                    height,
                    // opengl interprets this to mean that the texture should not be
                    // filled. Note that if the texture is orphaned and adopted, the new
                    // texture will be initialized with whatever garbage data was generated
                    // by opengl here
                    ptr::null(),
                    mem::align_of::<F::Data>(),
                );
            })
        }
        id = gl_draw.get_resource_generic::<Self>(tex, None);
        Texture2D {
            image_id: id,

            width: width,
            height: height,

            phantom: PhantomData,
        }
    }

    pub fn get_dimension(&self) -> (u32, u32) {
        (self.width as u32, self.height as u32)
    }

    #[allow(unused)]
    unsafe fn load_image(
        gl_draw: &mut GlDrawCore,
        gl: &Gl,
        tex: u32,
        width: u32,
        height: u32,
        ptr: *const F::Data,
        align: usize,
    ) {
        let data_size = mem::size_of::<F::Data>();
        let num_comps = F::Components::COMPONENTS;
        gl.BindTexture(gl::TEXTURE_2D, tex);
        gl.PixelStorei(gl::UNPACK_ALIGNMENT, align as i32);
        gl.TexImage2D(
            gl::TEXTURE_2D,
            0,
            F::INTERNAL_FORMAT as i32,
            width as i32,
            height as i32,
            0,
            F::Components::FORMAT,
            F::Data::TYPE,
            ptr as *const _,
        );

        gl.TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::NEAREST as i32);
        gl.TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::NEAREST as i32);
    }
}

pub struct ImageBindings<'a, T: ShaderArgsClass<ImageArgs>> {
    bindings: Vec<(*const (), fn(*const (), &Gl))>,
    phantom: PhantomData<&'a T>,
}

pub trait ArgsBindings<'a, S> {
    fn add_to_vec(self, vec: &mut Vec<(*const (), fn(*const (), &Gl))>);
}

impl<'a> ArgsBindings<'a, ()> for () {
    fn add_to_vec(self, _vec: &mut Vec<(*const (), fn(*const (), &Gl))>) {
        // do nothing
    }
}

impl<
        'a,
        R: 'a + BindTexture<S::Front>,
        T: RemoveFront<Front = &'a R>,
        S: ShaderArgsClass<ImageArgs> + RemoveFront,
    > ArgsBindings<'a, S> for T
where
    T::Remaining: ArgsBindings<'a, S::Remaining>,
{
    fn add_to_vec(self, vec: &mut Vec<(*const (), fn(*const (), &Gl))>) {
        let (f, r) = self.remove_front();
        unsafe {
            // convert an fn(&self, &Gl) to an fn(*const (), &Gl). This should
            // be ok, since this is how trait objects work internally, though
            // rust technically makes this UB (at least, its UB to call the function, not
            // to create it)
            vec.push((
                f as *const _ as *const _,
                std::mem::transmute::<unsafe fn(_, _) -> _, _>(R::bind),
            ));
        }
        r.add_to_vec(vec);
    }
}

impl<'a, S: ShaderArgsClass<ImageArgs>> ImageBindings<'a, S> {
    pub fn empty() -> ImageBindings<'a, ()> {
        ImageBindings {
            bindings: Vec::new(),
            phantom: PhantomData,
        }
    }

    pub fn new<
        F: BindTexture<<S as RemoveFront>::Front> + 'a,
        R: ArgsBindings<'a, <S as RemoveFront>::Remaining>,
        T: TypeIterator<&'a F, R>,
    >(
        t: T,
    ) -> ImageBindings<'a, S>
    where
        S: RemoveFront,
    {
        let mut v = Vec::with_capacity(<S as ShaderArgs>::NARGS);
        let (f, r) = t.yield_item();
        unsafe {
            // convert an fn(&self, &Gl) to an fn(*const (), &Gl). This should
            // be ok, since this is how trait objects work internally, though
            // rust technically makes this UB
            v.push((
                f as *const _ as *const _,
                std::mem::transmute::<unsafe fn(_, _) -> _, _>(F::bind),
            ));
        }
        r.add_to_vec(&mut v);
        ImageBindings {
            bindings: v,
            phantom: PhantomData,
        }
    }

    pub(crate) fn bind(&self, gl: &Gl) {
        for (i, (ptr, f)) in self.bindings.iter().enumerate() {
            unsafe {
                gl.ActiveTexture(gl::TEXTURE0 + i as u32);
            }
            f(*ptr, gl);
        }
    }
}

unsafe impl<F: TextureFormat> BindTexture<F::Texture2D> for Texture2D<F> {
    unsafe fn bind(&self, gl: &Gl) {
        let gl_draw = inner_gl_unsafe_static();

        gl.BindTexture(
            gl::TEXTURE_2D,
            gl_draw.resource_list[self.image_id as usize],
        );
    }
}

unsafe fn get_image(id: u32) -> GLuint {
    let gl_draw = inner_gl_unsafe();
    gl_draw.resource_list[id as usize]
}

pub fn texture<'a, T: ItemType<'a, Ty = <S::Ty as Sampler>::Arg>, S: ItemType<'a>>(
    sampler: S,
    arg: T,
) -> ProgramBuilderItem<'a, <S::Ty as Sampler>::Out, <(T::Expr, S::Expr) as ExprCombine>::Min>
where
    S::Ty: Sampler,
    (T::Expr, S::Expr): ExprCombine,
{
    let fmt_string = if shader::SCOPE_DERIVS.with(|x| x.get()) {
        "texture($, $)"
    } else {
        "textureLod($, $, 0)"
    };
    ProgramBuilderItem::create(
        VarString::format(fmt_string, vec![sampler.as_string(), arg.as_string()]),
        ItemRef::Expr,
    )
}

pub unsafe trait Sampler: ArgType {
    type Out: ArgType;
    type Arg: ArgType;
}

macro_rules! sampler {
    ($sampler:ident, $data_ty:expr, $data:ty, $arg:ty) => {
        #[derive(Clone, Copy)]
        pub struct $sampler {}

        unsafe impl ArgType for $sampler {
            fn data_type() -> DataType {
                $data_ty
            }
        }

        unsafe impl ArgParameter<ImageArgs> for $sampler {
            fn get_param() -> ImageArgs {
                ImageArgs
            }
        }

        unsafe impl Sampler for $sampler {
            type Out = $data;
            type Arg = $arg;
        }
    };
}

use super::shader::api::*;
use super::shader::{DataType, ItemRef, VarString};

sampler!(Sampler2D, DataType::Sampler2D, Float4, Float2);
sampler!(IntSampler2D, DataType::IntSampler2D, Int4, Float2);
sampler!(UIntSampler2D, DataType::UIntSampler2D, UInt4, Float2);
