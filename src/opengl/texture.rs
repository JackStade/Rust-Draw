use super::gl;
use super::shader;
use super::{inner_gl, inner_gl_unsafe, inner_gl_unsafe_static, GlDraw, GlDrawCore};
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

unsafe impl TextureComponents for RGBA {
    const FORMAT: GLuint = gl::RGBA;
    const COMPONENTS: u32 = 4;
}

pub unsafe trait TextureBufferType {
    type Data: GlDataType;
    type Components: TextureComponents;
}

pub unsafe trait TextureFormat: TextureBufferType {
    type Texture2D: ArgType;

    const INTERNAL_FORMAT: GLuint;
}

/// A normalized floating point texture.
///
/// The data is stored as an int, but converted to a float when accessed by the GPU. The data will
/// be mapped to [0, 1] for unsigned types and [-1, 1] for signed types.
///
/// `Data` can be one of:
/// * 'u8'
/// * 'i8'
/// * 'u16'
/// * 'i16'
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
);

/// An integer texture format, for when it is useful to be able to read integer values directly
/// in shaders.
///
/// `Data` can be one of:
/// * 'u8'
/// * 'i8'
/// * 'u16'
/// * 'i16'
/// * 'u32'
/// * 'i32'
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
    RGBA,
    IntSampler2D,
    i8,
    gl::RGBA8I,
    i16,
    gl::RGBA16I,
    i32,
    gl::RGBA32I,
);

/// A floating point texture format. The data is stored as a float and will not be altered when
/// accessed.
pub struct FloatTextureData<C: TextureComponents> {
    phantom: PhantomData<C>,
}

unsafe impl<C: TextureComponents> TextureBufferType for FloatTextureData<C> {
    type Data = f32;
    type Components = C;
}

unsafe impl TextureFormat for FloatTextureData<RGBA> {
    type Texture2D = FloatSampler2D;

    const INTERNAL_FORMAT: GLuint = gl::RGBA32F;
}

#[derive(Clone, Copy)]
pub struct DepthStencil24_8 {
    data: u32,
}

pub unsafe trait BindTexture<T> {
    unsafe fn bind(&self, gl: &Gl);
}

use super::GlResource;

pub struct Texture2D<F: TextureFormat> {
    image_id: u32,

    width: u32,
    height: u32,
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
        _context: &super::GlWindow,
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
    // the arguments to this function cannot come from a non-main thread
    let fmt_string = if unsafe { shader::SCOPE_DERIVS } {
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
use super::shader::{DataType, ItemRef, ProgramItem, VarString};

sampler!(Sampler2D, DataType::Sampler2D, Float4, Float2);
sampler!(IntSampler2D, DataType::IntSampler2D, Int4, Float2);
sampler!(UIntSampler2D, DataType::UIntSampler2D, UInt4, Float2);
sampler!(FloatSampler2D, DataType::FloatSampler2D, Float4, Float2);
