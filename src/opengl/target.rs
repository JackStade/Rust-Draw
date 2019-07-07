use super::gl;
use super::inner_gl_unsafe_static;
use super::shader::{
    api::*,
    traits::{ArgParameter, ArgType, OutputArgs, ShaderArgs, ShaderArgsClass},
};
use super::texture;
use super::GlWindow;
use crate::color;
use crate::tuple::{AttachFront, RemoveFront};
use gl::types::*;
use gl::Gl;
use texture::*;

pub unsafe trait RenderTarget<T: ShaderArgs> {
    /// Offscreen targets allow rendering to the currently active window, but rendering to a
    /// window's framebuffer directly requires activating that windows. This function
    /// activates the window in that case, and also otherwise prepares the target for
    /// rendering.
    unsafe fn bind_target(&mut self) -> &Gl;

    fn get_depth_bits(&self) -> u8;

    fn get_stencil_bits(&self) -> u8;

    /// A provided function that calls `glClear` on the render target
    fn background<C: color::Color>(&mut self, color: C) {
        let c = color.as_rgba();
        unsafe {
            let gl = self.bind_target();
            gl.ClearColor(c[0], c[1], c[2], c[3]);
            gl.Clear(gl::COLOR_BUFFER_BIT);
        }
    }

    fn clear_depth(&mut self, clear: f32) {
        unsafe {
            let gl = self.bind_target();
            gl.ClearDepth(clear as f64);
            gl.Clear(gl::DEPTH_BUFFER_BIT);
        }
    }

    fn clear_stencil(&mut self, clear: u8) {
        unsafe {
            let gl = self.bind_target();
            gl.ClearStencil(clear as i32);
            gl.Clear(gl::STENCIL_BUFFER_BIT);
        }
    }

    fn clear_depth_stencil(&mut self, depth: f32, clear: u8) {
        unsafe {
            let gl = self.bind_target();
            gl.ClearStencil(clear as i32);
            gl.ClearDepth(depth as f64);
            gl.Clear(gl::STENCIL_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);
        }
    }

    fn clear_depth_stencil_color<C: color::Color>(&mut self, depth: f32, clear: u8, color: C) {
        unsafe {
            let gl = self.bind_target();
            gl.ClearStencil(clear as i32);
            gl.ClearDepth(depth as f64);
            let c = color.as_rgba();
            gl.ClearColor(c[0], c[1], c[2], c[3]);
            gl.Clear(gl::COLOR_BUFFER_BIT | gl::STENCIL_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);
        }
    }
}

unsafe impl<'a> RenderTarget<(Float4,)> for super::WindowSurface<'a> {
    unsafe fn bind_target(&mut self) -> &Gl {
        super::activate_window(self.window);
        self.window.gl.BindFramebuffer(gl::FRAMEBUFFER, 0);
        let (width, height) = self.window.get_width_height();
        let scl = self.window.get_scale();
        self.window.gl.Viewport(0, 0, width * scl, height * scl);
        &*self.window.gl
    }

    fn get_depth_bits(&self) -> u8 {
        24
    }

    fn get_stencil_bits(&self) -> u8 {
        8
    }
}

pub(crate) mod map {
    use super::gl::types::*;
    use fnv::FnvHashMap;
    use glfw::ffi as glfw_raw;

    static mut FBO_NUM: usize = 1;

    /// This function must be called on the thread that owns the GlDraw
    pub(crate) unsafe fn get_fbo_id() -> usize {
        let n = FBO_NUM;
        FBO_NUM += 1;
        n
    }

    static mut FBO_MAP: *mut FnvHashMap<(usize, usize), GLuint> = 0 as *mut _;

    /// Add the vao to the map. This vao will be bound to the current window.
    #[inline]
    pub(crate) unsafe fn add_fbo(id: usize, fbo: GLuint) {
        if FBO_MAP.is_null() {
            let b = Box::new(
                FnvHashMap::<(usize, usize), GLuint>::with_capacity_and_hasher(
                    4,
                    Default::default(),
                ),
            );
            FBO_MAP = Box::into_raw(b);
        }
        (*FBO_MAP).insert((id, crate::opengl::CURRENT_WINDOW as usize), fbo);
    }

    /// Search for a vao corresponding to the current window and
    /// the id given.
    #[inline]
    pub(crate) unsafe fn get_fbo(id: usize) -> Option<GLuint> {
        if FBO_MAP.is_null() {
            return None;
        }
        (*FBO_MAP)
            .get(&(id, crate::opengl::CURRENT_WINDOW as usize))
            .map(|fbo| *fbo)
    }

    /// Clear all the vaos owned by a certain mesh. This should be
    /// called when the mesh is dropped.
    #[inline]
    pub(crate) unsafe fn clear_fbos(id: usize) {
        if !FBO_MAP.is_null() {
            (*FBO_MAP).retain(|key, _| key.0 != id);
        }
    }

    #[inline]
    pub(crate) unsafe fn clear_window_fbos(window_ptr: *mut glfw_raw::GLFWwindow) {
        if !FBO_MAP.is_null() {
            (*FBO_MAP).retain(|key, _| key.1 != window_ptr as usize);
        }
    }
}

pub unsafe trait DepthStencilAttachment {
    unsafe fn bind_attachments(&mut self, gl: &Gl);

    fn width_height(&self) -> (u32, u32);

    fn depth_bits(&self) -> u8;

    fn stencil_bits(&self) -> u8;
}

/// It is unsafe to implement this trait for a type that isn't
/// a reference (or a Copy type). Implementing this trait
/// for non-copy or non reference types can (and will) cause aliased mutables
/// and double drop errors.
pub unsafe trait FBOAttachments {
    type Args: ShaderArgsClass<OutputArgs>;

    unsafe fn bind_attachments<F: FnMut() -> u32>(&mut self, gl: &Gl, bindings: F);

    /// this takes a mutable reference for some kind of complicated reasons.
    fn width_height(&mut self) -> (u32, u32);
}

unsafe impl<'a, F: texture::TargetTexture> FBOAttachments for &'a mut texture::Texture2D<F>
where
    F::Target: ArgParameter<OutputArgs> + ArgType,
{
    type Args = (F::Target,);

    #[inline]
    unsafe fn bind_attachments<FN: FnMut() -> u32>(&mut self, gl: &Gl, mut bindings: FN) {
        let gl_draw = inner_gl_unsafe_static();
        gl.FramebufferTexture(
            gl::FRAMEBUFFER,
            gl::COLOR_ATTACHMENT0 + bindings(),
            gl_draw.resource_list[self.image_id as usize],
            0,
        );
    }

    #[inline]
    fn width_height(&mut self) -> (u32, u32) {
        (self.width, self.height)
    }
}

unsafe impl<'a, F: 'a + FBOAttachments, T: RemoveFront<Front = F>> FBOAttachments for T
where
    T::Remaining: FBOAttachments,
    F::Args: RemoveFront<Remaining = ()>,
    <T::Remaining as FBOAttachments>::Args: AttachFront<<F::Args as RemoveFront>::Front>,
    <<T::Remaining as FBOAttachments>::Args as AttachFront<<F::Args as RemoveFront>::Front>>::AttachFront:
        ShaderArgsClass<OutputArgs>,
{
    type Args = <<T::Remaining as FBOAttachments>::Args as AttachFront<<F::Args as RemoveFront>::Front>>::AttachFront;

    #[inline]
    unsafe fn bind_attachments<FN: FnMut() -> u32>(&mut self, gl: &Gl, mut bindings: FN) {
        // using `transmute` instead of `as` destroys the reference, preventing aliased mutables
        // amazingly, doing this is ok. Rust allows taking a reference from a tuple of references,
        // even though mutable references aren't copy, and the other mutable references still exist
        // elsewhere in memory. Since these mutable references are in a mutably borrowed state, the
        // aliasing rule doesn't seem to apply. It is neccesary to use transmute though so that the
        // copy of the references and the original reference don't exist at the same time
        let t = std::mem::transmute::<_, *const Self>(self).read();
        let (mut front, mut remaining) = t.remove_front();
        let b = bindings();
        front.bind_attachments(gl, || b);
        remaining.bind_attachments(gl, bindings);
    }

    #[inline]
    fn width_height(&mut self) -> (u32, u32) {
        unsafe {
            // this is why this function takes a mutable ref. See the paragraph above.
            let t = std::mem::transmute::<_, *const Self>(self).read();
            let (mut front, mut remaining) = t.remove_front();
            let (mut w, mut h) = front.width_height();
            let (width, height) = remaining.width_height();
            if width < w {
                w = width;
            }
            if height < h {
                h = height;
            }
            (w, h)
        }
    }
}

unsafe impl FBOAttachments for () {
    type Args = ();

    #[inline(always)]
    unsafe fn bind_attachments<F: FnMut() -> u32>(&mut self, _gl: &Gl, _bindings: F) {
        // do nothing
    }

    #[inline(always)]
    fn width_height(&mut self) -> (u32, u32) {
        (-1i32 as u32, -1i32 as u32)
    }
}

pub struct Framebuffer<T: FBOAttachments, D: DepthStencilAttachment> {
    id: usize,
    colors: T,
    depth_stencil: D,
    width_height: (u32, u32),
}

impl<T: FBOAttachments, D: DepthStencilAttachment> Drop for Framebuffer<T, D> {
    fn drop(&mut self) {
        unsafe {
            map::clear_fbos(self.id);
        }
    }
}

impl<T: FBOAttachments, D: DepthStencilAttachment> Framebuffer<T, D> {
    pub fn new(_context: super::ContextKey, mut t: T, d: D) -> Framebuffer<T, D> {
        let (mut w, mut h) = t.width_height();
        let (width, height) = d.width_height();
        if width < w {
            w = width;
        }
        if height < h {
            h = height;
        }
        Framebuffer {
            id: unsafe { map::get_fbo_id() },
            colors: t,
            depth_stencil: d,
            width_height: (w, h),
        }
    }
}

unsafe impl<T: FBOAttachments, D: DepthStencilAttachment> RenderTarget<T::Args>
    for Framebuffer<T, D>
{
    unsafe fn bind_target(&mut self) -> &Gl {
        let gl = &*gl::CURRENT;
        let fbo = if let Some(fbo) = map::get_fbo(self.id) {
            gl.BindFramebuffer(gl::FRAMEBUFFER, fbo);
            fbo
        } else {
            let mut fbo = 0;
            gl.GenFramebuffers(1, &mut fbo);
            map::add_fbo(self.id, fbo);
            gl.BindFramebuffer(gl::FRAMEBUFFER, fbo);
            let mut i = 0;
            self.colors.bind_attachments(gl, || {
                let x = i;
                i += 1;
                x
            });
            self.depth_stencil.bind_attachments(gl);
            fbo
        };
        gl.Viewport(0, 0, self.width_height.0 as i32, self.width_height.1 as i32);
        gl
    }

    fn get_depth_bits(&self) -> u8 {
        self.depth_stencil.depth_bits()
    }

    fn get_stencil_bits(&self) -> u8 {
        self.depth_stencil.stencil_bits()
    }
}

unsafe impl DepthStencilAttachment for () {
    #[inline(always)]
    unsafe fn bind_attachments(&mut self, _gl: &Gl) {
        // do nothing
    }

    #[inline(always)]
    fn depth_bits(&self) -> u8 {
        0
    }

    #[inline(always)]
    fn stencil_bits(&self) -> u8 {
        0
    }

    #[inline(always)]
    fn width_height(&self) -> (u32, u32) {
        (-1i32 as u32, -1i32 as u32)
    }
}

macro_rules! ds_attachment {
    ($f:ty, $db:expr, $sb:expr, $attach:expr) => {
        unsafe impl<'a> DepthStencilAttachment for &'a mut Texture2D<$f> {
            unsafe fn bind_attachments(&mut self, gl: &Gl) {
                let gl_draw = inner_gl_unsafe_static();
                gl.FramebufferTexture(
                    gl::FRAMEBUFFER,
                    $attach,
                    gl_draw.resource_list[self.image_id as usize],
                    0,
                );
            }

            #[inline(always)]
            fn depth_bits(&self) -> u8 {
                $db
            }

            #[inline(always)]
            fn stencil_bits(&self) -> u8 {
                $sb
            }

            #[inline]
            fn width_height(&self) -> (u32, u32) {
                (self.width, self.height)
            }
        }
    };
}

ds_attachment!(Depth16TextureData, 16, 0, gl::DEPTH_ATTACHMENT);
ds_attachment!(Depth24TextureData, 24, 0, gl::DEPTH_ATTACHMENT);
ds_attachment!(Depth32FloatTextureData, 32, 0, gl::DEPTH_ATTACHMENT);
ds_attachment!(
    Depth24Stencil8TextureData,
    24,
    8,
    gl::DEPTH_STENCIL_ATTACHMENT
);
ds_attachment!(
    Depth32FStencil8TextureData,
    32,
    8,
    gl::DEPTH_STENCIL_ATTACHMENT
);
