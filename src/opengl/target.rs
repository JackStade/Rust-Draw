use super::gl;
use super::gl::Gl;
use super::shader::{api::Float4, traits::ShaderArgs};

pub unsafe trait RenderTarget<T: ShaderArgs> {
    /// Offscreen targets allow rendering to the currently active window, but rendering to a
    /// window's framebuffer directly requires activating that windows. This function
    /// activates the window in that case. And also otherwise prepares the target for
    /// rendering.
    unsafe fn bind_target(&self) -> &Gl;

    fn get_depth_bits(&self) -> u8;

    fn get_stencil_bits(&self) -> u8;
}

unsafe impl RenderTarget<(Float4,)> for super::GlWindow {
    unsafe fn bind_target(&self) -> &Gl {
        super::activate_window(self);
        self.gl.BindFramebuffer(gl::FRAMEBUFFER, 0);
        &*self.gl
    }

    fn get_depth_bits(&self) -> u8 {
        24
    }

    fn get_stencil_bits(&self) -> u8 {
        8
    }
}
