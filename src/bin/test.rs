extern crate rust_draw as draw;

extern crate nalgebra as na;

use draw::opengl::shader;
use draw::opengl::shader::api::*;
use std::time::{Duration, Instant};
use draw::swizzle::SwizzleInPlace;
use draw::opengl::texture;
use draw::color;
use draw::opengl;
use draw::CoordinateSpace;

fn main() {
    let mut gl = unsafe { opengl::get_gl().unwrap() };

    let mut tex = Vec::with_capacity(4 * 128 * 128);
    for i in 0..128 {
        for k in 0..128 {
            if (i % 20 < 10) ^ (k % 20 < 10) {
                tex.extend_from_slice(&[255, 0, 255, 255]);
            } else {
                tex.extend_from_slice(&[0, 255, 255, 255]);
            }
        }
    }

    // let image = gl.load_image(128, 128, &tex[..]);
    let mut window = gl.new_window(800, 800, CoordinateSpace::PixelsTopLeft, "Test Window");
    let image =
        opengl::texture::Texture2D::<texture::TextureData<texture::RGBA, u8>>::new(
            &window, 128, 128, &tex,
        );
    gl.draw(&window);
    let c = window.get_window_pos(0.5, 0.5);
    let n = window.get_window_pos(0.75, 0.75);
    for i in 0..300 {
        std::thread::sleep(std::time::Duration::new(0, 10));
        window.background(color::Color8Bit {
            color: [255, 255, 255, 255],
        });
        let c = window.get_window_pos(0.5, 0.5);
        let n = window.get_window_pos(0.75, 0.75);
        // window.draw_triangle([0.0, 1.0, 1.0, 1.0], [c.0, c.1, 0.0, n.0, c.1, 0.0, c.0, n.1, 0.0]);
        window.draw_image(&image, c, n);

        gl.draw(&window);
    }
    std::mem::drop(window);
}
