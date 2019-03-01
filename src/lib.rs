use std::cmp::{max, min};
use std::fmt;
use std::num::FpCategory;

pub mod color;
pub mod opengl;
/// The swizzle module contains functions for safely reordering tuples and arrays
/// with compile time type checking.
#[allow(non_snake_case)]
pub mod swizzle;

use nalgebra as na;
use opengl::shader;
use shader::traits::*;
use swizzle::SwizzleInPlace;

pub fn test_window() {
    /*println!(
        "{}",
        shader::create_shader_string(|input: (shader::Float2, shader::Float3), uniforms: ()| {
            let s = input.1 / shader::float((3.0,));
            (input
                .0
                .map((shader::swizzle::r, shader::swizzle::g, shader::swizzle::r))
                + s
                + shader::float3((1.0, 1.0, 1.0)),)
        })
    );*/
    let mut array = [vec![0; 5], vec![1; 2], vec![3]];
    array.swizzle((swizzle::a, swizzle::c, swizzle::b));
    println!("{}", array[1][0]);
    let mut gl = opengl::get_gl().unwrap();
    let mut window = gl.new_window(800, 800, CoordinateSpace::PixelsTopLeft, "Test Window");
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

    let image = gl.load_image(128, 128, &tex[..]);
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
}

/// A coordinate space. Unless otherwise stated, up is positive y and positive z is coming out of the screen.
#[derive(Clone, Copy)]
pub enum CoordinateSpace {
    /// The coordinate origin is the center of the window. One unit is one pixel.
    PixelsCenter,
    /// The coordinate origin is the top left corner. Down is positive y. One unit is one pixel.
    PixelsTopLeft,
    /// The coordinate origin is the bottom left corner. One unit is one pixel.
    PixelsBottomLeft,
    /// The coordinate origin is the center. One unit in any direction reaches the edges of the window.
    WindowCenter,
    /// The coordinate origin is the top left. One unit in any direction reaches the edges of the window.
    WindowTopLeft,
    /// The coordinate origin is the bottom left. One unit in any direction reaches the edges of the window.
    WindowBottomLeft,
    /// The viewport is a 1x1 unit square with size equal to the width of the window. Origin is the center.
    ///
    /// Note that `WindowWidth`, `WindowHeight`, `SquareFill`, and `SquareFit` can cut off a portion of the screen.
    WindowWidth,
    /// The viewport is a 1x1 unit square with size equal to the height of the window. Origin is the center.
    ///
    /// Note that `WindowWidth`, `WindowHeight`, `SquareFill`, and `SquareFit` can cut off a portion of the screen.
    WindowHeight,
    /// The viewport is a 1x1 unit square size equal to max(width, height)
    ///
    /// Note that `WindowWidth`, `WindowHeight`, `SquareFill`, and `SquareFit` can cut off a portion of the screen.
    SquareFill,
    /// The viewport is a 1x1 unit square with size equal to min(width, height)
    ///
    /// Note that `WindowWidth`, `WindowHeight`, `SquareFill`, and `SquareFit` can cut off a portion of the screen.
    SquareFit,
}

impl CoordinateSpace {
    fn get_viewport(&self, width: i32, height: i32, scale: i32) -> (i32, i32, i32, i32) {
        let width = width * scale;
        let height = height * scale;
        let max = width.max(height);
        let min = width.min(height);
        match self {
            CoordinateSpace::WindowWidth => (0, (height - width) / 2, width, width),
            CoordinateSpace::WindowHeight => ((width - height) / 2, 0, height, height),
            CoordinateSpace::SquareFill => ((width - max) / 2, (height - max) / 2, max, max),
            CoordinateSpace::SquareFit => ((width - min) / 2, (height - min) / 2, min, min),
            _ => (0, 0, width, height),
        }
    }

    fn get_matrix(&self, width: i32, height: i32, scale: i32) -> na::Matrix4<f32> {
        let (diag, homo) = self.get_transform_data(width, height, scale);
        // column-major matrix
        na::Matrix4::<f32>::from_column_slice(&[
            diag[0], 0.0, 0.0, 0.0, //
            0.0, diag[1], 0.0, 0.0, //
            0.0, 0.0, diag[2], 0.0, //
            homo[0], homo[1], homo[2], 1.0, //
        ])
    }

    fn get_transform_data(&self, width: i32, height: i32, scale: i32) -> ([f32; 3], [f32; 3]) {
        let width = width as f32;
        let height = height as f32;
        let scale = scale as f32;
        let max = width.max(height);
        let min = width.min(height);
        let (diag, homo) = match self {
            CoordinateSpace::PixelsCenter => (
                [2.0 * scale / width, 2.0 * scale / height, 2.0 * scale / max],
                [0.0, 0.0, 0.0],
            ),
            CoordinateSpace::PixelsTopLeft => (
                [scale / width, -scale / height, 2.0 * scale / max],
                [-1.0, 1.0, 0.0],
            ),
            CoordinateSpace::PixelsBottomLeft => (
                [scale / width, scale / height, 2.0 * scale / max],
                [-1.0, -1.0, 0.0],
            ),
            CoordinateSpace::WindowTopLeft => ([2.0, -2.0, 1.0], [-1.0, 1.0, 1.0]),
            CoordinateSpace::WindowBottomLeft => ([2.0, 2.0, 1.0], [-1.0, -1.0, 1.0]),
            _ => ([1.0, 1.0, 1.0], [0.0, 0.0, 0.0]),
        };
        (diag, homo)
    }

    // Gets a position on the window. (0.0, 0.0) is bottom left and (1.0, 1.0) is top right
    fn get_window_pos(&self, x: f32, y: f32, width: i32, height: i32, scale: i32) -> (f32, f32) {
        let (diag, homo) = self.get_transform_data(width, height, scale);
        let viewport = self.get_viewport(width, height, scale);
        let viewport = (
            viewport.0 as f32,
            viewport.1 as f32,
            viewport.2 as f32,
            viewport.3 as f32,
        );
        let width = width as f32 * scale as f32;
        let height = height as f32 * scale as f32;
        let px = x * width;
        let py = y * height;
        let viewport_position = (px - viewport.0, py - viewport.1);
        let pos = (
            viewport_position.0 / viewport.2,
            viewport_position.1 / viewport.3,
        );
        let pos = (pos.0 * 2.0 - 1.0, pos.1 * 2.0 - 1.0);
        let pos = ((pos.0 - homo[0]) / diag[0], (pos.1 - homo[1]) / diag[1]);
        pos
    }

    // Gets a position on the viewport without applying the matrix inverse
    fn get_viewport_pos(&self, x: f32, y: f32, width: i32, height: i32, scale: i32) -> (f32, f32) {
        let (diag, homo) = self.get_transform_data(width, height, scale);
        let viewport = self.get_viewport(width, height, scale);
        let viewport = (
            viewport.0 as f32,
            viewport.1 as f32,
            viewport.2 as f32,
            viewport.3 as f32,
        );
        let width = width as f32 * scale as f32;
        let height = height as f32 * scale as f32;
        let px = x * width;
        let py = y * height;
        let viewport_position = (px - viewport.0, py - viewport.1);
        let pos = (
            viewport_position.0 / viewport.2,
            viewport_position.1 / viewport.3,
        );
        pos
    }
}

/// A type specifying how to interpret mesh data
#[derive(PartialEq, Eq, Clone, Copy)]
pub enum VertexDataType {
    Float,
    IntU8,
    IntI8,
    IntU16,
    IntI16,
    IntU32,
    IntI32,
    NormU8,
    NormI8,
    NormU16,
    NormI16,
    /// Note: normalizing 32 bit integers causes significant precision loss
    NormU32,
    NormI32,
}

pub trait Drawer {
    type Window: RenderWindow;

    /// Create a new window.
    fn new_window(&mut self, width: u32, height: u32) -> Self::Window;

    fn max_windows(&self) -> usize;

    /// Draws the window, updating the framebuffer in the window. The window will not
    /// be updated until this is called.
    fn draw(&mut self);
}

pub trait RenderWindow {}

pub struct Triangle3D {
    verts: [f32; 9],
}

/// A basic 2 dimensional polygon with vertices specified by the user
#[derive(Clone)]
pub struct UncheckedPoly {
    verts: Vec<(f32, f32)>,
}

/// A 2 dimensional polygon that enforces certain properties to be
/// true, namely that the polygon must be nonintersecting and have
/// CCW winding order.
pub struct Poly {
    verts: Vec<(f32, f32)>,
}

pub enum PolyCheckError {
    FloatError(UncheckedPoly),
    NonCCW(UncheckedPoly),
    Intersecting(UncheckedPoly),
}

impl fmt::Debug for PolyCheckError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            PolyCheckError::FloatError(_) => {
                write!(f, "One of the vertex coordinates is NaN or infinity")
            }
            PolyCheckError::NonCCW(_) => {
                write!(f, "The polygon not CCW, try calling `reverse_verts`")
            }
            PolyCheckError::Intersecting(_) => {
                write!(f, "The polygon intersects itself, try calling 'fix'")
            }
        }
    }
}

/// A triangle
pub struct Tri {
    verts: [(f32, f32); 3],
}

impl UncheckedPoly {
    pub fn new(verts: Vec<(f32, f32)>) -> Poly {
        Poly { verts: verts }
    }

    /// Check the poly, returning a checked poly if successful
    pub fn check(self) -> Result<Poly, PolyCheckError> {
        let n = self.verts.len();
        for i in 0..n {
            let vert = self.verts[i];
            let x_check = vert.0.classify();
            let y_check = vert.0.classify();
            // check to make sure that the coordinates are usable floats
            if x_check == FpCategory::Nan
                || x_check == FpCategory::Infinite
                || y_check == FpCategory::Nan
                || y_check == FpCategory::Infinite
            {
                return Err(PolyCheckError::FloatError(self));
            }
        }
        let mut area = 0.0;
        for i in 0..n {
            let vert = self.verts[i];
            let ni = (i + n) % n;
            let nvert = self.verts[ni];
            area += (nvert.0 - vert.0) * (vert.1 + nvert.1) / 2.0;
            // check for intersections
            for k in 0..n {
                if k != i && k != ni {
                    let v1 = self.verts[i];
                    let v2 = self.verts[(i + 1) % n];
                    if segment_intersect(vert, nvert, v1, v2) {
                        return Err(PolyCheckError::Intersecting(self));
                    }
                }
            }
        }
        if area < 0.0 {
            return Err(PolyCheckError::NonCCW(self));
        }
        Ok(Poly { verts: self.verts })
    }

    /// Reverse the verts. If a polygon fails to check because it is non CCW, then this will fix it
    pub fn reverse_verts(&mut self) {
        let n = self.verts.len();
        for i in 0..n / 2 {
            self.verts.swap(i, n - i - 1);
        }
    }

    pub fn fix(self) -> Vec<Poly> {
        panic!("Not implemented yet");
    }
}

// test if two line segments intersect
fn segment_intersect(a1: (f32, f32), a2: (f32, f32), b1: (f32, f32), b2: (f32, f32)) -> bool {
    line_test(a1, a2, b1, b2) && line_test(b1, b2, a1, a2)
}

// test if one line straddles another
fn line_test(a1: (f32, f32), a2: (f32, f32), b1: (f32, f32), b2: (f32, f32)) -> bool {
    let v1 = (b1.0 - a1.0, b1.1 - a1.1);
    let v2 = (b2.0 - a1.0, b2.1 - a1.1);
    let v = (a2.0 - a1.0, a2.1 - a1.1);
    let t1 = v.0 * v1.1 - v.1 * v1.0;
    let t2 = v.0 * v2.1 - v.1 * v2.0;
    (t2 < 0.0) ^ (t1 < 0.0)
}
