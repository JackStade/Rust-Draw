#![recursion_limit = "256"]

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
use shader::api::*;
use shader::traits::{Constant, ExprMin, IntoArgs, Varying};
use std::time::{Duration, Instant};
use swizzle::SwizzleInPlace;

/// A coordinate space.
///
/// Unless otherwise stated, up is positive y and positive z is coming out of the screen.
/// Positive x is always from left to right.
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
    /// The viewport is a 2x2 unit square with size equal to the width of the window. Origin is the center.
    WindowWidth,
    /// The viewport is a 2x2 unit square with size equal to the height of the window. Origin is the center.
    WindowHeight,
    /// The viewport is a 2x2 unit square size equal to max(width, height). Origin is the center.
    SquareFill,
    /// The viewport is a 2x2 unit square with size equal to min(width, height). Origin is the center.
    SquareFit,
}

impl CoordinateSpace {
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
            CoordinateSpace::WindowCenter => ([1.0, 1.0, 1.0], [0.0, 0.0, 0.0]),
            CoordinateSpace::WindowTopLeft => ([2.0, -2.0, 1.0], [-1.0, 1.0, 1.0]),
            CoordinateSpace::WindowBottomLeft => ([2.0, 2.0, 1.0], [-1.0, -1.0, 1.0]),
            CoordinateSpace::WindowWidth => ([1.0, width / height, 1.0], [0.0, 0.0, 0.0]),
            CoordinateSpace::WindowHeight => ([height / width, 1.0, 1.0], [0.0, 0.0, 0.0]),
            CoordinateSpace::SquareFit => ([min / width, min / height, 1.0], [0.0, 0.0, 0.0]),
            CoordinateSpace::SquareFill => ([max / width, max / height, 1.0], [0.0, 0.0, 0.0]),
        };
        (diag, homo)
    }
}
