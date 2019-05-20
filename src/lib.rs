#![recursion_limit = "256"]

pub mod color;
pub mod opengl;

#[allow(non_snake_case)]
pub mod tuple;

use nalgebra as na;

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
                [
                    2.0 * scale / width,
                    2.0 * -scale / height,
                    2.0 * scale / max,
                ],
                [-1.0, 1.0, 0.0],
            ),
            CoordinateSpace::PixelsBottomLeft => (
                [2.0 * scale / width, 2.0 * scale / height, 2.0 * scale / max],
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

pub mod internals {
    use std::hash::{Hash, Hasher};
    use std::marker::PhantomData;

    /// Similar to a reference counted pointer. This is used to make internal
    /// optimizations. It is exposed to the public api so that these optimizations
    /// can be used by extension libraries.
    ///
    /// The pointer stored by an ID will never change value as long as the ID is alive,
    /// so it can be used as an index into a hash map.
    pub struct ID {
        alive: *mut isize,
        phantom: PhantomData<std::rc::Rc<()>>,
    }

    // this is equal to a usize with only the highest bit set
    const UBIT: usize = std::isize::MIN as usize;

    impl ID {
        pub fn new() -> ID {
            let b = Box::new(0);
            let ptr = Box::into_raw(b);
            ID {
                alive: ptr,
                phantom: PhantomData,
            }
        }

        pub fn get_weak(&self) -> WeakID {
            unsafe {
                let val = self.alive.read();
                if (val as usize & !UBIT) + 1 == std::isize::MAX as usize {
                    panic!("Cannot allocate more than isize::MAX weak IDs.");
                }
                // this cannot overflow.
                self.alive.write(val + 1);
            }
            WeakID {
                alive: self.alive,
                phantom: PhantomData,
            }
        }

        pub fn alive(&self) -> bool {
            let count = unsafe { self.alive.read() as usize };
            count & UBIT == 0
        }
    }

    impl Hash for ID {
        fn hash<H: Hasher>(&self, state: &mut H) {
            // note: casts the pointer, which does not change
            state.write_usize(self.alive as usize);
        }
    }

    impl Drop for ID {
        fn drop(&mut self) {
            unsafe {
                let count = self.alive.read() as usize;
                if count as usize & !UBIT == 0 {
                    // drop the pointer
                    let _drop_box = Box::from_raw(self.alive);
                } else {
                    self.alive.write((count ^ UBIT) as isize);
                }
            }
        }
    }

    /// A weak pointer to an ID.
    pub struct WeakID {
        alive: *mut isize,
        phantom: PhantomData<std::rc::Rc<()>>,
    }

    impl Hash for WeakID {
        fn hash<H: Hasher>(&self, state: &mut H) {
            // note: casts the pointer, not the data behind the pointer,
            // which does not change
            state.write_usize(self.alive as usize);
        }
    }

    impl Clone for WeakID {
        fn clone(&self) -> WeakID {
            unsafe {
                let val = self.alive.read();
                if (val as usize & !UBIT) + 1 == std::isize::MAX as usize {
                    panic!("Cannot allocate more than isize::MAX weak IDs.");
                }
                // this cannot overflow.
                self.alive.write(val + 1);
            }
            WeakID {
                alive: self.alive,
                phantom: PhantomData,
            }
        }
    }

    impl PartialEq for WeakID {
        fn eq(&self, other: &WeakID) -> bool {
            self.alive as usize == other.alive as usize
        }
    }

    impl Eq for WeakID {}

    impl WeakID {
        pub fn alive(&self) -> bool {
            let count = unsafe { self.alive.read() as usize };
            count & UBIT == 0
        }
    }

    impl Drop for WeakID {
        fn drop(&mut self) {
            unsafe {
                let count = self.alive.read();
                // this can't over or under flow
                self.alive.write(count - 1);
                // isize::MIN means that the parent ID has been dropped and
                // there are no remaining Weak pointers
                if count == std::isize::MIN {
                    // drop the pointer
                    let _drop_box = Box::from_raw(self.alive);
                }
            }
        }
    }

    // until `alloc` gets stabilized, this will have to do
    #[inline]
    pub(crate) unsafe fn reallocate<T>(ptr: *mut T, old_len: usize, new_len: usize) -> *mut T {
        let mut vec = Vec::from_raw_parts(ptr, 0, old_len);
        vec.reserve_exact(new_len);
        let ptr = vec.as_mut_ptr();
        std::mem::forget(vec);
        ptr
    }
}
