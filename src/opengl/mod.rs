use glfw;

use nalgebra as na;
use std::cell::Cell;
use std::ffi::CString;
use std::fmt;
use std::marker::PhantomData;
use std::mem;
use std::os::raw::c_int;
use std::ptr;
use std::result::Result;
use std::str;
use std::thread;

use crate::color;
use crate::CoordinateSpace;

use self::glfw::ffi as glfw_raw;
use gl::types::*;
use gl::Gl;

/// The shader module contains lower level functions for creating and using shaders.
/// This is an advanced feature that can be difficult to use correctly.
///
/// Because it uses a complex system of types with many blanket implementations,
/// it is difficult to understand how this module works from the auto-doc. In
/// the future there will be a shader guide.
#[allow(non_snake_case)]
pub mod shader;

pub mod texture;

pub mod mesh;

pub mod target;

static mut GL_DRAW: *mut GlDrawCore = 0 as *mut _;

static mut CURRENT_WINDOW: *mut glfw_raw::GLFWwindow = 0 as *mut _;

static mut WINDOW_GLOBAL: WindowGlobal = WindowGlobal {
    num_windows: 0,
    total_windows: 0,
};

static mut DRAW_INIT: bool = false;

static mut GLFW_INIT: bool = false;

#[inline(always)]
#[allow(unused)]
fn inner_gl<'a>(_: &'a mut GlDraw) -> &'a mut GlDrawCore {
    unsafe { &mut *GL_DRAW }
}

#[inline(always)]
#[allow(unused)]
fn inner_gl_static<'a>(_: &'a GlDraw) -> &'a GlDrawCore {
    unsafe { &*GL_DRAW }
}

#[inline(always)]
#[allow(unused)]
unsafe fn inner_gl_unsafe<'a>() -> &'a mut GlDrawCore {
    &mut *GL_DRAW
}

#[inline(always)]
#[allow(unused)]
unsafe fn inner_gl_unsafe_static<'a>() -> &'a GlDrawCore {
    &*GL_DRAW
}

#[cfg(not(feature = "opengl41"))]
const GL_VERSION: i32 = 0;

#[cfg(all(not(feature = "opengl42"), feature = "opengl41"))]
const GL_VERSION: i32 = 1;

#[cfg(all(not(feature = "opengl43"), feature = "opengl42"))]
const GL_VERSION: i32 = 2;

#[cfg(all(not(feature = "opengl44"), feature = "opengl43"))]
const GL_VERSION: i32 = 3;

#[cfg(all(not(feature = "opengl45"), feature = "opengl44"))]
const GL_VERSION: i32 = 4;

#[cfg(all(not(feature = "opengl46"), feature = "opengl45"))]
const GL_VERSION: i32 = 5;

#[cfg(feature = "opengl46")]
const GL_VERSION: i32 = 6;

/// Initializes gl/glfw.
///
/// When only using this library, calling this function is completely safe. However, the gl/glfw api
/// is not specific to this crate, and use of either opengl or glfw by other crates can cause crashes
/// or unexpected behavior.
///
/// Calling this function is completely safe if no other crates that interact gl/glfw are being used.
pub unsafe fn get_gl() -> Result<GlDraw, InitError> {
    if let Some("main") = thread::current().name() {
    } else {
        return Err(InitError::NonMainThread);
    }
    let init = mem::replace(&mut GLFW_INIT, true);
    if init {
        return Err(InitError::AlreadyInitialized);
    }

    let draw_box = Box::new(GlDrawCore::new());
    GL_DRAW = Box::into_raw(draw_box);

    if glfw_raw::glfwInit() == glfw_raw::TRUE {
        // default opengl version
        glfw_raw::glfwWindowHint(glfw_raw::CONTEXT_VERSION_MAJOR, 4);
        glfw_raw::glfwWindowHint(glfw_raw::CONTEXT_VERSION_MINOR, GL_VERSION);
        // glfw options
        glfw_raw::glfwWindowHint(glfw_raw::OPENGL_FORWARD_COMPAT, true as c_int);
        glfw_raw::glfwWindowHint(
            glfw_raw::OPENGL_PROFILE,
            glfw::OpenGlProfileHint::Core as c_int,
        );
        glfw_raw::glfwWindowHint(glfw_raw::RESIZABLE, true as c_int);
        glfw_raw::glfwSetErrorCallback(Some(error_callback));
    } else {
        return Err(InitError::GlfwError);
    }

    Ok(GlDraw {
        phantom: PhantomData,
    })
}

pub enum InitError {
    NonMainThread,
    AlreadyInitialized,
    GlfwError,
}

impl fmt::Debug for InitError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            InitError::NonMainThread => write!(f, "GL must be initialized on the main thread."),
            InitError::AlreadyInitialized => write!(f, "GL has already been initialized."),
            InitError::GlfwError => write!(f, "An unknown error occured."),
        }
    }
}

struct WindowGlobal {
    num_windows: usize,
    total_windows: u32,
}

// the epoch is incremented every time resources are orphaned.
//
static mut EPOCH: u64 = 1;

struct GlDrawCore {
    // resources are objects that are shared between contexts, and have all relevant data stored
    // on the gpu. When all windows are closed it is neccesary to "orphan" these resources by
    // downloading the data to the cpu, so that the resources do not become invalid when a new window
    // opened.
    // These resources can only be used by functions that require a reference to a window, so
    // while there are no windows opened the resources are technically invalid, but cannot be used
    // during that time.
    // 0 represents an open space, since opengl functions cannot return 0 as a name. This array cannot
    // be reordered, since objects store references to the ids in it.
    resource_list: Vec<GLuint>,
    resource_search_start: usize,
    orphan_positions: Vec<GLuint>,
    // a list of the functions need to orphan/adopt each active resource
    resource_orphans: Vec<DynResource>,

    // a list of windows. This is used for resource sharing
    windows: Vec<*mut glfw_raw::GLFWwindow>,
}

const NUM_DRAW_BUFFERS: i32 = 3;

mod free_draw {
    use super::gl::types::*;
    use super::NUM_DRAW_BUFFERS;

    pub static mut DRAW_BUFFERS: [GLuint; NUM_DRAW_BUFFERS as usize] =
        [0; NUM_DRAW_BUFFERS as usize];
    pub static mut COLOR_SHADER: GLuint = 0;
    pub static mut TEX_SHADER: GLuint = 0;
    pub static mut COLOR_SHADER_COLOR: GLint = 0;
    pub static mut COLOR_SHADER_TRANSFORM: GLint = 0;
    pub static mut TEX_SHADER_TEX: GLint = 0;
    pub static mut TEX_SHADER_TRANSFORM: GLint = 0;
}

pub mod gl {
    use super::GlWindow;

    pub(super) static mut CURRENT: *const Gl = 0 as *const Gl;

    pub unsafe fn with_current<O, F: FnOnce(&Gl) -> O>(f: F) -> O {
        f(&*CURRENT)
    }

    /// Note: in order for opengl function calls to work correctly,
    /// the gl for the correct window must be used *and* that window
    /// must be currently active.
    ///
    /// On some operating system the function pointers for different windows
    /// may be the same, but this should not be relied upon. Calling a function
    /// pointer when that window is not active should be considered to be
    /// UB.
    pub unsafe fn get_gl<'a>(window: &'a GlWindow) -> &'a Gl {
        &window.gl
    }

    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

impl GlDrawCore {
    fn new() -> GlDrawCore {
        GlDrawCore {
            resource_list: Vec::new(),
            resource_search_start: 0,
            orphan_positions: Vec::new(),
            resource_orphans: Vec::new(),
            windows: Vec::new(),
        }
    }

    fn orphan_resources(&mut self) {
        for r in self.resource_orphans.iter_mut() {
            unsafe {
                r.orphan();
            }
        }
    }

    fn get_resource_generic<T: GlResource>(
        &mut self,
        name: GLuint,
        init_data: Option<*mut ()>,
    ) -> u32 {
        self.get_resource_id(
            name,
            T::adopt,
            T::drop_while_orphaned,
            T::cleanup,
            T::orphan,
            init_data,
        )
    }

    // note that a context is required in order to create gpu-side resources, so it is not
    // possible to create a resource in an orphan state
    fn get_resource_id(
        &mut self,
        name: GLuint,
        adopt_ptr: unsafe fn(*mut (), u32) -> Option<*mut ()>,
        drop_ptr: unsafe fn(*mut (), u32),
        cleanup_ptr: unsafe fn(*mut (), u32),
        orphan_ptr: unsafe fn(u32, *mut ()) -> *mut (),
        init_data: Option<*mut ()>,
    ) -> u32 {
        let mut found = None;
        let mut low_space = self.resource_search_start;
        for slot in self.resource_list[low_space..].iter_mut() {
            if *slot == 0 {
                *slot = name;
                self.orphan_positions[low_space] = self.resource_orphans.len() as u32;
                found = Some(low_space);
                break;
            }
            low_space += 1;
        }
        self.resource_search_start = low_space + 1;

        if found.is_none() {
            found = Some(self.resource_list.len());
            self.resource_list.push(name);
            self.orphan_positions
                .push(self.resource_orphans.len() as u32);
        }
        let found = found.unwrap();
        let ptr = match init_data {
            Some(p) => p,
            None => 0 as *mut (),
        };
        self.resource_orphans.push(DynResource {
            id: found as u32,
            ptr: ptr,
            adopt_ptr: adopt_ptr,
            drop_ptr: drop_ptr,
            cleanup_ptr: cleanup_ptr,
            orphan_ptr: orphan_ptr,
        });
        // this is theoretically a potential saftey violation, but all implementations will
        // crash long before the number of resources reaches 2^32
        found as u32
    }

    fn remove_resource(&mut self, id: u32) {
        let id = id as usize;
        if id < self.resource_search_start {
            self.resource_search_start = id;
        }
        self.resource_list[id] = 0;
        let pos = self.orphan_positions[id];
        // if the resource is in an orphan state, then a different destructor needs
        // to be called
        if unsafe { !DRAW_INIT } {
            unsafe { self.resource_orphans[pos as usize].drop_when_orphaned() };
        } else {
            unsafe { self.resource_orphans[pos as usize].cleanup() };
        }

        self.resource_orphans.swap_remove(pos as usize);

        // the previous last value is now in this position
        // if the value being removed is the last value in the array
        if (pos as usize) < self.resource_orphans.len() {
            let swap_id = self.resource_orphans[pos as usize].id;
            // update the indirect list
            self.orphan_positions[swap_id as usize] = pos;
        }
    }
}

pub trait GlResource {
    unsafe fn adopt(ptr: *mut (), id: u32) -> Option<*mut ()>;

    unsafe fn drop_while_orphaned(ptr: *mut (), id: u32);

    unsafe fn cleanup(ptr: *mut (), id: u32);

    unsafe fn orphan(id: u32, ptr: *mut ()) -> *mut ();
}

// note: this type has implicit state. It is either in an orphan state
// or an active state. Depending on the specific type of resource, the pointer
// may referece different data in these states.
struct DynResource {
    id: u32,
    ptr: *mut (),
    adopt_ptr: unsafe fn(*mut (), u32) -> Option<*mut ()>,
    drop_ptr: unsafe fn(*mut (), u32),
    cleanup_ptr: unsafe fn(*mut (), u32),
    orphan_ptr: unsafe fn(u32, *mut ()) -> *mut (),
}

impl DynResource {
    unsafe fn drop_when_orphaned(&mut self) {
        (self.drop_ptr)(self.ptr, self.id)
    }

    unsafe fn orphan(&mut self) {
        self.ptr = (self.orphan_ptr)(self.id, self.ptr);
    }

    unsafe fn adopt(&mut self) {
        if let Some(ptr) = (self.adopt_ptr)(self.ptr, self.id) {
            self.ptr = ptr;
        } // note: otherwise the pointer is left unchanged
    }

    unsafe fn cleanup(&mut self) {
        (self.cleanup_ptr)(self.ptr, self.id);
    }
}

/// A handle that represents the global state of GLFW
pub struct GlDraw {
    // rc::Rc makes this type non-send and non-sync
    phantom: PhantomData<std::rc::Rc<()>>,
}

impl Drop for GlDraw {
    fn drop(&mut self) {}
}

impl GlDraw {
    /// Create a new window. This window shares resources (textures, etc) with
    /// all other windows.
    ///
    /// Note that it is generally assumed that there are no more than a few windows. Trying to
    /// open more than 10 or so may drastically reduce performance and may cause things to break.
    ///
    /// Having multiple windows in general is likely to be unstable.
    pub fn new_window(&mut self, width: u32, height: u32, name: &str) -> GlWindow {
        let gl_draw = inner_gl(self);
        let root_ptr = if gl_draw.windows.len() == 0 {
            ptr::null_mut()
        } else {
            gl_draw.windows[0]
        };
        let old_context = unsafe { glfw_raw::glfwGetCurrentContext() };
        let window_ptr = unsafe {
            let window = glfw_raw::glfwCreateWindow(
                width as c_int,
                height as c_int,
                CString::new(name).unwrap().as_ptr(),
                ptr::null_mut(),
                root_ptr,
            );
            if window.is_null() {
                panic!("GLFW failed to create window.");
            }
            glfw_raw::glfwMakeContextCurrent(window);
            glfw_raw::glfwSetKeyCallback(window, Some(key_callback));
            glfw_raw::glfwSetFramebufferSizeCallback(window, Some(resize_callback));
            window
        };

        let init_gl = unsafe { DRAW_INIT };

        let gl;

        unsafe {
            // we really don't want a struct with hundreds of function pointers
            // to be sitting around on the stack
            gl = Box::new(gl::Gl::load_with(|symbol| {
                let c_str = CString::new(symbol.as_bytes());
                glfw_raw::glfwGetProcAddress(c_str.unwrap().as_ptr()) as *const _
            }));
        }

        unsafe {
            gl::CURRENT = &*gl as *const _;
        }

        if !init_gl {
            // load all OpenGL function pointers. This can only be done if there is a current
            // active context

            // this cannot be done until a context is linked
            let mut buffers = [0; NUM_DRAW_BUFFERS as usize];
            unsafe {
                gl.GenBuffers(NUM_DRAW_BUFFERS, (&mut buffers).as_mut_ptr());
                gl.PixelStorei(gl::UNPACK_ALIGNMENT, 1);

                free_draw::DRAW_BUFFERS = buffers;
            }
            let color_shader = shader::get_program(
                shader::COLOR_VERTEX_SHADER_SOURCE,
                shader::COLOR_FRAGMENT_SHADER_SOURCE,
            );
            let tex_shader = shader::get_program(
                shader::TEX_VERTEX_SHADER_SOURCE,
                shader::TEX_FRAGMENT_SHADER_SOURCE,
            );

            unsafe {
                let color_location =
                    gl.GetUniformLocation(color_shader, b"color\0".as_ptr() as *const _);
                let color_transform_location =
                    gl.GetUniformLocation(color_shader, b"transform\0".as_ptr() as *const _);
                let tex_transform_location =
                    gl.GetUniformLocation(tex_shader, b"transform\0".as_ptr() as *const _);
                let tex_sampler_location =
                    gl.GetUniformLocation(tex_shader, b"sampler\0".as_ptr() as *const _);

                free_draw::COLOR_SHADER_COLOR = color_location;
                free_draw::COLOR_SHADER_TRANSFORM = color_transform_location;
                free_draw::TEX_SHADER_TRANSFORM = tex_transform_location;
                free_draw::TEX_SHADER_TEX = tex_sampler_location;

                free_draw::COLOR_SHADER = color_shader;
                free_draw::TEX_SHADER = tex_shader;
            }

            for r in gl_draw.resource_orphans.iter_mut() {
                // this is the correct place to call adopt
                unsafe { r.adopt() }
            }

            unsafe { DRAW_INIT = true };
        }

        let mut pixels_width = width as i32;

        // call poll_events in order to initialize the window
        unsafe {
            glfw_raw::glfwPollEvents();
            glfw_raw::glfwGetFramebufferSize(window_ptr, &mut pixels_width, ptr::null_mut());
            if !old_context.is_null() {
                glfw_raw::glfwMakeContextCurrent(old_context);
            }
        }

        let window_data = unsafe { &mut WINDOW_GLOBAL };

        gl_draw.windows.push(window_ptr);

        let mut vao = 0;
        unsafe {
            gl.GenVertexArrays(1, &mut vao);
        }
        window_data.total_windows += 1;
        window_data.num_windows += 1;
        GlWindow {
            width: Cell::new(width as i32),
            height: Cell::new(height as i32),
            scale: Cell::new(pixels_width / width as i32),
            gl: gl,
            ptr: window_ptr,
            draw_vao: vao,
            phantom: PhantomData,
        }
    }

    pub fn draw<'a>(&'a mut self, window: &GlWindow) {
        unsafe {
            glfw_raw::glfwSwapBuffers(window.ptr);
            glfw_raw::glfwPollEvents();
            let mut width = 0;
            let mut height = 0;
            let mut scale = 1;
            glfw_raw::glfwGetWindowSize(window.ptr, &mut width, &mut height);
            glfw_raw::glfwGetFramebufferSize(window.ptr, &mut scale, 0 as *mut _);
            window.width.set(width);
            window.height.set(height);
            window.scale.set(scale / width);
        }

        let error = unsafe { window.gl.GetError() };

        match error {
            gl::INVALID_ENUM => println!("OpenGL - Invalid Enum"),
            gl::INVALID_VALUE => println!("OpenGL - Invalid Value"),
            gl::INVALID_OPERATION => println!("OpenGL - Invalid Operation"),
            gl::INVALID_FRAMEBUFFER_OPERATION => println!("OpenGL - Invalid Framebuffer Operation"),
            gl::OUT_OF_MEMORY => println!("OpenGL - Out of Memory"),
            gl::STACK_UNDERFLOW => println!("OpenGL - Stack Underflow"),
            gl::STACK_OVERFLOW => println!("OpenGL - Stack Overflow"),
            _ => {}
        }
    }
}

pub(crate) unsafe fn activate_window(window: &GlWindow) {
    if CURRENT_WINDOW != window.ptr {
        glfw_raw::glfwMakeContextCurrent(window.ptr);
        CURRENT_WINDOW = window.ptr;
        gl::CURRENT = (&*window.gl) as *const Gl;
    }
}

#[allow(unused)]
extern "C" fn key_callback(
    window: *mut glfw_raw::GLFWwindow,
    key: c_int,
    scancode: c_int,
    action: c_int,
    mods: c_int,
) {
    // not currently used
}

#[allow(unused)]
extern "C" fn resize_callback(window: *mut glfw_raw::GLFWwindow, width: c_int, height: c_int) {
    // not currently used
}

#[allow(unused)]
extern "C" fn error_callback(code: c_int, _: *const i8) {
    println!("GLFW Error: 0x{:X}", code);
}

pub struct GlWindow {
    width: Cell<i32>,
    height: Cell<i32>,
    scale: Cell<i32>,
    draw_vao: GLuint,
    gl: Box<Gl>,
    ptr: *mut glfw_raw::GLFWwindow,
    // rc::Rc is a non-send non-sync type
    phantom: PhantomData<std::rc::Rc<()>>,
}

pub(crate) struct WindowData {}

impl WindowData {
    #[allow(unused)]
    fn new() -> *mut WindowData {
        let data = Box::new(WindowData {});
        Box::into_raw(data)
    }
}

impl GlWindow {
    pub fn background<C: color::Color>(&self, color: C) {
        let c = color.as_rgba();
        let gl = &self.gl;
        unsafe {
            let old_context = glfw_raw::glfwGetCurrentContext();
            glfw_raw::glfwMakeContextCurrent(self.ptr);
            gl.ClearColor(c[0], c[1], c[2], c[3]);
            gl.Clear(gl::COLOR_BUFFER_BIT);
            glfw_raw::glfwMakeContextCurrent(old_context);
        }
    }

    pub fn drawer<'a>(&'a self, gl: &'a mut GlDraw, cs: CoordinateSpace) -> DrawingSurface<'a> {
        DrawingSurface {
            gl: gl,
            window: &self,
            coordinate_space: cs,
            transform: Cell::new(na::Matrix4::<f32>::identity()),
        }
    }

    pub fn get_width_height(&self) -> (i32, i32) {
        (self.width.get(), self.height.get())
    }

    pub fn get_scale(&self) -> i32 {
        self.scale.get()
    }

    /// Closes the window. Note that a window is automatically closed when dropped, so this function
    /// doesn't do anything other than calling the destructor.
    ///
    /// Using this function can cause a window to close before resources are dropped. This is behavior
    /// is not unsafe, but it causes data to be moved off to the graphics card in to temporary storage
    /// in main memory, which can be a slow operation.
    pub fn close(self) {}
}

impl Drop for GlWindow {
    fn drop(&mut self) {
        unsafe {
            if CURRENT_WINDOW == self.ptr {
                CURRENT_WINDOW = 0 as *mut _;
            }
        }
        window_cleanup(self.ptr);
        let global_data = unsafe { &mut WINDOW_GLOBAL };
        let gl_draw = unsafe { inner_gl_unsafe() };
        let pos = gl_draw
            .windows
            .iter()
            .position(|ptr| *ptr == self.ptr)
            .expect("Window was closed but window pointer not in list.");
        gl_draw.windows.swap_remove(pos);

        if global_data.num_windows == 1 {
            unsafe {
                DRAW_INIT = false;
                EPOCH += 1;
            }
            gl_draw.orphan_resources();
        }
        unsafe {
            glfw_raw::glfwDestroyWindow(self.ptr);
        }
        global_data.num_windows -= 1;
    }
}

#[inline]
fn window_cleanup(ptr: *mut glfw_raw::GLFWwindow) {
    unsafe {
        mesh::unsafe_api::clear_window_vaos(ptr);
    }
}

#[derive(Clone, Copy)]
pub struct DrawMode {
    mode: GLenum,
}

pub const TRIANGLES: DrawMode = DrawMode {
    mode: gl::TRIANGLES,
};

pub const TRIANGLE_STRIP: DrawMode = DrawMode {
    mode: gl::TRIANGLE_STRIP,
};

pub const TRIANGLE_FAN: DrawMode = DrawMode {
    mode: gl::TRIANGLE_FAN,
};

pub const LINES: DrawMode = DrawMode { mode: gl::LINES };

pub const LINE_LOOP: DrawMode = DrawMode {
    mode: gl::LINE_LOOP,
};

pub const LINE_STRIP: DrawMode = DrawMode {
    mode: gl::LINE_STRIP,
};

/// For the lifetime of the DrawingSurface, it is assumed that the active context
/// is the window.
pub struct DrawingSurface<'a> {
    gl: &'a mut GlDraw,
    window: &'a GlWindow,
    coordinate_space: CoordinateSpace,
    transform: Cell<na::Matrix4<f32>>,
}

impl<'a> DrawingSurface<'a> {
    pub fn draw(&mut self) {
        self.gl.draw(&self.window);
        // we know that the transform is at the base level because `with_transform`
        // borrows the surface
        let width = self.window.width.get();
        let height = self.window.height.get();
        self.transform.set(self.coordinate_space.get_matrix(
            width,
            height,
            self.window.scale.get(),
        ));
        unsafe {
            self.window.gl.Viewport(0, 0, width, height);
        }
    }

    pub fn background<C: color::Color>(&self, color: C) {
        let c = color.as_rgba();
        let gl = &self.window.gl;
        unsafe {
            gl.ClearColor(c[0], c[1], c[2], c[3]);
            gl.Clear(gl::COLOR_BUFFER_BIT);
        }
    }

    pub fn with_transform<F: FnOnce(&DrawingSurface)>(&self, transform: na::Matrix4<f32>, func: F) {
        let old_matrix = self.transform.get();
        self.transform.set(old_matrix * transform);
        func(&self);
        self.transform.set(old_matrix);
    }

    /// Gets the position in coordinate-space coordinates of a position on the window. Useful for
    /// uis, etc.
    ///
    /// The position (0.0, 0.0) will give the coordinates of the bottom left corner of the window,
    /// and the position (1.0, 1.0) will give the coordinates of the top right cornder of the window.
    ///
    /// In the rare case that a window has a non-invertible transformation, this will return `None`
    pub fn get_window_pos(&self, x: f32, y: f32, z: f32) -> Option<(f32, f32, f32)> {
        if let Some(inv) = self.transform.get().try_inverse() {
            let vec = inv * na::Vector4::new(x, y, z, 1.0);
            let nvec = vec / vec[3];
            Some((nvec[0], nvec[1], nvec[2]))
        } else {
            None
        }
    }

    pub fn get_transform(&self) -> na::Matrix4<f32> {
        self.transform.get()
    }

    pub fn draw_image(
        &mut self,
        image: &texture::BindTexture<texture::Sampler2D>,
        start: (f32, f32),
        end: (f32, f32),
    ) {
        let gl = &self.window.gl;
        let shader = unsafe { free_draw::TEX_SHADER };

        let verts = [
            start.0, start.1, 0.0, //
            end.0, start.1, 0.0, //
            end.0, end.1, 0.0, //
            start.0, end.1, 0.0, //
        ];

        let uv: [f32; 8] = [
            0.0, 1.0, //
            1.0, 1.0, //
            1.0, 0.0, //
            0.0, 0.0, //
        ];

        let elements: [u8; 6] = [0, 1, 2, 2, 3, 0];

        unsafe {
            gl.UseProgram(shader);
            gl.UniformMatrix4fv(
                free_draw::TEX_SHADER_TRANSFORM,
                1,
                gl::FALSE,
                self.transform.as_ptr() as *const _,
            );

            gl.Uniform1i(free_draw::TEX_SHADER_TEX, 0);

            gl.ActiveTexture(gl::TEXTURE0);
            image.bind(gl);

            gl.BindVertexArray(self.window.draw_vao);

            gl.BindBuffer(gl::ARRAY_BUFFER, free_draw::DRAW_BUFFERS[0]);
            gl.BufferData(
                gl::ARRAY_BUFFER,
                4 * 3 * 4,
                verts.as_ptr() as *const _,
                gl::STREAM_DRAW,
            );
            gl.VertexAttribPointer(0, 3, gl::FLOAT, gl::FALSE, 0, ptr::null());
            gl.EnableVertexAttribArray(0);

            gl.BindBuffer(gl::ARRAY_BUFFER, free_draw::DRAW_BUFFERS[1]);
            gl.BufferData(
                gl::ARRAY_BUFFER,
                4 * 2 * 4,
                uv.as_ptr() as *const _,
                gl::STREAM_DRAW,
            );
            gl.VertexAttribPointer(1, 2, gl::FLOAT, gl::FALSE, 0, ptr::null());
            gl.EnableVertexAttribArray(1);

            gl.BindBuffer(gl::ELEMENT_ARRAY_BUFFER, free_draw::DRAW_BUFFERS[2]);
            gl.BufferData(
                gl::ELEMENT_ARRAY_BUFFER,
                1 * 6,
                elements.as_ptr() as *const _,
                gl::STREAM_DRAW,
            );

            gl.DrawElements(gl::TRIANGLES, 6, gl::UNSIGNED_BYTE, ptr::null());
        }
    }

    pub fn draw_triangle(&self, tri: [f32; 9], color: &color::Color) {
        let gl = &self.window.gl;
        unsafe {
            let shader = free_draw::COLOR_SHADER;
            gl.UseProgram(shader);
            gl.Uniform4fv(
                free_draw::COLOR_SHADER_TRANSFORM,
                1,
                color.as_rgba().as_ptr() as *const _,
            );
            gl.UniformMatrix4fv(
                free_draw::COLOR_SHADER_TRANSFORM,
                1,
                gl::FALSE,
                self.transform.as_ptr() as *const _,
            );

            gl.BindVertexArray(self.window.draw_vao);
            gl.BindBuffer(gl::ARRAY_BUFFER, free_draw::DRAW_BUFFERS[0]);
            gl.BufferData(
                gl::ARRAY_BUFFER,
                4 * 9,
                tri.as_ptr() as *const _,
                gl::STREAM_DRAW,
            );
            gl.VertexAttribPointer(0, 3, gl::FLOAT, gl::FALSE, 0, ptr::null());
            gl.EnableVertexAttribArray(0);

            gl.DrawArrays(gl::TRIANGLES, 0, 3);
        }
    }
}
