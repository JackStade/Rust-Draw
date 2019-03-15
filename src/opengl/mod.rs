extern crate gl;
extern crate glfw;
extern crate parking_lot;

use nalgebra as na;
use std::ffi::CString;
use std::fmt;
use std::fs::File;
use std::marker::PhantomData;
use std::os::raw::{c_int, c_void};
use std::ptr;
use std::result::Result;
use std::str;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering, ATOMIC_BOOL_INIT, ATOMIC_USIZE_INIT};
use std::sync::mpsc::Receiver;
use std::thread;

use crate::color;
use crate::CoordinateSpace;

use self::gl::types::*;
use self::glfw::ffi as glfw_raw;
// parking lot's mutexes are much faster than the standard library ones
use self::parking_lot::Mutex;

/// The shader module contains lower level functions for creating and using shaders.
/// This is an advanced feature that can be difficult to use correctly.
///
/// Because it uses a complex system of types with many blanket implementations, 
/// it is difficult to understand how this module works from the auto-doc. In
/// the future there will be a shader guide.
#[allow(non_snake_case)]
pub mod shader;

static mut GL_DRAW: Option<GlDrawCore> = None;

static DRAW_INIT: AtomicBool = ATOMIC_BOOL_INIT;

/// Initializes gl/glfw.
pub fn get_gl() -> Result<GlDraw, InitError> {
    // check if there is already an active gl. It is unlikely that get_gl will be called by
    // two different threads at the same time, but it is neccesary to use atomics here to prevent
    // scary things from happening if it is
    let init = DRAW_INIT.swap(true, Ordering::AcqRel);
    if init {
        return Err(InitError::AlreadyInitialized);
    }
    if let Some("main") = thread::current().name() {
    } else {
        // TODO: this check currently doesn't work, cfg!(macos) is always false
        if cfg!(macos) {
            return Err(InitError::MacosNonMainThread);
        } else {
            // in theory, it is safe to run glfw on a seperate thread on other systems
            println!("Warning - GLFW should not be initialized on non-main thread.");
        }
    }
    unsafe {
        if glfw_raw::glfwInit() == glfw_raw::TRUE {
            // default opengl version
            glfw_raw::glfwWindowHint(glfw_raw::CONTEXT_VERSION_MAJOR, 4);
            glfw_raw::glfwWindowHint(glfw_raw::CONTEXT_VERSION_MINOR, 5);
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
    }
    // here we know that this is the GLFW thread, so no other threads can be accessing GL_DRAW
    unsafe {
        let gl_draw = GlDrawCore::init();
        GL_DRAW = Some(gl_draw);
    }
    Ok(GlDraw {
        phantom: PhantomData,
    })
}

pub enum InitError {
    MacosNonMainThread,
    AlreadyInitialized,
    GlfwError,
}

impl fmt::Debug for InitError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            InitError::MacosNonMainThread => {
                write!(f, "GL must be initialized on the main thread on macos.")
            }
            InitError::AlreadyInitialized => write!(f, "GL has already been initialized."),
            InitError::GlfwError => write!(f, "An unknown error occured."),
        }
    }
}

#[derive(Clone, Copy)]
pub struct TexPtr {
    unit: GLuint,
}

const NUM_DRAW_BUFFERS: i32 = 3;

struct GlDrawCore {
    init_gl: bool,

    num_windows: usize,
    total_windows: u32,
    windows: Vec<Option<WindowCore>>,
    current_window: u32,
    root_window: *mut glfw_raw::GLFWwindow,

    // the buffers to be used for free drawing
    draw_buffers: [GLuint; NUM_DRAW_BUFFERS as usize],
    // a vao that is attached to the window that is currently being drawn
    // vaos cannot be shared between contexts
    current_draw_vao: GLuint,

    // the default shader programs
    color_shader: GLuint,
    tex_shader: GLuint,

    color_shader_uniform_locations: [GLint; 2],
    tex_shader_uniform_locations: [GLint; 2],
}

impl GlDrawCore {
    fn init() -> GlDrawCore {
        GlDrawCore {
            num_windows: 0,
            total_windows: 0,

            init_gl: false,
            windows: Vec::new(),
            current_window: 0,
            root_window: ptr::null_mut(),

            draw_buffers: [0; NUM_DRAW_BUFFERS as usize],
            current_draw_vao: 0,

            color_shader: 0,
            tex_shader: 0,

            color_shader_uniform_locations: [0; 2],
            tex_shader_uniform_locations: [0; 2],
        }
    }

    fn use_window(&mut self, window_id: u32) {
        let window = self.windows[window_id as usize].as_ref().unwrap();
        unsafe {
            glfw_raw::glfwMakeContextCurrent(window.window_ptr);
        }
        self.current_window = window_id;
        self.current_draw_vao = window.vao;
    }
}

struct WindowCore {
    id: u32,
    // it is gaurunteed that no other window will ever have the
    // same permanent id, even after this one is closed
    permanent_id: u32,
    window_ptr: *mut glfw_raw::GLFWwindow,
    // vao for free drawing
    vao: GLuint,

    base_matrix: na::Matrix4<f32>,
    coordinate_space: CoordinateSpace,
    width: i32,
    height: i32,
    scale: i32,
}

impl WindowCore {
    fn init(
        id: u32,
        window_num: u32,
        window: *mut glfw_raw::GLFWwindow,
        width: i32,
        height: i32,
        scale: i32,
        coordinate_space: CoordinateSpace,
    ) -> WindowCore {
        let mut vao = 0;
        unsafe {
            gl::GenVertexArrays(1, &mut vao);
        }
        WindowCore {
            id: id,
            permanent_id: window_num,
            window_ptr: window,
            vao: vao,

            base_matrix: coordinate_space.get_matrix(width, height, scale),

            coordinate_space: coordinate_space,
            width: width,
            height: height,
            scale: scale,
        }
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

pub struct GlImage {
    gl_texture: GLuint,
    width: u32,
    height: u32,
    // images cannot be send or sync because they should not be dropped
    // on a different thread
    phantom: PhantomData<std::rc::Rc<()>>,
}

impl GlDraw {
    /// Create a new window. This window shares resources (textures, etc) with
    /// all other windows.
    ///
    /// Note that it is generally assumed that there are no more than a few windows. Trying to
    /// open more than 10 or so may drastically reduce performance and may cause things to break.
    /// Having multiple windows in general is likely to be very unstable.
    pub fn new_window(
        &mut self,
        width: u32,
        height: u32,
        space: CoordinateSpace,
        name: &str,
    ) -> GlWindow {
        let gl_draw;
        // GlDraw is not send, so new_window can only be called on the main thread
        unsafe {
            gl_draw = GL_DRAW.as_mut().unwrap();
        }
        let window_ptr = unsafe {
            let window = glfw_raw::glfwCreateWindow(
                width as c_int,
                height as c_int,
                CString::new(name).unwrap().as_ptr(),
                ptr::null_mut(),
                gl_draw.root_window,
            );
            if window.is_null() {
                panic!("GLFW failed to create window.");
            }
            glfw_raw::glfwMakeContextCurrent(window);
            glfw_raw::glfwSetKeyCallback(window, Some(key_callback));
            glfw_raw::glfwSetFramebufferSizeCallback(window, Some(resize_callback));
            window
        };

        if !gl_draw.init_gl {
            // load all OpenGL function pointers. This can only be done if there is a current
            // active context
            unsafe {
                gl::load_with(|symbol| {
                    let c_str = CString::new(symbol.as_bytes());
                    glfw_raw::glfwGetProcAddress(
                        c_str.unwrap().as_bytes_with_nul().as_ptr() as *const i8
                    ) as *const _
                });
                gl_draw.root_window = window_ptr;
            }

            // this cannot be done until a context is linked
            let mut buffers = [0; NUM_DRAW_BUFFERS as usize];
            unsafe {
                gl::GenBuffers(NUM_DRAW_BUFFERS, (&mut buffers).as_mut_ptr());
            }

            gl_draw.draw_buffers = buffers;

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
                    gl::GetUniformLocation(color_shader, b"color\0".as_ptr() as *const _);
                let color_transform_location =
                    gl::GetUniformLocation(color_shader, b"transform\0".as_ptr() as *const _);
                let tex_transform_location =
                    gl::GetUniformLocation(tex_shader, b"transform\0".as_ptr() as *const _);
                let tex_sampler_location =
                    gl::GetUniformLocation(tex_shader, b"sampler\0".as_ptr() as *const _);

                gl_draw.color_shader_uniform_locations = [color_location, color_transform_location];
                gl_draw.tex_shader_uniform_locations =
                    [tex_transform_location, tex_sampler_location];
            }

            gl_draw.color_shader = color_shader;
            gl_draw.tex_shader = tex_shader;

            gl_draw.init_gl = true;
        }

        // test if window is retina
        let mut view_dimensions: [GLint; 4] = [0, 0, width as i32, height as i32];
        unsafe {
            // values given by GetIntegerv will return values in screen pixels
            // width and height are in normalized pixels
            gl::GetIntegerv(gl::VIEWPORT, view_dimensions.as_mut_ptr());
        }
        // the pixel depth per normalized pixel is always an integer
        // Note: the scale can change during runtime when an external monitor is connected
        // currently, this is unaccounted for
        let scl = (view_dimensions[2] as u32) / width;
        let scl = scl as i32;

        gl_draw.num_windows += 1;

        let viewport = space.get_viewport(width as i32, height as i32, scl);
        unsafe { gl::Viewport(viewport.0, viewport.1, viewport.2, viewport.3) };
        // call poll_events in order to initialize the window
        unsafe {
            glfw_raw::glfwPollEvents();
        }
        let window_slots = gl_draw.windows.len();
        let mut slot = window_slots;
        let mut found = false;
        for i in 0..window_slots {
            if let None = gl_draw.windows[i] {
                slot = i;
                found = true;
                break;
            }
        }

        gl_draw.current_window = slot as u32;

        if !found {
            // add an extra slot
            gl_draw.windows.push(None);
        }
        let window = WindowCore::init(
            slot as u32,
            gl_draw.total_windows,
            window_ptr,
            width as i32,
            height as i32,
            scl,
            space,
        );

        gl_draw.current_draw_vao = window.vao;

        gl_draw.windows[slot] = Some(window);
        if gl_draw.root_window.is_null() {
            gl_draw.root_window = window_ptr;
        }
        unsafe {
            glfw_raw::glfwSetWindowUserPointer(window_ptr, slot as *mut _);
        }
        gl_draw.total_windows += 1;
        GlWindow {
            id: slot as u32,
            phantom: PhantomData,
        }
    }

    pub fn load_image(&mut self, width: u32, height: u32, bytes: &[u8]) -> GlImage {
        let mut tex = 0;
        unsafe {
            gl::GenTextures(1, &mut tex);
            gl::BindTexture(gl::TEXTURE_2D, tex);
            gl::TexImage2D(
                gl::TEXTURE_2D,
                0,
                gl::RGBA8 as i32,
                width as i32,
                height as i32,
                0,
                gl::RGBA,
                gl::UNSIGNED_BYTE,
                bytes.as_ptr() as *const _,
            );
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::NEAREST as i32);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::NEAREST as i32);
        }
        GlImage {
            gl_texture: tex,
            width: width,
            height: height,
            phantom: PhantomData,
        }
    }

    /// Draws the specified window. When there are multiple windows, it is reccomended to use draw_all
    pub fn draw(&mut self, window: &GlWindow) {
        let gl_draw;
        // draw can only be called on the thread that initialized gl, so this is safe
        unsafe {
            gl_draw = GL_DRAW.as_mut().unwrap();
        }
        let window_id = window.id;
        gl_draw.use_window(window_id);
        let window = gl_draw.windows[window_id as usize].as_mut().unwrap();

        unsafe {
            glfw_raw::glfwMakeContextCurrent(window.window_ptr);
            glfw_raw::glfwPollEvents();
            glfw_raw::glfwSwapBuffers(window.window_ptr);
        }

        let error = unsafe { gl::GetError() };

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

    // draws all windows currently open
    pub fn draw_all(&mut self) {}
}

pub mod gl_mesh {
    use crate::VertexDataType;
    use gl;
    use gl::types::*;
    use std::marker::PhantomData;
    use VertexDataType::*;

    /// Represents mesh data that has been written to opengl and stored in graphics memory.
    /// Using this directly is quite tricky, it is reccomended to use the included mesh types.
    pub struct GlMesh {
        vbos: Vec<GLuint>,
        // make this type non-send and non-sync
        phantom: PhantomData<std::rc::Rc<()>>,
    }

    impl GlMesh {
        /// Generates a new gl mesh.
        pub fn new_mesh(data: &[&[u8]]) -> GlMesh {
            let mut buffers = vec![0; data.len()];
            unsafe {
                gl::GenBuffers(data.len() as i32, buffers.as_mut_ptr() as *mut _);
                for (i, d) in data.iter().enumerate() {
                    gl::NamedBufferData(
                        buffers[i],
                        d.len() as isize,
                        d.as_ptr() as *const _,
                        gl::STATIC_DRAW,
                    );
                }
            }
            GlMesh {
                vbos: buffers,
                phantom: PhantomData,
            }
        }
    }

    pub struct UvMesh {
        mesh: GlMesh,
    }

    enum PointerType {
        Float,
        Int,
    }

    fn should_normalize(ty: VertexDataType) -> GLboolean {
        gl::TRUE
    }

    fn pointer_type(ty: VertexDataType) -> PointerType {
        match ty {
            Float => PointerType::Float,
            NormU8 => PointerType::Float,
            NormI8 => PointerType::Float,
            NormU16 => PointerType::Float,
            NormI16 => PointerType::Float,
            NormU32 => PointerType::Float,
            NormI32 => PointerType::Float,
            _ => PointerType::Int,
        }
    }

    fn gl_type(ty: VertexDataType) -> GLenum {
        match ty {
            Float => gl::FLOAT,
            IntU8 => gl::UNSIGNED_BYTE,
            IntI8 => gl::BYTE,
            IntU16 => gl::UNSIGNED_SHORT,
            IntI16 => gl::SHORT,
            IntU32 => gl::UNSIGNED_INT,
            IntI32 => gl::INT,
            NormU8 => gl::UNSIGNED_BYTE,
            NormI8 => gl::BYTE,
            NormU16 => gl::UNSIGNED_SHORT,
            NormI16 => gl::SHORT,
            NormU32 => gl::UNSIGNED_INT,
            NormI32 => gl::INT,
        }
    }
}

extern "C" fn key_callback(
    window: *mut glfw_raw::GLFWwindow,
    key: c_int,
    scancode: c_int,
    action: c_int,
    mods: c_int,
) {

}

extern "C" fn resize_callback(window: *mut glfw_raw::GLFWwindow, width: c_int, height: c_int) {
    let gl_draw;
    unsafe {
        gl_draw = GL_DRAW.as_mut().unwrap();
    }
    let window_ptr = window;
    let window_id = unsafe { glfw_raw::glfwGetWindowUserPointer(window) };
    let window = gl_draw.windows[window_id as usize].as_mut().unwrap();

    let scl = window.scale;

    window.width = width / scl;
    window.height = height / scl;

    let viewport = window
        .coordinate_space
        .get_viewport(width / scl, height / scl, scl);
    let matrix = window
        .coordinate_space
        .get_matrix(width / scl, height / scl, scl);
    window.base_matrix = matrix;

    unsafe {
        glfw_raw::glfwMakeContextCurrent(window_ptr);
        // gl::Viewport(0, 0, width, height);
        gl::Viewport(viewport.0, viewport.1, viewport.2, viewport.3);
    }
}

extern "C" fn error_callback(code: c_int, _: *const i8) {
    println!("Error: 0x{:X}", code);
}

pub struct GlWindow {
    id: u32,
    // rc::Rc is a non-send non-sync type
    phantom: PhantomData<std::rc::Rc<()>>,
}

// if you don't like this, then hope this gets implemented:
// https://github.com/rust-lang/rfcs/issues/1215
macro_rules! check_window {
    ($id:expr, $gl_draw:ident, $window:ident) => {
        let $gl_draw = unsafe { GL_DRAW.as_mut().unwrap() };
        if $gl_draw.current_window != $id {
            $gl_draw.use_window($id);
        }
        let $window = $gl_draw.windows[$id as usize].as_mut().unwrap();
    };
}

impl GlWindow {
    pub fn background<T: color::Color>(&mut self, color: T) {
        use color::Color;

        check_window!(self.id, gl_draw, window);

        let c = color.as_rgba();
        unsafe {
            gl::ClearColor(c[0], c[1], c[2], c[3]);
            gl::Clear(gl::COLOR_BUFFER_BIT);
        }
    }

    pub fn draw_triangle(&mut self, color: [f32; 4], tri: [f32; 9]) {
        check_window!(self.id, gl_draw, window);

        let shader = gl_draw.color_shader;

        unsafe {
            gl::UseProgram(shader);
            gl::Uniform4fv(
                gl_draw.color_shader_uniform_locations[0],
                1,
                color.as_ptr() as *const _,
            );
            gl::UniformMatrix4fv(
                gl_draw.color_shader_uniform_locations[1],
                1,
                gl::FALSE,
                window.base_matrix.as_ptr() as *const _,
            );

            gl::BindVertexArray(gl_draw.current_draw_vao);
            gl::BindBuffer(gl::ARRAY_BUFFER, gl_draw.draw_buffers[0]);
            gl::BufferData(
                gl::ARRAY_BUFFER,
                4 * 9,
                tri.as_ptr() as *const _,
                gl::STREAM_DRAW,
            );
            gl::VertexAttribPointer(0, 3, gl::FLOAT, gl::FALSE, 0, ptr::null());
            gl::EnableVertexAttribArray(0);

            gl::DrawArrays(gl::TRIANGLES, 0, 3);
        }
    }

    /// Draws an image. Note that the image may be inverted in some coordinate systems.
    pub fn draw_image(&mut self, image: &GlImage, start: (f32, f32), end: (f32, f32)) {
        check_window!(self.id, gl_draw, window);

        let shader = gl_draw.tex_shader;

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
            gl::UseProgram(shader);
            gl::UniformMatrix4fv(
                gl_draw.tex_shader_uniform_locations[0],
                1,
                gl::FALSE,
                window.base_matrix.as_ptr() as *const _,
            );

            gl::ActiveTexture(
                gl::TEXTURE0, /* + gl_draw.tex_shader_uniform_locations[1] as u32*/
            );
            gl::BindTexture(gl::TEXTURE_2D, image.gl_texture);

            gl::BindVertexArray(gl_draw.current_draw_vao);

            gl::BindBuffer(gl::ARRAY_BUFFER, gl_draw.draw_buffers[0]);
            gl::BufferData(
                gl::ARRAY_BUFFER,
                4 * 3 * 4,
                verts.as_ptr() as *const _,
                gl::STREAM_DRAW,
            );
            gl::VertexAttribPointer(0, 3, gl::FLOAT, gl::FALSE, 0, ptr::null());
            gl::EnableVertexAttribArray(0);

            gl::BindBuffer(gl::ARRAY_BUFFER, gl_draw.draw_buffers[1]);
            gl::BufferData(
                gl::ARRAY_BUFFER,
                4 * 2 * 4,
                uv.as_ptr() as *const _,
                gl::STREAM_DRAW,
            );
            gl::VertexAttribPointer(1, 2, gl::FLOAT, gl::FALSE, 0, ptr::null());
            gl::EnableVertexAttribArray(1);

            gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, gl_draw.draw_buffers[2]);
            gl::BufferData(
                gl::ELEMENT_ARRAY_BUFFER,
                1 * 6,
                elements.as_ptr() as *const _,
                gl::STREAM_DRAW,
            );

            gl::DrawElements(gl::TRIANGLES, 6, gl::UNSIGNED_BYTE, ptr::null());
        }
    }

    /// Transforms the window. Any drawing done on the window inside of `draw` will have this transform applied.
    pub fn draw_with_transform<T: FnMut(&mut GlWindow)>(
        &mut self,
        transform: na::Matrix4<f32>,
        mut draw: T,
    ) {
        check_window!(self.id, gl_draw, window);
        let mut old_matrix = transform * window.base_matrix;
        std::mem::swap(&mut old_matrix, &mut window.base_matrix);
        draw(self);
        std::mem::swap(&mut old_matrix, &mut window.base_matrix);
    }

    /// Gets the position in coordinate-space coordinates of a position on the window. Useful for
    /// uis, etc.
    ///
    /// The position (0.0, 0.0) will give the coordinates of the bottom left corner of the window,
    /// and the position (1.0, 1.0) will give the coordinates of the bottom right cornder of the window.
    pub fn get_window_pos(&self, x: f32, y: f32) -> (f32, f32) {
        let gl_draw = unsafe { GL_DRAW.as_ref().unwrap() };
        let window = gl_draw.windows[self.id as usize].as_ref().unwrap();
        window
            .coordinate_space
            .get_window_pos(x, y, window.width, window.height, window.scale)
    }
}

impl Drop for GlWindow {
    fn drop(&mut self) {
        let gl_draw;
        // gl windows aren't send so they can only be dropped on the thread
        // that owns the gl draw (should be the main thread)
        unsafe {
            gl_draw = GL_DRAW.as_mut().unwrap();
        }
        let mut window = None;
        std::mem::swap(&mut gl_draw.windows[self.id as usize], &mut window);
    }
}
