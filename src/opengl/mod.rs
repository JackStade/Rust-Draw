extern crate gl;
extern crate glfw;
extern crate parking_lot;

use nalgebra as na;
use std::ffi::CString;
use std::fmt;
use std::fs::File;
use std::marker::PhantomData;
use std::mem;
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

pub mod texture;

use texture::{BindTexture, ImageData};

static mut GL_DRAW: *mut GlDrawCore = 0 as *mut _;

static mut DRAW_INIT: bool = false;

#[inline(always)]
fn inner_gl<'a>(_: &'a mut GlDraw) -> &'a mut GlDrawCore {
    unsafe { &mut *GL_DRAW }
}

#[inline(always)]
fn inner_gl_static<'a>(_: &'a GlDraw) -> &'a GlDrawCore {
    unsafe { &*GL_DRAW }
}

#[inline(always)]
unsafe fn inner_gl_unsafe<'a>() -> &'a mut GlDrawCore {
    &mut *GL_DRAW
}

#[inline(always)]
unsafe fn inner_gl_unsafe_static<'a>() -> &'a GlDrawCore {
    &*GL_DRAW
}

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
    let init = mem::replace(&mut DRAW_INIT, true);
    if init {
        return Err(InitError::AlreadyInitialized);
    }

    let draw_box = Box::new(GlDrawCore::new());
    GL_DRAW = Box::into_raw(draw_box);

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
    // a list of the functions need to orphan/adopt each active resource
    orphan_positions: Vec<GLuint>,
    resource_orphans: Vec<DynResource>,

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
    fn new() -> GlDrawCore {
        GlDrawCore {
            num_windows: 0,
            total_windows: 0,

            init_gl: false,
            windows: Vec::new(),
            current_window: 0,
            root_window: ptr::null_mut(),

            resource_list: Vec::new(),
            resource_search_start: 0,
            orphan_positions: Vec::new(),
            resource_orphans: Vec::new(),

            draw_buffers: [0; NUM_DRAW_BUFFERS as usize],
            current_draw_vao: 0,

            color_shader: 0,
            tex_shader: 0,

            color_shader_uniform_locations: [0; 2],
            tex_shader_uniform_locations: [0; 2],
        }
    }

    fn orphan_resources(&mut self) {
        self.init_gl = false;
        for r in self.resource_orphans.iter_mut() {
            unsafe {
                r.orphan();
            }
        }
    }

    // note that a window is required in order to create new resources, so it is not
    // possible to create a resource in an orphan state
    fn get_resource_id(
        &mut self,
        name: GLuint,
        adopt_ptr: unsafe fn(*mut (), u32),
        drop_ptr: unsafe fn(*mut (), u32),
        orphan_ptr: fn(u32) -> *mut (),
    ) -> u32 {
        let mut found = None;
        let mut low_space = self.resource_search_start;
        for slot in self.resource_list[low_space..].iter_mut() {
            low_space += 1;
            if *slot == 0 {
                *slot = name;
                self.orphan_positions[low_space] = self.resource_orphans.len() as u32;
                found = Some(low_space);
                break;
            }
        }
        self.resource_search_start = low_space;

        if found.is_none() {
            found = Some(self.resource_list.len());
            self.resource_list.push(name);
            self.orphan_positions
                .push(self.resource_orphans.len() as u32);
        }
        let found = found.unwrap();
        self.resource_orphans.push(DynResource {
            id: found as u32,
            ptr: 0 as *mut _,
            adopt_ptr: adopt_ptr,
            drop_ptr: drop_ptr,
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
        if self.num_windows == 0 {
            unsafe { self.resource_orphans[pos as usize].drop_when_orphaned() };
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

    fn use_window(&mut self, window_id: u32) {
        let window = self.windows[window_id as usize].as_ref().unwrap();
        unsafe {
            glfw_raw::glfwMakeContextCurrent(window.window_ptr);
        }
        self.current_window = window_id;
        self.current_draw_vao = window.vao;
    }
}

pub trait GlResource {
    unsafe fn adopt(ptr: *mut (), id: u32);

    unsafe fn drop_while_orphaned(ptr: *mut (), id: u32);

    fn orphan(id: u32) -> *mut ();
}

// note: this type has implicit state. It is either in an orphan state
// or an active state. In an orphan state, ptr will contain data needed
// to reinitialzed the resource. In an active state, the pointer is invalid.
// this state is not stored in the object because it is assumed to be implied by
// the state of GlDrawCore
struct DynResource {
    id: u32,
    // in an orphaned state, this pointer will point to the data needed
    // otherwise, it will not be assumed to be unsafe
    // note that the pointer must be mutable in order to drop the memory
    // referenced by it
    ptr: *mut (),
    adopt_ptr: unsafe fn(*mut (), u32),
    drop_ptr: unsafe fn(*mut (), u32),
    orphan_ptr: fn(u32) -> *mut (),
}

impl DynResource {
    unsafe fn drop_when_orphaned(&self) {
        (self.drop_ptr)(self.ptr, self.id)
        // ptr is no longer valid
    }

    unsafe fn orphan(&mut self) {
        // ptr is invalid
        self.ptr = (self.orphan_ptr)(self.id);
        // ptr is now valid
    }

    unsafe fn adopt(&self) {
        (self.adopt_ptr)(self.ptr, self.id);
        // ptr is no longer valid
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

impl GlDraw {
    /// Create a new window. This window shares resources (textures, etc) with
    /// all other windows.
    ///
    /// Note that it is generally assumed that there are no more than a few windows. Trying to
    /// open more than 10 or so may drastically reduce performance and may cause things to break.
    ///
    /// Having multiple windows in general is likely to be unstable.
    pub fn new_window(
        &mut self,
        width: u32,
        height: u32,
        space: CoordinateSpace,
        name: &str,
    ) -> GlWindow {
        let gl_draw = inner_gl(self);
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
                gl::PixelStorei(gl::UNPACK_ALIGNMENT, 1);
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

            for r in gl_draw.resource_orphans.iter_mut() {
                // this is the correct place to call adopt
                unsafe { r.adopt() }
            }

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
            ptr: window_ptr,
            phantom: PhantomData,
        }
    }

    /* pub fn load_image(&mut self, width: u32, height: u32, bytes: &[u8]) -> GlImage {
        if bytes.len() < (4 * width * height) as usize {
            panic!("Slice provided is not long enough for an image of size {}x{}, must be at least {} bytes.", width, height, width * height * 4);
        };
        let gl_draw = inner_gl(self);

        if gl_draw.num_windows == 0 {
            let mut datastore = Vec::with_capacity((width * height) as usize);
            // we need to covert to a slice of u32
            unsafe {
                // need to be careful here because src isn't neccesarily aligned
                ptr::copy_nonoverlapping(
                    bytes.as_ptr(),
                    datastore.as_mut_ptr() as *mut u8,
                    (4 * width * height) as usize,
                );
                datastore.set_len((width * height) as usize);
            }
            let id = gl_draw.image(0);

            gl_draw.orphan_textures.push(ImageData {
                data: datastore,
                image_id: id,

                width: width,
                height: height,
            });

            return GlImage {
                image_id: id,

                width: width,
                height: height,
                phantom: PhantomData,
            };
        };
        let mut tex = 0;
        unsafe {
            gl::GenTextures(1, &mut tex);
            gl::BindTexture(gl::TEXTURE_2D, tex);
            gl::PixelStorei(gl::UNPACK_ALIGNMENT, 1);
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
        let id = gl_draw.image(tex);
        GlImage {
            image_id: id,

            width: width,
            height: height,
            phantom: PhantomData,
        }
    }*/

    /// Draws the specified window. When there are multiple windows, it is reccomended to use draw_all
    pub fn draw(&mut self, window: &GlWindow) {
        let gl_draw = inner_gl(self);
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
    pub fn draw_all(&mut self) {
        unimplemented!();
    }
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
    let gl_draw = unsafe { inner_gl_unsafe() };
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
        gl::Viewport(viewport.0, viewport.1, viewport.2, viewport.3);
    }
}

extern "C" fn error_callback(code: c_int, _: *const i8) {
    println!("Error: 0x{:X}", code);
}

pub struct GlWindow {
    id: u32,
    ptr: *mut glfw_raw::GLFWwindow,
    // rc::Rc is a non-send non-sync type
    phantom: PhantomData<std::rc::Rc<()>>,
}

// if you don't like this, then hope this gets implemented:
// https://github.com/rust-lang/rfcs/issues/1215
macro_rules! check_window {
    ($id:expr, $gl_draw:ident, $window:ident) => {
        let $gl_draw = unsafe { inner_gl_unsafe() };
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
    pub fn draw_image(
        &mut self,
        image: &texture::Texture2D<texture::TextureData<texture::RGBA, u8>>,
        start: (f32, f32),
        end: (f32, f32),
    ) {
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
            gl::BindTexture(gl::TEXTURE_2D, gl_draw.resource_list[image.get_id() as usize]);

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
    /// and the position (1.0, 1.0) will give the coordinates of the top right cornder of the window.
    pub fn get_window_pos(&self, x: f32, y: f32) -> (f32, f32) {
        let gl_draw = unsafe { inner_gl_unsafe_static() };
        let window = gl_draw.windows[self.id as usize].as_ref().unwrap();
        window
            .coordinate_space
            .get_window_pos(x, y, window.width, window.height, window.scale)
    }

    /// Closes the window. Note that a window is automatically closed when dropped, so this function
    /// doesn't do anything other than calling the destructor.
    pub fn close(self) {}
}

impl Drop for GlWindow {
    fn drop(&mut self) {
        let gl_draw = unsafe { inner_gl_unsafe() };
        if gl_draw.num_windows == 1 {
            gl_draw.orphan_resources();
        }
        unsafe {
            glfw_raw::glfwDestroyWindow(self.ptr);
        }
        let mut window = None;
        gl_draw.num_windows -= 1;
        std::mem::swap(&mut gl_draw.windows[self.id as usize], &mut window);
    }
}
