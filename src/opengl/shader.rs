#![allow(unused)]

use self::traits::*;
use std::cell::Cell;
use std::collections::VecDeque;
use std::ffi::CString;
use std::fmt;
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Sub};
use std::ptr;
use std::rc::Rc;
use std::str;

use super::gl;
use super::gl::types::*;

pub(super) const COLOR_VERTEX_SHADER_SOURCE: &[u8] = b"
#version 400 core

layout (location = 0) in vec3 aPos;

uniform mat4 transform;

void main() {
	gl_Position = transform*vec4(aPos.x, aPos.y, aPos.z, 1.0);
}
";

pub(super) const COLOR_FRAGMENT_SHADER_SOURCE: &[u8] = b"
#version 400 core

uniform vec4 color;

out vec4 FragColor;

void main() {
	FragColor = color;
}
";

pub(super) const TEX_VERTEX_SHADER_SOURCE: &[u8] = b"
#version 400 core

uniform mat4 transform;

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 uv;

out vec2 pass_uv;

void main() {
	gl_Position = transform*vec4(aPos.x, aPos.y, aPos.z, 1.0);
	pass_uv = uv;
}
";

pub(super) const TEX_FRAGMENT_SHADER_SOURCE: &[u8] = b"
#version 400 core

uniform sampler2D sampler;

in vec2 pass_uv;

out vec4 FragColor;

void main() {
	FragColor = texture(sampler, pass_uv);
}
";

pub struct VarBuilder {
    strings: Vec<String>,
    used_vars: usize,
    var_prefix: &'static str,
}

impl VarBuilder {
    fn new(prefix: &'static str) -> VarBuilder {
        VarBuilder {
            strings: Vec::new(),
            used_vars: 0,
            var_prefix: prefix,
        }
    }

    fn add(&mut self, expr: &VarExpr) -> usize {
        let mut s = self.format_var(&expr.var);
        let var_pos = self.used_vars;
        s = format!(
            "   {} {}{} = {};\n",
            expr.ty.gl_type(),
            self.var_prefix,
            var_pos,
            s
        );
        self.strings.push(s);
        self.used_vars += 1;
        var_pos
    }

    fn add_strings(&self, mut start: String) -> String {
        for s in &self.strings {
            start = format!("{}{}", start, s);
        }
        start
    }

    fn format_var(&mut self, string: &VarString) -> String {
        let mut string_pos = 0;
        let mut s = String::new();
        for (pos, ref var) in &string.vars {
            let var_pos = if var.key.get().is_none() {
                let v = self.add(&var);
                var.key.set(Some(v));
                v
            } else {
                var.key.get().unwrap()
            };
            s = format!(
                "{}{}{}{}",
                s,
                &string.string[string_pos..*pos],
                self.var_prefix,
                var_pos
            );
            string_pos = *pos;
        }
        s = format!("{}{}", s, &string.string[string_pos..]);
        s
    }
}

// Represents a variable declarartion.
#[derive(Clone, PartialEq)]
struct VarExpr {
    var: VarString,
    ty: DataType,
    key: Cell<Option<usize>>,
}

impl VarExpr {
    fn new(var: VarString, ty: DataType) -> VarExpr {
        VarExpr {
            var: var,
            ty: ty,
            key: Cell::new(None),
        }
    }
}

// A varstring is a type that is used interally for building a graph for the shader.
#[derive(Clone, PartialEq)]
pub struct VarString {
    string: String,
    // each value holds the position in the string and the name of the var
    // these are expected to be sorted
    vars: Vec<(usize, Rc<VarExpr>)>,
}

impl VarString {
    pub(crate) fn new<S: fmt::Display>(string: S) -> VarString {
        VarString {
            string: format!("{}", string),
            vars: Vec::new(),
        }
    }

    fn from_expr(expr: VarExpr) -> VarString {
        VarString {
            string: String::new(),
            vars: vec![(0, Rc::new(expr))],
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.string.len()
    }

    pub(crate) fn substring(&self, start: usize, end: usize) -> VarString {
        let s = format!("{}", &self.string[start..end]);
        let mut v = Vec::new();
        for (pos, var) in &self.vars {
            if *pos >= start && *pos < end {
                v.push((pos - start, var.clone()));
            }
        }
        VarString { string: s, vars: v }
    }

    pub(crate) fn format(formatter: &str, strings: Vec<VarString>) -> VarString {
        let mut frags = formatter.split("$");
        let mut string = if let Some(frag) = frags.next() {
            format!("{}", frag)
        } else {
            String::new()
        };
        let mut vars = Vec::with_capacity(strings.len());
        let mut dq = VecDeque::from(strings);
        while let Some(frag) = frags.next() {
            let s = dq.pop_front().unwrap();
            let mut dq2 = VecDeque::from(s.vars);
            while let Some(var) = dq2.pop_front() {
                vars.push((var.0 + string.len(), var.1));
            }
            string = format!("{}{}{}", string, s.string, frag);
        }
        VarString {
            string: string,
            vars: vars,
        }
    }
}

macro_rules! var_format {
    ($fmt:expr) => (
        VarString::new($fmt)
    );
    ($f0:expr, $($fmts:expr),*; $($string:expr),*) => (
        {
            let mut string = format!("{}", $f0);
            let mut vars = Vec::new();
            $(
                let s = $string;
                let mut dq = std::collections::VecDeque::from(s.vars);
                while let Some(var) = dq.pop_front() {
                    vars.push((var.0 + string.len(), var.1));
                }
                string = format!("{}{}{}", string, s.string, $fmts);
            )*
            VarString {
                string: string,
                vars: vars,
            }
        }
    )
}

// represents the main body of a glsl loop
#[derive(Clone, PartialEq)]
struct LoopIter {
    ty: DataType,
    // test is a repeat check dependent on the variables in the loop
    test: VarString,
}

/// Represents a glsl data type.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    Vec(PrimType, u8),
    Float2x2,
    Float2x3,
    Float2x4,
    Float3x2,
    Float3x3,
    Float3x4,
    Float4x2,
    Float4x3,
    Float4x4,
    Sampler2D,
    IntSampler2D,
    UIntSampler2D,
    FloatSampler2D,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PrimType {
    Float,
    Int,
    UInt,
    Boolean,
}

impl DataType {
    pub fn gl_type(self) -> &'static str {
        use PrimType::*;

        match self {
            DataType::Vec(Float, 1) => "float",
            DataType::Vec(Float, 2) => "vec2",
            DataType::Vec(Float, 3) => "vec3",
            DataType::Vec(Float, 4) => "vec4",
            DataType::Vec(Int, 1) => "int",
            DataType::Vec(Int, 2) => "ivec2",
            DataType::Vec(Int, 3) => "ivec3",
            DataType::Vec(Int, 4) => "ivec4",
            DataType::Vec(UInt, 1) => "uint",
            DataType::Vec(UInt, 2) => "uvec2",
            DataType::Vec(UInt, 3) => "uvec3",
            DataType::Vec(Boolean, 4) => "uvec4",
            DataType::Vec(Boolean, 1) => "bool",
            DataType::Vec(Boolean, 2) => "bvec2",
            DataType::Vec(Boolean, 3) => "bvec3",
            DataType::Vec(Boolean, 4) => "bvec4",
            DataType::Float2x2 => "mat2",
            DataType::Float2x3 => "mat2x3",
            DataType::Float2x4 => "mat2x4",
            DataType::Float3x2 => "mat3x2",
            DataType::Float3x3 => "mat3",
            DataType::Float3x4 => "mat3x4",
            DataType::Float4x2 => "mat4x2",
            DataType::Float4x3 => "mat4x3",
            DataType::Float4x4 => "mat4x4",
            DataType::Sampler2D => "sampler2D",
            DataType::IntSampler2D => "isampler2D",
            DataType::UIntSampler2D => "usampler2D",
            DataType::FloatSampler2D => "fsampler2D",
            _ => panic!("Unrecognized data type."),
        }
    }
}

/// This needs to be public because it is used by a function of
/// the ShaderArgs trait.
pub struct ShaderArgDataList {
    args: Vec<(DataType, VarString)>,
}

/// Another type that annoyingly needs to be public.
#[derive(PartialEq, Eq)]
pub struct ShaderArgList {
    args: Vec<DataType>,
}

pub use builtin_vars::{
    BuiltInFragInputs, BuiltInFragOutputs, BuiltInVertInputs, BuiltInVertOutputs,
};

use builtin_vars::BuiltInOutputs;

#[allow(unused)]
pub mod builtin_vars {
    use super::api::*;
    use super::traits::*;
    use super::ProgramBuilderItem;
    use super::{ItemRef, VarBuilder, VarString};
    use crate::tuple::{AttachBack, AttachFront, RemoveBack, RemoveFront};
    use std::marker::PhantomData;

    pub trait BuiltInOutputs {
        fn get_strings(self, v: &mut VarBuilder) -> String;
    }

    pub trait BuiltInOutput<'a, T: ArgType> {
        fn as_arg(self) -> Option<VarString>;
    }

    impl<'a, T: ArgType> BuiltInOutput<'a, T> for () {
        fn as_arg(self) -> Option<VarString> {
            None
        }
    }

    impl<'a, T: ArgType, S: ItemType<'a, Ty = T>> BuiltInOutput<'a, T> for S {
        fn as_arg(self) -> Option<VarString> {
            Some(self.as_string())
        }
    }

    fn create_in<'a, E: ExprType, T: ArgType>(s: &'static str) -> ProgramBuilderItem<'a, T, E> {
        ProgramBuilderItem::create(VarString::new(s), ItemRef::Static)
    }

    pub struct BuiltInVertInputs {
        pub vertex_id: ProgramBuilderItem<'static, Int, Varying>,
        pub instance_id: ProgramBuilderItem<'static, Int, Varying>,
        pub depth_near: ProgramBuilderItem<'static, Float, Uniform>,
        pub depth_far: ProgramBuilderItem<'static, Float, Uniform>,
        pub depth_diff: ProgramBuilderItem<'static, Float, Uniform>,
    }

    impl BuiltInVertInputs {
        pub(super) fn new() -> BuiltInVertInputs {
            BuiltInVertInputs {
                vertex_id: create_in("gl_VertexID"),
                instance_id: create_in("gl_InstanceID"),
                depth_near: create_in("gl_DepthRange.near"),
                depth_far: create_in("gl_DepthRange.far"),
                depth_diff: create_in("gl_DepthRange.diff"),
            }
        }
    }

    pub struct BuiltInVertOutputs<'a> {
        position: VarString,
        point_size: Option<VarString>,
        clip_distance: (),
        phantom: PhantomData<&'a ()>,
    }

    impl<'a> BuiltInOutputs for BuiltInVertOutputs<'a> {
        fn get_strings(self, builder: &mut VarBuilder) -> String {
            let mut string = format!("   gl_Position = {};\n", builder.format_var(&self.position));
            if let Some(s) = self.point_size {
                string = format!("{}   gl_PointSize = {};\n", string, builder.format_var(&s));
            }
            string
        }
    }

    impl<'a> BuiltInVertOutputs<'a> {
        pub fn position<T: ItemType<'a, Ty = Float4>>(position: T) -> BuiltInVertOutputs<'a> {
            BuiltInVertOutputs {
                position: position.as_string(),
                point_size: None,
                clip_distance: (),
                phantom: PhantomData,
            }
        }

        pub fn create<T: ItemType<'a, Ty = Float>, S: BuiltInOutput<'a, Float>>(
            position: T,
            point_size: S,
        ) -> BuiltInVertOutputs<'a> {
            BuiltInVertOutputs {
                position: position.as_string(),
                point_size: point_size.as_arg(),
                clip_distance: (),
                phantom: PhantomData,
            }
        }
    }

    pub struct BuiltInFragInputs {
        pub frag_coord: ProgramBuilderItem<'static, Float4, Varying>,
        pub point_coord: ProgramBuilderItem<'static, Float2, Varying>,
        pub primitive_id: ProgramBuilderItem<'static, Int, Varying>,
        pub clip_distance: (),
        pub depth_near: ProgramBuilderItem<'static, Float, Uniform>,
        pub depth_far: ProgramBuilderItem<'static, Float, Uniform>,
        pub depth_diff: ProgramBuilderItem<'static, Float, Uniform>,
    }

    impl BuiltInFragInputs {
        pub(super) fn new() -> BuiltInFragInputs {
            BuiltInFragInputs {
                frag_coord: create_in("gl_FragCoord"),
                point_coord: create_in("gl_PointCoord"),
                primitive_id: create_in("gl_PrimitiveID"),
                clip_distance: (),
                depth_near: create_in("gl_DepthRange.near"),
                depth_far: create_in("gl_DepthRange.far"),
                depth_diff: create_in("gl_DepthRange.diff"),
            }
        }
    }

    pub struct BuiltInFragOutputs<'a> {
        depth: Option<VarString>,
        discard: Option<VarString>,
        phantom: PhantomData<&'a ()>,
    }

    impl<'a> BuiltInOutputs for BuiltInFragOutputs<'a> {
        fn get_strings(self, builder: &mut VarBuilder) -> String {
            let mut string = String::new();
            if let Some(disc) = self.discard {
                string = format!("   if ({}) discard;\n", builder.format_var(&disc));
            }
            if let Some(dp) = self.depth {
                string = format!("{}   gl_FragDepth = {};\n", string, builder.format_var(&dp));
            }
            string
        }
    }

    impl<'a> BuiltInFragOutputs<'a> {
        pub fn empty() -> BuiltInFragOutputs<'static> {
            BuiltInFragOutputs {
                depth: None,
                discard: None,
                phantom: PhantomData,
            }
        }

        pub fn depth<T: ItemType<'a, Ty = Float>>(depth: T) -> BuiltInFragOutputs<'a> {
            unsafe {
                BuiltInFragOutputs {
                    depth: Some(depth.as_string()),
                    discard: None,
                    phantom: PhantomData,
                }
            }
        }

        pub fn create<T: BuiltInOutput<'a, Float>, S: BuiltInOutput<'a, Boolean>>(
            depth: T,
            discard: S,
        ) -> BuiltInFragOutputs<'a> {
            BuiltInFragOutputs {
                depth: depth.as_arg(),
                discard: discard.as_arg(),
                phantom: PhantomData,
            }
        }
    }
}

use super::GlResource;

pub struct ShaderProgram<In: ShaderArgs, Uniforms: ShaderArgs, Images: ShaderArgs, Out: ShaderArgs>
{
    pub(crate) uniform_locations: Vec<GLint>,
    pub(crate) image_locations: Vec<GLint>,
    pub(crate) program_id: u32,
    // need to make sure the type is not send or sync
    phantom: PhantomData<(In, Uniforms, Images, Out, std::rc::Rc<()>)>,
}

impl<In: ShaderArgs, Uniforms: ShaderArgs, Images: ShaderArgs, Out: ShaderArgs> Drop
    for ShaderProgram<In, Uniforms, Images, Out>
{
    fn drop(&mut self) {
        let gl_draw = unsafe { super::inner_gl_unsafe() };
        gl_draw.remove_resource(self.program_id);
    }
}

#[cfg(not(feature = "opengl41"))]
impl<In: ShaderArgs, Uniforms: ShaderArgs, Images: ShaderArgs, Out: ShaderArgs> GlResource
    for ShaderProgram<In, Uniforms, Images, Out>
{
    unsafe fn adopt(ptr: *mut (), id: u32) -> Option<*mut ()> {
        // we create a reference to the array to avoid dropping the CStrings
        let b = unsafe { &mut *(ptr as *mut [CString; 2]) };
        let gl_draw = unsafe { super::inner_gl_unsafe() };
        // since the underlying gl is only passed a pointer and not the slice length,
        // `as_bytes()` is equivalent to `as_bytes_with_nul()`
        let program = get_program(
            b.strings[0].as_bytes_with_nul(),
            b.strings[1].as_bytes_with_nul(),
        );
        ptr::write(b.id, 0);
        gl_draw.resource_list[id as usize] = program;
        None
    }

    unsafe fn drop_while_orphaned(ptr: *mut (), _id: u32) {
        let b = unsafe { Box::from_raw(ptr as *mut [CString; 2]) };
        // drop the box
    }

    unsafe fn cleanup(ptr: *mut (), _id: u32) {
        let b = unsafe { Box::from_raw(ptr as *mut [CString; 2]) };
        // drop the box
    }

    unsafe fn orphan(_id: u32, ptr: *mut ()) -> *mut () {
        // nothing needs to be done here, the data is stored in the pointer regardless
        // or orphan state
        ptr
    }
}

#[cfg(feature = "opengl41")]
impl<In: ShaderArgs, Uniforms: ShaderArgs, Images: ShaderArgs, Out: ShaderArgs> GlResource
    for ShaderProgram<In, Uniforms, Images, Out>
{
    unsafe fn adopt(ptr: *mut (), id: u32) -> Option<*mut ()> {
        let gl_draw = super::inner_gl_unsafe();
        let [data_len, format, drop_len] = ptr::read(ptr as *const [u32; 3]);
        let ptr = ptr as *mut u32;
        gl::with_current(|gl| {
            let program = gl.CreateProgram();
            gl.ProgramBinary(program, format, ptr.offset(3) as *const _, data_len as i32);
            gl_draw.resource_list[id as usize] = program;
        });
        let _drop_vec = Vec::from_raw_parts(ptr, drop_len as usize, drop_len as usize);
        // drop the data
        None
    }

    unsafe fn drop_while_orphaned(ptr: *mut (), _id: u32) {
        let [_, _, drop_len] = ptr::read(ptr as *const [u32; 3]);
        let _drop_vec = Vec::from_raw_parts(ptr, drop_len as usize, drop_len as usize);
        // drop the data
    }

    unsafe fn cleanup(_ptr: *mut (), _id: u32) {
        // nothing needs to be done here, since a shader does not
        // store any data in the pointer when not in an orphan state
    }

    unsafe fn orphan(id: u32, ptr: *mut ()) -> *mut () {
        gl::with_current(|gl| {
            let gl_draw = super::inner_gl_unsafe();
            let program = gl_draw.resource_list[id as usize];
            let mut len = 0;
            let mut format = 0;
            gl.GetProgramiv(program, gl::PROGRAM_BINARY_LENGTH, &mut len);
            // adding 3 to len rounds up if the binary length is not a multiple
            // of 4
            let mut buffer = Vec::<u32>::with_capacity(3 + (len as usize + 3) >> 2);
            let cap = buffer.capacity();
            let mut data_len = 0;
            let ptr = buffer.as_mut_ptr();
            std::mem::forget(buffer);
            gl.GetProgramBinary(
                program,
                len,
                &mut data_len,
                &mut format,
                ptr.offset(3) as *mut _,
            );
            ptr.write(data_len as u32);
            ptr.offset(1).write(format);
            ptr.offset(2).write(cap as u32);
            ptr as *mut ()
        })
    }
}

use super::mesh::Mesh;
use super::target::RenderTarget;
use super::texture::ImageBindings;
use super::DrawMode;
use super::GlWindow;
use render_options::RenderOptions;

impl<
        In: ShaderArgs,
        Uniforms: ShaderArgsClass<UniformArgs>,
        Images: ShaderArgsClass<ImageArgs>,
        Out: ShaderArgs,
    > ShaderProgram<In, Uniforms, Images, Out>
{
    pub fn draw<M: Mesh<In>, Target: RenderTarget<Out>, O: RenderOptions, F: Fn(M::Drawer)>(
        &self,
        _context: &GlWindow,
        mesh: &M,
        uniforms: &super::mesh::uniform::Uniforms<Uniforms>,
        images: ImageBindings<Images>,
        target: &Target,
        draw: F,
        mode: DrawMode,
        options: O,
    ) {
        unsafe {
            // this will always be the active gl after bind_target is called
            let gl = target.bind_target();
            // need to make sure the reference is destroyed before calling
            // set_uniforms or Mesh::bind because those functions might use
            // a mutable reference to the draw core (though it is probably always
            // a static reference).
            {
                let gl_draw = super::inner_gl_unsafe_static();
                gl.UseProgram(gl_draw.resource_list[self.program_id as usize]);
            }

            let mut slice = &self.uniform_locations[..];

            uniforms.set_uniforms(gl, || {
                let loc = slice[0];
                slice = &slice[1..];
                loc as u32
            });

            for i in 0..self.image_locations.len() {
                gl.Uniform1i(self.image_locations[i], i as i32);
            }

            images.bind(gl);

            mesh.bind(gl);

            let drawer = mesh.create_drawer(mode);

            options.with_options(gl, || draw(drawer));

            // we want to make sure any subsequent binds to
            // the the element array buffer don't effect vao state
            // (this is likely unecessary)
            gl.BindVertexArray(0);
        }
    }

    #[cfg(feature = "opengl41")]
    fn new(
        program: GLuint,
        uniform_locations: Vec<GLint>,
        image_locations: Vec<GLint>,
        _vsource: CString,
        _fsource: CString,
    ) -> ShaderProgram<In, Uniforms, Images, Out> {
        let gl_draw = unsafe { super::inner_gl_unsafe() };
        ShaderProgram {
            uniform_locations: uniform_locations,
            image_locations: image_locations,
            program_id: gl_draw.get_resource_generic::<Self>(program, None),
            phantom: PhantomData,
        }
    }

    #[cfg(not(feature = "opengl41"))]
    fn new(
        program: GLuint,
        uniform_locations: Vec<GLint>,
        image_locations: Vec<GLint>,
        vsource: CString,
        fsource: CString,
    ) -> ShaderProgram<In, Uniforms, Images, Out> {
        let gl_draw = super::inner_gl_unsafe();
        let b = Box::new([vsource, fsource]);
        let ptr = b.as_mut_ptr();
        mem::forget(b);
        ShaderProgram {
            uniform_locations: uniform_locations,
            image_locations: image_locations,
            program_id: gl_draw.get_resource_generic::<Self>(program, Some(ptr)),
            phantom: PhantomData,
        }
    }
}

pub mod render_options {
    use crate::opengl::Gl;
    use crate::opengl::GlWindow;
    use crate::tuple::{AttachFront, RemoveFront};

    pub unsafe trait RenderOptions {
        unsafe fn with_options<F: FnOnce()>(self, gl: &Gl, clo: F);
    }

    unsafe impl RenderOptions for () {
        unsafe fn with_options<F: FnOnce()>(self, gl: &Gl, clo: F) {
            clo();
        }
    }

    unsafe impl<T: RemoveFront> RenderOptions for T
    where
        T::Front: RenderOption,
        T::Remaining: RenderOptions,
    {
        unsafe fn with_options<F: FnOnce()>(self, gl: &Gl, clo: F) {
            let (front, remaining) = self.remove_front();
            let flags = front.set(gl);
            remaining.with_options(gl, clo);
            front.unset(gl, flags);
        }
    }

    pub unsafe trait RenderOption: Copy {
        unsafe fn set(self, gl: &Gl) -> u32;

        unsafe fn unset(self, gl: &Gl, flags: u32);
    }

    #[derive(Clone, Copy)]
    pub struct DisableDepth;

    pub const DISABLE_DEPTH: DisableDepth = DisableDepth;

    pub struct Depth {
        pub depth_enabled: bool,
    }

}

pub struct ProgramBuilder<'a> {
    phantom: PhantomData<&'a ()>,
}

/// This function is used to provide a ProgramBuilder to
/// a closure. This helps to ensure that the variables used to construct
/// a program don't outlive the invocation of `create_program`
pub fn build_program<T, F: FnOnce(ProgramBuilder) -> T>(f: F) -> T {
    let b = ProgramBuilder {
        phantom: PhantomData,
    };
    f(b)
}

// whether the current scope (from a shader compilation standpoint) has
// implicit derivatives

thread_local! {
    pub(crate) static SCOPE_DERIVS: Cell<bool> = Cell::new(false);
}

/// Create a shader program with a vertex and a fragment shader.
///
/// Shaders can be used to cause more advanced behavior to happen on the GPU. The
/// most common use is for lighting calculations. Shaders are a lower level feature
/// that can be difficult to use correctly.
pub fn create_program<
    'a,
    Uniforms: ShaderArgsClass<UniformArgs> + IntoWrapped<'a, Uniform>,
    Images: ShaderArgsClass<ImageArgs> + IntoWrapped<'a, Uniform>,
    In: ShaderArgsClass<TransparentArgs> + IntoWrapped<'a, Varying>,
    Pass: WrappedShaderArgs<'a>,
    Out: WrappedShaderArgs<'a>,
    VertFN: FnOnce(
        In::Wrapped,
        Uniforms::Wrapped,
        Images::Wrapped,
        BuiltInVertInputs,
    ) -> (Pass, BuiltInVertOutputs<'a>),
    FragFN: FnOnce(
        <Pass::Inner as IntoWrapped<'a, Varying>>::Wrapped,
        Uniforms::Wrapped,
        Images::Wrapped,
        BuiltInFragInputs,
    ) -> (Out, BuiltInFragOutputs<'a>),
>(
    _window: &super::GlWindow,
    builder: ProgramBuilder<'a>,
    vertex_shader_fn: VertFN,
    fragment_shader_fn: FragFN,
) -> ShaderProgram<In, Uniforms, Images, Out::Inner>
where
    Out::Inner:
        ShaderArgsClass<TransparentArgs> + ShaderArgsClass<OutputArgs> + IntoWrapped<'a, Varying>,
    Pass::Inner: ShaderArgsClass<TransparentArgs> + IntoWrapped<'a, Varying>,
{
    let v_string = CString::new(create_shader_string::<
        In,
        Uniforms,
        Images,
        Pass,
        BuiltInVertInputs,
        BuiltInVertOutputs,
        VertFN,
    >(
        vertex_shader_fn,
        BuiltInVertInputs::new(),
        "in",
        "u",
        "tex",
        "pass",
        true,
        false,
    ))
    .unwrap();
    SCOPE_DERIVS.with(|x| x.set(true));
    let f_string = CString::new(create_shader_string::<
        Pass::Inner,
        Uniforms,
        Images,
        Out,
        BuiltInFragInputs,
        BuiltInFragOutputs,
        FragFN,
    >(
        fragment_shader_fn,
        BuiltInFragInputs::new(),
        "pass",
        "u",
        "tex",
        "out",
        false,
        true,
    ))
    .unwrap();
    SCOPE_DERIVS.with(|x| x.set(false));
    let program = get_program(v_string.as_bytes_with_nul(), f_string.as_bytes_with_nul());
    let mut uniform_locations = vec![0; Uniforms::NARGS];
    let mut image_locations = vec![0; Images::NARGS];
    unsafe {
        gl::with_current(|gl| {
            if cfg!(feature = "opengl41") {
                gl.ProgramParameteri(
                    program,
                    gl::PROGRAM_BINARY_RETRIEVABLE_HINT,
                    gl::TRUE as i32,
                );
            }

            for i in 0..Uniforms::NARGS {
                uniform_locations[i] = gl.GetUniformLocation(
                    program,
                    CString::new(format!("u{}", i)).unwrap().as_ptr() as *const _,
                );
            }

            for i in 0..Images::NARGS {
                image_locations[i] = gl.GetUniformLocation(
                    program,
                    CString::new(format!("tex{}", i)).unwrap().as_ptr() as *const _,
                );
            }
        });
    }
    ShaderProgram::new(
        program,
        uniform_locations,
        image_locations,
        v_string,
        f_string,
    )
}

#[cfg(not(feature = "opengl41"))]
const VERSION: &str = "#version 400 core";

#[cfg(all(not(feature = "opengl42"), feature = "opengl41"))]
const VERSION: &str = "#version 410 core";

#[cfg(all(not(feature = "opengl43"), feature = "opengl42"))]
const VERSION: &str = "#version 420 core";

#[cfg(all(not(feature = "opengl44"), feature = "opengl43"))]
const VERSION: &str = "#version 430 core";

#[cfg(all(not(feature = "opengl45"), feature = "opengl44"))]
const VERSION: &str = "#version 440 core";

#[cfg(all(not(feature = "opengl46"), feature = "opengl45"))]
const VERSION: &str = "#version 450 core";

#[cfg(feature = "opengl46")]
const VERSION: &str = "#version 460 core";

fn create_shader_string<
    'a,
    In: ShaderArgsClass<TransparentArgs> + IntoWrapped<'a, Varying>,
    Uniforms: ShaderArgs + IntoWrapped<'a, Uniform>,
    Images: ShaderArgs + IntoWrapped<'a, Uniform>,
    Out: WrappedShaderArgs<'a>,
    T,
    S: builtin_vars::BuiltInOutputs,
    Shader: FnOnce(In::Wrapped, Uniforms::Wrapped, Images::Wrapped, T) -> (Out, S),
>(
    generator: Shader,
    gen_in: T,

    in_str: &str,
    uniform_str: &str,
    image_str: &str,
    out_str: &str,

    input_qualifiers: bool,
    output_qualifiers: bool,
) -> String
where
    Out::Inner: ShaderArgsClass<TransparentArgs>,
{
    let mut shader = format!("{}\n", VERSION);
    let in_args = In::map_args();
    let mut position = 0;
    for i in 0..In::NARGS {
        if input_qualifiers {
            shader = format!(
                "{}layout(location = {}) in {} {}{};\n",
                shader,
                position,
                in_args[i].gl_type(),
                in_str,
                i,
            );
            position += In::get_param(i).num_input_locations;
        } else {
            shader = format!("{}in {} {}{};\n", shader, in_args[i].gl_type(), in_str, i,);
        }
    }
    shader = format!("{}\n", shader);
    let uniform_args = Uniforms::map_args();
    for i in 0..Uniforms::NARGS {
        shader = format!(
            "{}uniform {} {}{};\n",
            shader,
            uniform_args[i].gl_type(),
            uniform_str,
            i
        );
    }
    shader = format!("{}\n", shader);
    let image_args = Images::map_args();
    for i in 0..Images::NARGS {
        shader = format!(
            "{}uniform {} {}{};\n",
            shader,
            image_args[i].gl_type(),
            image_str,
            i,
        );
    }
    shader = format!("{}\n", shader);
    let out_args = Out::Inner::map_args();
    let mut position = 0;
    for i in 0..Out::Inner::NARGS {
        if output_qualifiers {
            shader = format!(
                "{}layout(location = {}) out {} {}{};\n",
                shader,
                position,
                out_args[i].gl_type(),
                out_str,
                i,
            );
            position += Out::Inner::get_param(i).num_locations;
        } else {
            shader = format!(
                "{}out {} {}{};\n",
                shader,
                out_args[i].gl_type(),
                out_str,
                i,
            );
        }
    }
    // the create functions are marked as unsafe because it is neccesary to
    // ensure that the names created are defined in the shader.
    let (out, bout) = unsafe {
        let input = In::into_wrapped(in_str);
        let image = Images::into_wrapped(image_str);
        let uniform = Uniforms::into_wrapped(uniform_str);
        generator(input, uniform, image, gen_in)
    };
    shader = format!("{}\n\nvoid main() {{\n", shader);
    let mut builder = VarBuilder::new("var");
    let mut out_strings = out.get_strings(out_str, &mut builder);
    let bstring = bout.get_strings(&mut builder);
    shader = builder.add_strings(shader);
    shader = format!("{}{}\n", shader, bstring);
    shader = format!("{}{}\n", shader, out_strings);
    shader = format!("{}}}\n", shader);
    println!("{}\n", shader);
    shader
}

pub(super) fn get_program(vertex_source: &[u8], fragment_source: &[u8]) -> GLuint {
    unsafe {
        gl::with_current(|gl| {
            let vertex_shader = gl.CreateShader(gl::VERTEX_SHADER);
            gl.ShaderSource(
                vertex_shader,
                1,
                &(vertex_source.as_ptr() as *const i8),
                &((&vertex_source).len() as i32),
            );
            gl.CompileShader(vertex_shader);

            // check for shader compile errors
            let mut success = gl::FALSE as GLint;
            let mut info_log = Vec::with_capacity(512);
            let mut log_len = 0;
            gl.GetShaderiv(vertex_shader, gl::COMPILE_STATUS, &mut success);
            if success != gl::TRUE as GLint {
                gl.GetShaderInfoLog(
                    vertex_shader,
                    512,
                    &mut log_len,
                    info_log.as_mut_ptr() as *mut GLchar,
                );
                info_log.set_len(log_len as usize);
                println!(
                    "Vertex shader compilation failed.\n{}",
                    str::from_utf8(&info_log).unwrap()
                );
            }

            // fragment shader
            let fragment_shader = gl.CreateShader(gl::FRAGMENT_SHADER);
            gl.ShaderSource(
                fragment_shader,
                1,
                &(fragment_source.as_ptr() as *const i8),
                &((&fragment_source).len() as i32),
            );
            gl.CompileShader(fragment_shader);
            // check for shader compile errors
            gl.GetShaderiv(fragment_shader, gl::COMPILE_STATUS, &mut success);
            if success != gl::TRUE as GLint {
                gl.GetShaderInfoLog(
                    fragment_shader,
                    512,
                    &mut log_len,
                    info_log.as_mut_ptr() as *mut GLchar,
                );
                info_log.set_len(log_len as usize);
                println!(
                    "Fragment shader compilation failed.\n{}",
                    str::from_utf8(&info_log).unwrap()
                );
            }

            // link shaders
            let shader_program = gl.CreateProgram();
            gl.AttachShader(shader_program, vertex_shader);
            gl.AttachShader(shader_program, fragment_shader);
            gl.LinkProgram(shader_program);
            // check for linking errors
            gl.GetProgramiv(shader_program, gl::LINK_STATUS, &mut success);
            if success != gl::TRUE as GLint {
                gl.GetProgramInfoLog(
                    shader_program,
                    512,
                    &mut log_len,
                    info_log.as_mut_ptr() as *mut GLchar,
                );
                info_log.set_len(log_len as usize);
                println!(
                    "Program linking failed.\n{}",
                    str::from_utf8(&info_log).unwrap()
                );
            }
            gl.DeleteShader(vertex_shader);
            gl.DeleteShader(fragment_shader);
            shader_program
        })
    }
}

// shouldn't cause any problems
use api::*;
use traits::*;
use vec::*;

pub mod api {
    use super::traits::{Construct, ExprType};
    use super::vec;
    use super::ProgramBuilderItem;
    use vec::GlVec;
    use vec::{Vec1, Vec2, Vec3, Vec4};

    pub type Float = GlVec<vec::VecFloat, Vec1>;
    pub type Float2 = GlVec<vec::VecFloat, Vec2>;
    pub type Float3 = GlVec<vec::VecFloat, Vec3>;
    pub type Float4 = GlVec<vec::VecFloat, Vec4>;

    pub type Int = GlVec<vec::VecInt, Vec1>;
    pub type Int2 = GlVec<vec::VecInt, Vec2>;
    pub type Int3 = GlVec<vec::VecInt, Vec3>;
    pub type Int4 = GlVec<vec::VecInt, Vec4>;

    pub type UInt = GlVec<vec::VecUInt, Vec1>;
    pub type UInt2 = GlVec<vec::VecUInt, Vec2>;
    pub type UInt3 = GlVec<vec::VecUInt, Vec3>;
    pub type UInt4 = GlVec<vec::VecUInt, Vec4>;

    pub type Boolean = GlVec<vec::VecBoolean, Vec1>;
    pub type Boolean2 = GlVec<vec::VecBoolean, Vec2>;
    pub type Boolean3 = GlVec<vec::VecBoolean, Vec3>;
    pub type Boolean4 = GlVec<vec::VecBoolean, Vec4>;

    pub use super::{
        Float2x2, Float2x3, Float2x4, Float3x2, Float3x3, Float3x4, Float4x2, Float4x3, Float4x4,
    };

    pub type FragOutputs<'a> = super::BuiltInFragOutputs<'a>;
    pub type VertOutputs<'a> = super::BuiltInVertOutputs<'a>;

    macro_rules! named_construct {
        ($t:ty, $name:ident) => {
            pub fn $name<'a, E: ExprType, T: Construct<'a, $t, E>>(
                t: T,
            ) -> ProgramBuilderItem<'a, $t, E> {
                super::construct(t)
            }
        };
    }

    named_construct!(Float, float);
    named_construct!(Float2, float2);
    named_construct!(Float3, float3);
    named_construct!(Float4, float4);
    named_construct!(Int, int);
    named_construct!(Int2, int2);
    named_construct!(Int3, int3);
    named_construct!(Int4, int4);
    named_construct!(UInt, uint);
    named_construct!(UInt2, uint2);
    named_construct!(UInt3, uint3);
    named_construct!(UInt4, uint4);
    named_construct!(Boolean, boolean);
    named_construct!(Boolean2, boolean2);
    named_construct!(Boolean3, boolean3);
    named_construct!(Boolean4, boolean4);
    named_construct!(Float4x4, float4x4);
    named_construct!(Float4x3, float4x3);
    named_construct!(Float4x2, float4x2);
    named_construct!(Float3x4, float3x4);
    named_construct!(Float3x3, float3x3);
    named_construct!(Float3x2, float3x2);
    named_construct!(Float2x4, float2x4);
    named_construct!(Float2x3, float2x3);
    named_construct!(Float2x2, float2x2);

    pub mod map {
        pub use super::vec::{W, X, Y, Z};
    }
}

pub struct ProgramBuilderItem<'a, T: ArgType, E: ExprType> {
    item: ProgramItem,
    phantom: PhantomData<&'a (T, E)>,
}

impl<'a, T: ArgType, E: ExprType> Clone for ProgramBuilderItem<'a, T, E> {
    fn clone(&self) -> ProgramBuilderItem<'a, T, E> {
        ProgramBuilderItem {
            item: self.item.clone(),
            phantom: PhantomData,
        }
    }
}

impl<'a, T: ArgType, E: ExprType> ProgramBuilderItem<'a, T, E> {
    pub(crate) fn create(data: VarString, r: ItemRef) -> ProgramBuilderItem<'a, T, E> {
        ProgramBuilderItem {
            item: ProgramItem::new(data, T::data_type(), r),
            phantom: PhantomData,
        }
    }

    fn create_scoped(data: VarString, r: ItemRef, _scope: &'a ()) -> ProgramBuilderItem<'a, T, E> {
        ProgramBuilderItem {
            item: ProgramItem::new(data, T::data_type(), r),
            phantom: PhantomData,
        }
    }
}

macro_rules! item_ops {
    ($op:ident, $op_fn:ident, $op_sym:expr) => {
        impl<'a, L: ArgType, R: ArgType, E1: ExprType, E2: ExprType> $op<ProgramBuilderItem<'a, R, E1>>
            for ProgramBuilderItem<'a, L, E2>
        where
            L: $op<R>, (E1, E2): ExprCombine,
            L::Output: ArgType,
        {
            type Output = ProgramBuilderItem<'a, L::Output, <(E1, E2) as ExprCombine>::Min>;

            fn $op_fn(self, other: ProgramBuilderItem<'a, R, E1>) -> Self::Output {
                ProgramBuilderItem::create(
                    var_format!("(", $op_sym, ")"; self.as_string(), other.as_string()),
                    Expr,
                )
            }
        }

        impl<'a, L: ArgType, R: ArgType, E1: ExprType, E2: ExprType> $op<ProgramBuilderItem<'a, R, E1>>
            for &ProgramBuilderItem<'a, L, E2>
        where
            L: $op<R>, (E1, E2): ExprCombine,
            L::Output: ArgType,
        {
            type Output = ProgramBuilderItem<'a, L::Output, <(E1, E2) as ExprCombine>::Min>;

            fn $op_fn(self, other: ProgramBuilderItem<'a, R, E1>) -> Self::Output {
                ProgramBuilderItem::create(
                    var_format!("(", $op_sym, ")"; self.as_string(), other.as_string()),
                    Expr,
                )
            }
        }

        impl<'a, L: ArgType, R: ArgType, E1: ExprType, E2: ExprType> $op<&ProgramBuilderItem<'a, R, E1>>
            for ProgramBuilderItem<'a, L, E2>
        where
            L: $op<R>, (E1, E2): ExprCombine,
            L::Output: ArgType,
        {
            type Output = ProgramBuilderItem<'a, L::Output, <(E1, E2) as ExprCombine>::Min>;

            fn $op_fn(self, other: &ProgramBuilderItem<'a, R, E1>) -> Self::Output {
                ProgramBuilderItem::create(
                    var_format!("(", $op_sym, ")"; self.as_string(), other.as_string()),
                    Expr,
                )
            }
        }

        impl<'a, L: ArgType, R: ArgType, E1: ExprType, E2: ExprType> $op<&ProgramBuilderItem<'a, R, E1>>
            for &ProgramBuilderItem<'a, L, E2>
        where
            L: $op<R>, (E1, E2): ExprCombine,
            L::Output: ArgType,
        {
            type Output = ProgramBuilderItem<'a, L::Output, <(E1, E2) as ExprCombine>::Min>;

            fn $op_fn(self, other: &ProgramBuilderItem<'a, R, E1>) -> Self::Output {
                ProgramBuilderItem::create(
                    var_format!("(", $op_sym, ")"; self.as_string(), other.as_string()),
                    Expr,
                )
            }
        }
    };
}

item_ops!(Add, add, "+");
item_ops!(Sub, sub, "-");
item_ops!(Mul, mul, "*");
item_ops!(Div, div, "/");

#[allow(unused, unused_parens)]
pub mod traits {
    use super::gl;
    use super::ItemRef::*;
    use super::VarBuilder;

    use super::{
        DataType, ItemRef, ProgramBuilderItem, ShaderArgDataList, ShaderArgList, VarExpr, VarString,
    };
    use std::ops::{Add, Div, Mul, Neg, Sub};

    pub unsafe trait GlDataType: Copy {
        const TYPE: gl::types::GLenum;
    }

    unsafe impl GlDataType for u8 {
        const TYPE: gl::types::GLenum = gl::UNSIGNED_BYTE;
    }

    unsafe impl GlDataType for i8 {
        const TYPE: gl::types::GLenum = gl::BYTE;
    }

    unsafe impl GlDataType for u16 {
        const TYPE: gl::types::GLenum = gl::UNSIGNED_SHORT;
    }

    unsafe impl GlDataType for i16 {
        const TYPE: gl::types::GLenum = gl::SHORT;
    }

    unsafe impl GlDataType for u32 {
        const TYPE: gl::types::GLenum = gl::UNSIGNED_INT;
    }

    unsafe impl GlDataType for i32 {
        const TYPE: gl::types::GLenum = gl::INT;
    }

    unsafe impl GlDataType for f32 {
        const TYPE: gl::types::GLenum = gl::FLOAT;
    }

    pub unsafe trait ArgType: 'static + Clone {
        fn data_type() -> DataType;
    }

    pub unsafe trait ExprType: 'static {}

    /// This is a helper trait for implementing ExprMin
    pub trait ExprCombine {
        type Min: ExprType;
    }

    /// The trait exprmin is implemented for tuples of Expr types
    pub trait ExprMin {
        type Min: ExprType;
    }

    impl<T: crate::tuple::RemoveFront> ExprMin for T
    where
        T::Front: ExprType,
        T::Remaining: ExprMin,
        (T::Front, <T::Remaining as ExprMin>::Min): ExprCombine,
    {
        type Min = <(T::Front, <T::Remaining as ExprMin>::Min) as ExprCombine>::Min;
    }

    impl ExprMin for () {
        type Min = Constant;
    }

    impl ExprMin for Constant {
        type Min = Constant;
    }

    impl ExprMin for Uniform {
        type Min = Uniform;
    }

    impl ExprMin for Varying {
        type Min = Varying;
    }

    impl ExprCombine for (Constant, Constant) {
        type Min = Constant;
    }

    impl ExprCombine for (Constant, Uniform) {
        type Min = Uniform;
    }

    impl ExprCombine for (Constant, Varying) {
        type Min = Varying;
    }

    impl ExprCombine for (Uniform, Constant) {
        type Min = Uniform;
    }

    impl ExprCombine for (Uniform, Uniform) {
        type Min = Uniform;
    }

    impl ExprCombine for (Uniform, Varying) {
        type Min = Varying;
    }

    impl ExprCombine for (Varying, Constant) {
        type Min = Varying;
    }

    impl ExprCombine for (Varying, Uniform) {
        type Min = Varying;
    }

    impl ExprCombine for (Varying, Varying) {
        type Min = Varying;
    }

    #[derive(Clone, Copy)]
    pub struct Constant {}

    #[derive(Clone, Copy)]
    pub struct Uniform {}

    #[derive(Clone, Copy)]
    pub struct Varying {}

    unsafe impl ExprType for Constant {}

    unsafe impl ExprType for Uniform {}

    unsafe impl ExprType for Varying {}

    pub unsafe trait ArgClass: Copy {}

    pub unsafe trait ArgParameter<T: ArgClass> {
        fn get_param() -> T;
    }

    unsafe impl ArgClass for () {}

    unsafe impl<T: ArgType> ArgParameter<()> for T {
        fn get_param() {}
    }

    /// Most arg types, but not opaque types like samplers.
    #[derive(Clone, Copy)]
    pub struct TransparentArgs {
        // the number of locations can be different depening on where the type is used
        pub num_locations: u32,
        pub num_input_locations: u32,
    }

    unsafe impl ArgClass for TransparentArgs {}

    /// The types of args that can be used as fragment shader outputs. This includes all scalar
    /// and vector types except booleans, but not matrix types. All types that implement
    /// `ArgParameter<OutputArgs>` also implement `ArgParameter<TransparentArgs>`.
    #[derive(Clone, Copy)]
    pub struct OutputArgs;

    unsafe impl ArgClass for OutputArgs {}

    #[derive(Clone, Copy)]
    pub struct UniformArgs {
        pub num_elements: u32,
        pub array_count: u32,
        pub is_mat: bool,
        pub func: usize,
    }

    unsafe impl ArgClass for UniformArgs {}

    #[derive(Clone, Copy)]
    pub struct ImageArgs;

    unsafe impl ArgClass for ImageArgs {}

    /// ShaderArgs is a trait that is implemented for types of
    /// possible opengl argument sets
    pub unsafe trait ShaderArgs {
        const NARGS: usize;

        fn map_args() -> Vec<DataType>;
    }

    pub unsafe trait IntoWrapped<'a, E: ExprType> {
        type Wrapped: WrappedShaderArgs<'a>;

        unsafe fn into_wrapped(prefix: &str) -> Self::Wrapped;
    }

    pub unsafe trait WrappedShaderArgs<'a> {
        type Inner: ShaderArgs;
        type Min: ExprType;

        fn get_strings(self, prefix: &str, b: &mut VarBuilder) -> String;
    }

    unsafe impl<'a, T: ArgType, E: ExprType> WrappedShaderArgs<'a> for ProgramBuilderItem<'a, T, E> {
        type Inner = (T,);
        type Min = E;

        fn get_strings(self, prefix: &str, b: &mut VarBuilder) -> String {
            let s = b.format_var(&self.as_string());

            format!("   {}0 = {};", prefix, s)
        }
    }

    pub trait Construct<'a, T: ArgType, E: ExprType> {
        fn construct(self) -> ProgramBuilderItem<'a, T, E>;
    }

    /// Sometimes it is neccessary to restrict a type to only certain types
    /// of arguments.
    pub unsafe trait ShaderArgsClass<T>: ShaderArgs {
        fn get_param(i: usize) -> T;
    }

    pub unsafe trait ItemType<'a>: Sized {
        type Expr: ExprType;
        type Ty: ArgType;

        fn get_item(self) -> ProgramBuilderItem<'a, Self::Ty, Self::Expr>;

        fn as_string(self) -> VarString {
            self.get_item().item.data.into_inner()
        }
    }

    unsafe impl<'a, T: ArgType, E: ExprType> ItemType<'a> for ProgramBuilderItem<'a, T, E> {
        type Expr = E;
        type Ty = T;

        #[inline(always)]
        fn get_item(self) -> Self {
            self
        }
    }

    unsafe impl<'a, T: ArgType, E: ExprType> ItemType<'a> for &ProgramBuilderItem<'a, T, E> {
        type Expr = E;
        type Ty = T;

        #[inline(always)]
        fn get_item(self) -> ProgramBuilderItem<'a, T, E> {
            self.clone()
        }
    }

    macro_rules! impl_shader_args {
		($($name:ident),*; $num:expr) => (
            #[allow(unused_parens)]
			unsafe impl<$($name: ArgType),*> ShaderArgs for ($($name,)*) {
				const NARGS: usize = $num;

                fn map_args() -> Vec<DataType> {
                    vec![$($name::data_type()),*]
                }
			}

            unsafe impl<'a, $($name: ItemType<'a>),*> WrappedShaderArgs<'a> for
                ($($name,)*) where ($($name::Expr),*): ExprMin {
                type Inner = ($($name::Ty,)*);
                type Min = <($($name::Expr),*) as ExprMin>::Min;

                fn get_strings(self, prefix: &str, b: &mut VarBuilder) -> String {
                    let ($($name,)*) = self;
                    let ($($name,)*) = ($(b.format_var(&$name.as_string()),)*);

                    let mut i = 0;
                    let s = format!("");
                    $(
                        let s = format!("{}   {}{} = {};\n", s, prefix, i, $name);
                        i += 1;
                    )*
                    s
                }
            }

            unsafe impl<'a, E: ExprType + 'a, $($name: ArgType + 'a),*> IntoWrapped<'a, E> for ($($name,)*)
            where
                ($(ProgramBuilderItem<'a, $name, E>,)*): WrappedShaderArgs<'a>,
            {
                type Wrapped = ($(ProgramBuilderItem<'a, $name, E>),*);

                unsafe fn into_wrapped(prefix: &str) -> Self::Wrapped {
                    let mut i = 0;
                    ($({
                        // this doesn't need to reference $name at all,
                        // but if this wasn't here then it wouldn't know
                        // how many times to repeat the expression
                        let _ = stringify!($name);
                        let x = ProgramBuilderItem::create(VarString::new(format!("{}{}", prefix, i)), Static);
                        i += 1;
                        x
                    }),*)
                }
            }

            unsafe impl<T: ArgClass, $($name: ArgType + ArgParameter<T>),*> ShaderArgsClass<T> for ($($name,)*) {
                fn get_param(i: usize) -> T {
                    // note: create an array of function pointers and call the ith one
                    // this is likely faster and more optimizable than some alternatives
                    let a: [fn() -> T; $num] = [$($name::get_param),*];
                    (a[i])()
                }
            }
		)
	}

    macro_rules! args_set {
        ($($u:ident,)*;$n:expr) => (
            impl_shader_args!($($u),*;$n);

            args_set!(;$($u,)*;$n);
        );
        (;;$n:expr) => ();
        (;$u0:ident, $($u:ident,)*;$n:expr) => (
            args_set!($($u,)*; $n-1);
        )
    }

    #[cfg(feature = "longer_tuples")]
    args_set!(U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16,
        U17, U18, U19, U20, U21, U22, U23, U24, U25, U26, U27, U28, U29, U30, U31, U32,
        U33, U34, U35, U36, U37, U38, U39, U40, U41, U42, U43, U44, U45, U46, U47, U48,
        U49, U50, U51, U52, U53, U54, U55, U56, U57, U58, U59, U60, U61, U62, U63, U64,; 64);

    #[cfg(not(feature = "longer_tuples"))]
    args_set!(U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16,; 16);
}

pub mod vec {
    use super::traits::*;
    use super::ItemRef::*;
    use super::{DataType, PrimType, VarBuilder};
    use super::{ProgramBuilderItem, VarString};
    use crate::tuple::RemoveFront;
    use std::marker::PhantomData;
    use std::ops::{Add, Div, Mul, Sub};

    pub unsafe trait VecLen: 'static {
        const LEN: usize;
        type Next;
        type Prev;
    }

    pub unsafe trait VecType: 'static {
        const PTYPE: PrimType;
        type From: Copy;
    }

    pub struct GlVec<T: VecType, L: VecLen> {
        phantom: PhantomData<(T, L)>,
    }

    impl<T: VecType, L: VecLen> Clone for GlVec<T, L> {
        fn clone(&self) -> Self {
            GlVec {
                phantom: PhantomData,
            }
        }
    }

    unsafe impl<T: VecType, L: VecLen> ArgParameter<TransparentArgs> for GlVec<T, L> {
        fn get_param() -> TransparentArgs {
            TransparentArgs {
                num_locations: 1,
                num_input_locations: 1,
            }
        }
    }

    unsafe impl<L: VecLen> ArgParameter<OutputArgs> for GlVec<VecFloat, L> {
        fn get_param() -> OutputArgs {
            OutputArgs
        }
    }

    unsafe impl<L: VecLen> ArgParameter<OutputArgs> for GlVec<VecInt, L> {
        fn get_param() -> OutputArgs {
            OutputArgs
        }
    }

    unsafe impl<L: VecLen> ArgParameter<OutputArgs> for GlVec<VecUInt, L> {
        fn get_param() -> OutputArgs {
            OutputArgs
        }
    }

    macro_rules! vec_litteral {
        ($t:ty, $l:ty, $($c:ty),*; $($place:ident),*) => (
            impl Construct<'static, GlVec<$t, $l>, Constant> for ($($c),*) {
                fn construct(self) -> ProgramBuilderItem<'static, GlVec<$t, $l>, Constant> {
                    let ($($place),*) = self;
                    let s = format!("{}(", <GlVec<$t, $l>>::data_type().gl_type());
                    $(
                        let s = format!("{}{}, ", s, $place);
                    )*
                    // remove the trailing comma
                    let s = format!("{})", &s[..s.len()-2]);
                    ProgramBuilderItem::create(VarString::new(s), Static)
                }
            }
        )
    }

    macro_rules! vec_litterals {
        ($($t:ty, $c:ty),*) => (
            $(
                vec_litteral!($t, Vec1, $c; U1);
                vec_litteral!($t, Vec2, $c; U1);
                vec_litteral!($t, Vec3, $c; U1);
                vec_litteral!($t, Vec4, $c; U1);
                vec_litteral!($t, Vec2, $c, $c; U1, U2);
                vec_litteral!($t, Vec3, $c, $c, $c; U1, U2, U3);
                vec_litteral!($t, Vec4, $c, $c, $c, $c; U1, U2, U3, U4);
            )*
        )
    }

    vec_litterals!(VecFloat, f32, VecInt, i32, VecUInt, u32, VecBoolean, bool);

    #[derive(Clone, Copy)]
    pub struct VecFloat {
        _hidden: (),
    }

    #[derive(Clone, Copy)]
    pub struct VecInt {
        _hidden: (),
    }

    #[derive(Clone, Copy)]
    pub struct VecUInt {
        _hidden: (),
    }

    #[derive(Clone, Copy)]
    pub struct VecBoolean {
        _hidden: (),
    }

    unsafe impl VecType for VecFloat {
        const PTYPE: PrimType = PrimType::Float;
        type From = f32;
    }

    unsafe impl VecType for VecInt {
        const PTYPE: PrimType = PrimType::Int;
        type From = i32;
    }

    unsafe impl VecType for VecUInt {
        const PTYPE: PrimType = PrimType::UInt;
        type From = u32;
    }

    unsafe impl VecType for VecBoolean {
        const PTYPE: PrimType = PrimType::Boolean;
        type From = u32;
    }

    unsafe impl<T: VecType, L: VecLen> ArgType for GlVec<T, L> {
        fn data_type() -> DataType {
            DataType::Vec(T::PTYPE, L::LEN as u8)
        }
    }

    pub struct Vec0;

    pub struct Vec1;

    pub struct Vec2;

    pub struct Vec3;

    pub struct Vec4;

    unsafe impl VecLen for Vec1 {
        const LEN: usize = 1;
        type Next = Vec2;
        type Prev = Vec0;
    }

    unsafe impl VecLen for Vec2 {
        const LEN: usize = 2;
        type Next = Vec3;
        type Prev = Vec1;
    }

    unsafe impl VecLen for Vec3 {
        const LEN: usize = 3;
        type Next = Vec4;
        type Prev = Vec2;
    }

    unsafe impl VecLen for Vec4 {
        const LEN: usize = 4;
        type Next = ();
        type Prev = Vec3;
    }

    pub unsafe trait Maybe0VecLen {
        const LEN: usize;
        type Next;
    }

    unsafe impl Maybe0VecLen for Vec0 {
        const LEN: usize = 0;
        type Next = Vec1;
    }

    unsafe impl<T: VecLen> Maybe0VecLen for T {
        const LEN: usize = T::LEN;
        type Next = T::Next;
    }

    macro_rules! construct_vec {
        ($len:ty, $($t:ty),*;$($place:ident),*) => (
            impl<'a, T: VecType, $($place: ExprType),*>
                Construct<'a, GlVec<T, $len>, <($($place,)*) as ExprMin>::Min> for
                ($(ProgramBuilderItem<'a, GlVec<T, $t>, $place>),*)
            where
                ($($place,)*): ExprMin,
            {
                fn construct(self) ->
                    ProgramBuilderItem<'a, GlVec<T, $len>, <($($place,)*) as ExprMin>::Min>
                {
                    let ($($place,)*) = self;
                    let ($($place,)*) = ($($place.item.data.into_inner(),)*);
                    let s = var_format!(
                        format!("{}(", <GlVec<T, $len>>::data_type().gl_type()),
                        $({let _ = stringify!($place); ", "}),*;
                        $($place),*);
                    // remove the trailing comma
                    let s = s.substring(0, s.len()-2);
                    let s = var_format!("", ")"; s);
                    ProgramBuilderItem::create(s, Expr)
                }
            }
        )
    }

    construct_vec!(Vec4, Vec1, Vec1, Vec1, Vec1; U1, U2, U3, U4);
    construct_vec!(Vec4, Vec1, Vec1, Vec2; U1, U2, U3);
    construct_vec!(Vec4, Vec1, Vec2, Vec1; U1, U2, U3);
    construct_vec!(Vec4, Vec2, Vec1, Vec1; U1, U2, U3);
    construct_vec!(Vec4, Vec1, Vec3; U1, U2);
    construct_vec!(Vec4, Vec3, Vec1; U1, U2);
    construct_vec!(Vec3, Vec1, Vec1, Vec1; U1, U2, U3);
    construct_vec!(Vec3, Vec1, Vec2; U1, U2);
    construct_vec!(Vec3, Vec2, Vec1; U1, U2);
    construct_vec!(Vec2, Vec1, Vec1; U1, U2);

    impl<T: VecType, L: VecLen> Add<GlVec<T, L>> for GlVec<T, L> {
        type Output = GlVec<T, L>;

        fn add(self, rhs: GlVec<T, L>) -> GlVec<T, L> {
            // this implementation exists to generate an implementation
            // for ProgramBuilderItems
            unreachable!()
        }
    }

    impl<T: VecType, L: VecLen> Sub<GlVec<T, L>> for GlVec<T, L> {
        type Output = GlVec<T, L>;

        fn sub(self, rhs: GlVec<T, L>) -> GlVec<T, L> {
            unreachable!()
        }
    }

    impl<T: VecType, L: VecLen> Mul<GlVec<T, L>> for GlVec<T, Vec1> {
        type Output = GlVec<T, L>;

        fn mul(self, rhs: GlVec<T, L>) -> GlVec<T, L> {
            unreachable!()
        }
    }

    impl<T: VecType, L: VecLen> Div<GlVec<T, Vec1>> for GlVec<T, L> {
        type Output = GlVec<T, L>;

        fn div(self, rhs: GlVec<T, Vec1>) -> GlVec<T, L> {
            unreachable!()
        }
    }

    // annoying redundant implementations to avoid implementation conflicts

    impl<T: VecType> Mul<GlVec<T, Vec1>> for GlVec<T, Vec2> {
        type Output = GlVec<T, Vec2>;

        fn mul(self, rhs: GlVec<T, Vec1>) -> GlVec<T, Vec2> {
            unreachable!()
        }
    }

    impl<T: VecType> Mul<GlVec<T, Vec1>> for GlVec<T, Vec3> {
        type Output = GlVec<T, Vec3>;

        fn mul(self, rhs: GlVec<T, Vec1>) -> GlVec<T, Vec3> {
            unreachable!()
        }
    }

    impl<T: VecType> Mul<GlVec<T, Vec1>> for GlVec<T, Vec4> {
        type Output = GlVec<T, Vec4>;

        fn mul(self, rhs: GlVec<T, Vec1>) -> GlVec<T, Vec4> {
            unreachable!()
        }
    }

    pub unsafe trait MapFor<T: VecType, L: VecLen> {
        type L_Out: Maybe0VecLen;

        fn get_str() -> String;
    }

    unsafe impl<L: VecLen, T: VecType> MapFor<T, L> for () {
        type L_Out = Vec0;

        fn get_str() -> String {
            format!("")
        }
    }

    unsafe impl<L: VecLen, M: RemoveFront, T: VecType> MapFor<T, L> for M
    where
        <<M::Remaining as MapFor<T, L>>::L_Out as Maybe0VecLen>::Next: VecLen,
        M::Front: MapElement + MapElementFor<L>,
        M::Remaining: MapFor<T, L>,
    {
        type L_Out = <<M::Remaining as MapFor<T, L>>::L_Out as Maybe0VecLen>::Next;

        fn get_str() -> String {
            format!(
                "{}{}",
                <M::Front as MapElement>::V,
                <M::Remaining as MapFor<T, L>>::get_str()
            )
        }
    }

    pub trait MapElement {
        const V: &'static str;
    }

    pub unsafe trait MapElementFor<L: VecLen>: MapElement {}

    pub struct X;

    pub struct Y;

    pub struct Z;

    pub struct W;

    unsafe impl MapElementFor<Vec1> for X {}

    unsafe impl MapElementFor<Vec2> for X {}

    unsafe impl MapElementFor<Vec3> for X {}

    unsafe impl MapElementFor<Vec4> for X {}

    unsafe impl MapElementFor<Vec2> for Y {}

    unsafe impl MapElementFor<Vec3> for Y {}

    unsafe impl MapElementFor<Vec4> for Y {}

    unsafe impl MapElementFor<Vec3> for Z {}

    unsafe impl MapElementFor<Vec4> for Z {}

    unsafe impl MapElementFor<Vec4> for W {}

    impl MapElement for X {
        const V: &'static str = "x";
    }

    impl MapElement for Y {
        const V: &'static str = "y";
    }

    impl MapElement for Z {
        const V: &'static str = "z";
    }

    impl MapElement for W {
        const V: &'static str = "w";
    }
}

impl<'a, T: VecType, L: VecLen, E: ExprType> Construct<'a, GlVec<T, L>, E>
    for ProgramBuilderItem<'a, GlVec<T, Vec1>, E>
where
    GlVec<T, L>: ArgType,
{
    fn construct(self) -> ProgramBuilderItem<'a, GlVec<T, L>, E> {
        let s = var_format!(
            format!("{}(", <GlVec<T, L>>::data_type().gl_type()), ")";
            self.item.data.into_inner());
        ProgramBuilderItem::create(s, Expr)
    }
}

impl<'a, T: VecType, L: VecLen, E: ExprType> ProgramBuilderItem<'a, GlVec<T, L>, E> {
    pub fn map<M: MapFor<T, L>>(self) -> ProgramBuilderItem<'a, GlVec<T, M::L_Out>, E>
    where
        M::L_Out: VecLen,
    {
        if !cfg!(feature = "opengl42") && L::LEN == 1 {
            let s = var_format!(format!("{}(", <GlVec<T, L>>::data_type().gl_type()), ")"; self.as_string());
            ProgramBuilderItem::create(s, Expr)
        } else {
            let s = var_format!("", format!(".{}", M::get_str()); self.as_string());
            ProgramBuilderItem::create(s, Expr)
        }
    }
}

pub fn construct<'a, T: ArgType, E: ExprType, C: Construct<'a, T, E>>(
    c: C,
) -> ProgramBuilderItem<'a, T, E> {
    c.construct()
}

#[derive(Clone, Copy)]
pub enum ItemRef {
    // used for variable names and shader inputs
    Static,
    // an expression based on other variables
    Expr,
    // a reference to a single variable
    Var,
}

use ItemRef::{Expr, Static, Var};

/// An item that can be used as the data for a shader argument.
pub(crate) struct ProgramItem {
    data: Cell<VarString>,
    ref_type: Cell<ItemRef>,
    ty: DataType,
}

impl ProgramItem {
    pub fn create(data: VarString, ty: DataType) -> ProgramItem {
        ProgramItem {
            data: Cell::new(data),
            ref_type: Cell::new(Static),
            ty: ty,
        }
    }

    pub fn new(data: VarString, ty: DataType, r: ItemRef) -> ProgramItem {
        ProgramItem {
            data: Cell::new(data),
            ref_type: Cell::new(r),
            ty: ty,
        }
    }

    pub fn into_inner(self) -> VarString {
        self.data.into_inner()
    }
}

impl Clone for ProgramItem {
    fn clone(&self) -> ProgramItem {
        let ref_type = self.ref_type.get();
        let expression = self.data.replace(VarString::new(""));
        match ref_type {
            Expr => {
                let string = VarString::from_expr(VarExpr::new(expression, self.ty));
                self.data.set(string.clone());
                ProgramItem {
                    data: Cell::new(string),
                    ref_type: Cell::new(Var),
                    ty: self.ty,
                }
            }
            Static => {
                self.data.set(expression.clone());
                ProgramItem {
                    data: Cell::new(expression),
                    ref_type: Cell::new(Static),
                    ty: self.ty,
                }
            }
            Var => {
                self.data.set(expression.clone());
                ProgramItem {
                    data: Cell::new(expression),
                    ref_type: Cell::new(Var),
                    ty: self.ty,
                }
            }
        }
    }
}

macro_rules! vec_uniform {
    ($($vec_type:ident, $vec_len:ty, $func:ident, $n:expr;)*) => (
        $(
            unsafe impl ArgParameter<UniformArgs> for GlVec<$vec_type, $vec_len> {
                fn get_param() -> UniformArgs {
                    UniformArgs {
                        num_elements: $n,
                        array_count: 1,
                        is_mat: false,
                        func: gl::INDICES.$func,
                    }
                }
            }
        )*
    );
}

vec_uniform!(
    VecFloat, Vec4, Uniform4fv, 4;
    VecFloat, Vec3, Uniform3fv, 3;
    VecFloat, Vec2, Uniform2fv, 2;
    VecFloat, Vec1, Uniform1fv, 1;
    VecInt, Vec4, Uniform4iv, 4;
    VecInt, Vec3, Uniform4iv, 3;
    VecInt, Vec2, Uniform4iv, 2;
    VecInt, Vec1, Uniform4iv, 1;
    VecUInt, Vec4, Uniform4uiv, 4;
    VecUInt, Vec3, Uniform4uiv, 3;
    VecUInt, Vec2, Uniform4uiv, 2;
    VecUInt, Vec1, Uniform4uiv, 1;
);

macro_rules! impl_matrix {
    ($matrix_type:ident, $data:expr) => {
        #[derive(Clone)]
        pub struct $matrix_type {
            _hidden: (),
        }

        impl Add<$matrix_type> for $matrix_type {
            type Output = $matrix_type;

            fn add(self, rhs: $matrix_type) -> $matrix_type {
                unreachable!();
            }
        }

        unsafe impl ArgType for $matrix_type {
            fn data_type() -> DataType {
                $data
            }
        }
    };
}

macro_rules! matrix_subs {
    ($($vec:ident, $m1:ident, $m2:ident, $m3:ident,;)*) => (
        $(
            matrix_subs!($m1, $vec, U1, U2, U3, U4);
            matrix_subs!($m2, $vec, U1, U2, U3);
            matrix_subs!($m3, $vec, U1, U2);
        )*
    );
    ($mat_type:ident, $vec:ty, $($num:ident),*) => (
        impl<'a, $($num: ItemType<'a, Ty = $vec>),*>
            Construct<'a, $mat_type, <($($num,)*) as ExprMin>::Min> for ($($num,)*)
        where
            ($($num,)*): ExprMin,
        {
            fn construct(self) -> ProgramBuilderItem<'a, $mat_type, <($($num,)*) as ExprMin>::Min> {
                let ($($num,)*) = self;
                let ($($num,)*) = ($($num.as_string(),)*);
                let s = var_format!(
                    format!("{}(", $mat_type::data_type().gl_type()),
                    // hack to allow the macro to repeat the expression
                    $({let _ = stringify!($num); ","}),*;
                    $($num),*
                );
                // remove the trailing comma
                let s = s.substring(0, s.len()-2);
                let s = var_format!("", ")"; s);
                ProgramBuilderItem::create(s, Expr)
            }
        }
    )
}

macro_rules! arg_op {
    ($op:ident, $op_fn:ident, $t1:ident, $t2:ident, $out:ident) => {
        impl $op<$t2> for $t1 {
            type Output = $out;

            fn $op_fn(self, rhs: $t2) -> $out {
                unreachable!()
            }
        }
    };
}

macro_rules! mat_ops {
    (!$($vec:ident,)*;;) => ();
    ($($vec:ident,)*;;) => ();
    ($(;)* | $($($fmat:ident,)*;)*) => ();
    (!$($vec:ident,)*;$v0:ident, $($ev:ident,)*;$($m0:ident,)*;$($($mat:ident,)*;)*) => (
        $(
            arg_op!(Mul, mul, $m0, $vec, $v0);
            arg_op!(Mul, mul, $v0, $m0, $vec);
        )*

        mat_ops!(!$($vec,)*;$($ev,)*;$($($mat,)*;)*);
    );
    ($($vec:ident,)*;$v0:ident, $($ev:ident,)*;$($m0:ident,)*;$($($mat:ident,)*;)*) => (
        $(
            arg_op!(Mul, mul, $m0, $vec, $v0);
        )*

        mat_ops!($($vec,)*;$($ev,)*;$($($mat,)*;)*);
    );
    ($($m0:ident, $($mat:ident,)*;)* | $($($fmat:ident,)*;)*) => (
        mat_ops!($($m0,)*;$($m0,)*;$($($fmat,)*;)*);

        mat_ops!($($($mat,)*;)* | $($($fmat,)*;)*);
    )

}

macro_rules! matrix_param {
    (;$cols:expr) => ();
    ($mat:ident, $($m:ident,)*; $cols:expr) => (
        unsafe impl ArgParameter<TransparentArgs> for $mat {
            fn get_param() -> TransparentArgs {
                TransparentArgs { num_locations: 1, num_input_locations: $cols }
            }
        }
        matrix_param!($($m,)*;$cols - 1);
    )
}

macro_rules! create_matrix {
    ($t:ident, $($vec:ident, $($mat:ident,)*;)*) => (
        matrix_subs!($($vec, $($mat,)*;)*);
        mat_ops!($($($mat,)*;)* | $($($mat,)*;)*);
        mat_ops!(!$($vec,)*;$($vec,)*;$($($mat,)*;)*);

        $(
            matrix_param!($($mat,)*;4);
            $(
                arg_op!(Mul, mul, $t, $mat, $mat);
                arg_op!(Mul, mul, $mat, $t, $mat);
                impl_matrix!($mat, DataType::$mat);
            )*
        )*
    )
}

macro_rules! mat_uniform {
    ($($mat_type:ident, $func:ident, $n:expr;)*) => (
        $(
            unsafe impl ArgParameter<UniformArgs> for $mat_type {
                fn get_param() -> UniformArgs {
                    UniformArgs {
                        num_elements: $n,
                        array_count: 1,
                        is_mat: true,
                        func: gl::INDICES.$func,
                    }
                }
            }
        )*
    );
}

create_matrix!(Float,
    Float4, Float4x4, Float3x4, Float2x4,;
    Float3, Float4x3, Float3x3, Float2x3,;
    Float2, Float4x2, Float3x2, Float2x2,;);

mat_uniform!(
    Float4x4, UniformMatrix4fv, 16;
    Float4x3, UniformMatrix4x3fv, 12;
    Float4x2, UniformMatrix4x2fv, 8;
    Float3x4, UniformMatrix3x4fv, 12;
    Float3x3, UniformMatrix3fv, 9;
    Float3x2, UniformMatrix3x2fv, 6;
    Float2x4, UniformMatrix2x4fv, 8;
    Float2x3, UniformMatrix2x3fv, 6;
    Float2x2, UniformMatrix2fv, 4;
);
