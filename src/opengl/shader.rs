use self::traits::*;
use std::cell::Cell;
use std::collections::VecDeque;
use std::ffi::CString;
use std::fmt;
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul};
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

struct VarBuilder {
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
    fn new<S: fmt::Display>(string: S) -> VarString {
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

    fn format(formatter: &str, strings: Vec<VarString>) -> VarString {
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
                let mut dq = VecDeque::from(s.vars);
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
    Float,
    Float2,
    Float3,
    Float4,
    Int,
    Int2,
    Int3,
    Int4,
    UInt,
    UInt2,
    UInt3,
    UInt4,
    Boolean,
    Boolean2,
    Boolean3,
    Boolean4,
    Mat2x2,
    Mat2x3,
    Mat2x4,
    Mat3x2,
    Mat3x3,
    Mat3x4,
    Mat4x2,
    Mat4x3,
    Mat4x4,
    Sampler2D,
    IntSampler2D,
    UIntSampler2D,
    FloatSampler2D,
}

impl DataType {
    pub fn gl_type(self) -> &'static str {
        match self {
            DataType::Float => "float",
            DataType::Float2 => "vec2",
            DataType::Float3 => "vec3",
            DataType::Float4 => "vec4",
            DataType::Int => "int",
            DataType::Int2 => "ivec2",
            DataType::Int3 => "ivec3",
            DataType::Int4 => "ivec4",
            DataType::UInt => "uint",
            DataType::UInt2 => "uvec2",
            DataType::UInt3 => "uvec3",
            DataType::UInt4 => "uvec4",
            DataType::Boolean => "bool",
            DataType::Boolean2 => "bvec2",
            DataType::Boolean3 => "bvec3",
            DataType::Boolean4 => "bvec4",
            DataType::Mat2x2 => "mat2",
            DataType::Mat2x3 => "mat2x3",
            DataType::Mat2x4 => "mat2x4",
            DataType::Mat3x2 => "mat3x2",
            DataType::Mat3x3 => "mat3",
            DataType::Mat3x4 => "mat3x4",
            DataType::Mat4x2 => "mat4x2",
            DataType::Mat4x3 => "mat4x3",
            DataType::Mat4x4 => "mat4x4",
            DataType::Sampler2D => "sampler2D",
            DataType::IntSampler2D => "isampler2D",
            DataType::UIntSampler2D => "usampler2D",
            DataType::FloatSampler2D => "fsampler2D",
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
    use super::traits::*;
    use super::{Boolean, Float, Float2, Float3, Float4, Int};
    use super::{ItemRef, VarBuilder, VarString};
    use crate::tuple::{AttachBack, AttachFront, RemoveBack, RemoveFront};
    use std::marker::PhantomData;

    pub unsafe trait BuiltInOutput<T: ArgType> {
        unsafe fn as_t(self, scope: ScopeGaurd) -> (Option<T>, ScopeGaurd);
    }

    pub(super) trait BuiltInOutputs {
        fn get_strings(self, builder: &mut VarBuilder) -> String;
    }

    unsafe impl<T: ArgType> BuiltInOutput<T> for () {
        unsafe fn as_t(self, scope: ScopeGaurd) -> (Option<T>, ScopeGaurd) {
            (None, scope)
        }
    }

    unsafe impl<T: ArgType, S: IntoArg<Arg = T>> BuiltInOutput<T> for S {
        unsafe fn as_t(self, scope: ScopeGaurd) -> (Option<T>, ScopeGaurd) {
            let (arg, s) = self.into_arg();
            (Some(arg), s.merge(scope))
        }
    }

    fn create_in<S: ExprType<T>, T: ArgType>(s: &'static str) -> S {
        unsafe {
            S::from_t(
                T::create(VarString::new(s), ItemRef::Static),
                ScopeGaurd::Free,
            )
        }
    }

    fn create_in_with_scope<S: ExprType<T>, T: ArgType>(s: &'static str, scope: ScopeGaurd) -> S {
        unsafe { S::from_t(T::create(VarString::new(s), ItemRef::Static), scope) }
    }

    pub struct BuiltInVertInputs {
        pub vertex_id: Varying<Int>,
        pub instance_id: Varying<Int>,
        pub depth_near: Uniform<Float>,
        pub depth_far: Uniform<Float>,
        pub depth_diff: Uniform<Float>,
    }

    impl BuiltInVertInputs {
        pub(super) fn new(_scope: ScopeGaurd) -> BuiltInVertInputs {
            BuiltInVertInputs {
                vertex_id: create_in("gl_VertexID"),
                instance_id: create_in("gl_InstanceID"),
                depth_near: create_in("gl_DepthRange.near"),
                depth_far: create_in("gl_DepthRange.far"),
                depth_diff: create_in("gl_DepthRange.diff"),
            }
        }
    }

    pub struct BuiltInVertOutputs {
        position: Float4,
        point_size: Option<Float>,
        clip_distance: (),
        scope: ScopeGaurd,
    }

    impl BuiltInOutputs for BuiltInVertOutputs {
        fn get_strings(self, builder: &mut VarBuilder) -> String {
            let mut string = format!(
                "   gl_Position = {};\n",
                builder.format_var(&self.position.as_shader_data())
            );
            if let Some(s) = self.point_size {
                string = format!(
                    "{}   gl_PointSize = {};\n",
                    string,
                    builder.format_var(&s.as_shader_data())
                );
            }
            string
        }
    }

    impl BuiltInVertOutputs {
        pub fn position<T: IntoArg<Arg = Float4>>(position: T) -> BuiltInVertOutputs {
            unsafe {
                let (arg, scope) = position.into_arg();
                BuiltInVertOutputs {
                    position: arg,
                    point_size: None,
                    clip_distance: (),
                    scope: scope,
                }
            }
        }

        pub fn create<T: IntoArg<Arg = Float4>, S: BuiltInOutput<Float>>(
            position: T,
            point_size: S,
        ) -> BuiltInVertOutputs {
            unsafe {
                let (arg, scope) = position.into_arg();
                let (point, scope) = point_size.as_t(scope);
                BuiltInVertOutputs {
                    position: arg,
                    point_size: point,
                    clip_distance: (),
                    scope: scope,
                }
            }
        }
    }

    pub struct BuiltInFragInputs {
        pub frag_coord: Varying<Float4>,
        pub point_coord: Varying<Float2>,
        pub primitive_id: Varying<Int>,
        pub clip_distance: (),
        pub depth_near: Uniform<Float>,
        pub depth_far: Uniform<Float>,
        pub depth_diff: Uniform<Float>,
    }

    impl BuiltInFragInputs {
        pub(super) fn new(_scope: ScopeGaurd) -> BuiltInFragInputs {
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

    pub struct BuiltInFragOutputs {
        depth: Option<Float>,
        discard: Option<Boolean>,
        scope: ScopeGaurd,
    }

    impl BuiltInOutputs for BuiltInFragOutputs {
        fn get_strings(self, builder: &mut VarBuilder) -> String {
            let mut string = String::new();
            if let Some(disc) = self.discard {
                string = format!(
                    "   if ({}) discard;\n",
                    builder.format_var(&disc.as_shader_data())
                );
            }
            if let Some(dp) = self.depth {
                string = format!(
                    "{}   gl_FragDepth = {};\n",
                    string,
                    builder.format_var(&dp.as_shader_data())
                );
            }
            string
        }
    }

    impl BuiltInFragOutputs {
        pub fn empty() -> BuiltInFragOutputs {
            BuiltInFragOutputs {
                depth: None,
                discard: None,
                scope: ScopeGaurd::Free,
            }
        }

        pub fn depth<T: IntoArg<Arg = Float>>(depth: T) -> BuiltInFragOutputs {
            unsafe {
                let (arg, scope) = depth.into_arg();
                BuiltInFragOutputs {
                    depth: Some(arg),
                    discard: None,
                    scope: scope,
                }
            }
        }

        pub fn create<T: BuiltInOutput<Float>, S: BuiltInOutput<Boolean>>(
            depth: T,
            discard: S,
        ) -> BuiltInFragOutputs {
            unsafe {
                let scope = ScopeGaurd::Free;
                let (depth, scope) = depth.as_t(scope);
                let (discard, scope) = discard.as_t(scope);
                BuiltInFragOutputs {
                    depth: depth,
                    discard: discard,
                    scope: scope,
                }
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
    for ShaderProgram<In, Uniforms, Out>
{
    unsafe fn adopt(ptr: *mut (), id: u32) -> Option<*mut ()> {
        let b = unsafe { &mut *(ptr as *mut [CString; 2]) };
        let gl_draw = unsafe { super::inner_gl_unsafe() };
        // since the underlying gl is only passed a pointer and not the slice length,
        // `as_bytes()` is equivalent to `as_bytes_with_nul()`
        let program = get_program(b[0].as_bytes_with_nul(), b[1].as_bytes_with_nul());
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
        let program = gl::CreateProgram();
        gl::ProgramBinary(program, format, ptr.offset(3) as *const _, data_len as i32);
        gl_draw.resource_list[id as usize] = program;
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

    unsafe fn orphan(id: u32, _ptr: *mut ()) -> *mut () {
        let gl_draw = super::inner_gl_unsafe();
        let program = gl_draw.resource_list[id as usize];
        let mut len = 0;
        let mut format = 0;
        gl::GetProgramiv(program, gl::PROGRAM_BINARY_LENGTH, &mut len);
        // adding 3 to len rounds up if the binary length is not a multiple
        // of 4
        let mut buffer = Vec::<u32>::with_capacity(3 + (len as usize + 3) >> 2);
        let cap = buffer.capacity();
        let mut data_len = 0;
        let ptr = buffer.as_mut_ptr();
        std::mem::forget(buffer);
        gl::GetProgramBinary(
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
    }
}

impl<In: ShaderArgs, Uniforms: ShaderArgs, Images: ShaderArgs, Out: ShaderArgs>
    ShaderProgram<In, Uniforms, Images, Out>
{
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

/// Create a shader program with a vertex and a fragment shader.
///
/// Shaders can be used to cause more advanced behavior to happen on the GPU. The
/// most common use is for lighting calculations. Shaders are a lower level feature
/// that can be difficult to use correctly.
pub fn create_program<
    Uniforms: ShaderArgs + ShaderArgsClass<UniformArgs>,
    Images: ShaderArgs + ShaderArgsClass<ImageArgs>,
    In: ShaderArgs + ShaderArgsClass<TransparentArgs>,
    Pass: IntoArgs,
    Out: IntoArgs,
    VertFN: FnOnce(
        In::AsVarying,
        Uniforms::AsUniform,
        Images::AsUniform,
        BuiltInVertInputs,
    ) -> (Pass, BuiltInVertOutputs),
    FragFN: FnOnce(
        <Pass::Args as ShaderArgs>::AsVarying,
        Uniforms::AsUniform,
        Images::AsUniform,
        BuiltInFragInputs,
    ) -> (Out, BuiltInFragOutputs),
>(
    _window: &super::GlWindow,
    vertex_shader_fn: VertFN,
    fragment_shader_fn: FragFN,
) -> ShaderProgram<In, Uniforms, Images, Out::Args>
where
    Out::Args: ShaderArgs + ShaderArgsClass<TransparentArgs> + ShaderArgsClass<OutputArgs>,
    Pass::Args: ShaderArgsClass<TransparentArgs>,
{
    let v_scope = Rc::new(());
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
        BuiltInVertInputs::new(ScopeGaurd::Limited(v_scope.clone(), 0)),
        "in",
        "u",
        "tex",
        "pass",
        ScopeGaurd::Limited(v_scope.clone(), 0),
        true,
        false,
    ))
    .unwrap();
    let _ = Rc::try_unwrap(v_scope).expect("A value was moved out of the vertex shader generator and stored elsewhere. This is not allowed because it could cause generation of an invalid shader.");
    let f_scope = Rc::new(());
    let f_string = CString::new(create_shader_string::<
        Pass::Args,
        Uniforms,
        Images,
        Out,
        BuiltInFragInputs,
        BuiltInFragOutputs,
        FragFN,
    >(
        fragment_shader_fn,
        BuiltInFragInputs::new(ScopeGaurd::Limited(f_scope.clone(), 0)),
        "pass",
        "u",
        "tex",
        "out",
        ScopeGaurd::Limited(f_scope.clone(), 0),
        false,
        true,
    ))
    .unwrap();
    let _ = Rc::try_unwrap(f_scope).expect("A value was moved out of the fragment shader generator and stored elsewhere. This is not allowed because it could cause generation of an invalid shader.");
    let program = get_program(v_string.as_bytes_with_nul(), f_string.as_bytes_with_nul());
    if cfg!(feature = "opengl41") {
        unsafe {
            gl::ProgramParameteri(
                program,
                gl::PROGRAM_BINARY_RETRIEVABLE_HINT,
                gl::TRUE as i32,
            );
        }
    }
    let mut uniform_locations = vec![0; Uniforms::NARGS];
    for i in 0..Uniforms::NARGS {
        unsafe {
            uniform_locations[i] = gl::GetUniformLocation(
                program,
                CString::new(format!("u{}", i)).unwrap().as_ptr() as *const _,
            );
        }
    }
    let mut image_locations = vec![0; Images::NARGS];
    for i in 0..Images::NARGS {
        unsafe {
            image_locations[i] = gl::GetUniformLocation(
                program,
                CString::new(format!("tex{}", i)).unwrap().as_ptr() as *const _,
            );
        }
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
    In: ShaderArgs + ShaderArgsClass<TransparentArgs>,
    Uniforms: ShaderArgs,
    Images: ShaderArgs,
    Out: IntoArgs,
    T,
    S: builtin_vars::BuiltInOutputs,
    Shader: FnOnce(In::AsVarying, Uniforms::AsUniform, Images::AsUniform, T) -> (Out, S),
>(
    generator: Shader,
    gen_in: T,

    in_str: &'static str,
    uniform_str: &'static str,
    image_str: &'static str,
    out_str: &'static str,
    scope: ScopeGaurd,

    input_qualifiers: bool,
    output_qualifiers: bool,
) -> String
where
    Out::Args: ShaderArgsClass<TransparentArgs>,
{
    let mut shader = format!("{}\n", VERSION);
    let in_args = In::map_args().args;
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
    let uniform_args = Uniforms::map_args().args;
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
    let image_args = Images::map_args().args;
    for i in 0..Uniforms::NARGS {
        shader = format!(
            "{}uniform {} {}{};\n",
            shader,
            image_args[i].gl_type(),
            image_str,
            i,
        );
    }
    shader = format!("{}\n", shader);
    let out_args = Out::Args::map_args().args;
    let mut position = 0;
    for i in 0..Out::Args::NARGS {
        if output_qualifiers {
            shader = format!(
                "{}layout(location = {}) out {} {}{};\n",
                shader,
                position,
                out_args[i].gl_type(),
                out_str,
                i,
            );
            position += Out::Args::get_param(i).num_locations;
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
    let in_type = unsafe { In::create(in_str) };
    let uniform_type = unsafe { Uniforms::create(out_str) };
    let image_type = unsafe { Images::create(image_str) };
    let (out, bout) = unsafe {
        let input = in_type.as_varying(scope.clone());
        let image = image_type.as_uniform(scope.clone());
        let uniform = uniform_type.as_uniform(scope);
        let (out, builtin) = generator(input, uniform, image, gen_in);
        (out.into_args().0.map_data_args().args, builtin)
    };
    shader = format!("{}\n\nvoid main() {{\n", shader);
    let mut builder = VarBuilder::new("var");
    let mut out_strings = Vec::with_capacity(out.len());
    for i in 0..out.len() {
        out_strings.push(builder.format_var(&out[i].1));
    }
    let bstring = bout.get_strings(&mut builder);
    shader = builder.add_strings(shader);
    shader = format!("{}{}\n", shader, bstring);
    for i in 0..out.len() {
        shader = format!("{}   {}{} = {};\n", shader, out_str, i, out_strings[i]);
    }
    shader = format!("{}}}\n", shader);
    println!("{}\n", shader);
    shader
}

pub(super) fn get_program(vertex_source: &[u8], fragment_source: &[u8]) -> GLuint {
    unsafe {
        let vertex_shader = gl::CreateShader(gl::VERTEX_SHADER);
        gl::ShaderSource(
            vertex_shader,
            1,
            &(vertex_source.as_ptr() as *const i8),
            &((&vertex_source).len() as i32),
        );
        gl::CompileShader(vertex_shader);

        // check for shader compile errors
        let mut success = gl::FALSE as GLint;
        let mut info_log = Vec::with_capacity(512);
        let mut log_len = 0;
        gl::GetShaderiv(vertex_shader, gl::COMPILE_STATUS, &mut success);
        if success != gl::TRUE as GLint {
            gl::GetShaderInfoLog(
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
        let fragment_shader = gl::CreateShader(gl::FRAGMENT_SHADER);
        gl::ShaderSource(
            fragment_shader,
            1,
            &(fragment_source.as_ptr() as *const i8),
            &((&fragment_source).len() as i32),
        );
        gl::CompileShader(fragment_shader);
        // check for shader compile errors
        gl::GetShaderiv(fragment_shader, gl::COMPILE_STATUS, &mut success);
        if success != gl::TRUE as GLint {
            gl::GetShaderInfoLog(
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
        let shader_program = gl::CreateProgram();
        gl::AttachShader(shader_program, vertex_shader);
        gl::AttachShader(shader_program, fragment_shader);
        gl::LinkProgram(shader_program);
        // check for linking errors
        gl::GetProgramiv(shader_program, gl::LINK_STATUS, &mut success);
        if success != gl::TRUE as GLint {
            gl::GetProgramInfoLog(
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
        gl::DeleteShader(vertex_shader);
        gl::DeleteShader(fragment_shader);
        shader_program
    }
}

use self::swizzle::SwizzleMask;
/// Swizzling operations are an important part of glsl.
pub mod swizzle {
    use super::traits::ArgType;
    use crate::tuple;
    use tuple::TupleIndex;

    pub unsafe trait GlSZ {
        type S;

        type Set;

        const ARG: &'static str;
    }

    pub unsafe trait SwizzleMask<T4, T3, T2, T1, S> {
        type Swizzle;

        type Out: ArgType;

        fn get_vars() -> String;
    }

    pub unsafe trait SwizzleDepth<Vec> {}

    macro_rules! swizzle_set {
    	($set:ident, $($vals:ident),*;$($vars:ident),*;$($s:ty),*) => (
    		$(
    			unsafe impl GlSZ for $vals {
    				type S = $s;

    				type Set = $set;

    				const ARG: &'static str = stringify!($vars);
    			}
    		)*
    	);
        ($set:ident, $($vals:ident),*;$($vars:ident),*) => (
        	pub struct $set {}

        	$(
        		pub struct $vals {}

        		pub const $vals: $vals = $vals {};
        	)*

        	impl_swizzle!(;Vec4, Vec3, Vec2, Vec1,;$($vals,)*);

        	// swizzle_set!($set, $($vals),*;$($vars),*;tuple::T3, tuple::T2, tuple::T1, tuple::T0);
        )
    }

    macro_rules! impl_swizzle {
    	($($trait:ident),*;$($s:ident),*) => (
    		$(pub struct $trait {})*
    		swizzle_subs!(;$($trait,)*;$($s,)*);
    	);
    	(;;) => ();
    	(;$t0:ident, $($trait:ident,)*;$s0:ident, $($s:ident,)*) => (
			unsafe impl SwizzleDepth<$t0> for $s0 {}

			$(
				unsafe impl SwizzleDepth<$t0> for $s {}
			)*

			impl_swizzle!(;$($trait,)*;$($s,)*);
    	)
    }

    macro_rules! swizzle_subs {
    	($($top:ident,)*;;) => ();
    	($($top:ident,)*;$t0:ident, $($t:ident,)*;$s0:ident, $($s:ident,)*) => (
    		unsafe impl<S, T, $s0: GlSZ<Set=S> + SwizzleDepth<T>,$($s: GlSZ<Set=S> + SwizzleDepth<T>,)*
    		$($top,)*$t0: ArgType,$($t),*> SwizzleMask<$($top,)*$t0,$($t,)*T>
    		for ($s0,$($s,)*) {
    			type Swizzle = ($s0::S,$($s::S,)*);

    			type Out = $t0;

    			fn get_vars() -> String {
    				concat_args!($s0,$($s,)*)
    			}
    		}

    		swizzle_subs!($($top,)*$t0,;$($t,)*;$($s,)*);
    	)
    }

    macro_rules! concat_args {
    	($a0:ident, $($arg:ident,)*) => (
    		format!("{}{}", $a0::ARG, concat_args!($($arg,)*));
    	);
    	() => (
    		""
    	)
    }

    impl_swizzle!(Vec4, Vec3, Vec2, Vec1;T4, T3, T2, T1);

    swizzle_set!(XYZW, W, Z, Y, X; w, z, y, x);

    swizzle_set!(RGBA, A, B, G, R; a, b, g, r);

    swizzle_set!(STPQ, Q, P, T, S; q, p, t, s);
}

pub mod api {
    pub use super::traits::Map;
    pub use super::{Bool2Arg, Bool3Arg, Bool4Arg, BoolArg};
    pub use super::{Float2Arg, Float3Arg, Float4Arg, FloatArg};
    pub use super::{Int2Arg, Int3Arg, Int4Arg, IntArg};

    pub type FragOutputs = super::BuiltInFragOutputs;
    pub type VertOutputs = super::BuiltInVertOutputs;
}

#[allow(unused, unused_parens)]
pub mod traits {
    use super::swizzle::SwizzleMask;
    use super::{DataType, ItemRef, ShaderArgDataList, ShaderArgList, VarExpr, VarString};
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

    pub unsafe trait ArgType: Clone {
        /// Do not call this function.
        unsafe fn create(data: VarString, r: ItemRef) -> Self;

        fn data_type() -> DataType;

        fn as_shader_data(self) -> VarString;
    }

    pub unsafe trait ExprType<T: ArgType>: Clone {
        unsafe fn into_t(self) -> (T, ScopeGaurd);

        unsafe fn from_t(t: T, scope: ScopeGaurd) -> Self;
    }

    pub trait IntoArg {
        type Arg: ArgType;

        unsafe fn into_arg(self) -> (Self::Arg, ScopeGaurd);
    }

    pub trait IntoArgs {
        type Args: ShaderArgs;

        unsafe fn into_args(self) -> (Self::Args, ScopeGaurd);
    }

    pub trait IntoConstant<T: ArgType> {
        fn into_constant(self, scope: ScopeGaurd) -> Constant<T>;
    }

    pub trait IntoUniform<T: ArgType> {
        fn into_uniform(self, scope: ScopeGaurd) -> Uniform<T>;
    }

    pub trait IntoVarying<T: ArgType> {
        fn into_varying(self, scope: ScopeGaurd) -> Varying<T>;
    }

    impl<T: ArgType> IntoArg for Constant<T> {
        type Arg = T;

        unsafe fn into_arg(self) -> (Self::Arg, ScopeGaurd) {
            (self.arg, self.scope)
        }
    }

    impl<T: ArgType> IntoArg for Uniform<T> {
        type Arg = T;

        unsafe fn into_arg(self) -> (Self::Arg, ScopeGaurd) {
            (self.arg, self.scope)
        }
    }

    impl<T: ArgType> IntoArg for Varying<T> {
        type Arg = T;

        unsafe fn into_arg(self) -> (Self::Arg, ScopeGaurd) {
            (self.arg, self.scope)
        }
    }

    impl<T: ArgType> IntoArg for &Constant<T> {
        type Arg = T;

        unsafe fn into_arg(self) -> (Self::Arg, ScopeGaurd) {
            (self.arg.clone(), self.scope.clone())
        }
    }

    impl<T: ArgType> IntoArg for &Uniform<T> {
        type Arg = T;

        unsafe fn into_arg(self) -> (Self::Arg, ScopeGaurd) {
            (self.arg.clone(), self.scope.clone())
        }
    }

    impl<T: ArgType> IntoArg for &Varying<T> {
        type Arg = T;

        unsafe fn into_arg(self) -> (Self::Arg, ScopeGaurd) {
            (self.arg.clone(), self.scope.clone())
        }
    }

    macro_rules! wrapper_ops {
        ($op:ident, $op_func:ident) => {
            wrapper_impls!($op, $op_func, Constant, Constant, Constant);
            wrapper_impls!($op, $op_func, Constant, Uniform, Uniform);
            wrapper_impls!($op, $op_func, Uniform, Constant, Uniform);
            wrapper_impls!($op, $op_func, Uniform, Uniform, Uniform);
            wrapper_impls!($op, $op_func, Constant, Varying, Varying);
            wrapper_impls!($op, $op_func, Varying, Constant, Varying);
            wrapper_impls!($op, $op_func, Uniform, Varying, Varying);
            wrapper_impls!($op, $op_func, Varying, Uniform, Varying);
            wrapper_impls!($op, $op_func, Varying, Varying, Varying);
        };
    }

    macro_rules! wrapper_impls {
        ($op:ident, $op_func:ident, $l:ident, $r:ident, $o:ident) => {
            impl<O: ArgType, L: ArgType + $op<R, Output = O>, R: ArgType> $op<$r<R>> for $l<L> {
                type Output = $o<O>;

                fn $op_func(self, rhs: $r<R>) -> $o<O> {
                    $o {
                        arg: $op::$op_func(self.arg, rhs.arg),
                        scope: self.scope.merge(rhs.scope),
                    }
                }
            }

            impl<O: ArgType, L: ArgType + $op<R, Output = O> + Clone, R: ArgType> $op<$r<R>>
                for &$l<L>
            {
                type Output = $o<O>;

                fn $op_func(self, rhs: $r<R>) -> $o<O> {
                    let se = self.clone();
                    $o {
                        arg: $op::$op_func(se.arg, rhs.arg),
                        scope: se.scope.merge(rhs.scope),
                    }
                }
            }

            impl<O: ArgType, L: ArgType + $op<R, Output = O>, R: ArgType + Clone> $op<&$r<R>>
                for $l<L>
            {
                type Output = $o<O>;

                fn $op_func(self, rhs: &$r<R>) -> $o<O> {
                    let other = rhs.clone();
                    $o {
                        arg: $op::$op_func(self.arg, other.arg),
                        scope: self.scope.merge(other.scope),
                    }
                }
            }

            impl<O: ArgType, L: ArgType + $op<R, Output = O> + Clone, R: ArgType + Clone>
                $op<&$r<R>> for &$l<L>
            {
                type Output = $o<O>;

                fn $op_func(self, rhs: &$r<R>) -> $o<O> {
                    let se = self.clone();
                    let other = rhs.clone();
                    $o {
                        arg: $op::$op_func(se.arg, other.arg),
                        scope: se.scope.merge(other.scope),
                    }
                }
            }
        };
    }

    wrapper_ops!(Add, add);
    wrapper_ops!(Sub, sub);
    wrapper_ops!(Mul, mul);
    wrapper_ops!(Div, div);

    impl<T: ArgType> IntoConstant<T> for T {
        fn into_constant(self, scope: ScopeGaurd) -> Constant<T> {
            Constant {
                arg: self,
                scope: scope,
            }
        }
    }

    impl<T: ArgType> IntoUniform<T> for T {
        fn into_uniform(self, scope: ScopeGaurd) -> Uniform<T> {
            Uniform {
                arg: self,
                scope: scope,
            }
        }
    }

    impl<T: ArgType> IntoVarying<T> for T {
        fn into_varying(self, scope: ScopeGaurd) -> Varying<T> {
            Varying {
                arg: self,
                scope: scope,
            }
        }
    }

    pub trait ExprCombine<T: ArgType> {
        type Min: ExprType<T>;
    }

    pub trait ExprMin<T: ArgType> {
        type Min: ExprType<T>;
    }

    impl<A: ArgType, T: crate::tuple::RemoveFront> ExprMin<A> for T
    where
        T::Front: ExprCombine<A>,
        T::Remaining: ExprCombine<A>,
        (
            <T::Front as ExprCombine<A>>::Min,
            <T::Remaining as ExprCombine<A>>::Min,
        ): ExprCombine<A>,
    {
        type Min = <(
            <T::Front as ExprCombine<A>>::Min,
            <T::Remaining as ExprCombine<A>>::Min,
        ) as ExprCombine<A>>::Min;
    }

    impl<T: ArgType, S: ArgType> ExprCombine<T> for Constant<S> {
        type Min = Constant<T>;
    }

    impl<T: ArgType, S: ArgType> ExprCombine<T> for Uniform<S> {
        type Min = Uniform<T>;
    }

    impl<T: ArgType, S: ArgType> ExprCombine<T> for Varying<S> {
        type Min = Varying<T>;
    }

    impl<T: ArgType, S: ArgType> ExprCombine<T> for &Constant<S> {
        type Min = Constant<T>;
    }

    impl<T: ArgType, S: ArgType> ExprCombine<T> for &Uniform<S> {
        type Min = Uniform<T>;
    }

    impl<T: ArgType, S: ArgType> ExprCombine<T> for &Varying<S> {
        type Min = Varying<T>;
    }

    impl<T: ArgType, S: ArgType> ExprCombine<T> for (Constant<S>,) {
        type Min = Constant<T>;
    }

    impl<T: ArgType, S: ArgType> ExprCombine<T> for (Uniform<S>,) {
        type Min = Uniform<T>;
    }

    impl<T: ArgType, S: ArgType> ExprCombine<T> for (Varying<S>,) {
        type Min = Varying<T>;
    }

    impl<T: ArgType, S: ArgType> ExprCombine<T> for (&Constant<S>,) {
        type Min = Constant<T>;
    }

    impl<T: ArgType, S: ArgType> ExprCombine<T> for (&Uniform<S>,) {
        type Min = Uniform<T>;
    }

    impl<T: ArgType, S: ArgType> ExprCombine<T> for (&Varying<S>,) {
        type Min = Varying<T>;
    }

    impl<T: ArgType, L: ArgType, R: ArgType> ExprCombine<T> for (Constant<L>, Constant<R>) {
        type Min = Constant<T>;
    }

    impl<T: ArgType, L: ArgType, R: ArgType> ExprCombine<T> for (Uniform<L>, Constant<R>) {
        type Min = Uniform<T>;
    }

    impl<T: ArgType, L: ArgType, R: ArgType> ExprCombine<T> for (Constant<L>, Uniform<R>) {
        type Min = Uniform<T>;
    }

    impl<T: ArgType, L: ArgType, R: ArgType> ExprCombine<T> for (Uniform<L>, Uniform<R>) {
        type Min = Uniform<T>;
    }

    impl<T: ArgType, L: ArgType, R: ArgType> ExprCombine<T> for (Varying<L>, Constant<R>) {
        type Min = Varying<T>;
    }

    impl<T: ArgType, L: ArgType, R: ArgType> ExprCombine<T> for (Constant<L>, Varying<R>) {
        type Min = Varying<T>;
    }

    impl<T: ArgType, L: ArgType, R: ArgType> ExprCombine<T> for (Varying<L>, Uniform<R>) {
        type Min = Varying<T>;
    }

    impl<T: ArgType, L: ArgType, R: ArgType> ExprCombine<T> for (Uniform<L>, Varying<R>) {
        type Min = Varying<T>;
    }

    impl<T: ArgType, L: ArgType, R: ArgType> ExprCombine<T> for (Varying<L>, Varying<R>) {
        type Min = Varying<T>;
    }

    use std::rc::Rc;

    /// This type must be public because it is used in the methods for the public
    /// trait `ShaderArgs`
    #[derive(Clone)]
    pub enum ScopeGaurd {
        // free is used for built in inputs and constants, where
        // it won't break anything to use the input between invocations.
        // This is useful because constant initializers for types don't
        // have any way of accessing a scope, and indeed can be called outside
        // of a generation function
        Free,
        Limited(Rc<()>, usize),
    }

    impl ScopeGaurd {
        pub(super) fn merge(self, other: ScopeGaurd) -> ScopeGaurd {
            match (self, other) {
                (ScopeGaurd::Free, ScopeGaurd::Free) => ScopeGaurd::Free,
                (ScopeGaurd::Free, ScopeGaurd::Limited(rc, depth)) => {
                    ScopeGaurd::Limited(rc, depth)
                }
                (ScopeGaurd::Limited(rc, depth), ScopeGaurd::Free) => {
                    ScopeGaurd::Limited(rc, depth)
                }
                (ScopeGaurd::Limited(rc1, depth1), ScopeGaurd::Limited(rc2, depth2)) => {
                    if depth1 > depth2 {
                        ScopeGaurd::Limited(rc1, depth1)
                    } else {
                        ScopeGaurd::Limited(rc2, depth2)
                    }
                }
            }
        }
    }

    #[derive(Clone)]
    pub struct Constant<T: ArgType> {
        pub(super) arg: T,
        pub(super) scope: ScopeGaurd,
    }

    #[derive(Clone)]
    pub struct Uniform<T: ArgType> {
        pub(super) arg: T,
        pub(super) scope: ScopeGaurd,
    }

    #[derive(Clone)]
    pub struct Varying<T: ArgType> {
        pub(super) arg: T,
        pub(super) scope: ScopeGaurd,
    }

    unsafe impl<T: ArgType> ExprType<T> for Constant<T> {
        unsafe fn into_t(self) -> (T, ScopeGaurd) {
            (self.arg, self.scope)
        }

        unsafe fn from_t(t: T, scope: ScopeGaurd) -> Self {
            Constant {
                arg: t,
                scope: scope,
            }
        }
    }

    unsafe impl<T: ArgType> ExprType<T> for Uniform<T> {
        unsafe fn into_t(self) -> (T, ScopeGaurd) {
            (self.arg, self.scope)
        }

        unsafe fn from_t(t: T, scope: ScopeGaurd) -> Self {
            Uniform {
                arg: t,
                scope: scope,
            }
        }
    }

    unsafe impl<T: ArgType> ExprType<T> for Varying<T> {
        unsafe fn into_t(self) -> (T, ScopeGaurd) {
            (self.arg, self.scope)
        }

        unsafe fn from_t(t: T, scope: ScopeGaurd) -> Self {
            Varying {
                arg: t,
                scope: scope,
            }
        }
    }

    pub trait Map<T: ArgType, T1: ArgType, T2: ArgType, T3: ArgType, T4: ArgType, S>:
        ExprType<T>
    {
        fn map<SZ: SwizzleMask<T1, T2, T3, T4, S>>(
            &self,
            mask: SZ,
        ) -> <Self as ExprCombine<SZ::Out>>::Min
        where
            Self: ExprCombine<SZ::Out>;
    }

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
        // note: two layers of indirection are needed for a couple of reasons
        // first, the function pointers are re-loaded when the context is reset
        // second, this will allow get_param to be a constant function in the future
        pub func: *const *const std::os::raw::c_void,
    }

    unsafe impl ArgClass for UniformArgs {}

    #[derive(Clone, Copy)]
    pub struct ImageArgs;

    unsafe impl ArgClass for ImageArgs {}

    /// ShaderArgs is a trait that is implemented for types of
    /// possible opengl argument sets
    pub unsafe trait ShaderArgs {
        const NARGS: usize;

        /// Do not call this function.
        unsafe fn create(prefix: &'static str) -> Self;

        type AsUniform;

        type AsVarying;

        unsafe fn as_uniform(self, scope: ScopeGaurd) -> Self::AsUniform;

        unsafe fn as_varying(self, scope: ScopeGaurd) -> Self::AsVarying;

        fn map_args() -> ShaderArgList;

        fn map_data_args(self) -> ShaderArgDataList;
    }

    pub unsafe trait Construct<T: ArgType> {
        fn as_arg(self) -> T;
    }

    /// Sometimes it is neccessary to restrict a type to only certain types
    /// of arguments.
    pub unsafe trait ShaderArgsClass<T>: ShaderArgs {
        fn get_param(i: usize) -> T;
    }

    macro_rules! impl_shader_args {
		// a macro could be implemented that counts the number of arguments
		// that are passed to this macro, but that would be pretty unneccesary
		($($name:ident),*; $num:expr) => (
            #[allow(unused_parens)]
			unsafe impl<$($name: ArgType),*> ShaderArgs for ($($name,)*) {
				const NARGS: usize = $num;

				unsafe fn create(prefix: &'static str) -> Self {
					let n = 0;
					$(
						let $name = $name::create(VarString::new(format!("{}{}", prefix, n)), ItemRef::Static);
						let n = n + 1;
					)*
					($($name,)*)
				}

                type AsUniform = ($(Uniform<$name>),*);

                type AsVarying = ($(Varying<$name>),*);

                unsafe fn as_uniform(self, scope: ScopeGaurd) -> Self::AsUniform {
                    let ($($name,)*) = self;
                    ($(Uniform::from_t($name, scope.clone())),*)
                }

                unsafe fn as_varying(self, scope: ScopeGaurd) -> Self::AsVarying {
                    let ($($name,)*) = self;
                    ($(Varying::from_t($name, scope.clone())),*)
                }

				fn map_args() -> ShaderArgList {
					ShaderArgList {
						args: vec![$($name::data_type()),*],
					}
				}

				fn map_data_args(self) -> ShaderArgDataList {
					let ($($name,)*) = self;
					ShaderArgDataList {
						args: vec![$(($name::data_type(), $name.as_shader_data())),*]
					}
				}
			}

            impl<$($name: IntoArg),*> IntoArgs for ($($name),*) {
                type Args = ($($name::Arg,)*);

                unsafe fn into_args(self) -> (Self::Args, ScopeGaurd) {
                    let ($($name),*) = self;
                    let mut scope = ScopeGaurd::Free;
                    // I want to take a moment to thank Rust for allowing this
                    // type of syntax
                    (($({
                        let x = $name.into_arg();
                        scope = scope.merge(x.1);
                        x.0
                    },)*), scope)
                }
            }

            unsafe impl<T: ArgClass, $($name: ArgType + ArgParameter<T>),*> ShaderArgsClass<T> for ($($name,)*) {
                fn get_param(i: usize) -> T {
                    // note: create an array of function pointers and call the ith one
                    // this is likely faster and more optimizable.
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

// the complement of implementations for certain vec ops. Need to
// be careful not to redefine.
macro_rules! vec_ops_reverse {
    ($vec_type:ident, $s0:ident,) => ();
    ($vec_type:ident, $s0:ident, $($sub:ident,)+) => (
        impl Mul<last!($($sub,)*)> for $vec_type {
            type Output = $vec_type;

            fn mul(self, rhs: last!($($sub,)*)) -> $vec_type {
                let (l, r) = (self.data.data.into_inner(), rhs.data.data.into_inner());
                $vec_type::new(var_format!("(", " * ", ")"; l, r), Expr)
            }
        }
    )
}

macro_rules! vec_type {
    ($vec:ident, $vec_type:ident, $trait:ident, $data:expr, $($sub:ident,)*) => (
    	#[derive(Clone)]
        pub struct $vec_type {
            data: ProgramItem,
        }

        impl $vec_type {
            fn new(data: VarString, r: ItemRef) -> $vec_type {
                $vec_type {
                    data: ProgramItem::new(data, $data, r),
                }
            }
        }

        impl Mul<$vec_type> for last!($($sub,)*) {
            type Output = $vec_type;

            fn mul(self, rhs: $vec_type) -> $vec_type {
                let (l, r) = (self.data.data.into_inner(), rhs.data.data.into_inner());
                $vec_type::new(var_format!("(", " * ", ")"; l, r), Expr)
            }
        }

        impl Div<last!($($sub,)*)> for $vec_type {
            type Output = $vec_type;

            fn div(self, rhs: last!($($sub,)*)) -> $vec_type {
                let (l, r) = (self.data.data.into_inner(), rhs.data.data.into_inner());
                $vec_type::new(var_format!("(", " / ", ")"; l, r), Expr)
            }
        }

        vec_ops_reverse!($vec_type, $($sub,)*);

        impl Add for $vec_type {
        	type Output = $vec_type;

        	fn add(self, rhs: $vec_type) -> $vec_type {
        		let (l, r) = (self.data.data.into_inner(), rhs.data.data.into_inner());
                $vec_type::new(var_format!("(", " + ", ")"; l, r), Expr)
        	}
        }

        pub unsafe trait $trait {
            type Out: ExprType<$vec_type>;

            fn $vec(self) -> Self::Out;
        }

        unsafe impl<T: ShaderArgs + Construct<$vec_type>, S: IntoArgs<Args = T> + ExprMin<$vec_type>> $trait for S {
            type Out = S::Min;

            fn $vec(self) -> Self::Out {
                unsafe {
                    let (t, scope) = self.into_args();
                    S::Min::from_t(t.as_arg(), scope)
                }
            }
        }

        pub fn $vec<T: $trait>(args: T) -> T::Out {
        	args.$vec()
        }

        unsafe impl ArgType for $vec_type {
        	unsafe fn create(data: VarString, r: ItemRef) -> $vec_type {
        		$vec_type {
        			data: ProgramItem {
                        data: Cell::new(data),
                        ref_type: Cell::new(r),
                        ty: $data,
                    }
        		}
        	}

        	fn data_type() -> DataType {
        		$data
        	}

        	fn as_shader_data(self) -> VarString {
        		self.data.data.into_inner()
        	}
        }

        unsafe impl ArgParameter<TransparentArgs> for $vec_type {
            fn get_param() -> TransparentArgs {
                TransparentArgs { num_input_locations: 1, num_locations: 1 }
            }
        }

        subs!($trait, $vec, $vec_type ;  ; $($sub,)*);
    )
}

macro_rules! subs {
	($trait:ident, $vec:ident, $vec_type:ident;$($start:ident,)*;) => (
		unsafe impl Construct<$vec_type> for ($($start),*) {
            #[allow(unused_parens)]
			fn as_arg(self) -> $vec_type {
				match_from!(self, $($start,)*;u1, u2, u3, u4,;);
                $vec_type::new(var_format!("", "(", ")"; VarString::new($vec_type::data_type().gl_type()),
                        concat_enough!($($start,)* ; u1, u2, u3, u4,;)), Expr)
			}
		}
	);
	($trait:ident, $vec:ident, $vec_type:ident ; $($start:ident,)* ; $t0:ident, $($types:ident,)*) => (
		subs!($trait, $vec, $vec_type;$($start,)*$t0,;);
		subs_reverse!($trait, $vec, $vec_type;$($start,)*;;$($types,)*;$($types,)*);
	)
}

macro_rules! match_from {
	($var:ident, ; $($vals:ident,)* ; $($build:ident,)*) => (
		let ($($build),*) = $var;
	);
	($var:ident, $t1:ident, $($try:ident,)* ; $v1:ident, $($vals:ident,)* ; $($build:ident,)*) => (
		match_from!($var, $($try,)* ; $($vals,)* ; $($build,)*$v1,);
	)
}

macro_rules! concat_enough {
	(; $($vals:ident,)* ; $($build:ident,)*) => (
		concat_var!($($build.as_shader_data(),)*);
	);
	($t1:ident, $($try:ident,)* ; $v1:ident, $($vals:ident,)* ; $($build:ident,)*) => (
		concat_enough!($($try,)* ; $($vals,)* ; $($build,)*$v1,);
	)
}

macro_rules! concat_var {
    ($e:expr,) => (
        $e
    );
    ($e:expr, $($es:expr,)+) => (
        var_format!("", ", ", ""; $e, concat_var!($($es,)+))
    );
}

macro_rules! concat {
	($e:expr,) => (
		$e
	);
	($e:expr, $($es:expr,)+) => (
		format!("{}, {}", $e, concat!($($es,)+))
	);
}

// subs_repeat needs both a forward and backward list of types,
// so this macro generates those and passes it to subs_repeat
macro_rules! subs_reverse {
    ($trait:ident, $vec:ident, $vec_type:ident;$($start:ident,)*;$($rev:ident,)*;;$($ct:ident,)*) => (
        subs_repeat!($trait, $vec, $vec_type;$($start,)*;$($rev,)*;$($ct,)*);
    );
    ($trait:ident, $vec:ident, $vec_type:ident;$($start:ident,)*;
		$($rev:ident,)*;$t0:ident, $($types:ident,)*; $($ct:ident,)*) => (
        subs_reverse!($trait, $vec, $vec_type;$($start,)*;$t0,$($rev,)*;$($types,)*;$($ct,)*);
    );
}

macro_rules! subs_repeat {
	($trait:ident, $vec:ident, $vec_type:ident;$($start:ident,)*;;) => ();
	($trait:ident, $vec:ident, $vec_type:ident;$($start:ident,)*;$r0:ident, $($rev:ident,)*;$t0:ident, $($types:ident,)*) => (
		subs!($trait, $vec, $vec_type;$($start,)*$r0,;$t0,$($types,)*);
		subs_repeat!($trait, $vec, $vec_type;$($start,)*;$($rev,)*;$($types,)*);
	)
}

macro_rules! vec_types {
	(;;) => ();
	($t0:ident, $($type:ident,)*;$f0:ident, $($fn:ident,)*;$a0:ident, $($arg:ident,)*) => (
		vec_type!($f0, $t0, $a0, DataType::$t0, $t0, $($type,)*);

		vec_types!($($type,)*;$($fn,)*;$($arg,)*);
	)
}

macro_rules! vec_swizzle {
	($($types:ident),*) => (
		vec_swizzle!($($types,)*;$($types,)*;Vec4, Vec3, Vec2, Vec1,);
	);
    ($vec:ident,;$($types:ident,)*;$sz:ident,) => (
        #[cfg(feature = "opengl42")]
        impl<E: ExprType<$vec>> Map<$vec,$($types,)*swizzle::$sz> for E {
            /// Applies a swizzle mask to the vector type.
            fn map<T: SwizzleMask<$($types,)*swizzle::$sz>>(&self, _mask: T) -> <Self as ExprCombine<T::Out>>::Min
            where Self: ExprCombine<T::Out> {
                unsafe {
                    let (t, scope) = self.clone().into_t();
                    <Self as ExprCombine<T::Out>>::Min::from_t(
                        T::Out::create(var_format!("", ".", ""; t.data.data.into_inner(), VarString::new(T::get_vars())), Expr),
                        scope,
                    )
                }
            }
        }

        #[cfg(not(feature = "opengl42"))]
        impl<E: ExprType<$vec>> Map<$vec,$($types,)*swizzle::$sz> for E {
            /// Applies a swizzle mask to the vector type.
            fn map<T: SwizzleMask<$($types,)*swizzle::$sz>>(&self, _mask: T) -> <Self as ExprCombine<T::Out>>::Min
            where Self: ExprCombine<T::Out> {
                unsafe {
                    let (t, scope) = self.clone().into_t();
                    // convieniently, opengl allows syntax like `vec4(1.0)`
                    <Self as ExprCombine<T::Out>>::Min::from_t(
                        T::Out::create(var_format!("", "(", ")"; T::Out::data_type(), t.data.data.into_inner(), Expr),
                        scope,
                    ))
                }
            }
        }
    );
	($vec:ident, $($next:ident,)+;$($types:ident,)*;$sz:ident, $($s:ident,)*) => (
		impl<E: ExprType<$vec>> Map<$vec,$($types,)*swizzle::$sz> for E {
            /// Applies a swizzle mask to the vector type.
			fn map<T: SwizzleMask<$($types,)*swizzle::$sz>>(&self, _mask: T) -> <Self as ExprCombine<T::Out>>::Min
            where Self: ExprCombine<T::Out> {
                unsafe {
                    let (t, scope) = self.clone().into_t();
    				<Self as ExprCombine<T::Out>>::Min::from_t(
    					T::Out::create(var_format!("", ".", ""; t.data.data.into_inner(), VarString::new(T::get_vars())), Expr),
                        scope
    				)
                }
			}
		}

		vec_swizzle!($($next,)*;$($types,)*;$($s,)*);
	);
}

macro_rules! vec_litteral {
    ($tag:expr, $t:ty, $($arg:ident),*;$($f:ident),*;$($v:ident),*) => (
        vec_litteral!($tag, $($t, $arg, $f, $v,)*);
    );
    ($tag:expr,) => ();
    ($tag:expr, $t0:ty, $a0:ident, $f0:ident, $v0:ident, $($t:ty, $arg:ident, $f:ident, $v:ident,)*) => (
        unsafe impl $a0 for tup!($t0,$($t,)*) {
            type Out = Constant<$v0>;

            #[allow(unused_parens)]
            fn $f0(self) -> Constant<$v0> {
                let tup!($f0,$($f,)*) = self;
                Constant {
                    arg: $v0::new(var_format!("", "(", ")"; VarString::new($v0::data_type().gl_type()),
                        VarString::new(concat!(format!("{}{}", $f0, $tag),$(format!("{}{}", $f, $tag),)*))), Expr),
                    scope: ScopeGaurd::Free,
                }
            }
        }

        vec_litteral!($tag, $($t, $arg, $f, $v,)*);
    )
}

macro_rules! tup {
    ($($t:ident,)*) => (
        ($($t),*)
    );
    ($($t:ty,)*) => (
        ($($t),*)
    )
}

macro_rules! last {
    ($arg:ident,) => (
        $arg
    );
    ($a0:ident, $($args:ident,)+) => (
        last!($($args,)*)
    )
}

macro_rules! create_vec {
    ($ty:ty, $suffix:expr, $($obj:ident),*; $($func:ident),*; $($arg:ident),*) => {
        vec_types!($($obj,)*;$($func,)*;$($arg,)*);
        vec_swizzle!($($obj),*);
        vec_litteral!($suffix, $ty, $($arg),*;$($func),*;$($obj),*);
    }
}

macro_rules! vec_output {
    ($($vec_type:ident),*) => (
        $(
            unsafe impl ArgParameter<OutputArgs> for $vec_type {
                fn get_param() -> OutputArgs {
                    OutputArgs
                }
            }
        )*
    )
}

create_vec!(f32, "", Float4, Float3, Float2, Float;
    float4, float3, float2, float;
    Float4Arg, Float3Arg, Float2Arg, FloatArg);

create_vec!(i32, "", Int4, Int3, Int2, Int;
    int4, int3, int2, int;
    Int4Arg, Int3Arg, Int2Arg, IntArg);

create_vec!(u32, "", UInt4, UInt3, UInt2, UInt;
    uint4, uint3, uint2, uint;
    UInt4Arg, UInt3Arg, UInt2Arg, UIntArg);

vec_output!(Float4, Float3, Float2, Float, Int4, Int3, Int2, Int, UInt4, UInt3, UInt2, UInt);

create_vec!(bool, "", Boolean4, Boolean3, Boolean2, Boolean;
    boolean4, boolean3, boolean2, boolean;
    Bool4Arg,  Bool3Arg, Bool2Arg, BoolArg);

macro_rules! impl_matrix {
    ($matrix_type:ident, $matrix_fn:ident, $data:expr, $trait:ident) => (
        #[derive(Clone)]
        pub struct $matrix_type {
            data: ProgramItem,
        }

        impl $matrix_type {
            fn new(data: VarString, r: ItemRef) -> $matrix_type {
                $matrix_type {
                    data: ProgramItem::new(data, $data, r),
                }
            }
        }

        pub fn $matrix_fn<T: $trait>(t: T) -> T::Out {
            t.$matrix_fn()
        }

        impl Add<$matrix_type> for $matrix_type {
            type Output = $matrix_type;

            fn add(self, rhs: $matrix_type) -> $matrix_type {
                let (l, r) = (self.data.data.into_inner(), rhs.data.data.into_inner());
                $matrix_type::new(var_format!("(", " + ", ")"; l, r), Expr)
            }
        }

        pub unsafe trait $trait {
            type Out: ExprType<$matrix_type>;

            fn $matrix_fn(self) -> Self::Out;
        }

        unsafe impl<T: ShaderArgs + Construct<$matrix_type>, S: IntoArgs<Args = T> + ExprMin<$matrix_type>> $trait for S {
            type Out = S::Min;

            fn $matrix_fn(self) -> Self::Out {
                unsafe {
                    let (args, scope) = self.into_args();
                    S::Min::from_t(args.as_arg(), scope)
                }
            }
        }

        unsafe impl ArgType for $matrix_type {
            unsafe fn create(data: VarString, r: ItemRef) -> $matrix_type {
                $matrix_type {
                    data: ProgramItem {
                        data: Cell::new(data),
                        ref_type: Cell::new(r),
                        ty: $data,
                    }
                }
            }

            fn data_type() -> DataType {
                $data
            }

            fn as_shader_data(self) -> VarString {
                self.data.data.into_inner()
            }
        }
    )
}

macro_rules! matrix_subs {
    ($($vec:ident, $m1:ident, $m2:ident, $m3:ident,;)*) => (
        $(
            matrix_subs!($m1, $vec, $vec, $vec, $vec,);
            matrix_subs!($m2, $vec, $vec, $vec,);
            matrix_subs!($m3, $vec, $vec,);
        )*
    );
    ($mat_type:ident, $($vecs:ident,)*) => (
        unsafe impl Construct<$mat_type> for ($($vecs),*) {
            fn as_arg(self) -> $mat_type {
                match_from!(self, $($vecs,)*;u1, u2, u3, u4,;);
                $mat_type::new(var_format!("", "(", ")"; VarString::new($mat_type::data_type().gl_type()),
                        concat_enough!($($vecs,)* ; u1, u2, u3, u4,;)), Expr)
            }
        }
    )
}

macro_rules! arg_op {
    ($op:ident, $op_fn:ident, $t1:ident, $t2:ident, $out:ident) => (
        impl $op<$t2> for $t1 {
            type Output = $out;

            fn $op_fn(self, rhs: $t2) -> $out {
                let (l, r) = (self.data.data.into_inner(), rhs.data.data.into_inner());
                $out::new(var_format!("(", " * ", ")"; l, r), Expr)
            }
        }
    )
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
    ($t:ident, $($vec:ident, $($mat:ident, $mat_fn:ident, $trait:ident,)*;)*) => (
        matrix_subs!($($vec, $($mat,)*;)*);
        mat_ops!($($($mat,)*;)* | $($($mat,)*;)*);
        mat_ops!(!$($vec,)*;$($vec,)*;$($($mat,)*;)*);

        $(
            matrix_param!($($mat,)*;4);
            $(
                arg_op!(Mul, mul, $t, $mat, $mat);
                arg_op!(Mul, mul, $mat, $t, $mat);
                impl_matrix!($mat, $mat_fn, DataType::$mat, $trait);
            )*
        )*
    )
}

create_matrix!(Float, Float4, Mat4x4, mat4x4, Mat4x4Arg, Mat3x4, mat3x4, Mat3x4Arg, Mat2x4, mat2x4, Mat2x4Arg,;
    Float3, Mat4x3, mat4x3, Mat4x3Arg, Mat3x3, mat3x3, Mat3x3Arg, Mat2x3, mat2x3, Mat2x3Arg,;
    Float2, Mat4x2, mat4x2, Mat4x2Arg, Mat3x2, mat3x2, Mat3x2Arg, Mat2x2, mat2x2, Mat2x2Arg,;);
