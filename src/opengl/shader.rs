use self::traits::*;
use std::cmp;
use std::ffi::CString;
use std::fmt;
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul};
use std::ptr;
use std::str;

use super::gl;
use super::gl::types::*;

pub(super) const COLOR_VERTEX_SHADER_SOURCE: &[u8] = b"
#version 410 core

layout (location = 0) in vec3 aPos;

uniform mat4 transform;

void main() {
	gl_Position = transform*vec4(aPos.x, aPos.y, aPos.z, 1.0);
}
";

pub(super) const COLOR_FRAGMENT_SHADER_SOURCE: &[u8] = b"
#version 410 core

uniform vec4 color;

out vec4 FragColor;

void main() {
	FragColor = color;
}
";

pub(super) const TEX_VERTEX_SHADER_SOURCE: &[u8] = b"
#version 410 core

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
#version 410 core

uniform sampler2D sampler;

in vec2 pass_uv;

out vec4 FragColor;

void main() {
	FragColor = texture(sampler, pass_uv);
}
";

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
    Mat2,
    Mat3,
    Mat4,
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
            DataType::Mat2 => "mat2",
            DataType::Mat3 => "mat3",
            DataType::Mat4 => "mat4",
        }
    }
}

/// This needs to be public because it is used by a function of
/// the ShaderArgs trait.
pub struct ShaderArgDataList {
    args: Vec<(DataType, String)>,
}

/// Another type that annoyingly needs to be public.
#[derive(PartialEq, Eq)]
pub struct ShaderArgList {
    args: Vec<DataType>,
}

/// This trait marks an object that contains some parameters for how the
/// variables in a shader should be interpretted.
///
/// The role of this trait is complicated significantly by the fact that shaders have a number
/// of built in variables. This means that there are a number of different interfaces that need
/// to be described by this type:
///
/// * Uniforms: The uniforms that must be passed by the cpu, the 'Uniforms'
/// minus any built in uniforms used.
/// * PUniforms: The constant variables used by the shader program
/// * In: `PIn` minus the set of built in inputs used
/// * PIn: The inputs taken by the vertex shader.
/// * Out: `POut` minus any built in outputs used by the fragment shader.
/// * POut: The outputs of the fragment shader.
/// * VOut: The outputs of the vertex shader, including those passed to the fragment
/// shader and built in outputs.
/// * FIn: The inputs of the fragment shader, including those passed by the vertex shader
/// and built in inputs.
pub unsafe trait ProgramPrototype<
    Uniforms: ShaderArgs,
    PUniforms: ShaderArgs,
    In: ShaderArgs,
    PIn: ShaderArgs,
    Out: ShaderArgs,
    POut: ShaderArgs,
    VOut: ShaderArgs,
    FIn: ShaderArgs,
>
{
    // these functions serve to map the program set including
    // the input variables into the builtin vars and the program interface
    fn uniforms(i: usize) -> VarType;

    fn vert_inputs(i: usize) -> VarType;

    fn vert_outputs(i: usize) -> VarType;

    fn frag_inputs(i: usize) -> VarType;

    fn frag_outputs(i: usize) -> VarType;
}

/// A basic prototype for shaders that don't need to use built in variables other
/// than the vertex shader position output.
pub struct SimplePrototype<
    Uniforms: ShaderArgs,
    In: ShaderArgsClass<InterfaceArgs>,
    Pass: ShaderArgsClass<TransparentArgs>,
    Out: ShaderArgsClass<InterfaceArgs>,
> {
    phantom: PhantomData<(Uniforms, In, Pass, Out)>,
}

pub fn simple_prototype<
    U: ShaderArgs,
    I: ShaderArgsClass<InterfaceArgs>,
    P: ShaderArgsClass<TransparentArgs> + AttachFront<Float4>,
    O: ShaderArgsClass<InterfaceArgs>,
>() -> SimplePrototype<U, I, P, O> {
    SimplePrototype {
        phantom: PhantomData,
    }
}

use crate::swizzle::{AttachBack, AttachFront, RemoveBack, RemoveFront};

unsafe impl<
        U: ShaderArgs,
        I: ShaderArgsClass<InterfaceArgs>,
        P: ShaderArgsClass<TransparentArgs> + AttachFront<Float4>,
        O: ShaderArgsClass<InterfaceArgs>,
    > ProgramPrototype<U, U, I, I, O, O, P::AttachFront, P> for SimplePrototype<U, I, P, O>
where
    P::AttachFront: ShaderArgs,
{
    fn uniforms(i: usize) -> VarType {
        VarType::Declare("u", i)
    }

    fn vert_inputs(i: usize) -> VarType {
        VarType::Declare("i", i)
    }

    fn vert_outputs(i: usize) -> VarType {
        if i == 0 {
            VarType::Internal("gl_Position")
        } else {
            VarType::Declare("p", i - 1)
        }
    }

    fn frag_inputs(i: usize) -> VarType {
        VarType::Declare("p", i)
    }

    fn frag_outputs(i: usize) -> VarType {
        VarType::Declare("o", i)
    }
}

use builtin_vars::{
    BuiltInFragInputs, BuiltInFragOutputs, BuiltInUniforms, BuiltInVars, BuiltInVertInputs,
    BuiltInVertOutputs,
};

/// A full prototype is capable of referencing a full set of different
/// built in variables, like depth and stencils values, vertex id, sample mask,
/// etc.
pub struct FullPrototype<
    U: ShaderArgs,
    I: ShaderArgsClass<InterfaceArgs>,
    P: ShaderArgsClass<TransparentArgs>,
    O: ShaderArgsClass<InterfaceArgs>,
    UB: BuiltInVars<U>,
    IB: BuiltInVars<I>,
    VertPB: BuiltInVars<P>,
    FragPB: BuiltInVars<P>,
    OB: BuiltInVars<O>,
> {
    phantom: PhantomData<(U, I, P, O, UB, IB, VertPB, FragPB, OB)>,
}

unsafe impl<
        U: ShaderArgs,
        I: ShaderArgsClass<InterfaceArgs>,
        P: ShaderArgsClass<TransparentArgs>,
        O: ShaderArgsClass<InterfaceArgs>,
        UB: BuiltInVars<U>,
        IB: BuiltInVars<I>,
        VertPB: BuiltInVars<P>,
        FragPB: BuiltInVars<P>,
        OB: BuiltInVars<O>,
    > ProgramPrototype<U, UB::Tuple, I, IB::Tuple, O, OB::Tuple, VertPB::Tuple, FragPB::Tuple>
    for FullPrototype<U, I, P, O, UB, IB, VertPB, FragPB, OB>
{
    fn uniforms(i: usize) -> VarType {
        if (i < UB::LEN) {
            VarType::Internal(UB::get_name(i))
        } else {
            VarType::Declare("u", i - UB::LEN)
        }
    }

    fn vert_inputs(i: usize) -> VarType {
        if (i < IB::LEN) {
            VarType::Internal(IB::get_name(i))
        } else {
            VarType::Declare("i", i - IB::LEN)
        }
    }

    fn vert_outputs(i: usize) -> VarType {
        if (i < VertPB::LEN) {
            VarType::Internal(VertPB::get_name(i))
        } else {
            VarType::Declare("p", i - VertPB::LEN)
        }
    }

    fn frag_inputs(i: usize) -> VarType {
        if (i < FragPB::LEN) {
            VarType::Internal(FragPB::get_name(i))
        } else {
            VarType::Declare("p", i - FragPB::LEN)
        }
    }

    fn frag_outputs(i: usize) -> VarType {
        if (i < OB::LEN) {
            VarType::Internal(OB::get_name(i))
        } else {
            VarType::Declare("o", i - OB::LEN)
        }
    }
}

#[derive(Clone, Copy)]
pub struct ShaderParamSet<U: ShaderArgs, I: ShaderArgs, P: ShaderArgs, O: ShaderArgs> {
    phantom: PhantomData<(U, I, P, O)>,
}

impl<U: ShaderArgs, I: ShaderArgs, P: ShaderArgs, O: ShaderArgs> ShaderParamSet<U, I, P, O> {
    pub fn new() -> ShaderParamSet<U, I, P, O> {
        ShaderParamSet {
            phantom: PhantomData,
        }
    }
}

pub fn full_prototype<
    U: ShaderArgs,
    I: ShaderArgsClass<InterfaceArgs>,
    P: ShaderArgsClass<TransparentArgs>,
    O: ShaderArgsClass<InterfaceArgs>,
    UB: BuiltInVars<U>,
    IB: BuiltInVars<I>,
    VertPB: BuiltInVars<P>,
    FragPB: BuiltInVars<P>,
    OB: BuiltInVars<O>,
    UF: Fn(BuiltInUniforms) -> UB,
    IF: Fn(BuiltInVertInputs) -> IB,
    VPF: Fn(BuiltInVertOutputs) -> VertPB,
    FPF: Fn(BuiltInFragInputs) -> FragPB,
    OF: Fn(BuiltInFragOutputs) -> OB,
>(
    uniform_fn: UF,
    input_fn: IF,
    vertex_pass_fn: VPF,
    fragment_pass_fn: FPF,
    output_fn: OF,
    set: ShaderParamSet<U, I, P, O>,
) -> FullPrototype<U, I, P, O, UB, IB, VertPB, FragPB, OB> {
    FullPrototype {
        phantom: PhantomData,
    }
}

/// This is a helper function that generates a prototype that uses the position
/// vertex shader output and the depth fragment shader output.
pub fn depth_prototype<
    U: ShaderArgs,
    I: ShaderArgsClass<InterfaceArgs>,
    P: ShaderArgsClass<TransparentArgs>,
    O: ShaderArgsClass<InterfaceArgs>,
>(
    set: ShaderParamSet<U, I, P, O>,
) -> FullPrototype<
    U,
    I,
    P,
    O,
    (),
    (),
    (builtin_vars::BuiltInVar<Float4, builtin_vars::Position>,),
    (),
    (builtin_vars::BuiltInVar<Float, builtin_vars::Depth>,),
>
where
    (): BuiltInVars<U> + BuiltInVars<I> + BuiltInVars<P>,
    (builtin_vars::BuiltInVar<Float4, builtin_vars::Position>,): BuiltInVars<P>,
    (builtin_vars::BuiltInVar<Float, builtin_vars::Depth>,): BuiltInVars<O>,
{
    FullPrototype {
        phantom: PhantomData,
    }
}

pub mod builtin_vars {
    use super::{ArgType, ShaderArgs};
    use super::{Float, Float2, Float3, Float4, Int};
    use crate::swizzle::{AttachBack, AttachFront, RemoveBack, RemoveFront};
    use std::marker::PhantomData;

    pub struct BuiltInVar<T: ArgType, N: VarName> {
        phantom: PhantomData<(T, N)>,
    }

    impl<T: ArgType, N: VarName> BuiltInVar<T, N> {
        pub(crate) fn new() -> BuiltInVar<T, N> {
            BuiltInVar {
                phantom: PhantomData,
            }
        }
    }

    pub unsafe trait VarName {
        const NAME: &'static str;
    }

    struct Hidden;

    pub struct VertexID {
        hidden: Hidden,
    }

    pub struct InstanceID {
        hidden: Hidden,
    }

    pub struct Position {
        hidden: Hidden,
    }

    pub struct PointSize {
        hidden: Hidden,
    }

    pub struct Depth {
        hidden: Hidden,
    }

    pub struct FragCoord {
        hidden: Hidden,
    }

    pub struct PointCoord {
        hidden: Hidden,
    }

    unsafe impl VarName for VertexID {
        const NAME: &'static str = "gl_VertexID";
    }

    unsafe impl VarName for InstanceID {
        const NAME: &'static str = "gl_InstanceID";
    }

    unsafe impl VarName for Position {
        const NAME: &'static str = "gl_Position";
    }

    unsafe impl VarName for PointSize {
        const NAME: &'static str = "gl_PointSize";
    }

    unsafe impl VarName for Depth {
        const NAME: &'static str = "gl_FragDepth";
    }

    unsafe impl VarName for FragCoord {
        const NAME: &'static str = "gl_FragCoord";
    }

    unsafe impl VarName for PointCoord {
        const NAME: &'static str = "gl_PointCoord";
    }

    pub struct BuiltInUniforms {}

    pub struct BuiltInVertInputs {
        pub vertex_id: BuiltInVar<Int, VertexID>,
        pub instance_id: BuiltInVar<Int, InstanceID>,
    }

    pub struct BuiltInVertOutputs {
        pub position: BuiltInVar<Float4, Position>,
        pub point_size: BuiltInVar<Float4, PointSize>,
    }

    pub struct BuiltInFragInputs {
        pub frag_coord: BuiltInVar<Float4, FragCoord>,
        pub point_coord: BuiltInVar<Float2, PointCoord>,
    }

    pub struct BuiltInFragOutputs {
        pub depth: BuiltInVar<Float, Depth>,
    }

    pub unsafe trait BuiltInVars<A> {
        type Tuple: ShaderArgs;

        const LEN: usize;

        fn get_name(i: usize) -> &'static str;
    }

    pub struct RecursiveTuple<T, U> {
        _t: T,
        _u: U,
    }

    unsafe impl<T, A, F: ArgType, B: ArgType, FN: VarName, BN: VarName> BuiltInVars<A> for T
    where
        T: RemoveFront<Front = BuiltInVar<F, FN>>,
        T: RemoveBack<Back = BuiltInVar<B, BN>>,
        A: AttachFront<B>,
        <T as RemoveFront>::Remaining: BuiltInVars<A>,
        <T as RemoveBack>::Remaining: BuiltInVars<A::AttachFront>,
        <<T as RemoveBack>::Remaining as BuiltInVars<A::AttachFront>>::Tuple: ShaderArgs,
    {
        type Tuple = <<T as RemoveBack>::Remaining as BuiltInVars<A::AttachFront>>::Tuple;

        const LEN: usize = <T as RemoveBack>::Remaining::LEN + 1;

        fn get_name(i: usize) -> &'static str {
            if (i == 0) {
                return FN::NAME;
            }
            <T as RemoveFront>::Remaining::get_name(i - 1)
        }
    }

    unsafe impl<A: ShaderArgs> BuiltInVars<A> for () {
        type Tuple = A;

        const LEN: usize = 0;

        fn get_name(i: usize) -> &'static str {
            panic!("Name {} is out of bounds for this set of built in vars.", i);
        }
    }
}

/// A type that is used internally, but it is used by ProgramPrototype so
/// it has to be public.
#[derive(Clone, Copy)]
pub enum VarType {
    Declare(&'static str, usize),
    Internal(&'static str),
}

impl fmt::Display for VarType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            VarType::Declare(s, n) => write!(f, "{}{}", s, n),
            VarType::Internal(s) => write!(f, "{}", s),
        }
    }
}

pub struct ShaderProgram<In: ShaderArgs, Uniforms: ShaderArgs, Out: ShaderArgs> {
    uniform_locations: Vec<GLint>,
    program: GLuint,
    phantom: PhantomData<(In, Uniforms, Out)>,
}

/// Create a shader program with a vertex and a fragment shader.
///
/// Shaders can be used to cause more advanced behavior to happen on the GPU. The
/// most common use is for lighting calculations. Shaders are a lower level feature
/// that can be difficult to use correctly.
pub fn create_program<
    Uniforms: ShaderArgs,
    PUniforms: ShaderArgs,
    In: ShaderArgs,
    PIn: ShaderArgs,
    Out: ShaderArgs,
    POut: ShaderArgs,
    VOut: ShaderArgs,
    FIn: ShaderArgs,
    Vert: Fn(PIn, PUniforms) -> VOut + Sync,
    Frag: Fn(FIn, PUniforms) -> POut + Sync,
    Proto: ProgramPrototype<Uniforms, PUniforms, In, PIn, Out, POut, VOut, FIn>,
>(
    gl: &mut super::GlDraw,
    prototype: &Proto,
    vertex_shader_fn: Vert,
    fragment_shader_fn: Frag,
) -> ShaderProgram<In, Uniforms, Out> {
    let uniform_names = Proto::uniforms;
    let in_names = Proto::vert_inputs;
    let v_pass = Proto::vert_outputs;
    let f_pass = Proto::frag_inputs;
    let output_names = Proto::frag_outputs;
    let mut v_shader_string = CString::new(create_shader_string(
        vertex_shader_fn,
        in_names,
        uniform_names,
        v_pass,
        // vertex outputs don't have layout qualifiers, but the inputs do
        true,
        false,
    ))
    .unwrap();
    let mut f_shader_string = CString::new(create_shader_string(
        fragment_shader_fn,
        f_pass,
        uniform_names,
        output_names,
        // fragment inputs don't have layout qualifiers, but the outputs do
        false,
        true,
    ))
    .unwrap();
    let program = get_program(v_shader_string.as_bytes(), f_shader_string.as_bytes());

    let mut uniform_locations = vec![0; Uniforms::NARGS];
    for i in 0..PUniforms::NARGS {
        if let VarType::Declare(_, n) = uniform_names(i) {
            unsafe {
                uniform_locations[n] = gl::GetUniformLocation(
                    program,
                    CString::new(format!("{}", uniform_names(i)))
                        .unwrap()
                        .as_ptr() as *const _,
                );
            }
        }
    }

    ShaderProgram {
        uniform_locations: uniform_locations,
        program: program,
        phantom: PhantomData,
    }
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
    In: ShaderArgs,
    Uniforms: ShaderArgs,
    Out: ShaderArgs,
    Shader: Fn(In, Uniforms) -> Out,
>(
    generator: Shader,
    input_names: fn(usize) -> VarType,
    uniform_names: fn(usize) -> VarType,
    output_names: fn(usize) -> VarType,
    input_qualifiers: bool,
    output_qualifiers: bool,
) -> String {
    let mut shader = format!("{}\n", VERSION);
    let in_args = In::map_args().args;
    for i in 0..In::NARGS {
        if let VarType::Declare(_, n) = input_names(i) {
            if input_qualifiers {
                shader = format!(
                    "{}layout(location = {}) in {} {};\n",
                    shader,
                    n,
                    in_args[i].gl_type(),
                    input_names(i)
                );
            } else {
                shader = format!(
                    "{}in {} {};\n",
                    shader,
                    in_args[i].gl_type(),
                    input_names(i)
                );
            }
        }
    }
    shader = format!("{}\n", shader);
    let uniform_args = Uniforms::map_args().args;
    for i in 0..Uniforms::NARGS {
        if let VarType::Declare(_, n) = uniform_names(i) {
            shader = format!(
                "{}uniform {} {};\n",
                shader,
                uniform_args[i].gl_type(),
                uniform_names(i)
            );
        }
    }
    shader = format!("{}\n", shader);
    let out_args = Out::map_args().args;
    for i in 0..Out::NARGS {
        if let VarType::Declare(_, n) = output_names(i) {
            if output_qualifiers {
                shader = format!(
                    "{}layout(location = {}) out {} {};\n",
                    shader,
                    n,
                    out_args[i].gl_type(),
                    output_names(i)
                );
            } else {
                shader = format!(
                    "{}out {} {};\n",
                    shader,
                    out_args[i].gl_type(),
                    output_names(i)
                );
            }
        }
    }
    // the create function are marked as unsafe because it is neccesary to
    // ensure that the names created are defined in the shader.
    let in_type = unsafe { In::create(input_names) };
    let uniform_type = unsafe { Uniforms::create(output_names) };
    let out = generator(in_type, uniform_type).map_data_args().args;
    shader = format!("{}\n\nvoid main() {{\n", shader);
    for i in 0..out.len() {
        shader = format!("{}   {} = {};\n", shader, output_names(i), out[i].1);
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
    use super::super::super::swizzle as sz;
    use super::super::super::swizzle::{Swizzle, SZ};
    use super::traits::ArgType;

    pub unsafe trait GlSZ {
        type S: SZ;

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

        	swizzle_set!($set, $($vals),*;$($vars),*;sz::R3, sz::R2, sz::R1, sz::R0);
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

pub mod traits {
    use super::{DataType, ShaderArgDataList, ShaderArgList, VarType};
    pub use super::{Float2Arg, Float3Arg, Float4Arg, FloatArg};

    pub unsafe trait ArgType {
        /// Do not call this function.
        unsafe fn create(data: String) -> Self;

        fn data_type() -> DataType;

        fn as_shader_data(self) -> String;
    }

    pub unsafe trait ArgClass {}

    pub unsafe trait ArgParameter<T: ArgClass> {}

    unsafe impl ArgClass for () {}

    unsafe impl<T: ArgType> ArgParameter<()> for T {}

    /// Most arg types, but not opaque types like samplers.
    pub struct TransparentArgs;

    unsafe impl ArgClass for TransparentArgs {}

    /// The types of args that can be used as vertex shader inputs or
    /// fragment shader outputs. This includes all scalar and vector types, but
    /// not matrix types. All types that implement `ArgParameter<InterfaceArgs>`
    /// also implement `ArgParameter<TransparentArgs>`.
    pub struct InterfaceArgs;

    unsafe impl ArgClass for InterfaceArgs {}

    /// ShaderArgs is a trait that is implemented for types of
    /// possible opengl argument sets
    pub unsafe trait ShaderArgs {
        const NARGS: usize;

        /// Do not call this function.
        unsafe fn create(names: fn(usize) -> VarType) -> Self;

        fn map_args() -> ShaderArgList;

        fn map_data_args(self) -> ShaderArgDataList;
    }

    /// Sometimes it is neccessary to restrict a type to only certain types
    /// of arguments.
    pub unsafe trait ShaderArgsClass<T>: ShaderArgs {}

    macro_rules! impl_shader_args {
		// a macro could be implemented that counts the number of arguments
		// that are passed to this macro, but that would be pretty unneccesary
		($($name:ident),*; $num:expr) => (
			unsafe impl<$($name: ArgType),*> ShaderArgs for ($($name,)*) {
				const NARGS: usize = $num;

				unsafe fn create(names: fn(usize) -> VarType) -> Self {
					let n = 0;
					$(
						let $name = $name::create(format!("{}", names(n)));
						let n = n + 1;
					)*
					($($name,)*)
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

            unsafe impl<T: ArgClass, $($name: ArgType + ArgParameter<T>),*> ShaderArgsClass<T> for ($($name,)*) {}
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

    /*impl_shader_args! {; 0}
    impl_shader_args! {U1; 1 }
    impl_shader_args! {U1, U2; 2}
    impl_shader_args! {U1, U2, U3; 3}
    impl_shader_args! {U1, U2, U3, U4; 4}
    impl_shader_args! {U1, U2, U3, U4, U5; 5}
    impl_shader_args! {U1, U2, U3, U4, U5, U6; 6}*/
}

// the complement of implementations for certain vec ops. Need to
// be careful not to redefine.
macro_rules! vec_ops_reverse {
    ($vec_type:ident, $s0:ident,) => ();
    ($vec_type:ident, $s0:ident, $($sub:ident,)+) => (
        impl Mul<last!($($sub,)*)> for $vec_type {
            type Output = $vec_type;

            fn mul(self, rhs: last!($($sub,)*)) -> $vec_type {
                $vec_type {
                    data: format!("({} * {})", self.data, rhs.data),
                }
            }
        }
    )
}

macro_rules! vec_type {
    ($vec:ident, $vec_type:ident, $trait:ident, $data:expr, $($sub:ident,)*) => (
    	#[derive(Clone)]
        pub struct $vec_type {
            data: String,
        }

        impl Mul<$vec_type> for last!($($sub,)*) {
            type Output = $vec_type;

            fn mul(self, rhs: $vec_type) -> $vec_type {
                $vec_type {
                    data: format!("({} * {})", self.data, rhs.data),
                }
            }
        }

        impl Div<last!($($sub,)*)> for $vec_type {
            type Output = $vec_type;

            fn div(self, rhs: last!($($sub,)*)) -> $vec_type {
                $vec_type {
                    data: format!("({} / {})", self.data, rhs.data),
                }
            }
        }

        vec_ops_reverse!($vec_type, $($sub,)*);

        impl Add for $vec_type {
        	type Output = $vec_type;

        	fn add(self, rhs: $vec_type) -> $vec_type {
        		$vec_type {
        			data: format!("({} + {})", self.data, rhs.data),
        		}
        	}
        }

        pub unsafe trait $trait {
            fn $vec(self) -> $vec_type;
        }

        pub fn $vec<T: $trait>(args: T) -> $vec_type {
        	args.$vec()
        }

        unsafe impl ArgType for $vec_type {
        	unsafe fn create(data: String) -> Self {
        		$vec_type {
        			data: data,
        		}
        	}

        	fn data_type() -> DataType {
        		$data
        	}

        	fn as_shader_data(self) -> String {
        		self.data
        	}
        }

        unsafe impl ArgParameter<TransparentArgs> for $vec_type {}

        unsafe impl ArgParameter<InterfaceArgs> for $vec_type {}

        subs!($trait, $vec, $vec_type ;  ; $($sub,)*);
    )
}

macro_rules! subs {
	($trait:ident, $vec:ident, $vec_type:ident;$($start:ident,)*;) => (
		unsafe impl $trait for ($($start),*) {
			fn $vec(self) -> $vec_type {
				match_from!(self, $($start,)*;u1, u2, u3, u4,;);
				$vec_type {
					data: format!("{}({})",
						$vec_type::data_type().gl_type(), concat_enough!($($start,)* ; u1, u2, u3, u4,;)),
				}
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
		concat!($($build.as_shader_data(),)*);
	);
	($t1:ident, $($try:ident,)* ; $v1:ident, $($vals:ident,)* ; $($build:ident,)*) => (
		concat_enough!($($try,)* ; $($vals,)* ; $($build,)*$v1,);
	)
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
        impl $vec {
            /// Applies a swizzle mask to the vector type. Note that this is only availible for
            /// single item types when version >= opengl4.2.
            pub fn map<T: SwizzleMask<$($types,)*swizzle::$sz>>(self, mask: T) -> T::Out {
                unsafe {
                    T::Out::create(format!("{}.{}", self.data, T::get_vars()))
                }
            }
        }
    );
	($vec:ident, $($next:ident,)+;$($types:ident,)*;$sz:ident, $($s:ident,)*) => (
		impl $vec {
            /// Applies a swizzle mask to the vector type.
			pub fn map<T: SwizzleMask<$($types,)*swizzle::$sz>>(self, mask: T) -> T::Out {
				unsafe {
					T::Out::create(format!("{}.{}", self.data, T::get_vars()))
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
            fn $f0(self) -> $v0 {
                let tup!($f0,$($f,)*) = self;
                $v0 {
                    data: format!("{}({})", $v0::data_type().gl_type(),
                        concat!(format!("{}{}", $f0, $tag),$(format!("{}{}", $f, $tag),)*)),
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

macro_rules! first {
    ($arg:ident, $($args:ident,)*) => {
        $arg
    };
}

macro_rules! create_vec {
    ($ty:ty, $suffix:expr, $($obj:ident),*; $($func:ident),*; $($arg:ident),*) => {
        vec_types!($($obj,)*;$($func,)*;$($arg,)*);
        vec_swizzle!($($obj),*);
        vec_litteral!($suffix, $ty, $($arg),*;$($func),*;$($obj),*);
    }
}

create_vec!(f32, "", Float4, Float3, Float2, Float;
    float4, float3, float2, float;
    Float4Arg, Float3Arg, Float2Arg, FloatArg);

create_vec!(i32, "", Int4, Int3, Int2, Int;
    int4, int3, int2, int;
    Int4Arg, Int3Arg, Int2Arg, IntArg);

unsafe impl FloatArg for Int {
    fn float(self) -> Float {
        Float {
            data: format!("float({})", self.data),
        }
    }
}
