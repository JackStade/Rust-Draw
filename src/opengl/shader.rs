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
    fn uniforms(&self) -> &[VarType];

    fn vert_inputs(&self) -> &[VarType];

    fn vert_outputs(&self) -> &[VarType];

    fn frag_inputs(&self) -> &[VarType];

    fn frag_outputs(&self) -> &[VarType];
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
    Vert: Fn(PIn, PUniforms) -> VOut,
    Frag: Fn(FIn, PUniforms) -> POut,
    Proto: ProgramPrototype<Uniforms, PUniforms, In, PIn, Out, POut, VOut, FIn>,
>(
    gl: &mut super::GlDraw,
    prototype: Proto,
    vertex_shader_fn: Vert,
    fragment_shader_fn: Frag,
) -> ShaderProgram<In, Uniforms, Out> {
    let uniform_names = prototype.uniforms();
    let in_names = prototype.vert_inputs();
    let v_pass = prototype.vert_outputs();
    let f_pass = prototype.frag_inputs();
    let output_names = prototype.frag_outputs();
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

    let uniform_locations = if cfg!(feature = "opengl43") {
        // with higher versions of opengl, it isn't necessay to manually
        // query uniform locations
        Vec::new()
    } else {
        let mut v = vec![0; Uniforms::NArgs];
        for i in 0..PUniforms::NArgs {
            if let VarType::Declare(_, n) = uniform_names[i] {
                unsafe {
                    v[n] = gl::GetUniformLocation(
                        program,
                        CString::new(format!("{}", uniform_names[i]))
                            .unwrap()
                            .as_ptr() as *const _,
                    );
                }
            }
        }
        v
    };

    ShaderProgram {
        uniform_locations: uniform_locations,
        program: program,
        phantom: PhantomData,
    }
}

mod builtin_vars {
    const VERTEX_ID: &str = "gl_VertexID";
    const INSTANCE_ID: &str = "gl_InstanceID";
    const DRAW_ID: &str = "gl_DrawID";
    const BASE_VERTEX: &str = "gl_BaseVertex";
    const BASE_INSTANCE: &str = "gl_BaseInstance";

    const POSITION: &str = "gl_Position";
    const POINT_SIZE: &str = "gl_PointSize";
    const CLIP_DISTANCE: &str = "gl_ClipDistance";

    const COORD: &str = "gl_FragCoord";
    const FRONT_FACING: &str = "gl_FrontFacing";
    const POINT_COORD: &str = "gl_PointCoord";

    const SAMPLE_ID: &str = "gl_SampleID";
    const SAMPLE_POSITION: &str = "gl_SamplePosition";
    const SAMPLE_MASK_IN: &str = "gl_SampleMaskIn";

    const SAMPLE_MASK: &str = "gl_SampleMask";
    const DEPTH: &str = "gl_FragDepth";

    const DEPTH_PARAMS: &str = "gl_DepthRange";
    const NUM_SAMPLES: &str = "gl_NumSamples";
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
    input_names: &[VarType],
    uniform_names: &[VarType],
    output_names: &[VarType],
    input_qualifiers: bool,
    output_qualifiers: bool,
) -> String {
    let mut shader = format!("{}\n", VERSION);
    let in_args = In::map_args().args;
    for i in 0..In::NArgs {
        if let VarType::Declare(_, n) = input_names[i] {
            if input_qualifiers {
                shader = format!(
                    "{}layout(location = {}) in {} {};\n",
                    shader,
                    n,
                    in_args[i].gl_type(),
                    input_names[i]
                );
            } else {
                shader = format!(
                    "{}in {} {};\n",
                    shader,
                    in_args[i].gl_type(),
                    input_names[i]
                );
            }
        }
    }
    shader = format!("{}\n", shader);
    let uniform_args = Uniforms::map_args().args;
    for i in 0..Uniforms::NArgs {
        if let VarType::Declare(_, n) = uniform_names[i] {
            if cfg!(feature = "opengl43") {
                shader = format!(
                    "{}layout(location = {}) uniform {} {};\n",
                    shader,
                    n,
                    uniform_args[i].gl_type(),
                    uniform_names[i]
                );
            } else {
                shader = format!(
                    "{}uniform {} {};\n",
                    shader,
                    uniform_args[i].gl_type(),
                    uniform_names[i]
                );
            }
        }
    }
    shader = format!("{}\n", shader);
    let out_args = Out::map_args().args;
    for i in 0..Out::NArgs {
        if let VarType::Declare(_, n) = output_names[i] {
            if output_qualifiers {
                shader = format!(
                    "{}layout(location = {}) out {} {};\n",
                    shader,
                    n,
                    out_args[i].gl_type(),
                    output_names[i]
                );
            } else {
                shader = format!(
                    "{}out {} {};\n",
                    shader,
                    out_args[i].gl_type(),
                    output_names[i]
                );
            }
        }
    }
    // the create function are marked as unsafe because it is neccesary to
    // ensure that the names created are defined in the shader.
    let in_type = unsafe { In::create(&input_names) };
    let uniform_type = unsafe { Uniforms::create(&output_names) };
    let out = generator(in_type, uniform_type).map_data_args().args;
    shader = format!("{}\n\nvoid main() {{\n", shader);
    for i in 0..out.len() {
        shader = format!("{}   {} = {};\n", shader, output_names[i], out[i].1);
    }
    shader = format!("{}}}\n", shader);
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
        info_log.set_len(512 - 1); // subtract 1 to skip the trailing null character
        gl::GetShaderiv(vertex_shader, gl::COMPILE_STATUS, &mut success);
        if success != gl::TRUE as GLint {
            gl::GetShaderInfoLog(
                vertex_shader,
                512,
                ptr::null_mut(),
                info_log.as_mut_ptr() as *mut GLchar,
            );
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
                ptr::null_mut(),
                info_log.as_mut_ptr() as *mut GLchar,
            );
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
                ptr::null_mut(),
                info_log.as_mut_ptr() as *mut GLchar,
            );
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

        const Arg: &'static str;
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

    				const Arg: &'static str = stringify!($vars);
    			}
    		)*
    	);
        ($set:ident, $($vals:ident),*;$($vars:ident),*) => (
        	pub struct $set {}

        	$(
        		pub struct $vals {}

        		pub const $vars: $vals = $vals {};
        	)*

        	impl_swizzle!(;Vec4, Vec3, Vec2, Vec1,;$($vals,)*);

        	swizzle_set!($set, $($vals),*;$($vars),*;sz::D, sz::C, sz::B, sz::A);
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
    		format!("{}{}", $a0::Arg, concat_args!($($arg,)*));
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

/// This module contains traits that should be in scope in order to use
/// many shader building functions. Do not create new implimentations for these
/// traits. In general, it is not unsafe to call member functions of these traits,
/// but doing so is useless.
pub mod traits {
    use super::{DataType, ShaderArgDataList, ShaderArgList, VarType};
    pub use super::{Float2Arg, Float3Arg, Float4Arg, FloatArg};

    pub unsafe trait ArgType {
        /// Do not call this function.
        unsafe fn create(data: String) -> Self;

        fn data_type() -> DataType;

        fn as_shader_data(self) -> String;
    }

    /// A marker trait for non-opaque types. Types like samplers
    /// cannot be used as shader inputs.
    pub unsafe trait TransparentArg: ArgType {}

    /// ShaderArgs is a trait that is implemented for types of
    /// possible opengl arguments
    pub unsafe trait ShaderArgs {
        const NArgs: usize;

        /// Do not call this function.
        unsafe fn create(names: &[VarType]) -> Self;

        fn map_args() -> ShaderArgList;

        fn map_data_args(self) -> ShaderArgDataList;
    }

    /// Like shaderargs, but the args must be transparent glsl types. Samplers are
    /// not transparent, so they cannot be used as an output to a framebuffer for example.
    pub unsafe trait TransparentArgs: ShaderArgs {}

    macro_rules! impl_shader_args {
		// a macro could be implemented that counts the number of arguments
		// that are passed to this macro, but that would be pretty unneccesary
		($($name:ident),*; $num:expr) => (
			unsafe impl<$($name: ArgType),*> ShaderArgs for ($($name,)*) {
				const NArgs: usize = $num;

				unsafe fn create(names: &[VarType]) -> Self {
					let mut n = 0;
					$(
						let $name = $name::create(format!("{}", names[n]));
						n += 1;
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

			unsafe impl<$($name: TransparentArg),*> TransparentArgs for ($($name,)*) {}
		)
	}

    impl_shader_args! {; 0}
    impl_shader_args! {U1; 1 }
    impl_shader_args! {U1, U2; 2}
    impl_shader_args! {U1, U2, U3; 3}
    impl_shader_args! {U1, U2, U3, U4; 4}
    impl_shader_args! {U1, U2, U3, U4, U5; 5}
    impl_shader_args! {U1, U2, U3, U4, U5, U6; 6}
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
        unsafe impl $a0 for ($t0,$($t),*) {
            fn $f0(self) -> $v0 {
                let ($f0,$($f),*) = self;
                $v0 {
                    data: format!("{}({})", $v0::data_type().gl_type(),
                        concat!(format!("{}{}", $f0, $tag),$(format!("{}{}", $f, $tag),)*)),
                }
            }
        }

        vec_litteral!($tag, $($t, $arg, $f, $v,)*);
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

create_vec!(f32, "f", Float4, Float3, Float2, Float;
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
