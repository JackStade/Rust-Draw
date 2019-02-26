use self::traits::*;
use std::cmp;
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

pub enum ShaderType {
    Vertex,
    Fragment,
    #[cfg(feature = "opengl43")]
    Compute,
}

pub struct ProgramPrototype<In: ShaderArgs, Uniforms: ShaderArgs, Pass: ShaderArgs, Out: ShaderArgs>
{
    phantom: PhantomData<(In, Uniforms, Pass, Out)>,
}

pub struct ShaderProgram<In: ShaderArgs, Uniforms: ShaderArgs, Out: ShaderArgs> {
    program: GLuint,
    input_locations: Vec<GLuint>,
    uniform_locations: Vec<GLuint>,
    phantom: PhantomData<(In, Uniforms, Out)>,
}

pub fn create_program<
    In: ShaderArgs,
    Uniforms: ShaderArgs,
    Pass: ShaderArgs,
    Out: ShaderArgs,
    Vert: Fn(In) -> Pass,
    Frag: Fn(Pass) -> In,
>(
    proto: ProgramPrototype<In, Uniforms, Pass, Out>,
) -> ShaderProgram<In, Uniforms, Out>
where
    Vert: Sync,
    Frag: Sync,
{
    unimplemented!();
}

pub fn create_shader_string<
    In: ShaderArgs,
    Uniforms: ShaderArgs,
    Out: ShaderArgs,
    Shader: Fn(In, Uniforms) -> Out,
>(
    generator: Shader,
) -> String {
    let version = "#version 410 core\n";
    let mut shader = format!("{}\n", version);
    let in_args = In::map_args().args;
    for i in 0..In::NArgs {
        shader = format!("{}in {} i{};\n", shader, in_args[i].gl_type(), i);
    }
    shader = format!("{}\n", shader);
    let uniform_args = Uniforms::map_args().args;
    for i in 0..Uniforms::NArgs {
        shader = format!("{}uniform {} u{};\n", shader, uniform_args[i].gl_type(), i);
    }
    shader = format!("{}\n", shader);
    let out_args = Out::map_args().args;
    for i in 0..Out::NArgs {
        shader = format!("{}out {} o{};\n", shader, out_args[i].gl_type(), i);
    }
    // the new function are marked as unsafe because it is neccesary to
    // ensure that the names created are defined in the shader.
    let in_type = unsafe { In::create("i") };
    let uniform_type = unsafe { Uniforms::create("u") };
    let out = generator(in_type, uniform_type).map_data_args().args;
    shader = format!("{}\n\nvoid main() {{\n", shader);
    for i in 0..out.len() {
        shader = format!("{}   o{} = {};\n", shader, i, out[i].1);
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
/// traits.
pub mod traits {
    use super::{DataType, ShaderArgDataList, ShaderArgList};
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

    pub unsafe trait ShaderArgs {
        const NArgs: usize;

        /// Do not call this function.
        unsafe fn create(prefix: &str) -> Self;

        fn map_args() -> ShaderArgList;

        fn map_data_args(self) -> ShaderArgDataList;
    }

    /// Like shaderargs, but the args must be transparent glsl types.
    pub unsafe trait TransparentArgs: ShaderArgs {}

    macro_rules! impl_shader_args {
		// a macro could be implemented that counts the number of arguments
		// that are passed to this macro, but that would be pretty unneccesary
		($($name:ident),*; $num:expr) => (
			unsafe impl<$($name: ArgType),*> ShaderArgs for ($($name,)*) {
				const NArgs: usize = $num;

				unsafe fn create(prefix: &str) -> Self {
					let mut n = 0;
					$(
						let $name = $name::create(format!("{}{}", prefix, n));
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
