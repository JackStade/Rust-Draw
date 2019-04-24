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
        if i < UB::LEN {
            VarType::Internal(UB::get_name(i))
        } else {
            VarType::Declare("u", i - UB::LEN)
        }
    }

    fn vert_inputs(i: usize) -> VarType {
        if i < IB::LEN {
            VarType::Internal(IB::get_name(i))
        } else {
            VarType::Declare("i", i - IB::LEN)
        }
    }

    fn vert_outputs(i: usize) -> VarType {
        if i < VertPB::LEN {
            VarType::Internal(VertPB::get_name(i))
        } else {
            VarType::Declare("p", i - VertPB::LEN)
        }
    }

    fn frag_inputs(i: usize) -> VarType {
        if i < FragPB::LEN {
            VarType::Internal(FragPB::get_name(i))
        } else {
            VarType::Declare("p", i - FragPB::LEN)
        }
    }

    fn frag_outputs(i: usize) -> VarType {
        if i < OB::LEN {
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

#[allow(unused)]
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
    _set: ShaderParamSet<U, I, P, O>,
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

#[allow(unused)]
pub mod builtin_vars {
    use super::{ArgType, ShaderArgs};
    use super::{Float, Float2, Float3, Float4, Int};
    use crate::swizzle::{AttachBack, AttachFront, RemoveBack, RemoveFront};
    use std::marker::PhantomData;

    pub struct BuiltInVar<T: ArgType, N: VarName> {
        phantom: PhantomData<(T, N)>,
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

use super::GlResource;

pub struct ShaderProgram<In: ShaderArgs, Uniforms: ShaderArgs, Out: ShaderArgs> {
    uniform_locations: Vec<GLint>,
    program_id: u32,
    // need to make sure the type is not send or sync
    phantom: PhantomData<(In, Uniforms, Out, std::rc::Rc<()>)>,
}

#[cfg(not(feature = "opengl41"))]
impl<In: ShaderArgs, Uniforms: ShaderArgs, Out: ShaderArgs> GlResource
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
impl<In: ShaderArgs, Uniforms: ShaderArgs, Out: ShaderArgs> GlResource
    for ShaderProgram<In, Uniforms, Out>
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
        let mut data_len = 0;
        let ptr = buffer.as_mut_ptr();
        buffer.set_len(3);
        gl::GetProgramBinary(
            program,
            len,
            &mut data_len,
            &mut format,
            ptr.offset(3) as *mut _,
        );
        buffer[0] = data_len as u32;
        buffer[1] = format;
        buffer[2] = buffer.capacity() as u32;
        std::mem::forget(buffer);
        ptr as *mut ()
    }
}

impl<In: ShaderArgs, Uniforms: ShaderArgs, Out: ShaderArgs> ShaderProgram<In, Uniforms, Out> {
    #[cfg(feature = "opengl41")]
    fn new(
        program: GLuint,
        uniform_locations: Vec<GLint>,
        _vsource: CString,
        _fsource: CString,
    ) -> ShaderProgram<In, Uniforms, Out> {
        let gl_draw = unsafe { super::inner_gl_unsafe() };
        ShaderProgram {
            uniform_locations: uniform_locations,
            program_id: gl_draw.get_resource_generic::<Self>(program, None),
            phantom: PhantomData,
        }
    }

    #[cfg(not(feature = "opengl41"))]
    fn new(
        program: GLuint,
        uniform_locations: Vec<GLint>,
        vsource: CString,
        fsource: CString,
    ) -> ShaderProgram<In, Uniforms, Out> {
        let gl_draw = super::inner_gl_unsafe();
        let b = Box::new([vsource, fsource]);
        let ptr = b.as_mut_ptr();
        mem::forget(b);
        ShaderProgram {
            uniform_locations: uniform_locations,
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
    Uniforms: ShaderArgs,
    PUniforms: ShaderArgs,
    In: ShaderArgs,
    PIn: ShaderArgs,
    Out: ShaderArgs,
    POut: ShaderArgs,
    VOut: ShaderArgs,
    FIn: ShaderArgs,
    VGenOut: IntoArgs<Args = VOut>,
    FGenOut: IntoArgs<Args = POut>,
    Vert: Fn(PIn::AsVarying, PUniforms::AsUniform) -> VGenOut + Sync,
    Frag: Fn(FIn::AsVarying, PUniforms::AsUniform) -> FGenOut + Sync,
    Proto: ProgramPrototype<Uniforms, PUniforms, In, PIn, Out, POut, VOut, FIn>,
>(
    _window: &super::GlWindow,
    _prototype: &Proto,
    vertex_shader_fn: Vert,
    fragment_shader_fn: Frag,
) -> ShaderProgram<In, Uniforms, Out> {
    let uniform_names = Proto::uniforms;
    let in_names = Proto::vert_inputs;
    let v_pass = Proto::vert_outputs;
    let f_pass = Proto::frag_inputs;
    let output_names = Proto::frag_outputs;
    let v_shader_string = CString::new(create_shader_string::<PIn, PUniforms, VGenOut, Vert>(
        vertex_shader_fn,
        in_names,
        uniform_names,
        v_pass,
        // vertex outputs don't have layout qualifiers, but the inputs do
        true,
        false,
    ))
    .unwrap();
    let f_shader_string = CString::new(create_shader_string::<FIn, PUniforms, FGenOut, Frag>(
        fragment_shader_fn,
        f_pass,
        uniform_names,
        output_names,
        // fragment inputs don't have layout qualifiers, but the outputs do
        false,
        true,
    ))
    .unwrap();
    let program = get_program(
        v_shader_string.as_bytes_with_nul(),
        f_shader_string.as_bytes_with_nul(),
    );
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
    ShaderProgram::new(program, uniform_locations, v_shader_string, f_shader_string)
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
    Out: IntoArgs,
    Shader: Fn(In::AsVarying, Uniforms::AsUniform) -> Out,
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
        if let VarType::Declare(_, _) = uniform_names(i) {
            shader = format!(
                "{}uniform {} {};\n",
                shader,
                uniform_args[i].gl_type(),
                uniform_names(i)
            );
        }
    }
    shader = format!("{}\n", shader);
    let out_args = Out::Args::map_args().args;
    for i in 0..Out::Args::NARGS {
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
    let uniform_type = unsafe { Uniforms::create(uniform_names) };
    let out = unsafe {
        let input = in_type.as_varying();
        let output = uniform_type.as_uniform();
        generator(input, output).into_args().map_data_args().args
    };
    shader = format!("{}\n\nvoid main() {{\n", shader);
    let mut builder = VarBuilder::new("var");
    let mut out_strings = Vec::with_capacity(out.len());
    for i in 0..out.len() {
        out_strings.push(builder.format_var(&out[i].1));
    }
    shader = builder.add_strings(shader);
    for i in 0..out.len() {
        shader = format!("{}   {} = {};\n", shader, output_names(i), out_strings[i]);
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

pub mod api {
    pub use super::traits::Map;
    pub use super::{Bool2Arg, Bool3Arg, Bool4Arg, BoolArg};
    pub use super::{Float2Arg, Float3Arg, Float4Arg, FloatArg};
    pub use super::{Int2Arg, Int3Arg, Int4Arg, IntArg};
}

#[allow(unused, unused_parens)]
pub mod traits {
    use super::swizzle::SwizzleMask;
    use super::{DataType, ItemRef, ShaderArgDataList, ShaderArgList, VarExpr, VarString, VarType};
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
        unsafe fn into_t(self) -> T;

        unsafe fn from_t(t: T) -> Self;
    }

    pub trait IntoArg {
        type Arg: ArgType;

        unsafe fn into_arg(self) -> Self::Arg;
    }

    pub trait IntoArgs {
        type Args: ShaderArgs;

        unsafe fn into_args(self) -> Self::Args;
    }

    pub trait IntoConstant<T: ArgType> {
        fn into_constant(self) -> Constant<T>;
    }

    pub trait IntoUniform<T: ArgType> {
        fn into_uniform(self) -> Uniform<T>;
    }

    pub trait IntoVarying<T: ArgType> {
        fn into_varying(self) -> Varying<T>;
    }

    impl<T: ArgType> IntoArg for Constant<T> {
        type Arg = T;

        unsafe fn into_arg(self) -> Self::Arg {
            self.arg
        }
    }

    impl<T: ArgType> IntoArg for Uniform<T> {
        type Arg = T;

        unsafe fn into_arg(self) -> Self::Arg {
            self.arg
        }
    }

    impl<T: ArgType> IntoArg for Varying<T> {
        type Arg = T;

        unsafe fn into_arg(self) -> Self::Arg {
            self.arg
        }
    }

    impl<T: ArgType> IntoArg for &Constant<T> {
        type Arg = T;

        unsafe fn into_arg(self) -> Self::Arg {
            self.arg.clone()
        }
    }

    impl<T: ArgType> IntoArg for &Uniform<T> {
        type Arg = T;

        unsafe fn into_arg(self) -> Self::Arg {
            self.arg.clone()
        }
    }

    impl<T: ArgType> IntoArg for &Varying<T> {
        type Arg = T;

        unsafe fn into_arg(self) -> Self::Arg {
            self.arg.clone()
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
                    }
                }
            }

            impl<O: ArgType, L: ArgType + $op<R, Output = O> + Clone, R: ArgType> $op<$r<R>>
                for &$l<L>
            {
                type Output = $o<O>;

                fn $op_func(self, rhs: $r<R>) -> $o<O> {
                    $o {
                        arg: $op::$op_func(self.clone().arg, rhs.arg),
                    }
                }
            }

            impl<O: ArgType, L: ArgType + $op<R, Output = O>, R: ArgType + Clone> $op<&$r<R>>
                for $l<L>
            {
                type Output = $o<O>;

                fn $op_func(self, rhs: &$r<R>) -> $o<O> {
                    $o {
                        arg: $op::$op_func(self.arg, rhs.clone().arg),
                    }
                }
            }

            impl<O: ArgType, L: ArgType + $op<R, Output = O> + Clone, R: ArgType + Clone>
                $op<&$r<R>> for &$l<L>
            {
                type Output = $o<O>;

                fn $op_func(self, rhs: &$r<R>) -> $o<O> {
                    $o {
                        arg: $op::$op_func(self.clone().arg, rhs.clone().arg),
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
        fn into_constant(self) -> Constant<T> {
            Constant { arg: self }
        }
    }

    impl<T: ArgType> IntoUniform<T> for T {
        fn into_uniform(self) -> Uniform<T> {
            Uniform { arg: self }
        }
    }

    impl<T: ArgType> IntoVarying<T> for T {
        fn into_varying(self) -> Varying<T> {
            Varying { arg: self }
        }
    }

    pub trait ExprCombine<T: ArgType> {
        type Min: ExprType<T>;
    }

    pub trait ExprMin<T: ArgType> {
        type Min: ExprType<T>;
    }

    impl<A: ArgType, T: crate::swizzle::RemoveFront> ExprMin<A> for T
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

    #[derive(Clone)]
    pub struct Constant<T: ArgType> {
        pub(super) arg: T,
    }

    #[derive(Clone)]
    pub struct Uniform<T: ArgType> {
        pub(super) arg: T,
    }

    #[derive(Clone)]
    pub struct Varying<T: ArgType> {
        pub(super) arg: T,
    }

    unsafe impl<T: ArgType> ExprType<T> for Constant<T> {
        unsafe fn into_t(self) -> T {
            self.arg
        }

        unsafe fn from_t(t: T) -> Self {
            Constant { arg: t }
        }
    }

    unsafe impl<T: ArgType> ExprType<T> for Uniform<T> {
        unsafe fn into_t(self) -> T {
            self.arg
        }

        unsafe fn from_t(t: T) -> Self {
            Uniform { arg: t }
        }
    }

    unsafe impl<T: ArgType> ExprType<T> for Varying<T> {
        unsafe fn into_t(self) -> T {
            self.arg
        }

        unsafe fn from_t(t: T) -> Self {
            Varying { arg: t }
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

        type AsUniform;

        type AsVarying;

        unsafe fn as_uniform(self) -> Self::AsUniform;

        unsafe fn as_varying(self) -> Self::AsVarying;

        fn map_args() -> ShaderArgList;

        fn map_data_args(self) -> ShaderArgDataList;
    }

    pub unsafe trait Construct<T: ArgType> {
        fn as_arg(self) -> T;
    }

    /// Sometimes it is neccessary to restrict a type to only certain types
    /// of arguments.
    pub unsafe trait ShaderArgsClass<T>: ShaderArgs {}

    macro_rules! impl_shader_args {
		// a macro could be implemented that counts the number of arguments
		// that are passed to this macro, but that would be pretty unneccesary
		($($name:ident),*; $num:expr) => (
            #[allow(unused_parens)]
			unsafe impl<$($name: ArgType),*> ShaderArgs for ($($name,)*) {
				const NARGS: usize = $num;

				unsafe fn create(names: fn(usize) -> VarType) -> Self {
					let n = 0;
					$(
						let $name = $name::create(VarString::new(names(n)), ItemRef::Static);
						let n = n + 1;
					)*
					($($name,)*)
				}

                type AsUniform = ($(Uniform<$name>),*);

                type AsVarying = ($(Varying<$name>),*);

                unsafe fn as_uniform(self) -> Self::AsUniform {
                    let ($($name,)*) = self;
                    ($(Uniform::from_t($name)),*)
                }

                unsafe fn as_varying(self) -> Self::AsVarying {
                    let ($($name,)*) = self;
                    ($(Varying::from_t($name)),*)
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

                unsafe fn into_args(self) -> ($($name::Arg,)*) {
                    let ($($name),*) = self;
                    ($($name.into_arg(),)*)
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
                    S::Min::from_t(self.into_args().as_arg())
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

        unsafe impl ArgParameter<TransparentArgs> for $vec_type {}

        unsafe impl ArgParameter<InterfaceArgs> for $vec_type {}

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
                    <Self as ExprCombine<T::Out>>::Min::from_t(
                        T::Out::create(var_format!("", ".", ""; self.clone().into_t().data.data.into_inner(), VarString::new(T::get_vars())), Expr)
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
                    // convieniently, opengl allows syntax like `vec4(1.0)`
                    <Self as ExprCombine<T::Out>>::Min::from_t(
                        T::Out::create(var_format!("", "(", ")"; T::Out::data_type(), self.clone().into_t().data.data.into_inner(), Expr))
                    )
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
    				<Self as ExprCombine<T::Out>>::Min::from_t(
    					T::Out::create(var_format!("", ".", ""; self.clone().into_t().data.data.into_inner(), VarString::new(T::get_vars())), Expr)
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
                        VarString::new(concat!(format!("{}{}", $f0, $tag),$(format!("{}{}", $f, $tag),)*))), Expr)
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

create_vec!(f32, "", Float4, Float3, Float2, Float;
    float4, float3, float2, float;
    Float4Arg, Float3Arg, Float2Arg, FloatArg);

create_vec!(i32, "", Int4, Int3, Int2, Int;
    int4, int3, int2, int;
    Int4Arg, Int3Arg, Int2Arg, IntArg);

create_vec!(u32, "", UInt4, UInt3, UInt2, UInt;
    uint4, uint3, uint2, uint;
    UInt4Arg, UInt3Arg, UInt2Arg, UIntArg);

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
                    S::Min::from_t(self.into_args().as_arg())
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

        unsafe impl ArgParameter<TransparentArgs> for $matrix_type {}
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

macro_rules! create_matrix {
    ($t:ident, $($vec:ident, $($mat:ident, $mat_fn:ident, $trait:ident,)*;)*) => (
        matrix_subs!($($vec, $($mat,)*;)*);
        mat_ops!($($($mat,)*;)* | $($($mat,)*;)*);
        mat_ops!(!$($vec,)*;$($vec,)*;$($($mat,)*;)*);

        $(
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
