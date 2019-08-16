extern crate gl_generator;

use gl_generator::generators;
use gl_generator::{Api, Fallbacks, Generator, Profile, Registry};
use std::env;
use std::fs::File;
use std::io;
use std::path::Path;

#[cfg(not(feature = "opengl41"))]
const GL_VERSION: u8 = 0;

#[cfg(all(not(feature = "opengl42"), feature = "opengl41"))]
const GL_VERSION: u8 = 1;

#[cfg(all(not(feature = "opengl43"), feature = "opengl42"))]
const GL_VERSION: u8 = 2;

#[cfg(all(not(feature = "opengl44"), feature = "opengl43"))]
const GL_VERSION: u8 = 3;

#[cfg(all(not(feature = "opengl45"), feature = "opengl44"))]
const GL_VERSION: u8 = 4;

#[cfg(all(not(feature = "opengl46"), feature = "opengl45"))]
const GL_VERSION: u8 = 5;

#[cfg(feature = "opengl46")]
const GL_VERSION: u8 = 6;

fn main() {
    let dest = env::var("OUT_DIR").unwrap();
    let mut gl_file = File::create(&Path::new(&dest).join("bindings.rs")).unwrap();
    let mut blend_file = File::create(&Path::new(&dest).join("blend.rs")).unwrap();

    write_blend_gen(&mut blend_file).unwrap();

    Registry::new(Api::Gl, (4, GL_VERSION), Profile::Core, Fallbacks::All, [])
        .write_bindings(ArrayStructGenerator, &mut gl_file)
        .unwrap();
}

fn write_blend_gen<W: io::Write>(w: &mut W) -> io::Result<()> {
    write!(
        w,
        "#[derive(Clone, Copy)]\npub struct One;\n#[derive(Clone, Copy)]\npub struct Zero;\n\n"
    )?;
    writeln!(w, "pub const ZERO: Zero = Zero;")?;
    writeln!(w, "pub const ONE: One = One;")?;

    let srcdst = ["Src", "Dst"];
    let alphacolor = ["Color", "Alpha"];

    let mut base_types = Vec::with_capacity(4);
    let mut gl_enums = Vec::with_capacity(4);

    write!(w, "#[derive(Clone, Copy)]\npub struct SrcTimesDstColor;\n")?;
    write!(w, "#[derive(Clone, Copy)]\npub struct SrcTimesDstAlpha;\n")?;

    for p in &srcdst {
        for c in &alphacolor {
            let s = format!("{}{}", p, c);
            write!(w, "#[derive(Clone, Copy)]\npub struct {};\n", s)?;
            write!(
                w,
                "#[derive(Clone, Copy)]\npub struct {}WithParam{{p: u32}}\n",
                s
            )?;
            write!(w, "#[derive(Clone, Copy)]\npub struct OneMinus{};\n", s)?;
            write!(
                w,
                "pub const {up}_{uc}: {s} = {s};\n",
                up = p.to_uppercase(),
                uc = c.to_uppercase(),
                s = s,
            )?;
            let om = format!("OneMinus{}", s);
            write_op(w, "One", &s, &om, &om, "Sub")?;
            base_types.push(s);
            gl_enums.push(format!("{}_{}", p.to_uppercase(), c.to_uppercase()));
        }
    }

    for t1 in 0..4 {
        for t2 in 0..4 {
            let (rt, result) = if t1 == t2 {
                let r = format!("{}WithParam", base_types[t1]);
                (r.clone(), format!("{}{{p: gl::{}}}", r, gl_enums[t1]))
            } else if t1 % 2 == t2 % 2 {
                let r = format!("SrcTimesDst{}", alphacolor[t1 % 2]);
                (r.clone(), r)
            } else if t1 % 2 == 0 {
                let r = format!("{}WithParam", base_types[t1]);
                (r.clone(), format!("{}{{p: gl::{}}}", r, gl_enums[t2]))
            } else {
                // t1 % 2 == 1, t2 % 2 == 0
                let r = format!("{}WithParam", base_types[t2]);
                (r.clone(), format!("{}{{p: gl::{}}}", r, gl_enums[t1]))
            };
            let om = format!("OneMinus{}", base_types[t2]);
            let om_rt = format!("{}WithParam", base_types[t1]);
            let om_result = format!("{}{{p: gl::ONE_MINUS_{}}}", om_rt, gl_enums[t2]);
            write_op(w, &base_types[t1], &base_types[t2], &result, &rt, "Mul")?;
            write_op(w, &base_types[t1], &om, &om_result, &om_rt, "Mul")?;
            write_op(w, &om, &base_types[t1], &om_result, &om_rt, "Mul")?;
        }
        let result = format!("{}WithParam", base_types[t1]);
        let one_out = format!("{}{{p: gl::ONE}}", result);
        let zero_out = format!("{}{{p: gl::ZERO}}", result);
        write_op(w, &base_types[t1], "One", &one_out, &result, "Mul")?;
        write_op(w, "One", &base_types[t1], &one_out, &result, "Mul")?;
        write_op(w, &base_types[t1], "Zero", &zero_out, &result, "Mul")?;
        write_op(w, "Zero", &base_types[t1], &zero_out, &result, "Mul")?;
    }
    for c in &alphacolor {
        let out_ty = format!("{}Func", c);
        let ambig = format!("SrcTimesDst{}", c);
        let ambig_src = format!("gl::DST_{}", c.to_uppercase());
        let ambig_dst = format!("gl::SRC_{}", c.to_uppercase());

        write_op(
            w,
            &ambig,
            &ambig,
            &format!(
                "{}{{eqn: gl::FUNC_ADD, src: {}, dst: {}}}",
                out_ty, ambig_src, ambig_dst
            ),
            &out_ty,
            "Add",
        );
        write_op(
            w,
            &ambig,
            &ambig,
            // we are subtracting something from itself
            &format!(
                "{}{{eqn: gl::FUNC_ADD, src: gl::ZERO, dst: gl::ZERO}}",
                out_ty
            ),
            &out_ty,
            "Sub",
        );

        let src = format!("Src{}WithParam", c);
        let dst = format!("Dst{}WithParam", c);
        let src_val = "src.p";
        let dst_val = "dst.p";

        write_linear_ops(w, &out_ty, &src, &dst, src_val, dst_val)?;
        write_linear_ops(w, &out_ty, &src, &ambig, src_val, &ambig_dst)?;
        write_linear_ops(w, &out_ty, &ambig, &dst, &ambig_src, dst_val)?;
    }

    Ok(())
}

fn write_linear_ops<W: io::Write>(
    w: &mut W,
    out_ty: &str,
    src: &str,
    dst: &str,
    src_val: &str,
    dst_val: &str,
) -> io::Result<()> {
    let out_param_add = format!(
        "{}{{eqn: gl::FUNC_ADD, src: {}, dst: {}}}",
        out_ty, src_val, dst_val
    );
    let out_param_sub = format!(
        "{}{{eqn: gl::FUNC_SUBTRACT, src: {}, dst: {}}}",
        out_ty, src_val, dst_val
    );
    let out_param_rev_sub = format!(
        "{}{{eqn: gl::FUNC_REVERSE_SUBTRACT, src: {}, dst: {}}}",
        out_ty, src_val, dst_val
    );
    let f_vars = "let src = self;let dst = rhs;";
    let r_vars = "let src = rhs;let dst = self;";
    write_op(
        w,
        src,
        dst,
        &format!("{}{}", f_vars, out_param_add),
        out_ty,
        "Add",
    )?;
    write_op(
        w,
        dst,
        src,
        &format!("{}{}", r_vars, out_param_add),
        out_ty,
        "Add",
    )?;
    write_op(
        w,
        src,
        dst,
        &format!("{}{}", f_vars, out_param_sub),
        out_ty,
        "Sub",
    )?;
    write_op(
        w,
        dst,
        src,
        &format!("{}{}", r_vars, out_param_rev_sub),
        out_ty,
        "Sub",
    )
}

fn write_op<W: io::Write>(
    w: &mut W,
    t1: &str,
    t2: &str,
    out: &str,
    out_ty: &str,
    op: &str,
) -> io::Result<()> {
    writeln!(w, "impl std::ops::{}<{}> for {} {{", op, t2, t1)?;
    writeln!(w, "type Output = {};", out_ty)?;
    write!(
        w,
        "fn {}(self: {}, rhs: {}) -> {} {{{}}}",
        op.to_lowercase(),
        t1,
        t2,
        out_ty,
        out,
    )?;
    writeln!(w, "}}")
}

// note: the following code is derived from
// https://github.com/brendanzab/gl-rs/blob/master/gl_generator/generators/struct_gen.rs
// which is licenced under the apache license 2.0.

#[allow(missing_copy_implementations)]
struct ArrayStructGenerator;

impl Generator for ArrayStructGenerator {
    fn write<W>(&self, registry: &Registry, dest: &mut W) -> io::Result<()>
    where
        W: io::Write,
    {
        write_header(dest)?;
        write_type_aliases(registry, dest)?;
        write_enums(registry, dest)?;
        write_fnptr_struct_def(dest)?;
        write_panicking_fns(registry, dest)?;
        write_struct(registry, dest)?;
        write_impl(registry, dest)?;
        Ok(())
    }
}

/// Creates a `__gl_imports` module which contains all the external symbols that we need for the
///  bindings.
fn write_header<W>(dest: &mut W) -> io::Result<()>
where
    W: io::Write,
{
    writeln!(
        dest,
        r#"
        mod __gl_imports {{
            pub use std::mem;
            pub use std::marker::Send;
            pub use std::os::raw;
        }}
    "#
    )
}

/// Creates a `types` module which contains all the type aliases.
///
/// See also `generators::gen_types`.
fn write_type_aliases<W>(registry: &Registry, dest: &mut W) -> io::Result<()>
where
    W: io::Write,
{
    writeln!(
        dest,
        r#"
        pub mod types {{
            #![allow(non_camel_case_types, non_snake_case, dead_code, missing_copy_implementations)]
    "#
    )?;

    generators::gen_types(registry.api, dest)?;

    writeln!(dest, "}}")
}

/// Creates all the `<enum>` elements at the root of the bindings.
fn write_enums<W>(registry: &Registry, dest: &mut W) -> io::Result<()>
where
    W: io::Write,
{
    for enm in &registry.enums {
        generators::gen_enum_item(enm, "types::", dest)?;
    }

    Ok(())
}

/// Creates a `FnPtr` structure which contains the store for a single binding.
///
/// This differs from the implementation of StructGenerator in gl_generators in that
/// it ommits the is_loaded field of a function pointer to save space.
fn write_fnptr_struct_def<W>(dest: &mut W) -> io::Result<()>
where
    W: io::Write,
{
    writeln!(
        dest,
        "
        #[allow(dead_code)]
        #[derive(Clone, Copy)]
        pub struct FnPtr {{
            /// The function pointer that will be used when calling the function.
            f: *const __gl_imports::raw::c_void,
        }}
        impl FnPtr {{
            /// Creates a `FnPtr` from a load attempt.
            fn new(ptr: *const __gl_imports::raw::c_void) -> FnPtr {{
                if ptr.is_null() {{
                    FnPtr {{
                        f: missing_fn_panic as *const __gl_imports::raw::c_void,
                    }}
                }} else {{
                    FnPtr {{ f: ptr, }}
                }}
            }}
        }}
    "
    )
}

fn write_panicking_fns<W>(registry: &Registry, dest: &mut W) -> io::Result<()>
where
    W: io::Write,
{
    writeln!(
        dest,
        "#[inline(never)]
        fn missing_fn_panic() -> ! {{
            panic!(\"{api} function was not loaded\")
        }}",
        api = registry.api
    )
}

/// Creates a structure which stores all the `FnPtr` of the bindings.
///
/// The name of the struct corresponds to the namespace.
///
/// This is different from the StructGenerator in gl_generators in that the struct
/// generated stores one long array of function pointers. It also creates another struct
/// that stores the indices of each named function, and a constant of this struct.
fn write_struct<W>(registry: &Registry, dest: &mut W) -> io::Result<()>
where
    W: io::Write,
{
    writeln!(
        dest,
        "
        #[derive(Clone)]
        pub struct {api} {{",
        api = generators::gen_struct_name(registry.api)
    )?;

    writeln!(
        dest,
        "    pub fn_ptrs: [FnPtr; {len}],",
        len = registry.cmds.len()
    )?;

    writeln!(dest, "_priv: ()")?;

    writeln!(dest, "}}")?;

    // we know create a second struct which stores the index of each pointer
    // in the array
    writeln!(
        dest,
        "
        #[allow(non_camel_case_types, non_snake_case, dead_code)]
        #[derive(Clone)]
        pub struct {api}Indices {{",
        api = generators::gen_struct_name(registry.api)
    )?;

    for cmd in &registry.cmds {
        if let Some(v) = registry.aliases.get(&cmd.proto.ident) {
            writeln!(dest, "/// Fallbacks: {}", v.join(", "))?;
        }
        writeln!(dest, "pub {name}: usize,", name = cmd.proto.ident)?;
    }

    writeln!(dest, "}}")?;

    writeln!(
        dest,
        "pub const INDICES: {api}Indices = {api}Indices {{",
        api = generators::gen_struct_name(registry.api)
    )?;

    for (i, cmd) in registry.cmds.iter().enumerate() {
        writeln!(dest, "{name}: {index},", name = cmd.proto.ident, index = i)?;
    }

    writeln!(dest, "}};")
}

/// Creates the `impl` of the structure created by `write_struct`.
///
/// This is different from the implementation in gl_generators because
/// the struct is defined differently.
fn write_impl<W>(registry: &Registry, dest: &mut W) -> io::Result<()>
where
    W: io::Write,
{
    writeln!(dest,
                  "impl {api} {{
            /// Load each OpenGL symbol using a custom load function. This allows for the
            /// use of functions like `glfwGetProcAddress` or `SDL_GL_GetProcAddress`.
            ///
            /// ~~~ignore
            /// let gl = Gl::load_with(|s| glfw.get_proc_address(s));
            /// ~~~
            #[allow(dead_code, unused_variables)]
            pub fn load_with<F>(mut loadfn: F) -> {api} where F: FnMut(&'static str) -> *const __gl_imports::raw::c_void {{
                #[inline(never)]
                fn do_metaloadfn(loadfn: &mut FnMut(&'static str) -> *const __gl_imports::raw::c_void,
                                 symbol: &'static str,
                                 symbols: &[&'static str])
                                 -> *const __gl_imports::raw::c_void {{
                    let mut ptr = loadfn(symbol);
                    if ptr.is_null() {{
                        for &sym in symbols {{
                            ptr = loadfn(sym);
                            if !ptr.is_null() {{ break; }}
                        }}
                    }}
                    ptr
                }}
                let mut metaloadfn = |symbol: &'static str, symbols: &[&'static str]| {{
                    do_metaloadfn(&mut loadfn, symbol, symbols)
                }};
                {api} {{
                	fn_ptrs: [",
                  api = generators::gen_struct_name(registry.api))?;

    for cmd in &registry.cmds {
        writeln!(
            dest,
            "FnPtr::new(metaloadfn(\"{symbol}\", &[{fallbacks}])),",
            symbol = generators::gen_symbol_name(registry.api, &cmd.proto.ident),
            fallbacks = match registry.aliases.get(&cmd.proto.ident) {
                Some(fbs) => fbs
                    .iter()
                    .map(|name| format!("\"{}\"", generators::gen_symbol_name(registry.api, &name)))
                    .collect::<Vec<_>>()
                    .join(", "),
                None => format!(""),
            },
        )?;
    }

    writeln!(dest, "],")?;

    writeln!(dest, "_priv: ()")?;

    writeln!(
        dest,
        "}}
        }}"
    )?;

    for cmd in &registry.cmds {
        writeln!(dest,
            "#[allow(non_snake_case, unused_variables, dead_code)]
            #[inline] pub unsafe fn {name}(&self, {params}) -> {return_suffix} {{ \
                __gl_imports::mem::transmute::<_, extern \"system\" fn({typed_params}) -> {return_suffix}>\
                    (self.fn_ptrs[INDICES.{name}].f)({idents}) \
            }}",
            name = cmd.proto.ident,
            params = generators::gen_parameters(cmd, true, true).join(", "),
            typed_params = generators::gen_parameters(cmd, false, true).join(", "),
            return_suffix = cmd.proto.ty,
            idents = generators::gen_parameters(cmd, true, false).join(", "),
        )?;
    }

    writeln!(
        dest,
        "}}
        unsafe impl __gl_imports::Send for {api} {{}}",
        api = generators::gen_struct_name(registry.api)
    )
}
