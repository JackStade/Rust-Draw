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
    let mut file = File::create(&Path::new(&dest).join("bindings.rs")).unwrap();

    Registry::new(Api::Gl, (4, GL_VERSION), Profile::Core, Fallbacks::All, [])
        .write_bindings(ArrayStructGenerator, &mut file)
        .unwrap();
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

/// Creates a `panicking` module which contains one function per GL command.
///
/// These functions are the mocks that are called if the real function could not be loaded.
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
