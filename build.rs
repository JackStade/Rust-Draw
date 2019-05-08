extern crate gl_generator;

use gl_generator::{Api, Fallbacks, Profile, Registry, StructGenerator};
use std::env;
use std::fs::File;
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
        .write_bindings(StructGenerator, &mut file)
        .unwrap();
}
