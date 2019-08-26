# Rust Draw

Rust-Draw is a Rust library for graphics, built on OpenGL. It started out as an attempt to make something similar to the Processing library for java, which provides an extremely quick and easy to use way to prototype and experiment with graphics. Processing uses legacy opengl and is therefore not very performant, but a Rust library should be able to acheive some of the same things without sacrificing too much in terms of performance.

The library is very experimental, heavily using and abusing the type system to create extensive compile time checking. The hope is to leverage the versatility of opengl in an elegant and less error-prone way. This makes documenting and understanding the library a little bit of a nightmare, but code written using is almost like psuedo code in its simplicity. 

As of right now the library is a work in progress. While it is potentially quite powerful even in its current state, there is a lot of documentation that needs to be added at some point, as the API is not at all transparent. This library will probably never be as easy to use a Processing.