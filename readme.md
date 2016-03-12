Infomagr
========

Interactive ray tracer. Copyright 2016 Ruud van Asseldonk.

Requirements
------------

Hardware: a CPU that supports the AVX and FMA instructions is required. In
practice this means Haswell or later.

Software: a nightly version of the [Rust programming language][rust] is
required.

Compiling
---------

Normally Cargo can be used to build Rust projects, but I use AVX and FMA
instructions and LLVM has to be told to use these. There is no way to configure
this in `Cargo.toml`, so a more low-level command has to be used to compile. I
stuffed all of this away in a makefile, so now you can just run:

 * `make` to build in release mode.
 * `make run` to build and run the release executable.
 * `make bench` to build and run all benchmarks in release mode.
 * `make test` to build and run all tests in debug mode.

On Windows you might have to execute the commands in the makefile manually, but
everything should still compile and run.

[rust]: https://rust-lang.org
