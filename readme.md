Infomagr
========

Interactive ray tracer. Copyright 2016 Ruud van Asseldonk.

TL;DR
-----

Download the nightly version of [Rust](https://rust-lang.org), then `make run`.

Requirements
------------

Hardware: a CPU that supports the AVX instructions is required. In practice this
means Sandy Bridge or later. FMA instructions can be taken advantage of too,
those are Haswell or later.

Software: a recent nightly version of the
[Rust programming language](https://rust-lang.org) is required.

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

If you do not want to use the FMA instructions, remove the `+fma` from the
codegen options in the makefile.

On Windows you might have to execute the commands in the makefile manually, but
everything should still compile and run.

Controls
--------

 * Press `d` to toggle debug view.
   The green channel shows the number of AABB intersections,
   the blue channel shows the number of triangle intersections.
 * Press `q` to quit the application.
 * Press `s` to print statistics to the console.
 * Press `t` to write a trace to trace.json.
   It can be opened with Chrome by going to chrome://tracing.

About the code
--------------

Many structs represent eight instances at once, for SIMD. In that case the name
has been prefixed with `M` (for “multi”). The single-instance struct types have
the prefix `S` instead (for “single”).

The most interesting stuff is in `src/triangle.rs`, `src/aabb.rs`, `src/bvh.rs`,
and `src/renderer.rs`. The scene can be customized in `src/main.rs`.
