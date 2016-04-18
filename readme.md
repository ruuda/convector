Infomagr
========

Interactive ray tracer. Copyright 2016 Ruud van Asseldonk.

TL;DR
-----

Download the nightly version of [Rust](https://rust-lang.org),
then `cargo run --release`.

Requirements
------------

Hardware: a CPU that supports the AVX instructions is required. In practice this
means Sandy Bridge or later. FMA instructions can be taken advantage of too,
those are Haswell or later.

Software: a recent nightly version of the
[Rust programming language](https://rust-lang.org) is required.

Compiling and Running
---------------------

 * `cargo run --release` to build and run the release executable.
 * `cargo build --release` to build in release mode without running.
 * `cargo bench` to build and run all benchmarks in release mode.
 * `cargo test` to build and run all tests in debug mode.

If you do not want to use the FMA instructions, remove the `+fma` from the
codegen options in `.cargo/config`.

Controls
--------

 * Press `b` to toggle blending recent frames.
 * Press `d` to toggle debug view.
   The green channel shows the number of AABB intersections,
   the blue channel shows the number of triangle intersections.
 * Press `m` to toggle the median filter for noise reduction.
 * Press `q` to quit the application.
 * Press `r` to switch between realtime and accumulative rendering.
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
