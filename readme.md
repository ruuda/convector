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

I use fancy hardware instructions, but LLVM has to be told to use these
explicitly. When using Cargo normally, there is no way (yet) to do this, so
`cargo build` and `cargo bench` do not produce optimal code. Instead, one has to
use the more low-level `cargo rustc` command.

To compile and run the regular binary:

    $ FEATURES="+sse,+sse2,+sse3,+ssse3,+sse4.1,+sse4.2,+sse4a,+avx,+avx2,+fma"
    $ cargo rustc --release -- -C target-feature=$FEATURES
    $ target/release/infomagr

To benchmark and test:

    $ cargo rustc --release -- --test -C target-feature=$FEATURES
    $ target/release/infomagr --bench
    $ target/release/infomagr --test

Note that there is also `target-cpu=native`, but on my machine this produces
code containing illegal instructions.

[rust]: https://rust-lang.org
