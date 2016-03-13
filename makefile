# Makefile because Cargo does not support target features at the moment.
# Usage:
#
#  * `make`: build in release mode.
#  * `make run`: build and run the release executable.
#  * `make bench`: build and run all benchmarks in release mode.
#  * `make test`: build and run all tests in debug mode.

codegen_opts = -C target-feature=+avx,+fma

release: target/release/infomagr

bench: target/release/infomagr_bench
	target/release/infomagr_bench --bench

test: target/debug/infomagr_test
	target/debug/infomagr_test --test

run: target/release/infomagr
	target/release/infomagr

target/release/infomagr: Cargo.toml src/*.rs
	cargo rustc --release -- $(codegen_opts)

target/release/infomagr_bench: Cargo.toml src/*.rs
	cargo rustc --release -- --test -o target/release/infomagr_bench $(codegen_opts)

target/debug/infomagr_test: Cargo.toml src/*.rs
	cargo rustc -- --test -o target/debug/infomagr_test $(codegen_opts)
