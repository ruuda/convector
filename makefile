target/release/infomagr: src/*.rs
	cargo rustc --release -- -C target-feature=+sse,+sse2,+sse3,+ssse3,+sse4.1,+sse4.2,+avx,+avx2
