//! A mod with utility functions.

use alloc::heap;
use std::mem;

/// A structure that is 64 bytes in size (one cache line).
///
/// Due to the `repr(simd)` attribute, it is forced to be aligned to its size.
#[repr(simd)]
struct CacheLineSized {
    _bytes_00_07: u64,
    _bytes_08_15: u64,
    _bytes_16_23: u64,
    _bytes_24_31: u64,
    _bytes_32_39: u64,
    _bytes_40_47: u64,
    _bytes_48_55: u64,
    _bytes_56_63: u64,
}

/// Allocates a buffer for the specified number of elements, aligned to a cache
/// line.
pub fn cache_line_aligned_vec<T>(len: usize) -> Vec<T> {
    let size_of_t = mem::size_of::<T>();
    let size_of_line = 64;

    // The size of `T` must be a power of two to make sure that regardless of
    // `len`, rounding up the number of bytes requested to the cache line size
    // is a multiple of the size of `T`.
    assert_eq!(0, size_of_t & (size_of_t - 1));

    // Due to the allocation in multiples of the cache line size, round up the
    // number of bytes to a multiple of the cache line.
    let num_lines = (size_of_t * len + size_of_line - 1) / size_of_line;
    let capacity = (num_lines * size_of_line) / size_of_t;

    // Construct a new vector of cache lines. This will allocate the buffer with
    // the right alignment and size, but more importantly, the buffer will be
    // allocated by `Vec` itself. Constructructing a `Vec` from raw parts where
    // the buffer was allocated with `heap::allocate` does not work, this causes
    // a subsequent drop of the vector to crash on Windows.
    let mut vec: Vec<CacheLineSized> = Vec::with_capacity(num_lines);

    // Tear apart the vec and turn it into a new one with the same buffer, but
    // capacity updated for the size of `T`. The `mem::forget` prevents running
    // the destructor of `vec`, which would otherwise free the buffer when the
    // vector goes out of scope.
    unsafe {
        let buffer = vec.as_mut_ptr();
        mem::forget(vec);
        Vec::from_raw_parts(mem::transmute(buffer), 0, capacity)
    }
}

#[test]
fn cache_line_struct_is_64_bytes() {
    let size_of_line_struct = mem::size_of::<CacheLineSized>();
    assert_eq!(64, size_of_line_struct);
}

#[test]
fn free_cache_line_aligned_vec() {
    let mut vec: Vec<u16> = cache_line_aligned_vec(4096);
    vec.push(17);
    vec.push(19);
    // At this point, `vec` is dropped, and that should not crash. A previous
    // version did crash on Windows in `heap::deallocate` when the vector was
    // dropped.
}
