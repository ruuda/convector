//! A mod with utility functions.

use alloc::heap;
use std::mem;

/// Allocates a buffer for the specified number of elements, aligned to a cache
/// line.
pub fn cache_line_aligned_vec<T>(len: usize) -> Vec<T> {
    unsafe {
        let num_bytes = mem::size_of::<T>() * len;
        let cache_line_len = 64;
        let buffer = heap::allocate(num_bytes, cache_line_len);
        let ptr: *mut T = mem::transmute(buffer);
        Vec::from_raw_parts(ptr, 0, len)
    }
}
