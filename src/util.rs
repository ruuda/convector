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

/// Drops a vector that was constructed with `cache_line_aligned_vec` without
/// crashing on Windows.
pub fn drop_cache_line_aligned_vec<T>(mut v: Vec<T>) {
    unsafe {
        let ptr: *mut u8 = mem::transmute(v.as_mut_ptr());
        let num_bytes = v.capacity() * mem::size_of::<T>();

        // Prevent the destructor from freeing anything.
        mem::forget(v);

        // Then free manually.
        let cache_line_len = 64;
        heap::deallocate(ptr, num_bytes, cache_line_len);
    }
}

/// Transmutes the buffer of a vector into a buffer of elements with a different
/// type. The sizes of the types must be multiples of each other.
pub unsafe fn transmute_vec<T, U>(mut v: Vec<T>) -> Vec<U> {
    let cap_bytes = mem::size_of::<T>() * v.capacity();
    let len_bytes = mem::size_of::<T>() * v.len();

    let new_cap = cap_bytes / mem::size_of::<U>();
    let new_len = len_bytes / mem::size_of::<U>();

    assert_eq!(cap_bytes, new_cap * mem::size_of::<U>());
    assert_eq!(len_bytes, new_len * mem::size_of::<U>());

    let ptr: *mut U = mem::transmute(v.as_mut_ptr());

    // Prevent running the destructor of v, we are going to reuse its internals.
    mem::forget(v);

    Vec::from_raw_parts(ptr, new_len, new_cap)
}
