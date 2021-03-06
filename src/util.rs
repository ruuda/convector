// Convector -- An interactive CPU path tracer
// Copyright 2016 Ruud van Asseldonk

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License version 3. A copy
// of the License is available in the root of the repository.

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

/// Transmutes an immutable slice into a mutable slice.
#[allow(mutable_transmutes)]
pub unsafe fn make_mutable<T>(x: &[T]) -> &mut [T] {
    // UnsafeCell is a real pain to deal with; after 15 minutes I still did not
    // manage to write something that compiles. Just transmute the mutability
    // in.
    mem::transmute(x)
}

/// Builds a fixed-size slice by calling f for every index.
pub fn generate_slice8<T, F>(mut f: F) -> [T; 8]
    where F: FnMut(usize) -> T
{
    [f(0), f(1), f(2), f(3), f(4), f(5), f(6), f(7)]
}
