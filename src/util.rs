//! A mod with utility functions.

use std::mem;

/// Allocates a buffer for the specified number of elements, aligned to a cache
/// line.
pub fn cache_line_aligned_vec<T>(len: usize) -> Vec<T> {
    // This is a terrible, terrible, hack. It was the only way I could get cache
    // line-aligned allocation without control over the destructor to work on
    // Windows. Do not try this at home.
    //
    // The deal is this: most allocators, when you ask for memory the size
    // bigger than a certain treshold (often 4 MiB), will not do any allocation
    // at all, but just call `mmap` or `VirtualAlloc` to allocate pages
    // from the OS directly. A page boundary happens to be cache-line aligned,
    // so if we can trick the allocator into doing huge allocations, then we get
    // cache line alignment for free.

    // The gcd of the size of `T` and the cache line size must be 1, such that
    // the length of a buffer of cache lines can be expressed in the number of
    // elements of `T`. Because the cache line size is a power of two, this
    // means that the size of `T` must also be a power of two.
    let size_of_line = 64;
    let size_of_t = mem::size_of::<T>();
    assert_eq!(0, size_of_t & (size_of_t - 1));

    // Actually, pages are not always 4096 bytes, but that is not really
    // relevant here as long as the allocator will fall back to huge allocation.
    let page_size = 4096;
    let num_bytes = size_of_t * len;
    let num_pages = (num_bytes + page_size - 1) / page_size;

    // Ask for at least 4 MiB to trigger a huge allocation. This is not so bad
    // as it sounds, because if we do not touch the pages, they will not be
    // backed by physical RAM.
    let num_pages = if num_pages > 1024 { num_pages } else { 1024 };
    let num_elems = num_pages * page_size / size_of_t;

    // Create the vector, and pray that the allocator triggers huge allocation
    // and gives us a page aligned buffer. Even if that is not the case, we
    // might be lucky and get something that is still aligned to a cache line.
    let vec = Vec::with_capacity(num_elems);

    // Crash the program if the buffer was not aligned to a cache line.
    unsafe {
        let ptr: usize = mem::transmute(vec.as_ptr());
        assert_eq!(0, ptr & (size_of_line - 1));
    }

    vec
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
