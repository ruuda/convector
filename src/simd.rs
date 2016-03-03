//! This module makes AVX slightly less painful to work with.
//!
//! Note: compile with `cargo rustc -- -C target-feature=sse,sse2,avx,avx2` to
//! use this to the full extent. (Otherwise it will not use AVX but two SSE
//! adds, for instance.)

use std::ops::Add;

#[repr(simd)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct OctaF32(f32, f32, f32, f32, f32, f32, f32, f32);

impl Add<OctaF32> for OctaF32 {
    type Output = OctaF32;

    #[inline(always)]
    fn add(self, other: OctaF32) -> OctaF32 {
        unsafe { simd_add(self, other) }
    }
}

extern "platform-intrinsic" {
    // Note: `_mm256_add_ps` is called `simd_add` for some reason.
    fn simd_add<T>(x: T, y: T) -> T;
    /*
    fn x86_mm256_addsub_ps(x: OctaF32, y: OctaF32) -> OctaF32;
    fn x86_mm256_dp_ps(x: OctaF32, y: OctaF32, z: i32) -> OctaF32;
    fn x86_mm256_hadd_ps(x: OctaF32, y: OctaF32) -> OctaF32;
    fn x86_mm256_hsub_ps(x: OctaF32, y: OctaF32) -> OctaF32;
    fn x86_mm256_max_ps(x: OctaF32, y: OctaF32) -> OctaF32;
    fn x86_mm256_min_ps(x: OctaF32, y: OctaF32) -> OctaF32;
    fn x86_mm256_movemask_ps(x: OctaF32) -> i32;
    fn x86_mm256_permutevar_ps(x: OctaF32, y: i32x8) -> OctaF32;
    fn x86_mm256_rcp_ps(x: OctaF32) -> OctaF32;
    fn x86_mm256_rsqrt_ps(x: OctaF32) -> OctaF32;
    fn x86_mm256_sqrt_ps(x: OctaF32) -> OctaF32;
    fn x86_mm256_testc_ps(x: OctaF32, y: OctaF32) -> i32;
    fn x86_mm256_testnzc_ps(x: OctaF32, y: OctaF32) -> i32;
    fn x86_mm256_testz_ps(x: OctaF32, y: OctaF32) -> i32;
    */
}

#[test]
fn octa_f32_add_ps() {
    let a = OctaF32(0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0);
    let b = OctaF32(5.0, 6.0, 7.0, 8.0, 0.0, 1.0, 2.0, 3.0);
    let c = OctaF32(5.0, 6.0, 7.0, 8.0, 1.0, 3.0, 5.0, 7.0);
    assert_eq!(a + b, c);
}
