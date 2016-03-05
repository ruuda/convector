//! This module makes AVX slightly less painful to work with.
//!
//! Note: compile with `cargo rustc -- -C target-feature=sse,sse2,avx,avx2` to
//! use this to the full extent. (Otherwise it will not use AVX but two SSE
//! adds, for instance.)

use std::ops::{Add, Mul, Sub};

#[repr(simd)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct OctaF32(pub f32, pub f32, pub f32, pub f32, pub f32, pub f32, pub f32, pub f32);

impl OctaF32 {
    pub fn zero() -> OctaF32 {
        OctaF32(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    }

    /// Builds a octafloat by applying the function to the numbers 0..7.
    pub fn generate<F: FnMut(usize) -> f32>(mut f: F) -> OctaF32 {
        OctaF32(f(0), f(1), f(2), f(3), f(4), f(5), f(6), f(7))
    }
}

impl Add<OctaF32> for OctaF32 {
    type Output = OctaF32;

    #[inline(always)]
    fn add(self, other: OctaF32) -> OctaF32 {
        unsafe { simd_add(self, other) }
    }
}

impl Sub<OctaF32> for OctaF32 {
    type Output = OctaF32;

    #[inline(always)]
    fn sub(self, other: OctaF32) -> OctaF32 {
        unsafe { simd_sub(self, other) }
    }
}

impl Mul<OctaF32> for OctaF32 {
    type Output = OctaF32;

    #[inline(always)]
    fn mul(self, other: OctaF32) -> OctaF32 {
        unsafe { simd_mul(self, other) }
    }
}

extern "platform-intrinsic" {
    // This is `_mm256_add_ps` when compiled for AVX.
    fn simd_add<T>(x: T, y: T) -> T;

    // This is `_mm256_sub_ps` when compiled for AVX.
    fn simd_sub<T>(x: T, y: T) -> T;

    // This is `_mm256_mul_ps` when compiled for AVX.
    fn simd_mul<T>(x: T, y: T) -> T;

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
