//! This module makes AVX slightly less painful to work with.
//!
//! Note: compile with `cargo rustc -- -C target-feature=sse,sse2,avx,avx2` to
//! use this to the full extent. (Otherwise it will not use AVX but two SSE
//! adds, for instance.)

use std::ops::{Add, BitAnd, BitOr, Div, Mul, Sub};

#[cfg(test)]
use {bench, test};

#[repr(simd)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Mf32(pub f32, pub f32, pub f32, pub f32, pub f32, pub f32, pub f32, pub f32);


#[repr(simd)]
#[derive(Copy, Clone)]
pub struct Mi32(pub i32, pub i32, pub i32, pub i32, pub i32, pub i32, pub i32, pub i32);

pub type Mask = Mf32;

impl Mf32 {
    pub fn zero() -> Mf32 {
        Mf32::broadcast(0.0)
    }

    pub fn one() -> Mf32 {
        Mf32::broadcast(1.0)
    }

    /// A small value that can be used tolerate numerical imprecision.
    pub fn epsilon() -> Mf32 {
        Mf32::broadcast(1.0e-3)
    }

    /// Builds an mf32 by applying the function to the numbers 0..7.
    pub fn generate<F>(mut f: F) -> Mf32 where F: FnMut(usize) -> f32 {
        Mf32(f(0), f(1), f(2), f(3), f(4), f(5), f(6), f(7))
    }

    #[inline(always)]
    pub fn broadcast(x: f32) -> Mf32 {
        Mf32(x, x, x, x, x, x, x, x)
    }

    #[inline(always)]
    pub fn mul_add(self, factor: Mf32, term: Mf32) -> Mf32 {
        unsafe { x86_mm256_fmadd_ps(self, factor, term) }
    }

    #[inline(always)]
    pub fn mul_sub(self, factor: Mf32, term: Mf32) -> Mf32 {
        unsafe { x86_mm256_fmsub_ps(self, factor, term) }
    }

    /// Approximates 1 / self.
    #[inline(always)]
    pub fn recip(self) -> Mf32 {
        unsafe { x86_mm256_rcp_ps(self) }
    }

    /// Computes self / denom with best precision.
    #[inline(always)]
    pub fn div(self, denom: Mf32) -> Mf32 {
        unsafe { simd_div(self, denom) }
    }

    /// Approximates the reciprocal square root.
    #[inline(always)]
    pub fn rsqrt(self) -> Mf32 {
        unsafe { x86_mm256_rsqrt_ps(self) }
    }

    #[inline(always)]
    pub fn sqrt(self) -> Mf32 {
        unsafe { x86_mm256_sqrt_ps(self) }
    }

    #[inline(always)]
    pub fn max(self, other: Mf32) -> Mf32 {
        unsafe { x86_mm256_max_ps(self, other) }
    }

    #[inline(always)]
    pub fn min(self, other: Mf32) -> Mf32 {
        unsafe { x86_mm256_min_ps(self, other) }
    }

    #[inline(always)]
    pub fn geq(self, other: Mf32) -> Mask {
        // Operation 21 is a not less than comparison, unordered,
        // non-signalling.
        unsafe { x86_mm256_cmp_ps(self, other, 21) }
    }

    /// Returns whether any of the values is negative.
    #[inline(always)]
    pub fn all_sign_bits_positive(self) -> bool {
        // The mask contains the sign bits packed into an i32. If the mask is
        // zero, then all sign bits were 0, so all numbers were positive.
        unsafe { x86_mm256_movemask_ps(self) == 0 }
    }

    /// Returns whether all of the values not masked out are negative.
    ///
    /// Note that a value is negative if its sign bit is set.
    #[inline(always)]
    pub fn all_sign_bits_negative_masked(self, mask: Mask) -> bool {
        use std::mem::transmute;
        // The testc intrinsic computes `(not self) and mask`, and then returns
        // 1 if all resulting sign bits are 0, or 0 otherwise. If a value is
        // negative, the sign bit will be 1, so `not self` will have sign bit 0.
        // Mask out the values that we are not interested in, then testc returns
        // 1 if there were no positive values. Also, we know that the returned
        // value is either 0 or 1, so there is no need for a comparison, just
        // interpret the least significant byte of the result as a boolean.
        unsafe { transmute(x86_mm256_testc_ps(self, mask) as i8) }
    }

    /// Returns whether any of the values not masked out is positive.
    #[inline(always)]
    pub fn any_sign_bit_positive_masked(self, mask: Mask) -> bool {
        !self.all_sign_bits_negative_masked(mask)
    }

    /// Picks the component of self if the sign bit in the mask is 0,
    /// otherwise picks the component in other.
    #[inline(always)]
    pub fn pick(self, other: Mf32, mask: Mask) -> Mf32 {
        unsafe { x86_mm256_blendv_ps(self, other, mask) }
    }

    /// Converts floating-point numbers to 32-bit signed integers.
    #[inline(always)]
    pub fn into_mi32(self) -> Mi32 {
        unsafe { x86_mm256_cvtps_epi32(self) }
    }
}

impl Mi32 {
    pub fn zero() -> Mi32 {
        Mi32(0, 0, 0, 0, 0, 0, 0, 0)
    }

    /// Applies the function componentwise.
    pub fn map<F>(self, mut f: F) -> Mi32 where F: FnMut(i32) -> i32 {
        Mi32(f(self.0), f(self.1), f(self.2), f(self.2),
             f(self.4), f(self.5), f(self.6), f(self.7))
    }

    #[inline(always)]
    pub fn broadcast(x: i32) -> Mi32 {
        Mi32(x, x, x, x, x, x, x, x)
    }
}

impl Add<Mf32> for Mf32 {
    type Output = Mf32;

    #[inline(always)]
    fn add(self, other: Mf32) -> Mf32 {
        unsafe { simd_add(self, other) }
    }
}

impl BitAnd<Mask> for Mask {
    type Output = Mask;

    #[inline(always)]
    fn bitand(self, other: Mask) -> Mask {
        use std::mem::transmute;
        unsafe {
            let a: Mi32 = transmute(self);
            let b: Mi32 = transmute(other);
            let a_and_b = simd_and(a, b);
            transmute(a_and_b)
        }
    }
}

impl BitOr<Mi32> for Mi32 {
    type Output = Mi32;

    #[inline(always)]
    fn bitor(self, other: Mi32) -> Mi32 {
        unsafe { simd_or(self, other) }
    }
}

impl BitOr<Mask> for Mask {
    type Output = Mask;

    #[inline(always)]
    fn bitor(self, other: Mask) -> Mask {
        use std::mem::transmute;
        unsafe {
            let a: Mi32 = transmute(self);
            let b: Mi32 = transmute(other);
            let a_or_b = simd_or(a, b);
            transmute(a_or_b)
        }
    }
}

impl Div<Mf32> for Mf32 {
    type Output = Mf32;

    #[inline(always)]
    fn div(self, denom: Mf32) -> Mf32 {
        // Benchmarks show that _mm256_div_ps is as fast as a _mm256_rcp_ps
        // followed by a _mm256_mul_ps, so we might as well use the div.
        unsafe { simd_div(self, denom) }
    }
}

impl Sub<Mf32> for Mf32 {
    type Output = Mf32;

    #[inline(always)]
    fn sub(self, other: Mf32) -> Mf32 {
        unsafe { simd_sub(self, other) }
    }
}

impl Mul<Mf32> for Mf32 {
    type Output = Mf32;

    #[inline(always)]
    fn mul(self, other: Mf32) -> Mf32 {
        unsafe { simd_mul(self, other) }
    }
}

extern "platform-intrinsic" {
    // This is `_mm256_add_ps` when compiled for AVX.
    fn simd_add<T>(x: T, y: T) -> T;

    // This is `_mm256_and_ps` when compiled for AVX.
    fn simd_and<T>(x: T, y: T) -> T;

    // This is `_mm256_div_ps` when compiled for AVX.
    fn simd_div<T>(x: T, y: T) -> T;

    // This is `_mm256_sub_ps` when compiled for AVX.
    fn simd_sub<T>(x: T, y: T) -> T;

    // This is `_mm256_mul_ps` when compiled for AVX.
    fn simd_mul<T>(x: T, y: T) -> T;

    // This is `_mm256_or_ps` when compiled for AVX.
    fn simd_or<T>(x: T, y: T) -> T;

    fn x86_mm256_blendv_ps(x: Mf32, y: Mf32, mask: Mask) -> Mf32;
    fn x86_mm256_cmp_ps(x: Mf32, y: Mf32, op: i8) -> Mask;
    fn x86_mm256_cvtps_epi32(x: Mf32) -> Mi32;
    fn x86_mm256_max_ps(x: Mf32, y: Mf32) -> Mf32;
    fn x86_mm256_min_ps(x: Mf32, y: Mf32) -> Mf32;
    fn x86_mm256_movemask_ps(x: Mf32) -> i32;
    fn x86_mm256_rcp_ps(x: Mf32) -> Mf32;
    fn x86_mm256_rsqrt_ps(x: Mf32) -> Mf32;
    fn x86_mm256_sqrt_ps(x: Mf32) -> Mf32;
    fn x86_mm256_testc_ps(x: Mf32, y: Mf32) -> i32;
}

#[cfg(target_feature = "fma")]
extern "platform-intrinsic" {
    fn x86_mm256_fmadd_ps(x: Mf32, y: Mf32, z: Mf32) -> Mf32;
    fn x86_mm256_fmsub_ps(x: Mf32, y: Mf32, z: Mf32) -> Mf32;
}

// If the FMA instructions are not enabled, fall back to a separate mul and add
// or sub. These still use the AVX intrinsics.
#[cfg(not(target_feature = "fma"))]
unsafe fn x86_mm256_fmadd_ps(x: Mf32, y: Mf32, z: Mf32) -> Mf32 {
    x * y + z
}

#[cfg(not(target_feature = "fma"))]
unsafe fn x86_mm256_fmsub_ps(x: Mf32, y: Mf32, z: Mf32) -> Mf32 {
    x * y - z
}

#[test]
fn mf32_add_ps() {
    let a = Mf32(0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0);
    let b = Mf32(5.0, 6.0, 7.0, 8.0, 0.0, 1.0, 2.0, 3.0);
    let c = Mf32(5.0, 6.0, 7.0, 8.0, 1.0, 3.0, 5.0, 7.0);
    assert_eq!(a + b, c);
}

#[test]
fn mf32_fmadd_ps() {
    let a = Mf32(0.0,  1.0, 0.0,  2.0, 1.0, 2.0,  3.0,  4.0);
    let b = Mf32(5.0,  6.0, 7.0,  8.0, 0.0, 1.0,  2.0,  3.0);
    let c = Mf32(5.0,  6.0, 7.0,  8.0, 1.0, 3.0,  5.0,  7.0);
    let d = Mf32(5.0, 12.0, 7.0, 24.0, 1.0, 5.0, 11.0, 19.0);
    assert_eq!(a.mul_add(b, c), d);
}

#[test]
fn mf32_fmsub_ps() {
    let a = Mf32( 0.0, 1.0,  0.0, 2.0,  1.0,  2.0, 3.0, 4.0);
    let b = Mf32( 5.0, 6.0,  7.0, 8.0,  0.0,  1.0, 2.0, 3.0);
    let c = Mf32( 5.0, 6.0,  7.0, 8.0,  1.0,  3.0, 5.0, 7.0);
    let d = Mf32(-5.0, 0.0, -7.0, 8.0, -1.0, -1.0, 1.0, 5.0);
    assert_eq!(a.mul_sub(b, c), d);
}

#[test]
fn mf32_broadcast_ps() {
    let a = Mf32::broadcast(7.0);
    let b = Mf32(7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0);
    assert_eq!(a, b);
}

#[test]
fn mf32_any_sign_bit_positive_masked() {
    use std::mem::transmute;
    let a = Mf32(-2.0, -1.0, -0.0, 0.0, 1.0, 2.0, 3.0, 4.0);
    let f1: f32 = unsafe { transmute(0xffffffff_u32) };
    let f0: f32 = 0.0;

    assert!(a.any_sign_bit_positive_masked(Mf32(f1, f0, f1, f1, f1, f0, f0, f0)));
    assert!(a.any_sign_bit_positive_masked(Mf32(f1, f0, f1, f1, f0, f0, f0, f0)));
    assert!(!a.any_sign_bit_positive_masked(Mf32(f1, f0, f1, f0, f0, f0, f0, f0)));
    assert!(!a.any_sign_bit_positive_masked(Mf32(f1, f1, f0, f0, f0, f0, f0, f0)));
    assert!(a.any_sign_bit_positive_masked(Mf32(f1, f0, f0, f1, f0, f0, f0, f0)));
}

#[bench]
fn bench_mm256_div_ps_100(b: &mut test::Bencher) {
    let numers = bench::mf32_biunit(4096 / 8);
    let denoms = bench::mf32_biunit(4096 / 8);
    let mut frac_it = numers.iter().cycle().zip(denoms.iter().cycle());
    b.iter(|| {
        for _ in 0..100 {
            let (&numer, &denom) = frac_it.next().unwrap();
            test::black_box(numer.div(denom));
        }
    });
}

#[bench]
fn bench_mm256_rcp_ps_mm256_mul_ps_100(b: &mut test::Bencher) {
    let numers = bench::mf32_biunit(4096 / 8);
    let denoms = bench::mf32_biunit(4096 / 8);
    let mut frac_it = numers.iter().cycle().zip(denoms.iter().cycle());
    b.iter(|| {
        for _ in 0..100 {
            let (&numer, &denom) = frac_it.next().unwrap();
            test::black_box(numer * denom.recip());
        }
    });
}
