//! This module makes AVX slightly less painful to work with.
//!
//! Note: compile with `cargo rustc -- -C target-feature=sse,sse2,avx,avx2` to
//! use this to the full extent. (Otherwise it will not use AVX but two SSE
//! adds, for instance.)

use std::ops::{Add, BitAnd, BitOr, Div, Mul, Not, Sub};

#[cfg(test)]
use {bench, test};

#[repr(simd)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct OctaF32(pub f32, pub f32, pub f32, pub f32, pub f32, pub f32, pub f32, pub f32);

pub type Mask = OctaF32;

impl OctaF32 {
    pub fn zero() -> OctaF32 {
        OctaF32(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    }

    pub fn one() -> OctaF32 {
        OctaF32(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    }

    /// Builds an octafloat by applying the function to the numbers 0..7.
    pub fn generate<F: FnMut(usize) -> f32>(mut f: F) -> OctaF32 {
        OctaF32(f(0), f(1), f(2), f(3), f(4), f(5), f(6), f(7))
    }

    #[inline(always)]
    pub fn broadcast(x: f32) -> OctaF32 {
        // TODO: Investigate whether an intrinsic might be faster.
        // unsafe { x86_mm256_broadcast_ss(&x) }
        OctaF32(x, x, x, x, x, x, x, x)
    }

    pub fn as_slice(&self) -> &[f32; 8] {
        use std::mem;
        unsafe { mem::transmute(self) }
    }

    #[inline(always)]
    pub fn mul_add(self, factor: OctaF32, term: OctaF32) -> OctaF32 {
        unsafe { x86_mm256_fmadd_ps(self, factor, term) }
    }

    #[inline(always)]
    pub fn mul_sub(self, factor: OctaF32, term: OctaF32) -> OctaF32 {
        unsafe { x86_mm256_fmsub_ps(self, factor, term) }
    }

    /// Approximates 1 / self.
    #[inline(always)]
    pub fn recip(self) -> OctaF32 {
        unsafe { x86_mm256_rcp_ps(self) }
    }

    /// Computes self / denom with best precision.
    #[inline(always)]
    pub fn div(self, denom: OctaF32) -> OctaF32 {
        unsafe { simd_div(self, denom) }
    }

    /// Approximates the reciprocal square root.
    #[inline(always)]
    pub fn rsqrt(self) -> OctaF32 {
        unsafe { x86_mm256_rsqrt_ps(self) }
    }

    #[inline(always)]
    pub fn max(self, other: OctaF32) -> OctaF32 {
        unsafe { x86_mm256_max_ps(self, other) }
    }

    #[inline(always)]
    pub fn min(self, other: OctaF32) -> OctaF32 {
        unsafe { x86_mm256_min_ps(self, other) }
    }

    #[inline(always)]
    pub fn leq(self, other: OctaF32) -> Mask {
        // Operation 26 is a not greater than comparison, unordered,
        // non-signalling.
        unsafe { x86_mm256_cmp_ps(self, other, 26) }
    }

    #[inline(always)]
    pub fn geq(self, other: OctaF32) -> Mask {
        // Operation 21 is a not less than comparison, unordered,
        // non-signalling.
        unsafe { x86_mm256_cmp_ps(self, other, 21) }
    }

    /// Returns whether any of the values not masked out is positive.
    #[inline(always)]
    pub fn any_positive_masked(self, mask: Mask) -> bool {
        use std::mem::transmute;
        // The testc intrinsic computes `(not self) and mask`, and then returns
        // 1 if all resulting sign bits are 0, or 0 otherwise. If a value is
        // positive, the sign bit will be 0, so `not self` will have sign bit 1.
        // Mask out the values that we are not interested in, then testc returns
        // 1 if there were no positive values, so negate the result. Also, we
        // know that the returned value is either 0 or 1, so there is no need
        // for a comparison, just interpret the bytes as a boolean.
        let no_positive: bool = unsafe {
            transmute(x86_mm256_testc_ps(self, mask) as i8)
        };

        !no_positive
    }

    /// Picks the component of self if the sign bit in the mask is 0,
    /// otherwise picks the component in other.
    #[inline(always)]
    pub fn pick(self, other: OctaF32, mask: Mask) -> OctaF32 {
        unsafe { x86_mm256_blendv_ps(self, other, mask) }
    }
}

impl Add<OctaF32> for OctaF32 {
    type Output = OctaF32;

    #[inline(always)]
    fn add(self, other: OctaF32) -> OctaF32 {
        unsafe { simd_add(self, other) }
    }
}

impl BitAnd<Mask> for Mask {
    type Output = Mask;

    #[inline(always)]
    fn bitand(self, other: Mask) -> Mask {
        use std::mem::transmute;
        // The _mm256_and_ps intrinsic is not available as an LLVM intrinsic;
        // the bitwise and operations are different.
        // TODO: Verify that this emits a vandps instruction though.
        unsafe {
            let a: [u64; 4] = transmute(self);
            let b: [u64; 4] = transmute(other);
            let a_and_b = [a[0] & b[0], a[1] & b[1], a[2] & b[2], a[3] & b[3]];
            transmute(a_and_b)
        }
    }
}

impl BitOr<Mask> for Mask {
    type Output = Mask;

    #[inline(always)]
    fn bitor(self, other: Mask) -> Mask {
        use std::mem::transmute;
        // The _mm256_or_ps intrinsic is not available as an LLVM intrinsic;
        // the bitwise and operations are different.
        // TODO: Verify that this emits a vorps instruction though.
        unsafe {
            let a: [u64; 4] = transmute(self);
            let b: [u64; 4] = transmute(other);
            let a_or_b = [a[0] | b[0], a[1] | b[1], a[2] | b[2], a[3] | b[3]];
            transmute(a_or_b)
        }
    }
}

impl Div<OctaF32> for OctaF32 {
    type Output = OctaF32;

    #[inline(always)]
    fn div(self, denom: OctaF32) -> OctaF32 {
        // Benchmarks show that _mm256_div_ps is as fast as a _mm256_rcp_ps
        // followed by a _mm256_mul_ps, so we might as well use the div.
        unsafe { simd_div(self, denom) }
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

impl Not for Mask {
    type Output = Mask;

    #[inline(always)]
    fn not(self) -> Mask {
        use std::mem::transmute;
        // TODO: Verify that this generates concise assembly.
        unsafe {
            let a: [u64; 4] = transmute(self);
            let bitwise_not_a = [!a[0], !a[1], !a[2], !a[3]];
            transmute(bitwise_not_a)
        }
    }
}

extern "platform-intrinsic" {
    // This is `_mm256_add_ps` when compiled for AVX.
    fn simd_add<T>(x: T, y: T) -> T;

    // This is `_mm256_div_ps` when compiled for AVX.
    fn simd_div<T>(x: T, y: T) -> T;

    // This is `_mm256_sub_ps` when compiled for AVX.
    fn simd_sub<T>(x: T, y: T) -> T;

    // This is `_mm256_mul_ps` when compiled for AVX.
    fn simd_mul<T>(x: T, y: T) -> T;

    fn x86_mm256_blendv_ps(x: OctaF32, y: OctaF32, mask: Mask) -> OctaF32;
    fn x86_mm256_cmp_ps(x: OctaF32, y: OctaF32, op: i8) -> Mask;
    fn x86_mm256_fmadd_ps(x: OctaF32, y: OctaF32, z: OctaF32) -> OctaF32;
    fn x86_mm256_fmsub_ps(x: OctaF32, y: OctaF32, z: OctaF32) -> OctaF32;
    fn x86_mm256_max_ps(x: OctaF32, y: OctaF32) -> OctaF32;
    fn x86_mm256_min_ps(x: OctaF32, y: OctaF32) -> OctaF32;
    fn x86_mm256_rcp_ps(x: OctaF32) -> OctaF32;
    fn x86_mm256_rsqrt_ps(x: OctaF32) -> OctaF32;
    fn x86_mm256_testc_ps(x: OctaF32, y: OctaF32) -> i32;

    // TODO: Add the x86_mm256_broadcast_ss intrinsic to rustc and see if that
    // is faster than constructing the constant in Rust.
    // fn x86_mm256_broadcast_ss(x: &f32) -> OctaF32;

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

#[test]
fn octa_f32_fmadd_ps() {
    let a = OctaF32(0.0,  1.0, 0.0,  2.0, 1.0, 2.0,  3.0,  4.0);
    let b = OctaF32(5.0,  6.0, 7.0,  8.0, 0.0, 1.0,  2.0,  3.0);
    let c = OctaF32(5.0,  6.0, 7.0,  8.0, 1.0, 3.0,  5.0,  7.0);
    let d = OctaF32(5.0, 12.0, 7.0, 24.0, 1.0, 5.0, 11.0, 19.0);
    assert_eq!(a.mul_add(b, c), d);
}

#[test]
fn octa_f32_fmsub_ps() {
    let a = OctaF32( 0.0, 1.0,  0.0, 2.0,  1.0,  2.0, 3.0, 4.0);
    let b = OctaF32( 5.0, 6.0,  7.0, 8.0,  0.0,  1.0, 2.0, 3.0);
    let c = OctaF32( 5.0, 6.0,  7.0, 8.0,  1.0,  3.0, 5.0, 7.0);
    let d = OctaF32(-5.0, 0.0, -7.0, 8.0, -1.0, -1.0, 1.0, 5.0);
    assert_eq!(a.mul_sub(b, c), d);
}

#[test]
fn octa_f32_broadcast_ps() {
    let a = OctaF32::broadcast(7.0);
    let b = OctaF32(7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0);
    assert_eq!(a, b);
}

#[test]
fn octa_f32_any_positive_masked() {
    use std::mem::transmute;
    let a = OctaF32(-2.0, -1.0, -0.0, 0.0, 1.0, 2.0, 3.0, 4.0);
    let f1: f32 = unsafe { transmute(0xffffffff_u32) };
    let f0: f32 = 0.0;

    assert!(a.any_positive_masked(OctaF32(f1, f0, f1, f1, f1, f0, f0, f0)));
    assert!(a.any_positive_masked(OctaF32(f1, f0, f1, f1, f0, f0, f0, f0)));
    assert!(!a.any_positive_masked(OctaF32(f1, f0, f1, f0, f0, f0, f0, f0)));
    assert!(!a.any_positive_masked(OctaF32(f1, f1, f0, f0, f0, f0, f0, f0)));
    assert!(a.any_positive_masked(OctaF32(f1, f0, f0, f1, f0, f0, f0, f0)));
}

#[bench]
fn bench_mm256_div_ps_x100(b: &mut test::Bencher) {
    let numers = bench::octa_biunit(4096 / 8);
    let denoms = bench::octa_biunit(4096 / 8);
    let mut frac_it = numers.iter().cycle().zip(denoms.iter().cycle());
    b.iter(|| {
        for _ in 0..100 {
            let (&numer, &denom) = frac_it.next().unwrap();
            test::black_box(numer.div(denom));
        }
    });
}

#[bench]
fn bench_mm256_rcp_ps_mm256_mul_ps_x100(b: &mut test::Bencher) {
    let numers = bench::octa_biunit(4096 / 8);
    let denoms = bench::octa_biunit(4096 / 8);
    let mut frac_it = numers.iter().cycle().zip(denoms.iter().cycle());
    b.iter(|| {
        for _ in 0..100 {
            let (&numer, &denom) = frac_it.next().unwrap();
            test::black_box(numer * denom.recip());
        }
    });
}
