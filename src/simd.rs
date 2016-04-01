//! This module makes AVX slightly less painful to work with.
//!
//! Note: compile with `cargo rustc -- -C target-feature=sse,sse2,avx,avx2` to
//! use this to the full extent. (Otherwise it will not use AVX but two SSE
//! adds, for instance.)

use std::f32::consts::{FRAC_PI_2, PI};
use std::ops::{Add, BitAnd, BitOr, Div, Mul, Neg, Sub};

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

    /// Applies the function componentwise.
    ///
    /// This should not be used in hot code, as it executes f serially.
    pub fn map<F>(self, mut f: F) -> Mf32 where F: FnMut(f32) -> f32 {
        Mf32(f(self.0), f(self.1), f(self.2), f(self.3),
             f(self.4), f(self.5), f(self.6), f(self.7))
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

    #[inline(always)]
    pub fn neg_mul_add(self, factor: Mf32, term: Mf32) -> Mf32 {
        unsafe { x86_mm256_fnmadd_ps(self, factor, term) }
    }

    /// Computes the inverse cosine of self.
    ///
    /// This is based on a polynomial approximation of the inverse cosine.
    ///
    /// TODO: Document relative error.
    #[inline(always)]
    pub fn acos(self) -> Mf32 {
        // Sage code to generate the coefficients:
        //
        //     var('a, b, c')
        //
        //     def f(x):
        //         return pi/2 + a*x + b*x^3 + c*x^5
        //
        //     solve([f(1/3) == acos(1/3), f(2/3) == acos(2/3), f(1) == 0], a, b, c)[0]
        //
        // The absolute error in the polynomial is not distributed uniformly, it
        // is much bigger close to 1.0. Shifting the points to solve for can
        // reduce the error there at the cost of making it bigger elsewhere.

        let z = Mf32::broadcast(FRAC_PI_2); // pi / 2
        let a = Mf32::broadcast(-1.0295908346187686517475814855949);
        let b = Mf32::broadcast(0.16971176236809475557680747312938);
        let c = Mf32::broadcast(-0.71091725454422272306054767917421);

        let x = self;
        let x2 = self * self;
        let x3 = x * x2;
        let x5 = x3 * x2;

        // TODO: Does this require an extra term?

        x5.mul_add(c, x3.mul_add(b, x.mul_add(a, z)))
    }

    /// Computes the sine of self.
    ///
    /// Like `sin()`, but with one less term in the polynomial. Trades accuracy
    /// for performance.
    ///
    /// The absolute error is at most 0.02 on the interval (-pi, pi).
    /// The relative error is at most 0.8% on the interval (-2pi/3, 2pi/3).
    #[inline(always)]
    pub fn sin_fast(self) -> Mf32 {
        // Sage code to generate the coefficients:
        //
        //     var('a, b, c')
        //
        //     def f(x):
        //         return a*x + b*x^3 + c*x^5
        //
        //     solve([f(pi/3) == sin(pi/3), f(2*pi/3) == sin(2*pi/3), f(pi) == 0], a, b, c)[0]
        let x = self;
        let x2 = self * self;
        let x3 = x * x2;
        let x5 = x3 * x2;

        let a = Mf32::broadcast(0.99239201175922568912038769696334);
        let b = Mf32::broadcast(-0.15710989573225864252780806591220);
        let c = Mf32::broadcast(0.0057306818151060181989591272596327);

        // Due to associativity the result can be computed in several ways.
        // (Floating point numbers are not really associative, but the rounding
        // error due to that is much smaller than the error in the polynomial
        // approximation, so that is not a concern.) It is key to place the
        // parentheses such that the length of the dependency chain is
        // minimized. This can make a 6% difference in execution time. The
        // fused multiply-add is not faster than just doing separate
        // multiplications and adds, but it does save in code size.
        x5.mul_add(c, x3.mul_add(b, x * a))
    }

    /// Computes the sine of self.
    ///
    /// This is based on the Taylor expansion of the sine and takes into account
    /// only a few terms, so it is most accurate around 0. For values outside of
    /// the range (-pi, pi), it is better to add multiples of 2pi until the
    /// value lies inside this range.
    ///
    /// The relative error is about 0.01% at pi/2 and 0.25% at 2pi/3.
    #[inline(always)]
    pub fn sin(self) -> Mf32 {
        let x = self;
        let x2 = self * self;
        let x3 = x * x2;
        let x4 = x2 * x2;
        let x5 = x3 * x2;
        let x7 = x3 * x4;

        let f3 = Mf32::broadcast(-1.0 / 6.0);
        let f5 = Mf32::broadcast(1.0 / 120.0);
        let f7 = Mf32::broadcast(-1.0 / 5040.0);

        // x - x^3 / 6.0 + x^5 / 120.0 - x^7 / 5040.0
        // Like with `sin_fast()`, the dependency chain is the bottleneck here,
        // and using a fused-multiply-add is not really faster than just doing
        // the multiplications, but it does save in code size.
        x7.mul_add(f7, x5.mul_add(f5, x3.mul_add(f3, x)))
    }

    /// Approximates 1 / self.
    #[inline(always)]
    pub fn recip(self) -> Mf32 {
        unsafe { x86_mm256_rcp_ps(self) }
    }

    /// Flips the sign of self by flipping the sign bit.
    #[inline(always)]
    pub fn neg_xor(self) -> Mf32 {
        let sign_bit = Mi32::broadcast(0x80_00_00_00_u32 as i32);
        use std::mem::transmute;
        unsafe {
            let a: Mi32 = transmute(self);
            let minus_a = simd_xor(a, sign_bit);
            transmute(minus_a)
        }
    }

    /// Flips the sign of self by subtracting self from zero.
    #[inline(always)]
    pub fn neg_sub(self) -> Mf32 {
        Mf32::zero() - self
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
        Mi32(f(self.0), f(self.1), f(self.2), f(self.3),
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

impl Neg for Mf32 {
    type Output = Mf32;

    #[inline(always)]
    fn neg(self) -> Mf32 {
        // How to negate? Flip the sign bit or subtract from zero? According to
        // the benchmarks they are equally fast (though the compiler **does**
        // generate different code). A vxorps and vsubps instruction are both 4
        // bytes, so code size does not favor one either. According to the Intel
        // intrinsics guide the xor should have a lower latency, but it does
        // require loading a constant, and generating a zero can be done by de
        // decoder in 0 cycles. So either one is probably fine.
        self.neg_sub()
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

    // This is `_mm256_xor_ps` when compiled for AVX.
    fn simd_xor<T>(x: T, y: T) -> T;

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
    fn x86_mm256_fnmadd_ps(x: Mf32, y: Mf32, z: Mf32) -> Mf32;
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

#[cfg(not(target_feature = "fma"))]
unsafe fn x86_mm256_fnmadd_ps(x: Mf32, y: Mf32, z: Mf32) -> Mf32 {
    z - x * y
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
fn mf32_fnmadd_ps() {
    let a = Mf32(0.0, 1.0, 0.0,  2.0, 1.0, 2.0,  3.0,  4.0);
    let b = Mf32(5.0, 6.0, 7.0,  8.0, 0.0, 1.0,  2.0,  3.0);
    let c = Mf32(5.0, 6.0, 7.0,  8.0, 1.0, 3.0,  5.0,  7.0);
    let d = Mf32(5.0, 0.0, 7.0, -8.0, 1.0, 1.0, -1.0, -5.0);
    assert_eq!(a.neg_mul_add(b, c), d);
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

#[test]
fn mf32_sin() {
    let xs = bench::mf32_biunit(4096);
    for &x in &xs {
        let y = x * Mf32::broadcast(2.0 * PI / 3.0);

        // Approximate the sine using a Taylor expansion with AVX.
        let approx = y.sin();

        // Apply the regular sin function to every element, without AVX.
        let serial = y.map(|yi| yi.sin());

        // Compute the relative error.
        let error = Mf32::one() - (approx / serial);
        let abs_error = error.max(-error);

        // The relative error should not be greater than 0.25%.
        assert!((Mf32::broadcast(0.0025) - abs_error).all_sign_bits_positive(),
                "Error should be small but it is {:?} for the input {:?}", abs_error, y);
    }
}

#[test]
fn mf32_sin_fast() {
    let xs = bench::mf32_biunit(4096);
    for &x in &xs {
        let y = x * Mf32::broadcast(2.0 * PI / 3.0);

        // Approximate the sine using a Taylor expansion with AVX.
        let approx = y.sin_fast();

        // Apply the regular sin function to every element, without AVX.
        let serial = y.map(|yi| yi.sin());

        // Compute the relative error.
        let error = Mf32::one() - (approx / serial);
        let abs_error = error.max(-error);

        // The relative error should not be greater than 0.8%.
        assert!((Mf32::broadcast(0.008) - abs_error).all_sign_bits_positive(),
                "Error should be small but it is {:?} for the input {:?}", abs_error, y);
    }
}

#[bench]
fn bench_mm256_div_ps_1000(b: &mut test::Bencher) {
    let numers = bench::mf32_biunit(4096 / 8);
    let denoms = bench::mf32_biunit(4096 / 8);
    let mut frac_it = numers.iter().cycle().zip(denoms.iter().cycle());
    b.iter(|| {
        let (&numer, &denom) = frac_it.next().unwrap();
        for _ in 0..100 {
            test::black_box(test::black_box(numer).div(denom));
            test::black_box(test::black_box(numer).div(denom));
            test::black_box(test::black_box(numer).div(denom));
            test::black_box(test::black_box(numer).div(denom));
            test::black_box(test::black_box(numer).div(denom));
            test::black_box(test::black_box(numer).div(denom));
            test::black_box(test::black_box(numer).div(denom));
            test::black_box(test::black_box(numer).div(denom));
            test::black_box(test::black_box(numer).div(denom));
            test::black_box(test::black_box(numer).div(denom));
        }
    });
}

#[bench]
fn bench_mm256_rcp_ps_mm256_mul_ps_1000(b: &mut test::Bencher) {
    let numers = bench::mf32_biunit(4096 / 8);
    let denoms = bench::mf32_biunit(4096 / 8);
    let mut frac_it = numers.iter().cycle().zip(denoms.iter().cycle());
    b.iter(|| {
        let (&numer, &denom) = frac_it.next().unwrap();
        for _ in 0..100 {
            test::black_box(numer * test::black_box(denom).recip());
            test::black_box(numer * test::black_box(denom).recip());
            test::black_box(numer * test::black_box(denom).recip());
            test::black_box(numer * test::black_box(denom).recip());
            test::black_box(numer * test::black_box(denom).recip());
            test::black_box(numer * test::black_box(denom).recip());
            test::black_box(numer * test::black_box(denom).recip());
            test::black_box(numer * test::black_box(denom).recip());
            test::black_box(numer * test::black_box(denom).recip());
            test::black_box(numer * test::black_box(denom).recip());
            test::black_box(numer * test::black_box(denom).recip());
        }
    });
}

#[bench]
fn bench_negate_with_xor_1000(b: &mut test::Bencher) {
    let xs = bench::mf32_biunit(1024);
    let mut k = 0;
    b.iter(|| {
        let x = unsafe { xs.get_unchecked(k) };
        for _ in 0..100 {
            test::black_box(test::black_box(x).neg_xor());
            test::black_box(test::black_box(x).neg_xor());
            test::black_box(test::black_box(x).neg_xor());
            test::black_box(test::black_box(x).neg_xor());
            test::black_box(test::black_box(x).neg_xor());
            test::black_box(test::black_box(x).neg_xor());
            test::black_box(test::black_box(x).neg_xor());
            test::black_box(test::black_box(x).neg_xor());
            test::black_box(test::black_box(x).neg_xor());
            test::black_box(test::black_box(x).neg_xor());
        }
        k = (k + 1) % 1024;
    });
}

#[bench]
fn bench_negate_with_sub_1000(b: &mut test::Bencher) {
    let xs = bench::mf32_biunit(1024);
    let mut k = 0;
    b.iter(|| {
        let x = unsafe { xs.get_unchecked(k) };
        for _ in 0..100 {
            test::black_box(test::black_box(x).neg_sub());
            test::black_box(test::black_box(x).neg_sub());
            test::black_box(test::black_box(x).neg_sub());
            test::black_box(test::black_box(x).neg_sub());
            test::black_box(test::black_box(x).neg_sub());
            test::black_box(test::black_box(x).neg_sub());
            test::black_box(test::black_box(x).neg_sub());
            test::black_box(test::black_box(x).neg_sub());
            test::black_box(test::black_box(x).neg_sub());
            test::black_box(test::black_box(x).neg_sub());
        }
        k = (k + 1) % 1024;
    });
}

#[bench]
fn bench_sin_fast_1000(b: &mut test::Bencher) {
    let xs = bench::mf32_biunit(1024);
    let mut k = 0;
    b.iter(|| {
        let x = unsafe { xs.get_unchecked(k) };
        for _ in 0..100 {
            test::black_box(test::black_box(x).sin_fast());
            test::black_box(test::black_box(x).sin_fast());
            test::black_box(test::black_box(x).sin_fast());
            test::black_box(test::black_box(x).sin_fast());
            test::black_box(test::black_box(x).sin_fast());
            test::black_box(test::black_box(x).sin_fast());
            test::black_box(test::black_box(x).sin_fast());
            test::black_box(test::black_box(x).sin_fast());
            test::black_box(test::black_box(x).sin_fast());
            test::black_box(test::black_box(x).sin_fast());
        }
        k = (k + 1) % 1024;
    });
}

#[bench]
fn bench_sin_1000(b: &mut test::Bencher) {
    let xs = bench::mf32_biunit(1024);
    let mut k = 0;
    b.iter(|| {
        let x = unsafe { xs.get_unchecked(k) };
        for _ in 0..100 {
            test::black_box(test::black_box(x).sin());
            test::black_box(test::black_box(x).sin());
            test::black_box(test::black_box(x).sin());
            test::black_box(test::black_box(x).sin());
            test::black_box(test::black_box(x).sin());
            test::black_box(test::black_box(x).sin());
            test::black_box(test::black_box(x).sin());
            test::black_box(test::black_box(x).sin());
            test::black_box(test::black_box(x).sin());
            test::black_box(test::black_box(x).sin());
        }
        k = (k + 1) % 1024;
    });
}

// TODO: DRY up these benchmarks.
#[bench]
fn bench_acos_1000(b: &mut test::Bencher) {
    let xs = bench::mf32_biunit(1024);
    let mut k = 0;
    b.iter(|| {
        let x = unsafe { xs.get_unchecked(k) };
        for _ in 0..100 {
            test::black_box(test::black_box(x).acos());
            test::black_box(test::black_box(x).acos());
            test::black_box(test::black_box(x).acos());
            test::black_box(test::black_box(x).acos());
            test::black_box(test::black_box(x).acos());
            test::black_box(test::black_box(x).acos());
            test::black_box(test::black_box(x).acos());
            test::black_box(test::black_box(x).acos());
            test::black_box(test::black_box(x).acos());
            test::black_box(test::black_box(x).acos());
        }
        k = (k + 1) % 1024;
    });
}
