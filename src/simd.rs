//! This module makes AVX slightly less painful to work with.
//!
//! Note: compile with AVX and FMA target features to use this to the full
//! extent.

use std::f32::consts;
use std::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Neg, Sub};

#[cfg(test)]
use {bench, test};

#[repr(simd)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Mf32(pub f32, pub f32, pub f32, pub f32, pub f32, pub f32, pub f32, pub f32);


#[repr(simd)]
#[derive(Copy, Clone)]
pub struct Mi32(pub i32, pub i32, pub i32, pub i32, pub i32, pub i32, pub i32, pub i32);

#[repr(simd)]
#[derive(Copy, Clone)]
pub struct Mu64(pub u64, pub u64, pub u64, pub u64);

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

    /// Returns the i-th coordinate. Index must be in the range 0-7 (inclusive).
    #[inline(always)]
    pub fn get_coord(self, i: usize) -> f32 {
        match i {
            0 => self.0,
            1 => self.1,
            2 => self.2,
            3 => self.3,
            4 => self.4,
            5 => self.5,
            6 => self.6,
            7 => self.7,
            _ => panic!("index out of bounds"),
        }
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

    /// Approximates the inverse cosine of self.
    ///
    /// This is based on a rational function approximation of the inverse
    /// cosine. The polynomials are of lower degree than the polynomials used in
    /// `acos()`, trading accuracy for performance.
    ///
    /// The absolute error is at most 0.075 radians (4.3 degrees) on the entire
    /// domain. The relative error is at most 3% on the interval (-0.9, 0.9).
    #[inline(always)]
    pub fn acos_fast(self) -> Mf32 {
        // Sage code to generate the coefficients:
        //
        //     var('a, b, c')
        //
        //     def f(x):
        //         return pi/2 + (a*x + b*x^3) / (1 + c*x^2)
        //
        //     soln = solve([f(3/7) == acos(3/7),
        //                   f(6/7) == acos(6/7),
        //                   f(7/7) == acos(7/7)],
        //                  a, b, c, solution_dict=True)[0]
        //
        //     for (name, val) in soln.iteritems():
        //         print name, val.n(106)
        //
        // The absolute error in this approximation is not distributed
        // uniformly, it is larger at the steep tail than at the almost straight
        // part around 0, therefore the points to match have not been chosen
        // uniformly either. Note that the pi/2 is outside of the fraction to
        // keep the numerator odd and the denominator even.

        let z = Mf32::broadcast(consts::FRAC_PI_2); // pi / 2
        let a = Mf32::broadcast(-1.010967650783108321215466002435);
        let b = Mf32::broadcast(0.7187490034089083954058416974792);
        let c = Mf32::broadcast(-0.8139678312270743216751354761334);

        let x = self;
        let x2 = self * self;
        let x3 = x * x2;

        let numer = x3.mul_add(b, a * x);
        let denom = x2.mul_add(c, Mf32::one());

        numer.mul_add(denom.recip_fast(), z)
    }

    /// Approximates the inverse cosine of self.
    ///
    /// This is based on a rational function approximation of the inverse
    /// cosine.
    ///
    /// The absolute error is at most 0.017 radians (0.96 degrees) on the entire
    /// domain. The relative error is at most 2.2% on the interval (-0.9, 0.9).
    #[inline(always)]
    pub fn acos(self) -> Mf32 {
        // Evaluate a function of the form
        //
        //     pi        ax + bx^3
        //     --  +  ---------------
        //      2     1 + cx^2 + dx^4
        //
        //  A rational function is much better at approximating the inverse
        //  cosine than a polynomial, because a polynomial has a hard time
        //  approximating the part where the derivative goes to infinity. The
        //  script to find the coefficients that minimize the worst error is
        //  in tools/approx_acos.py.

        let z = Mf32::broadcast(consts::FRAC_PI_2); // pi / 2
        let a = Mf32::broadcast(-0.939115566365855);
        let b = Mf32::broadcast(0.9217841528914573);
        let c = Mf32::broadcast(-1.2845906244690837);
        let d = Mf32::broadcast(0.295624144969963174);

        let x = self;
        let x2 = self * self;
        let x3 = x * x2;
        let x4 = x2 * x2;

        let numer = x3.mul_add(b, a * x);
        let denom = x4.mul_add(d, x2.mul_add(c, Mf32::one()));

        numer.mul_add(denom.recip_fast(), z)
    }

    /// Approximates the cosine of self.
    ///
    /// This is based on a polynomial approximation of the cosine. It has been
    /// fitted to minimize the error on the domain (-pi, pi). For values outside
    /// of that range, multiples of 2pi must be added until the value lies
    /// inside this range.
    ///
    /// The absolute error is at most 0.0021 on the interval (-pi, pi).
    /// The relative error is at most 0.5% on the interval (-pi/3, pi/3).
    #[inline(always)]
    pub fn cos(self) -> Mf32 {
        // Evaluate a polynomial of degree 6 with only even terms. The script to
        // find the coefficients that minimize the worst error is in
        // tools/approx_cos.py.

        let a = Mf32::broadcast(-0.496000299455);
        let b = Mf32::broadcast(0.0392596924214);
        let c = Mf32::broadcast(-0.000966231179636657107);

        let x2 = self * self;
        let x4 = x2 * x2;
        let x6 = x2 * x4;

        x6.mul_add(c, x4.mul_add(b, x2.mul_add(a, Mf32::one())))
    }

    /// Approximates the sine of self.
    ///
    /// This is based on a polynomial approximation of the sine. It has been
    /// fitted to minimize the error on the domain (-pi, pi). For values outside
    /// of that range, multiples of 2pi must be added until the value lies
    /// inside this range.
    ///
    /// The absolute error is at most 0.0082 on the interval (-pi, pi).
    /// The relative error is at most 0.18% on the interval (-2pi/3, 2pi/3).
    #[inline(always)]
    pub fn sin(self) -> Mf32 {
        // Evaluate a polynomial of degree 5 with only odd terms. The script to
        // find the coefficients that minimize the worst error is in
        // tools/approx_sin.py.

        let a = Mf32::broadcast(0.982006669405);
        let b = Mf32::broadcast(-0.152175915294);
        let c = Mf32::broadcast(0.00533738081738135917);

        let x = self;
        let x2 = self * self;
        let x3 = x * x2;
        let x5 = x3 * x2;

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

    // TODO: Document / what to do about this? Do I keep it, or do I use the
    // more accurate version (that is still quite fast). When I also need the
    // cosine, the cost of computing powers of x is amortized over the two, so I
    // might get away with the polynomials.
    #[inline(always)]
    pub fn sin_alt(self) -> Mf32 {
        let a = Mf32::broadcast(4.0 / consts::PI);
        let b = Mf32::broadcast(-4.0 / (consts::PI * consts::PI));
        (b * self).mul_add(self.abs(), a * self)
    }

    /// Approximates 1 / self. Precision is poor but it is fast.
    #[inline(always)]
    pub fn recip_fast(self) -> Mf32 {
        unsafe { x86_mm256_rcp_ps(self) }
    }

    /// Approximates 1 / self. Precision is acceptable. It is reasonably fast.
    ///
    /// This is based on the hardware fast reciprocal approximation plus one
    /// Newton iteration. It is slower than `recip_fast()`, but it is also a lot
    /// more precise. A multiplication with `recip_precise()` is still faster
    /// than doing a true division. The error with respect to doing a true
    /// division is less than 0.0001%.
    #[inline(always)]
    pub fn recip_precise(self) -> Mf32 {
        let x0 = self.recip_fast();
        self.neg_mul_add(x0 * x0, x0 + x0)
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

    /// Takes the absolute value of self.
    pub fn abs(self) -> Mf32 {
        use std::mem::transmute;

        // Create a mask with all bits set except the sign bit.
        let mask: f32 = unsafe { transmute(0x7f_ff_ff_ff_u32) };

        // Set the sign bit to zero (indicating positive).
        Mf32::broadcast(mask) & self
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

    /// Returns whether all sign bits are positive (all sign bits are 0).
    #[inline(always)]
    pub fn all_sign_bits_positive(self) -> bool {
        // The mask contains the sign bits packed into an i32. If the mask is
        // zero, then all sign bits were 0, so all numbers were positive.
        unsafe { x86_mm256_movemask_ps(self) == 0 }
    }

    /// Returns whether all sign bits are negative (all sign bits are 1).
    #[inline(always)]
    pub fn all_sign_bits_negative(self) -> bool {
        // The mask contains the sign bits packed into an i32. If the mask is
        // all ones, then all sign bits were 1, so all numbers were negative.
        unsafe { x86_mm256_movemask_ps(self) == 0xff }
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

    /// Returns whether all components are finite.
    ///
    /// This is slow, use only for diagnostic purposes.
    pub fn all_finite(self) -> bool {
        self.0.is_finite() && self.1.is_finite() &&
        self.2.is_finite() && self.3.is_finite() &&
        self.4.is_finite() && self.5.is_finite() &&
        self.6.is_finite() && self.7.is_finite()
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

    /// Converts signed integers into floating-point numbers.
    #[inline(always)]
    pub fn into_mf32(self) -> Mf32 {
        unsafe { x86_mm256_cvtepi32_ps(self) }
    }

    /// Returns the i-th coordinate. Index must be in the range 0-7 (inclusive).
    #[inline(always)]
    pub fn get_coord(self, i: usize) -> i32 {
        match i {
            0 => self.0,
            1 => self.1,
            2 => self.2,
            3 => self.3,
            4 => self.4,
            5 => self.5,
            6 => self.6,
            7 => self.7,
            _ => panic!("index out of bounds"),
        }
    }
}

impl Mask {
    pub fn ones() -> Mask {
        use std::mem::transmute;
        let ones: f32 = unsafe { transmute(0xffffffff_u32) };
        Mf32::broadcast(ones)
    }
}

impl Mu64 {
    /// Applies the function componentwise.
    pub fn map<F>(self, mut f: F) -> Mu64 where F: FnMut(u64) -> u64 {
        Mu64(f(self.0), f(self.1), f(self.2), f(self.3))
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

impl BitAnd<Mi32> for Mi32 {
    type Output = Mi32;

    #[inline(always)]
    fn bitand(self, other: Mi32) -> Mi32 {
        unsafe {
            simd_and(self, other)
        }
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

impl BitOr<Mi32> for Mi32 {
    type Output = Mi32;

    #[inline(always)]
    fn bitor(self, other: Mi32) -> Mi32 {
        unsafe { simd_or(self, other) }
    }
}

impl BitXor<Mask> for Mask {
    type Output = Mask;

    #[inline(always)]
    fn bitxor(self, other: Mask) -> Mask {
        use std::mem::transmute;
        unsafe {
            let a: Mi32 = transmute(self);
            let b: Mi32 = transmute(other);
            let a_xor_b = simd_xor(a, b);
            transmute(a_xor_b)
        }
    }
}

impl BitXor<Mu64> for Mu64 {
    type Output = Mu64;

    #[inline(always)]
    fn bitxor(self, other: Mu64) -> Mu64 {
        unsafe { simd_xor(self, other) }
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

impl Mul<Mu64> for Mu64 {
    type Output = Mu64;

    #[inline(always)]
    fn mul(self, other: Mu64) -> Mu64 {
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
    fn x86_mm256_cvtepi32_ps(x: Mi32) -> Mf32;
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
fn mf32_recip_precise() {
    let xs = bench::mf32_biunit(4096);
    for &x in &xs {
        // Use a true division, which is slow but also precise.
        let expected = Mf32::one() / x;

        let approx = x.recip_precise();

        // Compute the relative error.
        let error = Mf32::one() - (approx / expected);
        let abs_error = error.abs();

        // The relative error should not be greater than 0.0001%.
        assert!((Mf32::broadcast(1e-6) - abs_error).all_sign_bits_positive(),
                "Error should be small but it is {:?} for the input {:?}", abs_error, x);
    }
}

#[test]
fn mf32_cos() {
    let xs = bench::mf32_biunit(4096);
    for &x in &xs {
        let y = x * Mf32::broadcast(consts::PI);

        // Approximate the sine using a polynomial with AVX.
        let approx = y.cos();

        // Apply the regular sin function to every element, without AVX.
        let serial = y.map(|yi| yi.cos());

        // Compute the absolute error.
        let error = approx - serial;
        let abs_error = error.abs();

        // The absolute error should not be greater than 0.0082.
        assert!((Mf32::broadcast(0.0021) - abs_error).all_sign_bits_positive(),
                "Error should be small but it is {:?} for the input {:?}", abs_error, y);
    }
}

#[test]
fn mf32_sin() {
    let xs = bench::mf32_biunit(4096);
    for &x in &xs {
        let y = x * Mf32::broadcast(consts::PI);

        // Approximate the sine using a polynomial with AVX.
        let approx = y.sin();

        // Apply the regular sin function to every element, without AVX.
        let serial = y.map(|yi| yi.sin());

        // Compute the absolute error.
        let error = approx - serial;
        let abs_error = error.abs();

        // The absolute error should not be greater than 0.0082.
        assert!((Mf32::broadcast(0.0082) - abs_error).all_sign_bits_positive(),
                "Error should be small but it is {:?} for the input {:?}", abs_error, y);
    }
}

#[test]
fn mf32_acos_fast() {
    let xs = bench::mf32_biunit(4096);
    for &x in &xs {
        // Approximate the inverse cosine with AVX instructions.
        let approx = x.acos_fast();

        // Apply the regular acos function to every element, without AVX.
        let serial = x.map(|xi| xi.acos());

        // Compute the absolute error.
        let error = approx - serial;
        let abs_error = error.abs();

        // The absolute error should not be greater than 0.075.
        assert!((Mf32::broadcast(0.075) - abs_error).all_sign_bits_positive(),
                "Error should be small but it is {:?} for the input {:?}", abs_error, x);
    }
}

#[test]
fn mf32_acos() {
    let xs = bench::mf32_biunit(4096);
    for &x in &xs {
        // Approximate the inverse cosine with AVX instructions.
        let y = x * Mf32::broadcast(0.98);
        let approx = y.acos();

        // Apply the regular acos function to every element, without AVX.
        let serial = y.map(|yi| yi.acos());

        // Compute the absolute error.
        let error = approx - serial;
        let abs_error = error.abs();

        // The absolute error should not be greater than 0.017.
        assert!((Mf32::broadcast(0.017) - abs_error).all_sign_bits_positive(),
                "Error should be small but it is {:?} for the input {:?}", abs_error, x);
    }
}

#[test]
fn verify_abs() {
    let x = Mf32(1.0, -1.0, 0.0, -0.0, 2.0, 3.0, 5.0, -7.0);
    let y = Mf32(1.0, 1.0, 0.0, 0.0, 2.0, 3.0, 5.0, 7.0);
    assert_eq!(x.abs(), y);
}

macro_rules! unroll_10 {
    { $x: block } => {
        $x $x $x $x $x $x $x $x $x $x
    }
}

#[bench]
fn bench_div_precise_1000(b: &mut test::Bencher) {
    let numers = bench::mf32_biunit(4096 / 8);
    let denoms = bench::mf32_biunit(4096 / 8);
    let mut frac_it = numers.iter().cycle().zip(denoms.iter().cycle());
    b.iter(|| {
        let (&numer, &denom) = frac_it.next().unwrap();
        for _ in 0..100 {
            unroll_10! {{
                test::black_box(test::black_box(numer).div(denom));
            }};
        }
    });
}

#[bench]
fn bench_div_recip_fast_1000(b: &mut test::Bencher) {
    let numers = bench::mf32_biunit(4096 / 8);
    let denoms = bench::mf32_biunit(4096 / 8);
    let mut frac_it = numers.iter().cycle().zip(denoms.iter().cycle());
    b.iter(|| {
        let (&numer, &denom) = frac_it.next().unwrap();
        for _ in 0..100 {
            unroll_10! {{
                test::black_box(test::black_box(numer) * test::black_box(denom).recip_fast());
            }};
        }
    });
}

#[bench]
fn bench_div_recip_precise_1000(b: &mut test::Bencher) {
    let numers = bench::mf32_biunit(4096 / 8);
    let denoms = bench::mf32_biunit(4096 / 8);
    let mut frac_it = numers.iter().cycle().zip(denoms.iter().cycle());
    b.iter(|| {
        let (&numer, &denom) = frac_it.next().unwrap();
        for _ in 0..100 {
            unroll_10! {{
                test::black_box(test::black_box(numer) * test::black_box(denom).recip_precise());
            }};
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
            unroll_10! {{
                test::black_box(test::black_box(x).neg_xor());
            }};
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
            unroll_10! {{
                test::black_box(test::black_box(x).neg_sub());
            }};
        }
        k = (k + 1) % 1024;
    });
}

#[bench]
fn bench_cos_1000(b: &mut test::Bencher) {
    let xs = bench::mf32_biunit(1024);
    let mut k = 0;
    b.iter(|| {
        let x = unsafe { xs.get_unchecked(k) };
        for _ in 0..100 {
            unroll_10! {{
                test::black_box(test::black_box(x).cos());
            }};
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
            unroll_10! {{
                test::black_box(test::black_box(x).sin());
            }};
        }
        k = (k + 1) % 1024;
    });
}

#[bench]
fn bench_sin_alt_1000(b: &mut test::Bencher) {
    let xs = bench::mf32_biunit(1024);
    let mut k = 0;
    b.iter(|| {
        let x = unsafe { xs.get_unchecked(k) };
        for _ in 0..100 {
            unroll_10! {{
                test::black_box(test::black_box(x).sin_alt());
            }};
        }
        k = (k + 1) % 1024;
    });
}

#[bench]
fn bench_acos_fast_1000(b: &mut test::Bencher) {
    let xs = bench::mf32_biunit(1024);
    let mut k = 0;
    b.iter(|| {
        let x = unsafe { xs.get_unchecked(k) };
        for _ in 0..100 {
            unroll_10! {{
                test::black_box(test::black_box(x).acos_fast());
            }};
        }
        k = (k + 1) % 1024;
    });
}

#[bench]
fn bench_acos_1000(b: &mut test::Bencher) {
    let xs = bench::mf32_biunit(1024);
    let mut k = 0;
    b.iter(|| {
        let x = unsafe { xs.get_unchecked(k) };
        for _ in 0..100 {
            unroll_10! {{
                test::black_box(test::black_box(x).acos());
            }};
        }
        k = (k + 1) % 1024;
    });
}
