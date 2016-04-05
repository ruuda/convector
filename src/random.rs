//! Functions for generating random numbers fast.
//!
//! To do Monte Carlo integration you need random numbers. Lots of them, but not
//! necessarily high-quality random numbers. Not online casino or cryptography-
//! grade random numbers. So it is possible to do a lot better than conventional
//! RNGs.

use simd::{Mf32, Mu64};

pub struct Rng {
    state: Mu64,
}

impl Rng {

    /// Creates a new random number generator.
    ///
    /// The initial state is taken from the thread-local random number generator
    /// (from the rand crate).
    pub fn seeded_from_tls() -> Rng {
        // TODO: seed. But should **not** seed randomly, must take an element of
        // maximal order.
        Rng {
            state: Mu64(8388581, 8388587, 8388593, 8388617)
        }
    }

    /// Updates the state and returns the old state.
    fn next(&mut self) -> Mu64 {
        // The function x -> n * x is a bijection of Z/mZ if n and m are
        // coprime, hence the sequence (a, na, 2na, ...) mod m repeats after
        // phi(m) values. If we take m = 2^64 and n any number not divisible by
        // 2, then this is we get a periodic sequence. The period depends on a
        // and it is at most (2^64 - 1)/3, the largest divisor of the order of
        // (Z/mZ)*.
        // Empirical observation: the sequence (a, na, 2na, ...) mod m appears
        // random if a is not too small (if there is no wraparound we just get
        // the multiples of a) or too large (if a is close to m, then it is
        // minus a small value mod m).
        let old_state = self.state;

        // Take a prime near sqrt(2^64) as factor.
        let prime = 4_294_967_387;

        // Multiply with the constant. Wraparound is intended.
        self.state = self.state * Mu64(prime, prime, prime, prime);

        old_state
    }

    /// Returns 8 random numbers distributed uniformly over the closed interval
    /// [0, 1].
    pub fn sample_unit(&mut self) -> Mf32 {
        use std::mem::transmute;

        // This method employs a bit of bit trickery to construct floating point
        // numbers. If the exponent part is fixed, then the mantissa part works
        // just like an integer, but implicitly a base value is added (there is
        // an implicit leading 1). So we can generate random integers in that
        // range just fine. Then do regular floating-point math to put those
        // numbers in the interval [0, 1].

        // A mask for the 23-bit mantissa part of a 32-bit float.
        let mask: u32 = 0x7f_ff_ff;

        // The exponent that indicates 0.
        let exponent: u32 = 127 << 23;

        // The minimum and maximum value that we will generate.
        let min_val: f32 = unsafe { transmute(exponent) };
        let max_val: f32 = unsafe { transmute(exponent | mask) };
        let rlen = 1.0 / (max_val - min_val);

        let mask_mf32 = Mf32::broadcast(unsafe { transmute(mask) });
        let exponent_mf32 = Mf32::broadcast(unsafe { transmute(exponent) });

        // Generate a random float distributed uniformly between min_val and
        // max_val.
        let bits: Mf32 = unsafe { transmute(self.next()) };
        let x = (bits & mask_mf32) | exponent_mf32;

        // Scale to the range [0.0, 1.0].
        (x - Mf32::broadcast(min_val)) * Mf32::broadcast(rlen)
    }

    /// Returns 8 random numbers distributed uniformly over the closed interval
    /// [-1, 1].
    pub fn sample_biunit(&mut self) -> Mf32 {
        use std::mem::transmute;

        // See `sample_unit()` for how this works.
        let mask: u32 = 0x7f_ff_ff;
        let exponent: u32 = 127 << 23;
        let min_val: f32 = unsafe { transmute(exponent) };
        let max_val: f32 = unsafe { transmute(exponent | mask) };
        let half_val = 0.5 * (min_val + max_val);
        let mask_mf32 = Mf32::broadcast(unsafe { transmute(mask) });
        let exponent_mf32 = Mf32::broadcast(unsafe { transmute(exponent) });
        let rlen = 0.5 / (max_val - min_val);
        let bits: Mf32 = unsafe { transmute(self.next()) };
        let x = (bits & mask_mf32) | exponent_mf32;

        (x - Mf32::broadcast(half_val)) * Mf32::broadcast(rlen)

    }
}

#[test]
fn sample_unit_is_in_interval() {
    let mut rng = Rng::seeded_from_tls();

    for _ in 0..4096 {
        let x = rng.sample_unit();
        assert!(x.all_sign_bits_positive(), "{:?} should be >= 0");
        assert!((Mf32::one() - x).all_sign_bits_positive(), "{:?} should be <= 1");
    }
}

#[test]
fn sample_biunit_is_in_interval() {
    let mut rng = Rng::seeded_from_tls();

    for _ in 0..4096 {
        let x = rng.sample_biunit();
        assert!((Mf32::one() + x).all_sign_bits_positive(), "{:?} should be >= -1");
        assert!((Mf32::one() - x).all_sign_bits_positive(), "{:?} should be <= 1");
    }
}
