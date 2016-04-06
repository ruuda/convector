//! Functions for generating random numbers fast.
//!
//! To do Monte Carlo integration you need random numbers. Lots of them, but not
//! necessarily high-quality random numbers. Not online casino or cryptography-
//! grade random numbers. So it is possible to do a lot better than conventional
//! RNGs.

use simd::{Mf32, Mi32, Mu64};
use std::i32;

// A theorem that is used intensively in this file: if n and m are coprime, then
// the map x -> n * x is a bijection of Z/mZ. In practice m is a power of two
// (2^64 in this case), so anything not divisible by two will do for n, but we
// might as well take a prime.
//
// With that you can build a simple and fast hash function for integers:
// multiply with a number coprime to 2. On a computer you get the "modulo a
// power of two" for free. For more details on why this works pretty well,
// Knuth has an entire section devoted to it in Volume 3 of TAOCP.

pub struct Rng {
    state: Mu64,
}

impl Rng {

    /// Creates a new random number generator.
    ///
    /// The generator is seeded from three 32-bit integers, suggestively called
    /// x, y, and i (for frame number). These three values are hashed together,
    /// and that is used as the seed.
    pub fn with_seed(x: u32, y: u32, i: u32) -> Rng {
        // The constants here are all primes near 2^32. It is important that the
        // four values in the final multiplication are distinct, otherwise the
        // sequences will produce the same values. The values `x`, `y`, and `i`
        // are hashed with different functions to ensure that a permutation of
        // (x, y, i) results in a different seed, otherwise patterns would
        // appear because the range of x and y is similar.
        let a = x as u64 * 4294705031;
        let b = y as u64 * 4294836181;
        let c = i as u64 * 4294442983;
        let seed = a + b + c;
        let primes = Mu64(4295032801, 4295098309, 4295229439, 4295491573);

        Rng {
            state: Mu64(seed, seed, seed, seed) * primes,
        }
    }

    /// Updates the state and returns the old state.
    fn next(&mut self) -> Mu64 {
        let old_state = self.state;

        // Again, this is really nothing more than iteratively hashing the
        // state. It is faster than e.g. xorshift, and the quality of the
        // random numbers is still good enough. To demonstrate that it is
        // sufficient that the factor is coprime to 2 I picked a composite
        // number here. Try multiplying it by two and observe how the state
        // reaches 0 after a few iterations.

        let factor = 3 * 5 * 7 * 4294967387;
        self.state = self.state * Mu64(factor, factor, factor, factor);

        old_state
    }

    /// Returns 8 random numbers distributed uniformly over the half-open
    /// interval [0, 1).
    pub fn sample_unit(&mut self) -> Mf32 {
        use std::mem::transmute;

        let mi32: Mi32 = unsafe { transmute(self.next()) };
        let range = Mf32::broadcast(0.5 / i32::MIN as f32);
        let half = Mf32::broadcast(0.5);

        mi32.into_mf32().mul_add(range, half)
    }

    /// Returns 8 random numbers distributed uniformly over the half-open
    /// interval [-1, 1).
    pub fn sample_biunit(&mut self) -> Mf32 {
        use std::mem::transmute;

        let mi32: Mi32 = unsafe { transmute(self.next()) };
        let range = Mf32::broadcast(1.0 / i32::MIN as f32);

        mi32.into_mf32() * range
    }
}

#[test]
fn sample_unit_is_in_interval() {
    let mut rng = Rng::with_seed(2, 5, 7);

    for _ in 0..4096 {
        let x = rng.sample_unit();
        assert!(x.all_sign_bits_positive(), "{:?} should be >= 0");
        assert!((Mf32::one() - x).all_sign_bits_positive(), "{:?} should be <= 1");
    }
}

#[test]
fn sample_biunit_is_in_interval() {
    let mut rng = Rng::with_seed(2, 5, 7);

    for _ in 0..4096 {
        let x = rng.sample_biunit();
        assert!((Mf32::one() + x).all_sign_bits_positive(), "{:?} should be >= -1");
        assert!((Mf32::one() - x).all_sign_bits_positive(), "{:?} should be <= 1");
    }
}
