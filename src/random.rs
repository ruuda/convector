//! Functions for generating random numbers fast.
//!
//! To do Monte Carlo integration you need random numbers. Lots of them, but not
//! necessarily high-quality random numbers. Not online casino or cryptography-
//! grade random numbers. So it is possible to do a lot better than conventional
//! RNGs.

use simd::{Mf32, Mi32, Mu64};
use std::f32::consts;
use std::i32;
use vector3::MVector3;

#[cfg(test)]
use test;

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

    /// Returns 8 random numbers distributed uniformly over the half-open
    /// interval [-pi, pi).
    pub fn sample_angle(&mut self) -> Mf32 {
        use std::mem::transmute;

        let mi32: Mi32 = unsafe { transmute(self.next()) };
        let range = Mf32::broadcast(consts::PI / i32::MIN as f32);

        mi32.into_mf32() * range
    }

    /// Returns a random unit vector in the hemisphere around the positive
    /// z-axis, drawn from a cosine-weighted distribution.
    pub fn sample_hemisphere_vector(&mut self) -> MVector3 {
        let phi = self.sample_angle();
        let r_sqr = self.sample_unit();

        // Instead of the full square root, we could also do a fast inverse
        // square root approximation and a reciprocal approximation. It is less
        // precise, but according to the Intel intrinsics guide, that would take
        // 14 cycles instead of 21. However, we need to compute the polynomials
        // for sin and cos anyway and that takes time, so it is not a problem to
        // take the slow but precise square root: by the time we need it, plenty
        // of cycles will have passed. Pipelining to the rescue here.
        let r = r_sqr.sqrt();
        let x = phi.sin() * r;
        let y = phi.cos() * r; // TODO: cos is a bottleneck, do I need the precision?
        let z = (Mf32::one() - r_sqr).sqrt();

        // TODO: Perhaps it would be faster to use a less precise sin and cos,
        // but normalize the vector in the end?
        MVector3::new(x, y, z)
    }

    /// Returns a random unit vector in the hemisphere around the positive
    /// z-axis, drawn from a cosine-weighted distribution.
    ///
    /// This method uses a different sampling method than
    /// `sample_hemisphere_vector`. Benchmarks show that it is not faster, and
    /// with a small probability this function returns a wrong result too, so it
    /// should not be used at all. It is kept here for comparison purposes.
    fn sample_hemisphere_vector_reject(&mut self) -> MVector3 {
        // This function uses rejection sampling without branching: sample two
        // points in a square, and if the second one is not inside a circle,
        // take the first one instead. The probability that both points do not
        // lie in a circle is (1 - pi/4)^2, about 4.6%. To reduce that
        // probability further you can take more samples.
        let x0 = self.sample_biunit();
        let y0 = self.sample_biunit();
        let r0 = x0.mul_add(x0, y0 * y0);

        let x1 = self.sample_biunit();
        let y1 = self.sample_biunit();
        let r1 = x1.mul_add(x1, y1 * y1);

        // If r1 > 1, then the point lies outside of a unit disk, so the sign
        // bit of this value will be positive, indicating that we should pick
        // point 0 instead of point 1.
        let pick_01 = Mf32::one() - r1;

        let x = x0.pick(x1, pick_01);
        let y = y0.pick(y1, pick_01);
        let r = r0.pick(r1, pick_01);

        let z = (Mf32::one() - r).sqrt();

        MVector3::new(x, y, z)
    }
}

#[test]
fn sample_unit_is_in_interval() {
    let mut rng = Rng::with_seed(2, 5, 7);

    for _ in 0..4096 {
        let x = rng.sample_unit();
        assert!(x.all_sign_bits_positive(), "{:?} should be >= 0", x);
        assert!((Mf32::one() - x).all_sign_bits_positive(), "{:?} should be <= 1", x);
    }
}

#[test]
fn sample_biunit_is_in_interval() {
    let mut rng = Rng::with_seed(2, 5, 7);

    for _ in 0..4096 {
        let x = rng.sample_biunit();
        assert!((Mf32::one() + x).all_sign_bits_positive(), "{:?} should be >= -1", x);
        assert!((Mf32::one() - x).all_sign_bits_positive(), "{:?} should be <= 1", x);
    }
}

#[test]
fn sample_angle_is_in_interval() {
    let mut rng = Rng::with_seed(2, 5, 7);

    for _ in 0..4096 {
        let x = rng.sample_angle();
        assert!((Mf32::broadcast(consts::PI) + x).all_sign_bits_positive(), "{:?} should be >= -pi", x);
        assert!((Mf32::broadcast(consts::PI) - x).all_sign_bits_positive(), "{:?} should be <= pi", x);
    }
}

#[test]
fn sample_hemisphere_vector_has_unit_norm() {
    let mut rng = Rng::with_seed(2, 5, 7);

    for _ in 0..4096 {
        let v = rng.sample_hemisphere_vector();
        let r = v.norm_squared().sqrt();
        assert!((r - Mf32::broadcast(0.991)).all_sign_bits_positive(), "{:?} should be ~1", r);
        assert!((Mf32::broadcast(1.009) - r).all_sign_bits_positive(), "{:?} should be ~1", r);
    }
}

macro_rules! unroll_10 {
    { $x: block } => {
        $x $x $x $x $x $x $x $x $x $x
    }
}

#[bench]
fn bench_sample_unit_1000(b: &mut test::Bencher) {
    let mut rng = Rng::with_seed(2, 5, 7);
    b.iter(|| {
        for _ in 0..100 {
            unroll_10! {{
                test::black_box(rng.sample_unit());
            }};
        }
    });
}

#[bench]
fn bench_sample_hemisphere_vector_1000(b: &mut test::Bencher) {
    let mut rng = Rng::with_seed(2, 5, 7);
    b.iter(|| {
        for _ in 0..100 {
            unroll_10! {{
                test::black_box(rng.sample_hemisphere_vector());
            }};
        }
    });
}

#[bench]
fn bench_sample_hemisphere_vector_reject_1000(b: &mut test::Bencher) {
    let mut rng = Rng::with_seed(2, 5, 7);
    b.iter(|| {
        for _ in 0..100 {
            unroll_10! {{
                test::black_box(rng.sample_hemisphere_vector_reject());
            }};
        }
    });
}
