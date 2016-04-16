//! Implements vectors in R3.

use simd::{Mask, Mf32};
use std::f32;
use std::fmt;
use std::ops::{Add, Sub, Neg, Mul};

#[cfg(test)]
use {bench, test};

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SVector3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct MVector3 {
    pub x: Mf32,
    pub y: Mf32,
    pub z: Mf32,
}

#[derive(Copy, Clone, Debug)]
pub enum Axis {
    X,
    Y,
    Z,
}

impl SVector3 {
    pub fn new(x: f32, y: f32, z: f32) -> SVector3 {
        SVector3 { x: x, y: y, z: z }
    }

    pub fn zero() -> SVector3 {
        SVector3::new(0.0, 0.0, 0.0)
    }

    pub fn one() -> SVector3 {
        SVector3::new(1.0, 1.0, 1.0)
    }

    #[inline(always)]
    pub fn cross_naive(self: SVector3, other: SVector3) -> SVector3 {
        let (a, b) = (self, other);
        SVector3 {
            x: a.y * b.z - a.z * b.y,
            y: a.z * b.x - a.x * b.z,
            z: a.x * b.y - a.y * b.x,
        }
    }

    #[inline(always)]
    pub fn cross_fma(self: SVector3, other: SVector3) -> SVector3 {
        let (a, b) = (self, other);
        SVector3 {
            x: a.y.mul_add(b.z, -a.z * b.y),
            y: a.z.mul_add(b.x, -a.x * b.z),
            z: a.x.mul_add(b.y, -a.y * b.x),
        }
    }

    pub fn cross(self, other: SVector3) -> SVector3 {
        // Benchmarks show that the FMA version is faster than the
        // naive version (1.9 ns vs 2.1 ns on my Skylake i7). **However**
        // the "fma" codegen feature must be enabled, otherwise the naive
        // version is faster.
        self.cross_fma(other)
    }

    #[inline(always)]
    pub fn dot_naive(self, other: SVector3) -> f32 {
        let (a, b) = (self, other);
        a.x * b.x + a.y * b.y + a.z * b.z
    }

    #[inline(always)]
    pub fn dot_fma(self, other: SVector3) -> f32 {
        let (a, b) = (self, other);
        a.x.mul_add(b.x, a.y.mul_add(b.y, a.z * b.z))
    }

    pub fn dot(self, other: SVector3) -> f32 {
        // Benchmarks show that the naive version is faster than the FMA version
        // when the "fma" codegen feature is not enabled, but when it is the
        // performance is similar. The FMA version appears to be slightly more
        // stable.
        self.dot_fma(other)
    }

    pub fn norm_squared(self) -> f32 {
        self.dot(self)
    }

    pub fn normalized(self) -> SVector3 {
        let norm_squared = self.norm_squared();
        if norm_squared == 0.0 {
            self
        } else {
            let rnorm = norm_squared.sqrt().recip();
            SVector3 {
                x: self.x * rnorm,
                y: self.y * rnorm,
                z: self.z * rnorm,
            }
        }
    }

    pub fn get_coord(self, axis: Axis) -> f32 {
        match axis {
            Axis::X => self.x,
            Axis::Y => self.y,
            Axis::Z => self.z,
        }
    }

    /// Returns the coordinatewise minimum of the two vectors.
    pub fn min(self, other: SVector3) -> SVector3 {
        SVector3 {
            x: f32::min(self.x, other.x),
            y: f32::min(self.y, other.y),
            z: f32::min(self.z, other.z),
        }
    }

    /// Returns the coordinatewise maximum of the two vectors.
    pub fn max(self, other: SVector3) -> SVector3 {
        SVector3 {
            x: f32::max(self.x, other.x),
            y: f32::max(self.y, other.y),
            z: f32::max(self.z, other.z),
        }
    }
}

impl MVector3 {
    pub fn new(x: Mf32, y: Mf32, z: Mf32) -> MVector3 {
        MVector3 { x: x, y: y, z: z }
    }

    pub fn zero() -> MVector3 {
        MVector3::new(Mf32::zero(), Mf32::zero(), Mf32::zero())
    }

    pub fn broadcast(a: SVector3) -> MVector3 {
        MVector3 {
            x: Mf32::broadcast(a.x),
            y: Mf32::broadcast(a.y),
            z: Mf32::broadcast(a.z),
        }
    }

    /// Builds an mvector by applying the function to the numbers 0..7.
    ///
    /// Note: this is essentially a transpose, avoid in hot code.
    pub fn generate<F>(mut f: F) -> MVector3 where F: FnMut(usize) -> SVector3 {
        MVector3 {
            x: Mf32::generate(|i| f(i).x),
            y: Mf32::generate(|i| f(i).y),
            z: Mf32::generate(|i| f(i).z),
        }
    }

    #[inline(always)]
    pub fn cross_naive(self, other: MVector3) -> MVector3 {
        let (a, b) = (self, other);
        MVector3 {
            x: a.y * b.z - a.z * b.y,
            y: a.z * b.x - a.x * b.z,
            z: a.x * b.y - a.y * b.x,
        }
    }

    #[inline(always)]
    pub fn cross_fma(self, other: MVector3) -> MVector3 {
        let (a, b) = (self, other);
        MVector3 {
            x: a.y.mul_sub(b.z, a.z * b.y),
            y: a.z.mul_sub(b.x, a.x * b.z),
            z: a.x.mul_sub(b.y, a.y * b.x),
        }
    }

    pub fn cross(self, other: MVector3) -> MVector3 {
        // Benchmarks show that the FMA version is faster than the
        // naive version (2.1 ns vs 2.4 ns on my Skylake i7).
        self.cross_fma(other)
    }

    #[inline(always)]
    pub fn dot_naive(self, other: MVector3) -> Mf32 {
        let (a, b) = (self, other);
        a.x * b.x + a.y * b.y + a.z * b.z
    }

    #[inline(always)]
    pub fn dot_fma(self, other: MVector3) -> Mf32 {
        let (a, b) = (self, other);
        a.x.mul_add(b.x, a.y.mul_add(b.y, a.z * b.z))
    }

    pub fn dot(self, other: MVector3) -> Mf32 {
        // Benchmarks show no performance difference between the naive version
        // and the FMA version. Use the naive one because it is more portable.
        self.dot_naive(other)
    }

    /// Given a vector in the hemisphere with pole at the positive z-axis,
    /// rotates the vector into the hemisphere with pole given by the normal.
    pub fn rotate_hemisphere(self, normal: MVector3) -> MVector3 {
        // If the z-component of the normal is near -1, we might divide by 0. To
        // avoid this, if the z-component is negative, flip the normal. Then we
        // end up in the wrong hemisphere, so at the end, flip the computed
        // vector again.
        let n = normal.pick(-normal, normal.z);

        // One option here would be to take the cross product of the normal and
        // an up vector, and the cross product of the normal with that vector,
        // to get a new orthonormal basis. Then use the old coordinates in this
        // new basis. The method below -- however not as simple -- requires less
        // arithmetic operations.
        // Based on https://math.stackexchange.com/a/61550/6873.
        let v = self;

        // Using the fast reciprocal instead of the precise one does hurt
        // precision, but this is used only after the first bounce, so it is
        // less of an issue, and the performance difference is significant. The
        // inaccurate version is about 15% faster.
        let rz = (Mf32::one() + n.z).recip_fast();

        let c = n.x * n.y * rz;
        let x = v.x.mul_sub(n.y.mul_add(n.y * rz, n.z), v.y.mul_sub(c, v.z * n.x));
        let y = v.x.neg_mul_add(c, v.y.mul_add(n.x.mul_add(n.x * rz, n.z), v.z * n.y));
        let z = v.x.neg_mul_add(n.x, v.y.neg_mul_add(n.y, v.z * n.z));

        let result = MVector3::new(x, y, z);

        // If we flipped the normal, flip the result too.
        result.pick(-result, normal.z)
    }

    /// Scalar multiplication and vector add using fused multiply-add.
    pub fn mul_add(self, factor: Mf32, other: MVector3) -> MVector3 {
        MVector3 {
            x: self.x.mul_add(factor, other.x),
            y: self.y.mul_add(factor, other.y),
            z: self.z.mul_add(factor, other.z),
        }
    }

    /// Scalar multiplication with -factor and vector add using fused multiply-add.
    pub fn neg_mul_add(self, factor: Mf32, other: MVector3) -> MVector3 {
        MVector3 {
            x: self.x.neg_mul_add(factor, other.x),
            y: self.y.neg_mul_add(factor, other.y),
            z: self.z.neg_mul_add(factor, other.z),
        }
    }

    /// Scalar multiplication and vector subtract using fused multiply-subtract.
    pub fn mul_sub(self, factor: Mf32, other: MVector3) -> MVector3 {
        MVector3 {
            x: self.x.mul_sub(factor, other.x),
            y: self.y.mul_sub(factor, other.y),
            z: self.z.mul_sub(factor, other.z),
        }
    }

    /// Multiplies two vectors coordinatewise.
    pub fn mul_coords(self, factors: MVector3) -> MVector3 {
        MVector3 {
            x: self.x * factors.x,
            y: self.y * factors.y,
            z: self.z * factors.z,
        }
    }

    /// Returns ||self|| * ||self||.
    pub fn norm_squared(self) -> Mf32 {
        self.dot(self)
    }

    /// Returns 1 / ||self||.
    pub fn rnorm(self) -> Mf32 {
        self.norm_squared().rsqrt()
    }

    pub fn normalized(self) -> MVector3 {
        let rnorm = self.rnorm();
        MVector3 {
            x: self.x * rnorm,
            y: self.y * rnorm,
            z: self.z * rnorm,
        }
    }

    /// Clamps every coordinate to 1.0 if it exceeds 1.0.
    pub fn clamp_one(self) -> MVector3 {
        MVector3 {
            x: Mf32::one().min(self.x),
            y: Mf32::one().min(self.y),
            z: Mf32::one().min(self.z),
        }
    }

    /// Picks self if the sign bit of mask is 0, or picks other if it is 1.
    pub fn pick(self, other: MVector3, mask: Mask) -> MVector3 {
        MVector3 {
            x: self.x.pick(other.x, mask),
            y: self.y.pick(other.y, mask),
            z: self.z.pick(other.z, mask),
        }
    }

    /// Returns whether all components are finite.
    ///
    /// This is slow, use only for diagnostic purposes.
    pub fn all_finite(self) -> bool {
        self.x.all_finite() && self.y.all_finite() && self.z.all_finite()
    }
}

impl Add for SVector3 {
    type Output = SVector3;

    fn add(self, other: SVector3) -> SVector3 {
        SVector3 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

impl Add for MVector3 {
    type Output = MVector3;

    fn add(self, other: MVector3) -> MVector3 {
        MVector3 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

impl Sub for SVector3 {
    type Output = SVector3;

    fn sub(self, other: SVector3) -> SVector3 {
        SVector3 {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

impl Sub for MVector3 {
    type Output = MVector3;

    fn sub(self, other: MVector3) -> MVector3 {
        MVector3 {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

impl Neg for SVector3 {
    type Output = SVector3;

    fn neg(self) -> SVector3 {
        SVector3 {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

impl Neg for MVector3 {
    type Output = MVector3;

    fn neg(self) -> MVector3 {
        MVector3 {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

impl Mul<f32> for SVector3 {
    type Output = SVector3;

    fn mul(self, a: f32) -> SVector3 {
        SVector3 {
            x: self.x * a,
            y: self.y * a,
            z: self.z * a,
        }
    }
}

impl Mul<Mf32> for MVector3 {
    type Output = MVector3;

    fn mul(self, a: Mf32) -> MVector3 {
        MVector3 {
            x: self.x * a,
            y: self.y * a,
            z: self.z * a,
        }
    }
}

impl fmt::Display for SVector3 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "({}, {}, {})", self.x, self.y, self.z)
    }
}

#[cfg(test)]
fn assert_mvectors_equal(expected: MVector3, computed: MVector3, margin: f32) {
    // Test that the vectors are equal, to within floating point inaccuracy
    // margins.
    let error = (computed - expected).norm_squared();
    assert!((Mf32::broadcast(margin * margin) - error).all_sign_bits_positive(),
            "expected: {:?}, computed: {:?}", expected, computed);
}

#[test]
fn verify_rotate_hemisphere() {
    let x = MVector3::new(Mf32::one(), Mf32::zero(), Mf32::zero());
    let y = MVector3::new(Mf32::zero(), Mf32::one(), Mf32::zero());
    let z = MVector3::new(Mf32::zero(), Mf32::zero(), Mf32::one());

    // If we rotate z -> y, then a vector along x does not change.
    assert_mvectors_equal(x, x.rotate_hemisphere(y), 1e-3);

    // Same for z -> x, then y does not change.
    assert_mvectors_equal(y, y.rotate_hemisphere(x), 1e-3);

    // If we rotate z -> y about the x-axis, then y rotates to -z.
    assert_mvectors_equal(-z, y.rotate_hemisphere(y), 1e-3);

    // If we rotate z -> x about the y-axis, then x rotates to -z.
    assert_mvectors_equal(-z, x.rotate_hemisphere(x), 1e-3);

    // A starting normal of positive z is assumed, so picking that should not
    // change anything.
    assert_mvectors_equal(x, x.rotate_hemisphere(z), 1e-3);
    assert_mvectors_equal(y, y.rotate_hemisphere(z), 1e-3);
}

#[test]
fn rotate_hemisphere_is_orthogonal() {
    let ns = bench::mvectors_on_unit_sphere(4096);
    let xs = bench::mvectors_on_unit_sphere(4096);
    let ys = bench::mvectors_on_unit_sphere(4096);

    for (&n, (&x, &y)) in ns.iter().zip(xs.iter().zip(ys.iter())) {
        let sum_before = x + y;
        let sum_mapped = sum_before.rotate_hemisphere(n);
        let n2_before = sum_before.norm_squared();
        let n2_after = sum_mapped.norm_squared();

        // An orthogonal map does not change the length of vectors.
        let error = (n2_before - n2_after) * (n2_before - n2_after);
        assert!((Mf32::broadcast(1e-5) - error).all_sign_bits_positive(),
            "expected equal norm, got {:?} and {:?}", n2_before, n2_after);

        // Also, an orthogonal map is linear.
        let sum_after = x.rotate_hemisphere(n) + y.rotate_hemisphere(n);
        assert_mvectors_equal(sum_mapped, sum_after, 1e-3);
    }
}

#[test]
fn rotate_hemisphere_extrema() {
    let x = MVector3::new(Mf32::one(), Mf32::zero(), Mf32::zero());
    let y = MVector3::new(Mf32::zero(), Mf32::one(), Mf32::zero());
    let z = MVector3::new(Mf32::zero(), Mf32::zero(), Mf32::one());

    // The rotation code breaks down at a normal vector -z due to division by
    // zero. This edge case should be handled correctly.

    // If we rotate z -> -z, then a vector along x flips sign.
    assert_eq!(x.rotate_hemisphere(-z), -x);

    // A vector along y should change sign too, because the hemisphere is
    // flipped if the z-component of the normal is negative.
    assert_eq!(y.rotate_hemisphere(-z), -y);

    // The z-axis itself should just rotate along.
    assert_eq!(z.rotate_hemisphere(-z), -z);
}

#[test]
fn rotate_hemisphere_random() {
    use random::Rng;
    let x = MVector3::new(Mf32::one(), Mf32::zero(), Mf32::zero());
    let epsilon = Mf32::broadcast(0.0001);
    let mut rng = Rng::with_seed(1, 2, 3);
    let mut had_negative_y = false;
    let mut had_positive_y = false;
    let mut had_negative_z = false;
    let mut had_positive_z = false;
    for _ in 0..4096 {
        let v = rng.sample_hemisphere_vector();
        let w = v.rotate_hemisphere(x);
        assert!((w.x + epsilon).all_sign_bits_positive(),
            "when rotating {:?} to the positive x-axis, no x-coordinate should be negative, \
             but the result is {:?}", v, w);

        // After rotation, (y, z) should lie on a circle, and every sign should
        // occur for these two coordinates.
        had_negative_y = had_negative_y || !(epsilon + w.y).all_sign_bits_positive();
        had_positive_y = had_positive_y || !(epsilon - w.y).all_sign_bits_positive();
        had_negative_z = had_negative_z || !(epsilon + w.z).all_sign_bits_positive();
        had_positive_z = had_positive_z || !(epsilon - w.z).all_sign_bits_positive();
    }
    assert!(had_negative_y);
    assert!(had_positive_y);
    assert!(had_negative_z);
    assert!(had_positive_z);
}

macro_rules! unroll_10 {
    { $x: block } => {
        $x $x $x $x $x $x $x $x $x $x
    }
}

#[bench]
fn bench_scross_naive_1000(bencher: &mut test::Bencher) {
    let vectors = bench::svector3_pairs(4096);
    let mut vectors_it = vectors.iter().cycle();
    bencher.iter(|| {
        let &(a, b) = vectors_it.next().unwrap();
        for _ in 0..100 {
            unroll_10! {{
                test::black_box(test::black_box(a).cross_naive(test::black_box(b)));
            }};
        }
    });
}

#[bench]
fn bench_scross_fma_1000(bencher: &mut test::Bencher) {
    let vectors = bench::svector3_pairs(4096);
    let mut vectors_it = vectors.iter().cycle();
    bencher.iter(|| {
        let &(a, b) = vectors_it.next().unwrap();
        for _ in 0..100 {
            unroll_10! {{
                test::black_box(test::black_box(a).cross_fma(test::black_box(b)));
            }};
        }
    });
}

#[bench]
fn bench_mcross_naive_1000(bencher: &mut test::Bencher) {
    let vectors = bench::mvector3_pairs(4096 / 8);
    let mut vectors_it = vectors.iter().cycle();
    bencher.iter(|| {
        let &(a, b) = vectors_it.next().unwrap();
        for _ in 0..100 {
            unroll_10! {{
                test::black_box(test::black_box(a).cross_naive(test::black_box(b)));
            }};
        }
    });
}

#[bench]
fn bench_mcross_fma_1000(bencher: &mut test::Bencher) {
    let vectors = bench::mvector3_pairs(4096 / 8);
    let mut vectors_it = vectors.iter().cycle();
    bencher.iter(|| {
        let &(a, b) = vectors_it.next().unwrap();
        for _ in 0..100 {
            unroll_10! {{
                test::black_box(test::black_box(a).cross_fma(test::black_box(b)));
            }};
        }
    });
}

#[bench]
fn bench_sdot_naive_1000(bencher: &mut test::Bencher) {
    let vectors = bench::svector3_pairs(4096);
    let mut vectors_it = vectors.iter().cycle();
    bencher.iter(|| {
        let &(a, b) = vectors_it.next().unwrap();
        for _ in 0..100 {
            unroll_10! {{
                test::black_box(test::black_box(a).dot_naive(test::black_box(b)));
            }};
        }
    });
}

#[bench]
fn bench_sdot_fma_1000(bencher: &mut test::Bencher) {
    let vectors = bench::svector3_pairs(4096);
    let mut vectors_it = vectors.iter().cycle();
    bencher.iter(|| {
        let &(a, b) = vectors_it.next().unwrap();
        for _ in 0..100 {
            unroll_10! {{
                test::black_box(test::black_box(a).dot_fma(test::black_box(b)));
            }};
        }
    });
}

#[bench]
fn bench_mdot_naive_1000(bencher: &mut test::Bencher) {
    let vectors = bench::mvector3_pairs(4096 / 8);
    let mut vectors_it = vectors.iter().cycle();
    bencher.iter(|| {
        let &(a, b) = vectors_it.next().unwrap();
        for _ in 0..100 {
            unroll_10! {{
                test::black_box(test::black_box(a).dot_naive(test::black_box(b)));
            }};
        }
    });
}

#[bench]
fn bench_mdot_fma_1000(bencher: &mut test::Bencher) {
    let vectors = bench::mvector3_pairs(4096 / 8);
    let mut vectors_it = vectors.iter().cycle();
    bencher.iter(|| {
        let &(a, b) = vectors_it.next().unwrap();
        for _ in 0..100 {
            unroll_10! {{
                test::black_box(test::black_box(a).dot_fma(test::black_box(b)));
            }};
        }
    });
}

#[bench]
fn bench_rotate_hemisphere_1000(bencher: &mut test::Bencher) {
    let vectors = bench::mvector3_pairs(4096 / 8);
    let mut vectors_it = vectors.iter().cycle();
    bencher.iter(|| {
        let &(v, n) = vectors_it.next().unwrap();
        for _ in 0..100 {
            unroll_10! {{
                test::black_box(test::black_box(v).rotate_hemisphere(test::black_box(n)));
            }};
        }
    });
}
