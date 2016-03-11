//! Implements vectors in R3.

use simd::{Mask, Mf32};
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

#[derive(Copy, Clone)]
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

    pub fn norm(self) -> f32 {
        self.norm_squared().sqrt()
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

    pub fn reflect(self, normal: SVector3) -> SVector3 {
        self - normal * 2.0 * normal.dot(self)
    }

    pub fn get_coord(self, axis: Axis) -> f32 {
        match axis {
            Axis::X => self.x,
            Axis::Y => self.y,
            Axis::Z => self.z,
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

    #[inline(always)]
    pub fn mul_add_naive(self, factor: Mf32, other: MVector3) -> MVector3 {
        self * factor + other
    }

    #[inline(always)]
    pub fn mul_add_fma(self, factor: Mf32, other: MVector3) -> MVector3 {
        MVector3 {
            x: self.x.mul_add(factor, other.x),
            y: self.y.mul_add(factor, other.y),
            z: self.z.mul_add(factor, other.z),
        }
    }

    /// Scalar multiplication and vector add using fused multiply-add.
    pub fn mul_add(self, factor: Mf32, other: MVector3) -> MVector3 {
        self.mul_add_fma(factor, other)
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

// These benchmarks all measure ten operations per iteration, because the
// benchmark framework reports times in nanoseconds, which is too coarse for
// these operations.

#[bench]
fn bench_scross_naive_10(bencher: &mut test::Bencher) {
    let vectors = bench::svector3_pairs(4096);
    let mut vectors_it = vectors.iter().cycle();
    bencher.iter(|| {
        for _ in 0..10 {
            let &(a, b) = vectors_it.next().unwrap();
            test::black_box(a.cross_naive(b));
        }
    });
}

#[bench]
fn bench_scross_fma_10(bencher: &mut test::Bencher) {
    let vectors = bench::svector3_pairs(4096);
    let mut vectors_it = vectors.iter().cycle();
    bencher.iter(|| {
        for _ in 0..10 {
            let &(a, b) = vectors_it.next().unwrap();
            test::black_box(a.cross_fma(b));
        }
    });
}

#[bench]
fn bench_mcross_naive_10(bencher: &mut test::Bencher) {
    let vectors = bench::mvector3_pairs(4096 / 8);
    let mut vectors_it = vectors.iter().cycle();
    bencher.iter(|| {
        for _ in 0..10 {
            let &(a, b) = vectors_it.next().unwrap();
            test::black_box(a.cross_naive(b));
        }
    });
}

#[bench]
fn bench_mcross_fma_10(bencher: &mut test::Bencher) {
    let vectors = bench::mvector3_pairs(4096 / 8);
    let mut vectors_it = vectors.iter().cycle();
    bencher.iter(|| {
        for _ in 0..10 {
            let &(a, b) = vectors_it.next().unwrap();
            test::black_box(a.cross_fma(b));
        }
    });
}

#[bench]
fn bench_sdot_naive_10(bencher: &mut test::Bencher) {
    let vectors = bench::svector3_pairs(4096);
    let mut vectors_it = vectors.iter().cycle();
    bencher.iter(|| {
        for _ in 0..10 {
            let &(a, b) = vectors_it.next().unwrap();
            test::black_box(a.dot_naive(b));
        }
    });
}

#[bench]
fn bench_sdot_fma_10(bencher: &mut test::Bencher) {
    let vectors = bench::svector3_pairs(4096);
    let mut vectors_it = vectors.iter().cycle();
    bencher.iter(|| {
        for _ in 0..10 {
            let &(a, b) = vectors_it.next().unwrap();
            test::black_box(a.dot_fma(b));
        }
    });
}

#[bench]
fn bench_mdot_naive_10(bencher: &mut test::Bencher) {
    let vectors = bench::mvector3_pairs(4096 / 8);
    let mut vectors_it = vectors.iter().cycle();
    bencher.iter(|| {
        for _ in 0..10 {
            let &(a, b) = vectors_it.next().unwrap();
            test::black_box(a.dot_naive(b));
        }
    });
}

#[bench]
fn bench_mdot_fma_10(bencher: &mut test::Bencher) {
    let vectors = bench::mvector3_pairs(4096 / 8);
    let mut vectors_it = vectors.iter().cycle();
    bencher.iter(|| {
        for _ in 0..10 {
            let &(a, b) = vectors_it.next().unwrap();
            test::black_box(a.dot_fma(b));
        }
    });
}
