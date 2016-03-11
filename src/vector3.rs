//! Implements vectors in R3.

use simd::OctaF32;
use std::ops::{Add, Sub, Neg, Mul};

#[cfg(test)]
use {bench, test};

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Vector3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct OctaVector3 {
    pub x: OctaF32,
    pub y: OctaF32,
    pub z: OctaF32,
}

#[derive(Copy, Clone)]
pub enum Axis {
    X,
    Y,
    Z,
}

impl Vector3 {
    pub fn new(x: f32, y: f32, z: f32) -> Vector3 {
        Vector3 { x: x, y: y, z: z }
    }

    pub fn zero() -> Vector3 {
        Vector3::new(0.0, 0.0, 0.0)
    }

    #[inline(always)]
    pub fn cross_naive(self: Vector3, other: Vector3) -> Vector3 {
        let (a, b) = (self, other);
        Vector3 {
            x: a.y * b.z - a.z * b.y,
            y: a.z * b.x - a.x * b.z,
            z: a.x * b.y - a.y * b.x,
        }
    }

    #[inline(always)]
    pub fn cross_fma(self: Vector3, other: Vector3) -> Vector3 {
        let (a, b) = (self, other);
        Vector3 {
            x: a.y.mul_add(b.z, -a.z * b.y),
            y: a.z.mul_add(b.x, -a.x * b.z),
            z: a.x.mul_add(b.y, -a.y * b.x),
        }
    }

    pub fn cross(self, other: Vector3) -> Vector3 {
        // Benchmarks show that the FMA version is faster than the
        // naive version (1.9 ns vs 2.1 ns on my Skylake i7). **However**
        // the "fma" codegen feature must be enabled, otherwise the naive
        // version is faster.
        self.cross_fma(other)
    }

    #[inline(always)]
    pub fn dot_naive(self, other: Vector3) -> f32 {
        let (a, b) = (self, other);
        a.x * b.x + a.y * b.y + a.z * b.z
    }

    #[inline(always)]
    pub fn dot_fma(self, other: Vector3) -> f32 {
        let (a, b) = (self, other);
        a.x.mul_add(b.x, a.y.mul_add(b.y, a.z * b.z))
    }

    pub fn dot(self, other: Vector3) -> f32 {
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

    pub fn normalized(self) -> Vector3 {
        let norm_squared = self.norm_squared();
        if norm_squared == 0.0 {
            self
        } else {
            let norm = norm_squared.sqrt();
            Vector3 {
                x: self.x / norm,
                y: self.y / norm,
                z: self.z / norm,
            }
        }
    }

    pub fn reflect(self, normal: Vector3) -> Vector3 {
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

impl OctaVector3 {
    pub fn new(x: OctaF32, y: OctaF32, z: OctaF32) -> OctaVector3 {
        OctaVector3 { x: x, y: y, z: z }
    }

    pub fn zero() -> OctaVector3 {
        OctaVector3::new(OctaF32::zero(), OctaF32::zero(), OctaF32::zero())
    }

    pub fn broadcast(a: Vector3) -> OctaVector3 {
        OctaVector3 {
            x: OctaF32::broadcast(a.x),
            y: OctaF32::broadcast(a.y),
            z: OctaF32::broadcast(a.z),
        }
    }

    /// Builds a octavector by applying the function to the numbers 0..7.
    ///
    /// Note: this is essentially a transpose, avoid in hot code.
    pub fn generate<F: FnMut(usize) -> Vector3>(mut f: F) -> OctaVector3 {
        OctaVector3 {
            x: OctaF32::generate(|i| f(i).x),
            y: OctaF32::generate(|i| f(i).y),
            z: OctaF32::generate(|i| f(i).z),
        }
    }

    #[inline(always)]
    pub fn cross_naive(self, other: OctaVector3) -> OctaVector3 {
        let (a, b) = (self, other);
        OctaVector3 {
            x: a.y * b.z - a.z * b.y,
            y: a.z * b.x - a.x * b.z,
            z: a.x * b.y - a.y * b.x,
        }
    }

    #[inline(always)]
    pub fn cross_fma(self, other: OctaVector3) -> OctaVector3 {
        let (a, b) = (self, other);
        OctaVector3 {
            x: a.y.mul_sub(b.z, a.z * b.y),
            y: a.z.mul_sub(b.x, a.x * b.z),
            z: a.x.mul_sub(b.y, a.y * b.x),
        }
    }

    pub fn cross(self, other: OctaVector3) -> OctaVector3 {
        // Benchmarks show that the FMA version is faster than the
        // naive version (2.1 ns vs 2.4 ns on my Skylake i7).
        self.cross_fma(other)
    }

    #[inline(always)]
    pub fn dot_naive(self, other: OctaVector3) -> OctaF32 {
        let (a, b) = (self, other);
        a.x * b.x + a.y * b.y + a.z * b.z
    }

    #[inline(always)]
    pub fn dot_fma(self, other: OctaVector3) -> OctaF32 {
        let (a, b) = (self, other);
        a.x.mul_add(b.x, a.y.mul_add(b.y, a.z * b.z))
    }

    pub fn dot(self, other: OctaVector3) -> OctaF32 {
        // Benchmarks show no performance difference between the naive version
        // and the FMA version. Use the naive one because it is more portable.
        self.dot_naive(other)
    }

    #[inline(always)]
    pub fn mul_add_naive(self, factor: OctaF32, other: OctaVector3) -> OctaVector3 {
        self * factor + other
    }

    #[inline(always)]
    pub fn mul_add_fma(self, factor: OctaF32, other: OctaVector3) -> OctaVector3 {
        OctaVector3 {
            x: self.x.mul_add(factor, other.x),
            y: self.y.mul_add(factor, other.y),
            z: self.z.mul_add(factor, other.z),
        }
    }

    /// Scalar multiplication and vector add using fused multiply-add.
    pub fn mul_add(self, factor: OctaF32, other: OctaVector3) -> OctaVector3 {
        self.mul_add_fma(factor, other)
    }

    pub fn norm_squared(self) -> OctaF32 {
        self.dot(self)
    }

    pub fn normalized(self) -> OctaVector3 {
        let rnorm = self.norm_squared().rsqrt();
        OctaVector3 {
            x: self.x * rnorm,
            y: self.y * rnorm,
            z: self.z * rnorm,
        }
    }

    /// Clamps every coordinate to 1.0 if it exceeds 1.0.
    pub fn clamp_one(self) -> OctaVector3 {
        OctaVector3 {
            x: OctaF32::one().min(self.x),
            y: OctaF32::one().min(self.y),
            z: OctaF32::one().min(self.z),
        }
    }
}

impl Add for Vector3 {
    type Output = Vector3;

    fn add(self, other: Vector3) -> Vector3 {
        Vector3 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

impl Add for OctaVector3 {
    type Output = OctaVector3;

    fn add(self, other: OctaVector3) -> OctaVector3 {
        OctaVector3 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

impl Sub for Vector3 {
    type Output = Vector3;

    fn sub(self, other: Vector3) -> Vector3 {
        Vector3 {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

impl Sub for OctaVector3 {
    type Output = OctaVector3;

    fn sub(self, other: OctaVector3) -> OctaVector3 {
        OctaVector3 {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

impl Neg for Vector3 {
    type Output = Vector3;

    fn neg(self) -> Vector3 {
        Vector3 {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

impl Mul<f32> for Vector3 {
    type Output = Vector3;

    fn mul(self, a: f32) -> Vector3 {
        Vector3 {
            x: self.x * a,
            y: self.y * a,
            z: self.z * a,
        }
    }
}

impl Mul<OctaF32> for OctaVector3 {
    type Output = OctaVector3;

    fn mul(self, a: OctaF32) -> OctaVector3 {
        OctaVector3 {
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
fn bench_cross_naive_x10(bencher: &mut test::Bencher) {
    let vectors = bench::vector3_pairs(4096);
    let mut vectors_it = vectors.iter().cycle();
    bencher.iter(|| {
        for _ in 0..10 {
            let &(a, b) = vectors_it.next().unwrap();
            test::black_box(a.cross_naive(b));
        }
    });
}

#[bench]
fn bench_cross_fma_x10(bencher: &mut test::Bencher) {
    let vectors = bench::vector3_pairs(4096);
    let mut vectors_it = vectors.iter().cycle();
    bencher.iter(|| {
        for _ in 0..10 {
            let &(a, b) = vectors_it.next().unwrap();
            test::black_box(a.cross_fma(b));
        }
    });
}

#[bench]
fn bench_octa_cross_naive_x10(bencher: &mut test::Bencher) {
    let vectors = bench::octa_vector3_pairs(4096 / 8);
    let mut vectors_it = vectors.iter().cycle();
    bencher.iter(|| {
        for _ in 0..10 {
            let &(a, b) = vectors_it.next().unwrap();
            test::black_box(a.cross_naive(b));
        }
    });
}

#[bench]
fn bench_octa_cross_fma_x10(bencher: &mut test::Bencher) {
    let vectors = bench::octa_vector3_pairs(4096 / 8);
    let mut vectors_it = vectors.iter().cycle();
    bencher.iter(|| {
        for _ in 0..10 {
            let &(a, b) = vectors_it.next().unwrap();
            test::black_box(a.cross_fma(b));
        }
    });
}

#[bench]
fn bench_dot_naive_x10(bencher: &mut test::Bencher) {
    let vectors = bench::vector3_pairs(4096);
    let mut vectors_it = vectors.iter().cycle();
    bencher.iter(|| {
        for _ in 0..10 {
            let &(a, b) = vectors_it.next().unwrap();
            test::black_box(a.dot_naive(b));
        }
    });
}

#[bench]
fn bench_dot_fma_x10(bencher: &mut test::Bencher) {
    let vectors = bench::vector3_pairs(4096);
    let mut vectors_it = vectors.iter().cycle();
    bencher.iter(|| {
        for _ in 0..10 {
            let &(a, b) = vectors_it.next().unwrap();
            test::black_box(a.dot_fma(b));
        }
    });
}

#[bench]
fn bench_octa_dot_naive_x10(bencher: &mut test::Bencher) {
    let vectors = bench::octa_vector3_pairs(4096 / 8);
    let mut vectors_it = vectors.iter().cycle();
    bencher.iter(|| {
        for _ in 0..10 {
            let &(a, b) = vectors_it.next().unwrap();
            test::black_box(a.dot_naive(b));
        }
    });
}

#[bench]
fn bench_octa_dot_fma_x10(bencher: &mut test::Bencher) {
    let vectors = bench::octa_vector3_pairs(4096 / 8);
    let mut vectors_it = vectors.iter().cycle();
    bencher.iter(|| {
        for _ in 0..10 {
            let &(a, b) = vectors_it.next().unwrap();
            test::black_box(a.dot_fma(b));
        }
    });
}
