// This code is adapted from vector3 in Robigo Luculenta (of which I am the
// author), licensed under the GNU General Public License version 3.
// See https://github.com/ruud-v-a/robigo-luculenta/blob/master/src/vector3.rs

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

    pub fn octa_dot(self, other: OctaVector3) -> OctaF32 {
        // Benchmarks show no performance difference between the naive version
        // and the FMA version. Use the naive one because it is more portable.
        self.dot_naive(other)
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
