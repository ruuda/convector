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

#[inline(always)]
pub fn cross_naive(a: Vector3, b: Vector3) -> Vector3 {
    Vector3 {
        x: a.y * b.z - a.z * b.y,
        y: a.z * b.x - a.x * b.z,
        z: a.x * b.y - a.y * b.x,
    }
}

#[inline(always)]
pub fn cross_fma(a: Vector3, b: Vector3) -> Vector3 {
    Vector3 {
        x: a.y.mul_add(b.z, -a.z * b.y),
        y: a.z.mul_add(b.x, -a.x * b.z),
        z: a.x.mul_add(b.y, -a.y * b.x),
    }
}

pub fn cross(a: Vector3, b: Vector3) -> Vector3 {
    // Benchmarks show that the naive version is faster than the
    // FMA version (2 ns vs 6 ns on my Skylake i7).
    cross_naive(a, b)
}

#[inline(always)]
pub fn dot_naive(a: Vector3, b: Vector3) -> f32 {
    a.x * b.x + a.y * b.y + a.z * b.z
}

#[inline(always)]
pub fn dot_fma(a: Vector3, b: Vector3) -> f32 {
    a.x.mul_add(b.x, a.y.mul_add(b.y, a.z * b.z))
}

pub fn dot(a: Vector3, b: Vector3) -> f32 {
    // Benchmarks show that the naive version is faster than the
    // FMA version (1 ns vs 4 ns on my Skylake i7). This makes sense:
    // the naive version has less data dependencies.
    dot_naive(a, b)
}

impl Vector3 {
    pub fn new(x: f32, y: f32, z: f32) -> Vector3 {
        Vector3 { x: x, y: y, z: z }
    }

    pub fn zero() -> Vector3 {
        Vector3::new(0.0, 0.0, 0.0)
    }

    pub fn norm_squared(self) -> f32 {
        dot(self, self)
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
        self - normal * 2.0 * dot(normal, self)
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

#[bench]
fn bench_cross_naive(bencher: &mut test::Bencher) {
    let vectors = bench::vector3_pairs(4096);
    let mut vectors_it = vectors.iter().cycle();
    bencher.iter(|| {
        let &(a, b) = vectors_it.next().unwrap();
        test::black_box(cross_naive(a, b));
    });
}

#[bench]
fn bench_cross_fma(bencher: &mut test::Bencher) {
    let vectors = bench::vector3_pairs(4096);
    let mut vectors_it = vectors.iter().cycle();
    bencher.iter(|| {
        let &(a, b) = vectors_it.next().unwrap();
        test::black_box(cross_fma(a, b));
    });
}

#[bench]
fn bench_dot_naive(bencher: &mut test::Bencher) {
    let vectors = bench::vector3_pairs(4096);
    let mut vectors_it = vectors.iter().cycle();
    bencher.iter(|| {
        let &(a, b) = vectors_it.next().unwrap();
        test::black_box(dot_naive(a, b));
    });
}

#[bench]
fn bench_dot_fma(bencher: &mut test::Bencher) {
    let vectors = bench::vector3_pairs(4096);
    let mut vectors_it = vectors.iter().cycle();
    bencher.iter(|| {
        let &(a, b) = vectors_it.next().unwrap();
        test::black_box(dot_fma(a, b));
    });
}
