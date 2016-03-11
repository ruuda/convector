//! This module generates test data for the benchmarks.

use aabb::Aabb;
use rand;
use rand::Rng;
use rand::distributions::{IndependentSample, Range};
use ray::{OctaRay, Ray};
use simd::OctaF32;
use std::f32::consts;
use vector3::{OctaVector3, SVector3};

/// Generates n random OctaF32s in the range [-1, 1).
pub fn octa_biunit(n: usize) -> Vec<OctaF32> {
    let mut vectors = Vec::with_capacity(n);
    let mut rng = rand::thread_rng();
    let range = Range::new(-1.0, 1.0);
    for _ in 0..n {
        vectors.push(OctaF32::generate(|_| range.ind_sample(&mut rng)));
    }
    vectors
}

/// Generates n vectors distributed uniformly on the unit sphere.
pub fn points_on_sphere(n: usize) -> Vec<SVector3> {
    let mut rng = rand::thread_rng();
    let phi_range = Range::new(0.0, 2.0 * consts::PI);
    let cos_theta_range = Range::new(-1.0_f32, 1.0);
    let mut vectors = Vec::with_capacity(n);
    for _ in 0..n {
        let phi = phi_range.ind_sample(&mut rng);
        let theta = cos_theta_range.ind_sample(&mut rng).acos();
        let vector = SVector3 {
            x: phi.cos() * theta.sin(),
            y: phi.sin() * theta.sin(),
            z: theta.cos(),
        };
        vectors.push(vector);
    }
    vectors
}

/// Generates n times 8 vectors distributed uniformly on the unit sphere.
pub fn octa_points_on_sphere(n: usize) -> Vec<OctaVector3> {
    let mut vectors = Vec::with_capacity(n);
    for _ in 0..n {
        let p = points_on_sphere(8);
        let x = OctaF32::generate(|i| p[i].x);
        let y = OctaF32::generate(|i| p[i].y);
        let z = OctaF32::generate(|i| p[i].z);
        vectors.push(OctaVector3::new(x, y, z));
    }
    vectors
}

/// Generates n pairs of nonzero vectors.
pub fn svector3_pairs(n: usize) -> Vec<(SVector3, SVector3)> {
    let mut a = points_on_sphere(n);
    let mut b = points_on_sphere(n);
    let pairs = a.drain(..).zip(b.drain(..)).collect();
    pairs
}

/// Generates n times 8 pairs of nonzero vectors.
pub fn octa_vector3_pairs(n: usize) -> Vec<(OctaVector3, OctaVector3)> {
    let mut a = octa_points_on_sphere(n);
    let mut b = octa_points_on_sphere(n);
    let pairs = a.drain(..).zip(b.drain(..)).collect();
    pairs
}

/// Generates rays with origin on a sphere, pointing to the origin.
pub fn rays_inward(radius: f32, n: usize) -> Vec<Ray> {
    points_on_sphere(n).iter().map(|&x| Ray::new(x * radius, -x)).collect()
}

/// Generates a random AABB and n rays of which m intersect the box.
pub fn aabb_with_rays(n: usize, m: usize) -> (Aabb, Vec<Ray>) {
    let origin = SVector3::new(-1.0, -1.0, -1.0);
    let size = SVector3::new(2.0, 2.0, 2.0);
    let aabb = Aabb::new(origin, size);
    let up = SVector3::new(0.0, 0.0, 1.0);
    let mut rays = rays_inward(16.0, n);

    // Offset the m-n rays that should not intersect the box in a direction
    // perpendicular to the ray.
    for i in m..n {
        let p = rays[i].origin + up.cross(rays[i].direction).normalized() * 16.0;
        rays[i].origin = p;
    }

    // Shuffle the intersecting and non-intersecting rays to confuse the branch
    // predictor.
    rand::thread_rng().shuffle(&mut rays[..]);

    (aabb, rays)
}

/// Generates a random AABB and n rays of which m intersect the box,
/// packed per 8 rays. N must be a multiple of 8.
pub fn aabb_with_octa_rays(n: usize, m: usize) -> (Aabb, Vec<OctaRay>) {
    assert_eq!(0, n & 7); // Must be a multiple of 8.
    let (aabb, rays) = aabb_with_rays(n, m);
    let octa_rays = rays.chunks(8)
                        .map(|rs| OctaRay::generate(|i| rs[i].clone()))
                        .collect();
    (aabb, octa_rays)
}

#[test]
fn aabb_with_rays_respects_probability() {
    let (aabb, rays) = aabb_with_rays(4096, 2048);
    let mut n = 0;
    for ray in &rays {
        if aabb.intersect(ray) {
            n += 1;
        }
    }
    assert_eq!(2048, n);
}
