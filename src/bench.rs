//! This module generates test data for the benchmarks.

use aabb::Aabb;
use rand;
use rand::Rng;
use rand::distributions::{IndependentSample, Range};
use ray::{MRay, SRay};
use simd::Mf32;
use std::f32::consts;
use triangle::Triangle;
use vector3::{MVector3, SVector3};

/// Generates n random Mf32s in the range [-1, 1).
pub fn mf32_biunit(n: usize) -> Vec<Mf32> {
    let mut vectors = Vec::with_capacity(n);
    let mut rng = rand::thread_rng();
    let range = Range::new(-1.0, 1.0);
    for _ in 0..n {
        vectors.push(Mf32::generate(|_| range.ind_sample(&mut rng)));
    }
    vectors
}

/// Generates n vectors distributed uniformly on the unit sphere.
pub fn points_on_sphere_s(n: usize) -> Vec<SVector3> {
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
pub fn points_on_sphere_m(n: usize) -> Vec<MVector3> {
    let mut vectors = Vec::with_capacity(n);
    for _ in 0..n {
        let p = points_on_sphere_s(8);
        let x = Mf32::generate(|i| p[i].x);
        let y = Mf32::generate(|i| p[i].y);
        let z = Mf32::generate(|i| p[i].z);
        vectors.push(MVector3::new(x, y, z));
    }
    vectors
}

/// Generates n pairs of nonzero vectors.
pub fn svector3_pairs(n: usize) -> Vec<(SVector3, SVector3)> {
    let mut a = points_on_sphere_s(n);
    let mut b = points_on_sphere_s(n);
    let pairs = a.drain(..).zip(b.drain(..)).collect();
    pairs
}

/// Generates n times 8 pairs of nonzero vectors.
pub fn mvector3_pairs(n: usize) -> Vec<(MVector3, MVector3)> {
    let mut a = points_on_sphere_m(n);
    let mut b = points_on_sphere_m(n);
    let pairs = a.drain(..).zip(b.drain(..)).collect();
    pairs
}

/// Generates rays with origin on a sphere, pointing to the origin.
pub fn srays_inward(radius: f32, n: usize) -> Vec<SRay> {
    points_on_sphere_s(n).iter().map(|&x| SRay::new(x * radius, -x)).collect()
}

/// Generates a random AABB and n rays of which m intersect the box.
pub fn aabb_with_srays(n: usize, m: usize) -> (Aabb, Vec<SRay>) {
    let origin = SVector3::new(-1.0, -1.0, -1.0);
    let far = SVector3::new(1.0, 1.0, 1.0);
    let aabb = Aabb::new(origin, far);
    let up = SVector3::new(0.0, 0.0, 1.0);
    let mut rays = srays_inward(16.0, n);

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
pub fn aabb_with_mrays(n: usize, m: usize) -> (Aabb, Vec<MRay>) {
    assert_eq!(0, n & 7); // Must be a multiple of 8.
    let (aabb, srays) = aabb_with_srays(n, m);
    let mrays = srays.chunks(8)
                     .map(|rs| MRay::generate(|i| rs[i].clone()))
                     .collect();
    (aabb, mrays)
}

/// Generates n triangles with vertices on the unit sphere.
pub fn triangles(n: usize) -> Vec<Triangle> {
    let v0s = points_on_sphere_s(n);
    let v1s = points_on_sphere_s(n);
    let v2s = points_on_sphere_s(n);
    v0s.iter()
       .zip(v1s.iter().zip(v2s.iter()))
       .map(|(&v0, (&v1, &v2))| Triangle::new(v0, v1, v2))
       .collect()
}

/// Generates n bounding boxes with two vertices on the unit sphere.
pub fn aabbs(n: usize) -> Vec<Aabb> {
    let v0s = points_on_sphere_s(n);
    let v1s = points_on_sphere_s(n);
    v0s.iter()
       .zip(v1s.iter())
       .map(|(&v0, &v1)| Aabb::new(SVector3::min(v0, v1), SVector3::max(v0, v1)))
       .collect()
}

/// Generates n mrays originating from a sphere of radius 10, pointing inward.
pub fn mrays_inward(n: usize) -> Vec<MRay> {
    let origins = points_on_sphere_m(n);
    let dests = points_on_sphere_m(n);
    origins.iter()
           .zip(dests.iter())
           .map(|(&from, &to)| {
               let origin = from * Mf32::broadcast(10.0);
               let direction = (to - origin).normalized();
               MRay::new(origin, direction)
           })
           .collect()
}

/// Generates n mrays originating from a sphere of radius 10, pointing inward.
/// The rays share the origin and point roughly in the same direction.
pub fn mrays_inward_coherent(n: usize) -> Vec<MRay> {
    let origins = points_on_sphere_s(n);
    let dests = points_on_sphere_m(n);
    origins.iter()
           .zip(dests.iter())
           .map(|(&from, &to)| {
               let origin = MVector3::broadcast(from * 10.0);
               let dest = to * Mf32::broadcast(0.5);
               let direction = (dest - origin).normalized();
               MRay::new(origin, direction)
           })
           .collect()
}

#[test]
fn aabb_with_srays_respects_probability() {
    let (aabb, rays) = aabb_with_srays(4096, 2048);
    let mut n = 0;
    for ray in &rays {
        let mray = MRay::broadcast(ray);
        if aabb.intersect(&mray).any() {
            n += 1;
        }
    }
    assert_eq!(2048, n);
}
