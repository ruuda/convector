//! This module generates test data for the benchmarks.

use aabb::Aabb;
use material::SMaterial;
use quaternion::{MQuaternion, SQuaternion};
use rand;
use rand::Rng;
use rand::distributions::{IndependentSample, Range};
use ray::{MRay, SRay};
use simd::Mf32;
use std::f32::consts;
use triangle::Triangle;
use vector3::{MVector3, SVector3};

/// Generates n random Mf32s in the range [0, 1).
pub fn mf32_unit(n: usize) -> Vec<Mf32> {
    let mut mf32s = Vec::with_capacity(n);
    let mut rng = rand::thread_rng();
    let range = Range::new(0.0, 1.0);
    for _ in 0..n {
        mf32s.push(Mf32::generate(|_| range.ind_sample(&mut rng)));
    }
    mf32s
}

/// Generates n random Mf32s in the range [-1, 1).
pub fn mf32_biunit(n: usize) -> Vec<Mf32> {
    let mut mf32s = Vec::with_capacity(n);
    let mut rng = rand::thread_rng();
    let range = Range::new(-1.0, 1.0);
    for _ in 0..n {
        mf32s.push(Mf32::generate(|_| range.ind_sample(&mut rng)));
    }
    mf32s
}

/// Generates n vectors distributed uniformly on the unit sphere.
pub fn svectors_on_unit_sphere(n: usize) -> Vec<SVector3> {
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
pub fn mvectors_on_unit_sphere(n: usize) -> Vec<MVector3> {
    let mut vectors = Vec::with_capacity(n);
    for _ in 0..n {
        let p = svectors_on_unit_sphere(8);
        let x = Mf32::generate(|i| p[i].x);
        let y = Mf32::generate(|i| p[i].y);
        let z = Mf32::generate(|i| p[i].z);
        vectors.push(MVector3::new(x, y, z));
    }
    vectors
}

/// Generates n quaternions uniformly distributed over the unit sphere.
pub fn unit_squaternions(n: usize) -> Vec<SQuaternion> {
    let mut rng = rand::thread_rng();
    let range = Range::new(-1.0_f32, 1.0);
    let mut quaternions = Vec::with_capacity(n);

    let mut i = 0;
    while i < n {
        let a = range.ind_sample(&mut rng);
        let b = range.ind_sample(&mut rng);
        let c = range.ind_sample(&mut rng);
        let d = range.ind_sample(&mut rng);

        // Use rejection sampling because I do not know how to sample a 4D unit
        // sphere uniformly.
        let norm_squared = a * a + b * b + c * c + d * d;
        if norm_squared > 1.0 { continue }

        let norm = norm_squared.sqrt();
        let q = SQuaternion::new(a / norm, b / norm, c / norm, d / norm);
        quaternions.push(q);

        i += 1;
    }

    quaternions
}

/// Generates n times 8 quaternions uniformly distributed over the unit sphere.
pub fn unit_mquaternions(n: usize) -> Vec<MQuaternion> {
    let mut quaternions = Vec::with_capacity(n);
    for _ in 0..n {
        let q = unit_squaternions(8);
        let a = Mf32::generate(|i| q[i].a);
        let b = Mf32::generate(|i| q[i].b);
        let c = Mf32::generate(|i| q[i].c);
        let d = Mf32::generate(|i| q[i].d);
        quaternions.push(MQuaternion::new(a, b, c, d));
    }
    quaternions
}

/// Generates n pairs of nonzero vectors.
pub fn svector3_pairs(n: usize) -> Vec<(SVector3, SVector3)> {
    let mut a = svectors_on_unit_sphere(n);
    let mut b = svectors_on_unit_sphere(n);
    let pairs = a.drain(..).zip(b.drain(..)).collect();
    pairs
}

/// Generates n times 8 pairs of nonzero vectors.
pub fn mvector3_pairs(n: usize) -> Vec<(MVector3, MVector3)> {
    let mut a = mvectors_on_unit_sphere(n);
    let mut b = mvectors_on_unit_sphere(n);
    let pairs = a.drain(..).zip(b.drain(..)).collect();
    pairs
}

/// Generates rays with origin on a sphere, pointing to the origin.
pub fn srays_inward(radius: f32, n: usize) -> Vec<SRay> {
    svectors_on_unit_sphere(n).iter().map(|&x| SRay::new(x * radius, -x)).collect()
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
    let v0s = svectors_on_unit_sphere(n);
    let v1s = svectors_on_unit_sphere(n);
    let v2s = svectors_on_unit_sphere(n);
    v0s.iter()
       .zip(v1s.iter().zip(v2s.iter()))
       .map(|(&v0, (&v1, &v2))| Triangle::new(v0, v1, v2, SMaterial::white()))
       .collect()
}

/// Generates n bounding boxes with two vertices on the unit sphere.
pub fn aabbs(n: usize) -> Vec<Aabb> {
    let v0s = svectors_on_unit_sphere(n);
    let v1s = svectors_on_unit_sphere(n);
    v0s.iter()
       .zip(v1s.iter())
       .map(|(&v0, &v1)| Aabb::new(SVector3::min(v0, v1), SVector3::max(v0, v1)))
       .collect()
}

/// Generates n mrays originating from a sphere of radius 10, pointing inward.
pub fn mrays_inward(n: usize) -> Vec<MRay> {
    let origins = mvectors_on_unit_sphere(n);
    let dests = mvectors_on_unit_sphere(n);
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
    let origins = svectors_on_unit_sphere(n);
    let dests = mvectors_on_unit_sphere(n);
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
