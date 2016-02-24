//! This module generates test data for the benchmarks.

use rand;
use rand::distributions::{IndependentSample, Range};
use ray::Ray;
use std::f32::consts;
use vector3::Vector3;

/// Generates n vectors distributed uniformly on the unit sphere.
fn points_on_sphere(n: usize) -> Vec<Vector3> {
    let mut rng = rand::thread_rng();
    let phi_range = Range::new(0.0, 2.0 * consts::PI);
    let cos_theta_range = Range::new(-1.0_f32, 1.0);
    let mut vectors = Vec::with_capacity(n);
    for _ in 0..n {
        let phi = phi_range.ind_sample(&mut rng);
        let theta = cos_theta_range.ind_sample(&mut rng).acos();
        let vector = Vector3 {
            x: phi.cos() * theta.sin(),
            y: phi.sin() * theta.sin(),
            z: theta.cos(),
        };
        vectors.push(vector);
    }
    vectors
}

/// Generates rays with origin on a sphere, pointing to the origin.
fn rays_inward(radius: f32, n: usize) -> Vec<Ray> {
    points_on_sphere(n).iter().map(|&x| Ray::new(x * radius, -x)).collect()
}
