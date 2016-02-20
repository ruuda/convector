use bvh::Bvh;
use ray::{Intersection, Ray};
use std::f32::consts::PI;
use vector3::Vector3;
use wavefront::Mesh;

pub struct Camera {
    pub position: Vector3,
    // TODO: Add quaternion orientation.

    /// Distance such that a vector at `(1, 0, screen_distance)` makes an angle
    /// of the desired field of view with `(-1, 0, screen_distance)`.
    pub screen_distance: f32,
}

impl Camera {
    /// Creates a new camera for the desired horizontal field of view in
    /// radians.
    pub fn with_fov(fov: f32) -> Camera {
        Camera {
            position: Vector3::zero(),
            screen_distance: 1.0 / (fov / 2.0).sin(),
        }
    }

    /// Returns a camera ray for the given screen coordinate.
    ///
    /// Values for x are in the range (-1, 1), the scale is uniform in both
    /// directions.
    pub fn get_ray(&self, x: f32, y: f32) -> Ray {
        let direction = Vector3::new(x, y, -self.screen_distance).normalized();
        // TODO: Transform direction with orientation quaternion.
        Ray {
            origin: self.position,
            direction: direction,
        }
    }
}

pub struct Light {
    pub position: Vector3,
}

pub struct Scene {
    pub bvh: Bvh,
    pub lights: Vec<Light>,
    pub camera: Camera,
}

/// Returns the closest of two intersections.
fn closest(i1: Option<Intersection>, i2: Option<Intersection>) -> Option<Intersection> {
    match (i1, i2) {
        (None, None) => None,
        (None, Some(isect)) => Some(isect),
        (Some(isect), None) => Some(isect),
        (Some(isect1), Some(isect2)) => if isect1.distance < isect2.distance {
            Some(isect1)
        } else {
            Some(isect2)
        }
    }
}

impl Scene {
    pub fn from_mesh(mesh: &Mesh) -> Scene {
        Scene {
            bvh: Bvh::from_mesh(mesh),
            lights: Vec::new(),
            camera: Camera::with_fov(PI * 0.6),
        }
    }

    pub fn intersect(&self, ray: &Ray) -> Option<Intersection> {
        let mut result = None;
        {
            // TODO: Can I make this more ergonomic? Perhaps an iterator instead of a closure after
            // all?
            let result_ref = &mut result;
            self.bvh.traverse(ray, |triangle| {
                *result_ref = closest(result_ref.clone(), triangle.intersect(ray));
            });
        }
        result
    }
}
