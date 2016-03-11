use bvh::Bvh;
use ray::{Intersection, OctaIntersection, OctaRay, Ray};
use simd::OctaF32;
use std::f32::consts::PI;
use vector3::{OctaVector3, Vector3};
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

    /// Returns a camera ray for the given screen coordinates.
    ///
    /// Values for x are in the range (-1, 1), the scale is uniform in both
    /// directions.
    pub fn get_octa_ray(&self, x: OctaF32, y: OctaF32) -> OctaRay {
        let dist = OctaF32::broadcast(-self.screen_distance);
        let origin = OctaVector3::broadcast(self.position);
        let direction = OctaVector3::new(x, y, dist).normalized();
        // TODO: Transform direction with orientation quaternion.
        OctaRay {
            origin: origin,
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

impl Scene {
    pub fn from_mesh(mesh: &Mesh) -> Scene {
        Scene {
            bvh: Bvh::from_mesh(mesh),
            lights: Vec::new(),
            camera: Camera::with_fov(PI * 0.6),
        }
    }

    pub fn intersect(&self, ray: &Ray) -> Option<Intersection> {
        self.bvh.intersect_nearest(ray)
    }

    pub fn intersect_nearest(&self, octa_ray: &OctaRay) -> OctaIntersection {
        let huge_distance = OctaF32::broadcast(1.0e5);
        let far_away = OctaIntersection {
            position: octa_ray.direction.mul_add(huge_distance, octa_ray.origin),
            normal: octa_ray.direction,
            distance: huge_distance,
            // TODO: Set sky/far away material.
        };
        self.bvh.intersect_nearest_octa(octa_ray, far_away)
    }
}
