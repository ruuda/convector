use bvh::Bvh;
use ray::{OctaIntersection, OctaRay};
use simd::Mf32;
use std::f32::consts::PI;
use vector3::{OctaVector3, SVector3};
use wavefront::Mesh;

pub struct Camera {
    pub position: SVector3,
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
            position: SVector3::zero(),
            screen_distance: 1.0 / (fov / 2.0).sin(),
        }
    }

    /// Returns a camera ray for the given screen coordinates.
    ///
    /// Values for x are in the range (-1, 1), the scale is uniform in both
    /// directions.
    pub fn get_octa_ray(&self, x: Mf32, y: Mf32) -> OctaRay {
        let dist = Mf32::broadcast(-self.screen_distance);
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
    pub position: SVector3,
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

    pub fn intersect_nearest(&self, octa_ray: &OctaRay) -> OctaIntersection {
        let huge_distance = Mf32::broadcast(1.0e5);
        let far_away = OctaIntersection {
            position: octa_ray.direction.mul_add(huge_distance, octa_ray.origin),
            normal: octa_ray.direction,
            distance: huge_distance,
            // TODO: Set sky/far away material.
        };
        self.bvh.intersect_nearest_octa(octa_ray, far_away)
    }
}
