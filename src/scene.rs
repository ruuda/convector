use bvh::Bvh;
use ray::{MIntersection, MRay};
use simd::{Mask, Mf32};
use std::f32::consts::PI;
use vector3::{MVector3, SVector3};
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
    pub fn get_ray(&self, x: Mf32, y: Mf32) -> MRay {
        let dist = Mf32::broadcast(-self.screen_distance);
        let origin = MVector3::broadcast(self.position);
        let direction = MVector3::new(x, y, dist).normalized();
        // TODO: Transform direction with orientation quaternion.
        MRay {
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
    pub fn from_meshes(meshes: &[Mesh]) -> Scene {
        Scene {
            bvh: Bvh::from_meshes(meshes),
            lights: Vec::new(),
            camera: Camera::with_fov(PI * 0.3),
        }
    }

    /// Returns the interections with the shortest distance along the ray.
    ///
    /// Intersects the sky if no other geometry was intersected.
    pub fn intersect_nearest(&self, ray: &MRay) -> MIntersection {
        let huge_distance = Mf32::broadcast(1.0e5);
        let far_away = MIntersection {
            position: ray.direction.mul_add(huge_distance, ray.origin),
            normal: ray.direction,
            distance: huge_distance,
            // TODO: Set sky/far away material.
        };
        self.bvh.intersect_nearest(ray, far_away)
    }

    /// Returns whether there is any geometry along the ray.
    ///
    /// This is intended for occlusion testing. The exact location of the
    /// intersection is not computed.
    pub fn intersect_any(&self, ray: &MRay, max_dist: Mf32) -> Mask {
        self.bvh.intersect_any(ray, max_dist)
    }
}
