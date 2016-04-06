use bvh::Bvh;
use material::MMaterial;
use quaternion::{MQuaternion, SQuaternion, rotate};
use ray::{MIntersection, MRay};
use simd::Mf32;
use std::f32::consts::PI;
use vector3::{MVector3, SVector3};
use wavefront::Mesh;

pub struct Camera {
    pub position: SVector3,

    /// Distance such that a vector at `(1, 0, screen_distance)` makes an angle
    /// of the desired field of view with `(-1, 0, screen_distance)`.
    screen_distance: f32,

    orientation: SQuaternion,
}

impl Camera {
    /// Creates a camera at the origin with 60 degrees field of view.
    pub fn new() -> Camera {
        Camera {
            position: SVector3::zero(),
            screen_distance: 1.0 / (PI / 6.0).sin(),
            orientation: SQuaternion::new(1.0, 0.0, 0.0, 0.0),
        }
    }

    /// Sets the desired horizontal field of view in radians.
    pub fn set_fov(&mut self, fov: f32) {
        self.screen_distance = 1.0 / (fov / 2.0).sin();
    }

    /// Sets the rotation of the camera in the xz-plane.
    pub fn set_rotation(&mut self, radians: f32) {
        let x = (radians * 0.5).cos();
        let y = (radians * 0.5).sin();
        self.orientation = SQuaternion::new(x, 0.0, -y, 0.0);
    }

    /// Returns a camera ray for the given screen coordinates.
    ///
    /// Values for x are in the range (-1, 1), the scale is uniform in both
    /// directions.
    pub fn get_ray(&self, x: Mf32, y: Mf32) -> MRay {
        let dist = Mf32::broadcast(-self.screen_distance);
        let origin = MVector3::broadcast(self.position);
        let dir_src = MVector3::new(x, y, dist).normalized();
        let dir = rotate(&dir_src, &MQuaternion::broadcast(self.orientation));

        MRay {
            origin: origin,
            direction: dir,
            active: Mf32::zero(),
        }
    }
}

pub struct Light {
    pub position: SVector3,

    /// Power for the red, green, and blue components.
    pub power: SVector3,
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
            camera: Camera::new(),
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
            material: MMaterial::sky(),
            tex_coords: (Mf32::zero(), Mf32::zero()),
        };
        self.bvh.intersect_nearest(ray, far_away)
    }

    /// Returns the number of AABBs and triangles intersected to find the
    /// nearest intersection.
    pub fn intersect_debug(&self, ray: &MRay) -> (u32, u32) {
        let huge_distance = Mf32::broadcast(1.0e5);
        let far_away = MIntersection {
            position: ray.direction.mul_add(huge_distance, ray.origin),
            normal: ray.direction,
            distance: huge_distance,
            material: MMaterial::sky(),
            tex_coords: (Mf32::zero(), Mf32::zero()),
        };
        self.bvh.intersect_debug(ray, far_away)
    }
}
