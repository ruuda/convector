use bvh::Bvh;
use material::MMaterial;
use quaternion::{MQuaternion, SQuaternion, rotate};
use ray::{MIntersection, MRay};
use simd::Mf32;
use std::f32::consts::PI;
use vector3::{MVector3, SVector3};
use wavefront::Mesh;

pub struct Camera {
    position: SVector3,
    position_delta: SVector3,

    orientation: SQuaternion,
    orientation_delta: SQuaternion,

    /// Distance such that a vector at `(1, 0, screen_distance)` makes an angle
    /// of the desired field of view with `(-1, 0, screen_distance)`.
    screen_distance: f32,
}

impl Camera {
    /// Creates a camera at the origin with 60 degrees field of view.
    pub fn new() -> Camera {
        Camera {
            position: SVector3::zero(),
            position_delta: SVector3::zero(),
            orientation: SQuaternion::new(1.0, 0.0, 0.0, 0.0),
            orientation_delta: SQuaternion::new(0.0, 0.0, 0.0, 0.0),
            screen_distance: 1.0 / (PI / 6.0).sin(),
        }
    }

    /// Sets the position of the camera at the beginning of the frame, and the
    /// offset such that position + delta is the position at the end of the
    /// frame.
    pub fn set_position(&mut self, position: SVector3, delta: SVector3) {
        self.position = position;
        self.position_delta = delta;
    }

    /// Sets the orientation of the camera at the beginning of the frame, and
    /// the delta such that orientation + delta normalized is the orientation at
    /// the end of the frame.
    pub fn set_orientation(&mut self, orientation: SQuaternion, delta: SQuaternion) {
        self.orientation = orientation;
        self.orientation_delta = delta;
    }

    /// Sets the desired horizontal field of view in radians.
    pub fn set_fov(&mut self, fov: f32) {
        self.screen_distance = 1.0 / (fov / 2.0).sin();
    }

    /// Sets the rotation of the camera in the xz-plane.
    pub fn set_rotation(&mut self, radians: f32, delta: f32) {
        let x = (radians * 0.5).cos();
        let y = (radians * 0.5).sin();
        self.orientation = SQuaternion::new(x, 0.0, -y, 0.0);

        let x_delta = 0.5 * -(radians * 0.5).sin() * delta;
        let y_delta = 0.5 * (radians * 0.5).cos() * delta;
        self.orientation_delta = SQuaternion::new(x_delta, 0.0, -y_delta, 0.0);
    }

    /// Returns a camera ray for the given screen coordinates.
    ///
    /// Values for x are in the range (-1, 1), the scale is uniform in both
    /// directions. The time ranges from 0.0 at the beginning of the frame to
    /// 1.0 at the end of the frame.
    pub fn get_ray(&self, x: Mf32, y: Mf32, t: Mf32) -> MRay {
        let origin = MVector3::broadcast(self.position);
        let origin_delta = MVector3::broadcast(self.position_delta);
        let origin = origin_delta.mul_add(t, origin);

        let orientation = MQuaternion::broadcast(self.orientation);
        let orientation_delta = MQuaternion::broadcast(self.orientation_delta);
        let orientation = orientation.interpolate(&orientation_delta, t);

        let dist = Mf32::broadcast(-self.screen_distance);
        let dir_src = MVector3::new(x, y, dist).normalized();
        let dir = rotate(&dir_src, &orientation);

        MRay {
            origin: origin,
            direction: dir,
            active: Mf32::zero(),
        }
    }
}

pub struct Scene {
    pub bvh: Bvh,
    pub camera: Camera,
}

impl Scene {
    pub fn from_meshes(meshes: &[Mesh]) -> Scene {
        Scene {
            bvh: Bvh::from_meshes(meshes),
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
