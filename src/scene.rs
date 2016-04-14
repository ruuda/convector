use bvh::Bvh;
use material::{MDirectSample, MMaterial};
use quaternion::{MQuaternion, SQuaternion, rotate};
use random::Rng;
use ray::{MIntersection, MRay};
use simd::Mf32;
use std::f32::consts::PI;
use util::generate_slice8;
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
    pub camera: Camera,

    /// Bounding volume hierarchy of all triangles in the scene.
    bvh: Bvh,

    /// Indices into the BVH's triangle list, of triangles that have a material
    /// eligible for direct sampling.
    direct_sample: Vec<u32>,
}

impl Scene {
    pub fn from_meshes(meshes: &[Mesh]) -> Scene {
        let bvh = Bvh::from_meshes(meshes);

        let mut direct_sample = Vec::new();
        for i in 0..bvh.triangles.len() {
            if bvh.triangles[i].material.is_direct_sample() {
                direct_sample.push(i as u32);
            }
        }

        Scene {
            camera: Camera::new(),
            bvh: bvh,
            direct_sample: direct_sample,
        }
    }

    pub fn print_stats(&self) {
        self.bvh.print_stats();

        println!("scene statistics:");
        println!("  triangles eligible for direct sampling: {} / {} ({:0.1}%)",
            self.direct_sample.len(), self.bvh.triangles.len(),
            100.0 * self.direct_sample.len() as f32 / self.bvh.triangles.len() as f32);
    }

    /// Returns 8 random points on 8 random triangles eligible for direct
    /// sampling.
    pub fn get_direct_sample(&self, rng: &mut Rng) -> MDirectSample {
        let random_bits = rng.sample_u32();

        // Pick a random direct sampling triangle for every coordinate. This has
        // to be done serially, unfortunately.  Doing the full range modulo the
        // valid range introduces a slight bias towards lower indices, but the
        // u32 range is so vast in comparison with the number of direct sampling
        // triangles, that the effect is negligible.
        // TODO: Are the bounds checks a bottleneck here?
        let indices = generate_slice8(|i| random_bits[i] % self.direct_sample.len() as u32);
        let tri_indices = generate_slice8(|i| self.direct_sample[indices[i] as usize]);
        let tris = generate_slice8(|i| &self.bvh.triangles[tri_indices[i] as usize]);

        // Gather the vertices of the triangles into SIMD vectors, so from now
        // on we are not serial any more.
        let v0 = MVector3::generate(|i| tris[i].v0);
        let v1 = MVector3::generate(|i| tris[i].v1);
        let v2 = MVector3::generate(|i| tris[i].v2);

        let e1 = v0 - v2;
        let e2 = v1 - v0;
        let normal_denorm = e1.cross(e2);
        let cross_norm_recip = normal_denorm.norm_squared().rsqrt();
        let normal = normal_denorm * cross_norm_recip;
        let area = Mf32::broadcast(0.5) * cross_norm_recip.recip_fast();

        let u = rng.sample_unit();
        let v = rng.sample_unit();
        // If u + v > 1, the point lies outside of the triangle, and s will have
        // negative sign. If the point is inside the triangle, s will have
        // positive sign.
        let s = (Mf32::one() - u) - v;
        // If the point lies outside the triangle, it lies in the other half of
        // the parallellogram, so transform the coordinates to get them into the
        // correct triangle again.
        let u = u.pick(Mf32::one() - u, s);
        let v = v.pick(Mf32::one() - v, s);

        let p = e2.mul_add(v, e1.neg_mul_add(u, v0));

        let ds = MDirectSample {
            position: p,
            normal: normal,
            area: area,
        };

        // Prevent NaNs from creeping in, and ensure that the sample is valid.
        debug_assert!(normal.all_finite());
        debug_assert!(area.all_finite());
        debug_assert!(area.all_sign_bits_positive(), "area must be positive");

        ds
    }

    /// Returns the number of triangles eligible for direct sampling.
    pub fn direct_sample_num(&self) -> usize {
        self.direct_sample.len()
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
