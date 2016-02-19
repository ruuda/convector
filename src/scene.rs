use bvh::{Triangle};
use std::f32::consts::PI;
use vector3::{Intersection, Ray, Vector3, cross, dot};
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
    // The only primitive is the triangle, there are no spheres or other shapes.
    // This avoids having to dispatch on the primitive type to intersect an
    // object. It avoids a virtual method call. This in turn enables the
    // triangle intersection code to be inlined.
    pub geometry: Vec<Triangle>,
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
    pub fn new() -> Scene {
        Scene {
            geometry: Vec::new(),
            lights: Vec::new(),
            camera: Camera::with_fov(PI * 0.6),
        }
    }

    pub fn add_mesh(&mut self, mesh: &Mesh) {
        for &(i1, i2, i3) in &mesh.triangles {
            let v1 = mesh.vertices[i1 as usize];
            let v2 = mesh.vertices[i2 as usize];
            let v3 = mesh.vertices[i3 as usize];
            self.geometry.push(Triangle::new(v1, v2, v3));
        }
    }

    pub fn intersect(&self, ray: &Ray) -> Option<Intersection> {
        let mut result = None;
        for triangle in &self.geometry {
            result = closest(result, triangle.intersect(ray));
        }
        result
    }
}
