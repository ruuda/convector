use std::f32::consts::PI;
use vector3::{Intersection, Ray, Vector3, cross, dot};
use wavefront::Mesh;

pub struct Triangle {
    pub origin: Vector3,
    pub left: Vector3,
    pub right: Vector3,
}

impl Triangle {
    pub fn new(v1: Vector3, v2: Vector3, v3: Vector3) -> Triangle {
        Triangle {
            origin: v1,
            left: v2 - v1,
            right: v3 - v1,
        }
    }

    pub fn intersect(&self, ray: &Ray) -> Option<Intersection> {
        // All points P on the plane in which the triangle lies satisfy the
        // equation (P . normal) = c for a unique constant c determined by the
        // plane. (The dot denotes the dot product here.) To intersect the ray
        // with the plane, solve the equation (O + tD) . normal = c, where O
        // is the origin of the ray and D the direction. Note: if the ray
        // direction D is normalized, then t is the distance from the ray origin
        // to the plane.
        let normal = cross(self.left, self.right).normalized();
        let t = (dot(self.origin, normal) - dot(ray.origin, normal)) /
            dot(ray.direction, normal);

        // Do not intersect backwards. Also, if t = 0.0 then the ray originated
        // from this triangle, so in that case we don't want to intersect it.
        if t <= 0.0 {
            return None
        }

        // Compute the position of the intersection relative to the triangle
        // origin.
        let isect_pos = ray.origin + ray.direction * t;
        let isect_rel = isect_pos - self.origin;

        // Express the location of the intersection in terms of the basis for
        // the plane given by (triangle.left, triangle.right).
        let d = dot(self.left, self.right);
        let l2 = self.left.norm_squared();
        let r2 = self.right.norm_squared();
        let denom = d * d - l2 * r2;
        let u = (d * dot(isect_rel, self.right) - r2 * dot(isect_rel, self.left)) / denom;
        let v = (d * dot(isect_rel, self.left) - l2 * dot(isect_rel, self.right)) / denom;

        // In this coordinate system, the triangle is the set of points such
        // { (u, v) in plane | u >= 0 and v >= 0 and u + v <= 1 }

        if u < 0.0 || v < 0.0 || u + v > 1.0 {
            None
        } else {
            let isect = Intersection {
                position: isect_pos,
                normal: normal,
                distance: t,
            };
            Some(isect)
        }
    }
}

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

#[test]
fn intersect_triangle() {
    let triangle = Triangle::new(
        Vector3::new(0.0, 1.0, 1.0),
        Vector3::new(-1.0, -1.0, 1.0),
        Vector3::new(1.0, -1.0, 1.0)
    );

    let r1 = Ray {
        origin: Vector3::zero(),
        direction: Vector3::new(0.0, 0.0, 1.0),
    };

    let r2 = Ray {
        origin: Vector3::new(-1.0, 0.0, 0.0),
        direction: Vector3::new(0.0, 0.0, 1.0),
    };

    assert!(triangle.intersect(&r1).is_some());
    assert!(triangle.intersect(&r2).is_none());
}
