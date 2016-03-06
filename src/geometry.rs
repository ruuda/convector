//! This module implement the triangle primitive and related geometry functions.
//!
//! The only primitive is the triangle, there are no spheres or other shapes.
//! This avoids having to dispatch on the primitive type to intersect an object.
//! It avoids a virtual method call, which in turn enables the triangle
//! intersection code to be inlined.

use aabb::Aabb;
use ray::{Intersection, Ray};
use vector3::{Vector3, dot};

#[derive(Clone, Debug)]
pub struct Triangle {
    pub v1: Vector3,
    pub v2: Vector3,
    pub v3: Vector3,
    pub aabb: Aabb,
}

impl Triangle {
    pub fn new(v1: Vector3, v2: Vector3, v3: Vector3) -> Triangle {
        Triangle {
            v1: v1,
            v2: v2,
            v3: v3,
            aabb: Aabb::enclose_points(&[v1, v2, v3]),
        }
    }

    pub fn barycenter(&self) -> Vector3 {
        (self.v1 + self.v2 + self.v3) * 3.0f32.recip()
    }

    pub fn intersect(&self, ray: &Ray) -> Option<Intersection> {
        let e1 = self.v2 - self.v1;
        let e2 = self.v3 - self.v1;

        // All points P on the plane in which the triangle lies satisfy the
        // equation (P . normal) = c for a unique constant c determined by the
        // plane. (The dot denotes the dot product here.) To intersect the ray
        // with the plane, solve the equation (O + tD) . normal = c, where O
        // is the origin of the ray and D the direction. Note: if the ray
        // direction D is normalized, then t is the distance from the ray origin
        // to the plane.
        let normal = e1.cross(e2).normalized();
        let t = (dot(self.v1, normal) - dot(ray.origin, normal)) /
            dot(ray.direction, normal);

        // Do not intersect backwards. Also, if t = 0.0 then the ray originated
        // from this triangle, so in that case we don't want to intersect it.
        if t <= 0.0 {
            return None
        }

        // Compute the position of the intersection relative to the triangle
        // origin.
        let isect_pos = ray.origin + ray.direction * t;
        let isect_rel = isect_pos - self.v1;

        // Express the location of the intersection in terms of the basis for
        // the plane given by (e1, e2).
        let d = dot(e1, e2);
        let e1_nsq = e1.norm_squared();
        let e2_nsq = e2.norm_squared();
        let denom = d * d - e1_nsq * e2_nsq;
        let u = (d * dot(isect_rel, e2) - e2_nsq * dot(isect_rel, e1)) / denom;
        let v = (d * dot(isect_rel, e1) - e1_nsq * dot(isect_rel, e2)) / denom;

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
