//! Implements a bounding volume hierarchy.

use std::f32;
use vector3::{Intersection, Ray, Vector3, cross, dot};

/// An axis-aligned bounding box.
pub struct Aabb {
    pub origin: Vector3,
    pub size: Vector3,
}

pub struct Triangle {
    pub v1: Vector3,
    pub v2: Vector3,
    pub v3: Vector3,
    pub aabb: Aabb,
}

pub struct BvhNode {
    aabb: Aabb,
    children: Vec<BvhNode>,
    geometry: Vec<Triangle>,
}

impl Aabb {
    pub fn new(origin: Vector3, size: Vector3) -> Aabb {
        Aabb {
            origin: origin,
            size: size,
        }
    }

    /// Returns the smalles axis-aligned bounding box that contains all input
    /// points.
    pub fn enclose_points(points: &[Vector3]) -> Aabb {
        assert!(points.len() > 0);

        let mut min = points[0];
        let mut max = points[0];

        for point in points.iter().skip(1) {
            min.x = f32::min(min.x, point.x);
            min.y = f32::min(min.y, point.y);
            min.z = f32::min(min.z, point.z);
            max.x = f32::max(max.x, point.x);
            max.y = f32::max(max.y, point.y);
            max.z = f32::max(max.z, point.z);
        }

        Aabb::new(min, max - min)
    }

    /// Returns the smallest bounding box that contains all input boxes.
    pub fn enclose_aabbs(a: &Aabb, b: &Aabb) -> Aabb {
        let xmin = f32::min(a.origin.x, b.origin.x);
        let ymin = f32::min(a.origin.y, b.origin.y);
        let zmin = f32::min(a.origin.z, b.origin.z);
        let xmax = f32::max(a.origin.x + a.size.x, b.origin.x + b.size.x);
        let ymax = f32::max(a.origin.y + a.size.y, b.origin.y + b.size.y);
        let zmax = f32::max(a.origin.z + a.size.z, b.origin.z + b.size.z);
        let origin = Vector3::new(xmin, ymin, zmax);
        let size = Vector3::new(xmax - xmin, ymax - ymin, zmax - zmin);
        Aabb::new(origin, size)
    }

    /// Returns whether the ray intersects the bounding box.
    pub fn intersect(&self, ray: &Ray) -> bool {
        // TODO: Division by zero?
        // TODO: Simd the **** out of this.
        let nxy = ray.direction.x / ray.direction.y;
        let nyx = ray.direction.y / ray.direction.x;
        let nzx = ray.direction.z / ray.direction.x;
        let d_near = self.origin - ray.origin;
        let d_far = d_near + self.size;
        let ix1 = d_near.y * nxy - self.origin.x;
        let ix2 = d_far.y * nxy - self.origin.x;
        let iy1 = d_near.x * nyx - self.origin.y;
        let iy2 = d_far.x * nyx - self.origin.y;
        let iz1 = d_near.x * nzx - self.origin.z;
        let iz2 = d_far.x * nzx - self.origin.z;

        let in_x1 = (0.0 <= ix1) && (ix1 <= self.size.x);
        let in_x2 = (0.0 <= ix2) && (ix2 <= self.size.x);
        let in_y1 = (0.0 <= iy1) && (iy1 <= self.size.y);
        let in_y2 = (0.0 <= iy2) && (iy2 <= self.size.y);
        let in_z1 = (0.0 <= iz1) && (iz1 <= self.size.z);
        let in_z2 = (0.0 <= iz2) && (iz2 <= self.size.z);

        (in_x1 || in_x2) && (in_y1 || in_y2) && (in_z1 || in_z2)
    }
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
        let normal = cross(e1, e2).normalized();
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

#[test]
fn intersect_aabb() {
    let aabb = Aabb {
        origin: Vector3::new(0.0, 1.0, 2.0),
        size: Vector3::new(1.0, 2.0, 3.0),
    };

    let r1 = Ray {
        origin: Vector3::zero(),
        direction: Vector3::new(2.0, 3.0, 5.0).normalized(),
    };

    let r2 = Ray {
        origin: Vector3::zero(),
        direction: Vector3::new(1.0, 4.0, 5.0).normalized(),
    };

    assert!(aabb.intersect(&r1));
    assert!(aabb.intersect(&r2));
}
