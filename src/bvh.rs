//! Implements a bounding volume hierarchy.

use std::f32;
use vector3::{Axis, Intersection, Ray, Vector3, cross, dot};

/// An axis-aligned bounding box.
#[derive(Clone)]
pub struct Aabb {
    pub origin: Vector3,
    pub size: Vector3,
}

#[derive(Clone)]
pub struct Triangle {
    pub v1: Vector3,
    pub v2: Vector3,
    pub v3: Vector3,
    pub aabb: Aabb,
}

/// One node in a bounding volume hierarchy.
struct BvhNode {
    aabb: Aabb,
    children: Vec<BvhNode>,
    geometry: Vec<Triangle>,
}

/// A bounding volume hierarchy.
pub struct Bvh {
    root: BvhNode,
}

// Intersecting an axis-aligned bounding box is tricky to implement. Below is a
// macro for intersecting a ray with the two faces of the AABB parallel to the
// xy-plane. By cyclically permuting x, y, and z, it is possible to intersect
// all faces, saving a third of the code.
macro_rules! intersect_aabb {
    ($aabb: ident, $ray: ident, $x: ident, $y: ident, $z: ident) => {
        if $ray.direction.$z != 0.0 {
            let origin_z1 = $aabb.origin.$z - $ray.origin.$z;
            let origin_z2 = origin_z1 + $aabb.size.$z;
            let t1 = origin_z1 / $ray.direction.$z;
            let t2 = origin_z2 / $ray.direction.$z;
            let i1 = $ray.origin + $ray.direction * t1 - $aabb.origin;
            let i2 = $ray.origin + $ray.direction * t2 - $aabb.origin;
            let in_x1 = (0.0 <= i1.$x) && (i1.$x <= $aabb.size.$x);
            let in_x2 = (0.0 <= i2.$x) && (i2.$x <= $aabb.size.$x);
            let in_y1 = (0.0 <= i1.$y) && (i1.$y <= $aabb.size.$y);
            let in_y2 = (0.0 <= i1.$y) && (i1.$y <= $aabb.size.$y);
            if (t1 >= 0.0 && in_x1 && in_y1) || (t2 >= 0.0 && in_x2 && in_y2) {
                return true
            }
        }
    }
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
        // TODO: Simd the **** out of this.
        intersect_aabb!(self, ray, x, y, z);
        intersect_aabb!(self, ray, y, z, x);
        intersect_aabb!(self, ray, z, x, y);
        false
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

fn build_bvh_node(triangles: &mut [Triangle]) -> BvhNode {
    let mut aabb = Aabb::new(Vector3::zero(), Vector3::zero());

    // Compute the bounding box that encloses all triangles.
    for triangle in triangles.iter() {
        aabb = Aabb::enclose_aabbs(&aabb, &triangle.aabb);
    }

    // Ideally every node would contain two triangles, so splitting less than
    // four triangles does not make sense; make a leaf node in that case.
    if triangles.len() < 4 {
        return BvhNode {
            aabb: aabb,
            children: Vec::new(),
            geometry: triangles.iter().cloned().collect(),
        }
    }

    // Split along the axis in which the box is largest.
    let mut size = aabb.size.x;
    let mut axis = Axis::X;

    if aabb.size.y > size {
        size = aabb.size.y;
        axis = Axis::Y;
    }

    if aabb.size.z > size {
        size = aabb.size.z;
        axis = Axis::Z;
    }

    // TODO: Sort all triangles along this axis.
    // TODO: Partition along this axis.
    let split_point = triangles.len() / 2;
    let (left_triangles, right_triangles) = triangles.split_at_mut(split_point);
    let left_node = build_bvh_node(left_triangles);
    let right_node = build_bvh_node(right_triangles);
    BvhNode {
        aabb: aabb,
        children: vec![left_node, right_node],
        geometry: Vec::new(),
    }
}

impl Bvh {
    fn build(mut triangles: Vec<Triangle>) -> Bvh {
        // TODO: Use rayon for data parallelism here.
        let root = build_bvh_node(&mut triangles);
        Bvh {
            root: root,
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

    // Intersects forwards but not backwards.
    let r1 = Ray {
        origin: Vector3::zero(),
        direction: Vector3::new(2.0, 3.0, 5.0).normalized(),
    };
    assert!(aabb.intersect(&r1));
    assert!(!aabb.intersect(&-r1));

    // Intersects forwards but not backwards.
    let r2 = Ray {
        origin: Vector3::zero(),
        direction: Vector3::new(1.0, 4.0, 5.0).normalized(),
    };
    assert!(aabb.intersect(&r2));
    assert!(!aabb.intersect(&-r2));

    // Intersects neither forwards nor backwards.
    let r3 = Ray {
        origin: Vector3::zero(),
        direction: Vector3::new(2.0, 3.0, 0.0).normalized(),
    };
    assert!(!aabb.intersect(&r3));
    assert!(!aabb.intersect(&-r3));

    // Intersects both forwards and backwards (origin is inside the aabb).
    let r4 = Ray {
        origin: Vector3::new(0.2, 1.2, 2.2),
        direction: Vector3::new(1.0, 1.0, 0.0).normalized(),
    };
    assert!(aabb.intersect(&r4));
    assert!(aabb.intersect(&-r4));

    // Intersects both forwards and backwards (origin is inside the aabb).
    let r5 = Ray {
        origin: Vector3::new(0.0, 2.0, 3.5),
        direction: Vector3::new(0.0, 0.0, 1.0).normalized(),
    };
    assert!(aabb.intersect(&r5));
    assert!(aabb.intersect(&-r5));
}
