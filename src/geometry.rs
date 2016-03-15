//! This module implement the triangle primitive and related geometry functions.
//!
//! The only primitive is the triangle, there are no spheres or other shapes.
//! This avoids having to dispatch on the primitive type to intersect an object.
//! It avoids a virtual method call, which in turn enables the triangle
//! intersection code to be inlined.

use aabb::Aabb;
use ray::{MIntersection, MRay};
use simd::Mf32;
use vector3::{MVector3, SVector3};

#[derive(Clone, Debug)]
pub struct Triangle {
    pub v1: SVector3,
    pub v2: SVector3,
    pub v3: SVector3,
    pub aabb: Aabb,
}

impl Triangle {
    pub fn new(v1: SVector3, v2: SVector3, v3: SVector3) -> Triangle {
        Triangle {
            v1: v1,
            v2: v2,
            v3: v3,
            aabb: Aabb::enclose_points(&[v1, v2, v3]),
        }
    }

    pub fn barycenter(&self) -> SVector3 {
        (self.v1 + self.v2 + self.v3) * 3.0f32.recip()
    }

    pub fn intersect_full(&self, ray: &MRay, isect: MIntersection) -> MIntersection {
        // TODO: Switch to edge representation so this is computed already.
        let e1 = MVector3::broadcast(self.v2 - self.v1);
        let e2 = MVector3::broadcast(self.v3 - self.v1);
        let v1 = MVector3::broadcast(self.v1);

        // All points P on the plane in which the triangle lies satisfy the
        // equation (P . normal) = c for a unique constant c determined by the
        // plane. (The dot denotes the dot product here.) To intersect the ray
        // with the plane, solve the equation (O + tD) . normal = c, where O
        // is the origin of the ray and D the direction. Note: if the ray
        // direction D is normalized, then t is the distance from the ray origin
        // to the plane.
        let normal = e1.cross(e2).normalized(); // TODO: Can precompute at the cost of cache pressure. Is it worth it?
        let t = (v1.dot(normal) - ray.origin.dot(normal)) /
            ray.direction.dot(normal);

        // Compute the position of the intersection relative to the triangle
        // origin.
        let isect_pos = ray.direction.mul_add(t, ray.origin);
        let isect_rel = isect_pos - v1;

        // Express the location of the intersection in terms of the basis for
        // the plane given by (e1, e2).
        let d = e1.dot(e2);
        let e1_nsq = e1.norm_squared();
        let e2_nsq = e2.norm_squared();
        let factor = d.mul_sub(d, e1_nsq * e2_nsq).recip();
        let u = d.mul_sub(isect_rel.dot(e2), e2_nsq * isect_rel.dot(e1)) * factor;
        let v = d.mul_sub(isect_rel.dot(e1), e1_nsq * isect_rel.dot(e2)) * factor;

        // In this coordinate system, the triangle is the set of points such
        // { (u, v) in plane | u >= 0 and v >= 0 and u + v <= 1 }

        // We need t to be positive, because we should not intersect backwards.
        // Also, u and v need to be positive. We can abuse the vblendvps
        // instruction, which considers only the sign bit, so if t, u, v all
        // have sign bit set to 0 (positive), then their bitwise or will have so
        // too.
        let mask_positive = t | u | v;

        // We also need u + v < 1.0. With the vblendps instruction we are going
        // to use a sign bit of 0 in the mask as "take new value" and 1 as "keep
        // previous intersection", so do a greater-than comparison.
        let mask_uv = (u + v).geq(Mf32::one());

        // The intersection also needs to be closer than any previous
        // intersection. (Again, do the reverse comparison because sign bit 1
        // means discard intersection.)
        let mask_closer = t.geq(isect.distance);

        let new_isect = MIntersection {
            position: isect_pos,
            normal: normal,
            distance: t
        };

        // Per ray, pick the new intersection if it is closer and if it was
        // indeed an intersection of the triangle, or pick the previous
        // intersection otherwise.
        new_isect.pick(&isect, mask_positive | mask_uv | mask_closer)
    }
}

#[test]
fn intersect_triangle() {
    use ray::SRay;

    let triangle = Triangle::new(
        SVector3::new(0.0, 1.0, 1.0),
        SVector3::new(-1.0, -1.0, 1.0),
        SVector3::new(1.0, -1.0, 1.0)
    );

    let r1 = SRay {
        origin: SVector3::zero(),
        direction: SVector3::new(0.0, 0.0, 1.0),
    };

    let r2 = SRay {
        origin: SVector3::new(-1.0, 0.0, 0.0),
        direction: SVector3::new(0.0, 0.0, 1.0),
    };

    let ray = MRay::generate(|i| if i % 2 == 0 { r1.clone() } else { r2.clone() });

    let far = MIntersection {
        position: MVector3::zero(),
        normal: MVector3::zero(),
        distance: Mf32::broadcast(1e5),
    };

    let isect = triangle.intersect_full(&ray, far);

    assert!(isect.distance.0 < 1.01);
    assert!(isect.distance.0 > 0.99);
    assert_eq!(isect.distance.1, 1e5);
}
