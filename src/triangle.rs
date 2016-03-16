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

#[cfg(test)]
use {bench, test};

#[derive(Clone, Debug)]
pub struct Triangle {
    pub v0: SVector3,
    pub v1: SVector3,
    pub v2: SVector3,
    pub aabb: Aabb,
}

impl Triangle {
    pub fn new(v0: SVector3, v1: SVector3, v2: SVector3) -> Triangle {
        Triangle {
            v0: v0,
            v1: v1,
            v2: v2,
            aabb: Aabb::enclose_points(&[v0, v1, v2]),
        }
    }

    pub fn barycenter(&self) -> SVector3 {
        (self.v0 + self.v1 + self.v2) * 3.0f32.recip()
    }

    pub fn intersect(&self, ray: &MRay, isect: MIntersection) -> MIntersection {
        // One would expect that if the triangle were represented as
        // (v0, e1, e2) instead of (v0, v1, v2), that would be faster because we
        // could avoid the subtractions here. My measurements show that the
        // converse is true.
        // TODO: Add a proper benchmark.
        let v0 = MVector3::broadcast(self.v0);
        let e1 = MVector3::broadcast(self.v0 - self.v2);
        let e2 = MVector3::broadcast(self.v1 - self.v0);

        // All points P on the plane in which the triangle lies satisfy the
        // equation (P . normal) = c for a unique constant c determined by the
        // plane. (The dot denotes the dot product here.) To intersect the ray
        // with the plane, solve the equation (O + tD) . normal = c, where O
        // is the origin of the ray and D the direction. Note: if the ray
        // direction D is normalized, then t is the distance from the ray origin
        // to the plane. There is no need to normalize the triangle normal at
        // this point, because it appears both in the numerator and denominator.
        let normal_denorm = e1.cross(e2);
        let from_ray = v0 - ray.origin;
        let denom = ray.direction.dot(normal_denorm).recip();
        let t = from_ray.dot(normal_denorm) * denom;

        // If the potential intersection is further away than the current
        // intersection for all of the rays, it is possible to early out. This
        // cranks up the number of branches from 209M/s to 256M/s and the
        // misprediction rate from 0.66% to 1.11%. Surprisingly, there is no
        // significant effect on the framerate. It appears that the early out
        // wins almost exactly cancel the mispredict penalty on my Skylake i7.
        // I opt for not poisioning the branch prediction cache here.

        // if (t - isect.distance).all_sign_bits_positive() {
        //     return isect
        // }

        // Express the location of the intersection in terms of the basis for
        // the plane given by (-e1, e2). The computation of u and v is based on
        // the method in this paper (there they are called alpha and beta):
        // https://www.cs.utah.edu/~aek/research/triangle.pdf
        let cross = ray.direction.cross(from_ray);
        let u = cross.dot(e2) * denom;
        let v = cross.dot(e1) * denom;

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
            position: ray.direction.mul_add(t, ray.origin),
            normal: normal_denorm.normalized(),
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

    let isect = triangle.intersect(&ray, far);

    println!("distance is {}", isect.distance.0);
    assert!(isect.distance.0 < 1.01);
    assert!(isect.distance.0 > 0.99);
    assert_eq!(isect.distance.1, 1e5);

    let up = MVector3::new(Mf32::zero(), Mf32::zero(), Mf32::one());
    let should_be_origin = isect.position - up;
    let should_be_zero = should_be_origin.norm_squared();
    assert!(should_be_zero.0 < 0.01);
}

#[bench]
fn bench_intersect_8_mrays_per_tri(b: &mut test::Bencher) {
    let rays = bench::mrays_inward(4096 / 8);
    let tris = bench::triangles(4096);
    let mut rays_it = rays.iter().cycle();
    let mut tris_it = tris.iter().cycle();
    b.iter(|| {
        let triangle = tris_it.next().unwrap();
        for _ in 0..8 {
            let ray = rays_it.next().unwrap();
            let isect = MIntersection {
                position: MVector3::zero(),
                normal: MVector3::zero(),
                distance: Mf32::broadcast(1e5),
            };
            test::black_box(triangle.intersect(&ray, isect));
        }
    });
}

#[bench]
fn bench_intersect_8_tris_per_mray(b: &mut test::Bencher) {
    let rays = bench::mrays_inward(4096 / 8);
    let tris = bench::triangles(4096);
    let mut rays_it = rays.iter().cycle();
    let mut tris_it = tris.iter().cycle();
    b.iter(|| {
        let ray = rays_it.next().unwrap();
        let mut isect = MIntersection {
            position: MVector3::zero(),
            normal: MVector3::zero(),
            distance: Mf32::broadcast(1e5),
        };
        for _ in 0..8 {
            let triangle = tris_it.next().unwrap();
            isect = triangle.intersect(&ray, isect);
        }
        test::black_box(isect);
    });
}
