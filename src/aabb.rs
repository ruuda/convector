//! This module implements axis-aligned bounding boxes and related functions.

use ray::Ray;
use vector3::Vector3;

#[cfg(test)]
use {bench, test};

/// An axis-aligned bounding box.
#[derive(Clone, Debug)]
pub struct Aabb {
    pub origin: Vector3,
    pub size: Vector3,
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
        let origin = Vector3::new(xmin, ymin, zmin);
        let size = Vector3::new(xmax - xmin, ymax - ymin, zmax - zmin);
        Aabb::new(origin, size)
    }

    /// Returns the center of the bounding box.
    pub fn center(&self) -> Vector3 {
        self.origin + self.size * 0.5
    }

    /// Returns whether the ray intersects the bounding box.
    pub fn intersect(&self, ray: &Ray) -> bool {
        // My measurements show that this is the fastest method to intersect an
        // AABB by a factor 2 in the frame time.
        // TODO: Add benchmarks to verify.
        // TODO: It seems that marking this method #[inline(always)] makes it
        // a bit faster too (2.8 fps vs 2.9 fps).
        self.intersect_flavor_planes(ray)

        // Alternative:
        // self.intersect_flavor_slab(ray).is_some()
    }

    /// Intersects the AABB by intersecting six planes and testing the bounds.
    #[inline(always)]
    fn intersect_flavor_planes(&self, ray: &Ray) -> bool {
        // TODO: Simd the **** out of this.
        intersect_aabb!(self, ray, x, y, z);
        intersect_aabb!(self, ray, y, z, x);
        intersect_aabb!(self, ray, z, x, y);
        false
    }

    /// Intersects the AABB by clipping the t values inside.
    #[inline(always)]
    fn intersect_flavor_slab(&self, ray: &Ray) -> Option<f32> {
        // TODO: The reciprocal could be precomputed per ray.
        let xinv = ray.direction.x.recip();
        let yinv = ray.direction.y.recip();
        let zinv = ray.direction.z.recip();

        let d1 = self.origin - ray.origin;
        let d2 = d1 + self.size;

        let txmin = f32::min(d1.x * xinv, d2.x * xinv);
        let txmax = f32::max(d1.x * xinv, d2.x * xinv);

        let tymin = f32::min(d1.y * yinv, d2.y * yinv);
        let tymax = f32::max(d1.y * yinv, d2.y * yinv);

        let tzmin = f32::min(d1.z * zinv, d2.z * zinv);
        let tzmax = f32::max(d1.z * zinv, d2.z * zinv);

        // The minimum t in all dimension is the maximum of the per-axis minima.
        let tmin = f32::max(txmin, f32::max(tymin, tzmin));
        let tmax = f32::min(txmax, f32::min(tymax, tzmax));

        if tmax >= tmin && tmax >= 0.0 {
            Some(tmin)
        } else {
            None
        }
    }
}

#[test]
fn aabb_enclose_aabbs() {
    let a = Aabb::new(Vector3::new(1.0, 2.0, 3.0), Vector3::new(4.0, 5.0, 6.0));
    let b = Aabb::new(Vector3::new(0.0, 3.0, 2.0), Vector3::new(9.0, 3.0, 7.0));
    let ab = Aabb::enclose_aabbs(&a, &b);
    assert_eq!(ab.origin, Vector3::new(0.0, 2.0, 2.0));
    assert_eq!(ab.size, Vector3::new(9.0, 5.0, 7.0));
}

#[test]
fn aabb_center() {
    let aabb = Aabb::new(Vector3::new(1.0, 2.0, 3.0), Vector3::new(4.0, 5.0, 6.0));
    assert_eq!(aabb.center(), Vector3::new(3.0, 4.5, 6.0));
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

#[bench]
fn bench_intersect_aabb_flavor_planes_p100(b: &mut test::Bencher) {
    let (aabb, rays) = bench::aabb_with_rays(4096, 4096);
    let mut rays_it = rays.iter().cycle();
    b.iter(|| {
        let isect = aabb.intersect_flavor_planes(rays_it.next().unwrap());
        test::black_box(isect);
    });
}

#[bench]
fn bench_intersect_aabb_flavor_planes_p50(b: &mut test::Bencher) {
    let (aabb, rays) = bench::aabb_with_rays(4096, 2048);
    let mut rays_it = rays.iter().cycle();
    b.iter(|| {
        let isect = aabb.intersect_flavor_planes(rays_it.next().unwrap());
        test::black_box(isect);
    });
}

#[bench]
fn bench_intersect_aabb_flavor_slab_p100(b: &mut test::Bencher) {
    let (aabb, rays) = bench::aabb_with_rays(4096, 4096);
    let mut rays_it = rays.iter().cycle();
    b.iter(|| {
        let isect = aabb.intersect_flavor_slab(rays_it.next().unwrap());
        test::black_box(isect);
    });
}

#[bench]
fn bench_intersect_aabb_flavor_slab_p50(b: &mut test::Bencher) {
    let (aabb, rays) = bench::aabb_with_rays(4096, 2048);
    let mut rays_it = rays.iter().cycle();
    b.iter(|| {
        let isect = aabb.intersect_flavor_slab(rays_it.next().unwrap());
        test::black_box(isect);
    });
}
