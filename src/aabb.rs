//! This module implements axis-aligned bounding boxes and related functions.

use ray::MRay;
use simd::{Mask, Mf32};
use vector3::{MVector3, SVector3};

#[cfg(test)]
use {bench, test};

/// An axis-aligned bounding box.
#[derive(Clone, Debug)]
pub struct Aabb {
    pub origin: SVector3,

    /// The origin plus the size.
    pub far: SVector3,
}

/// Caches AABB intersection distances.
pub struct MAabbIntersection {
    // The AABB was intersected by the line defined by the ray if tmax > tmin.
    // The mask contains the result of this comparison. If tmax is negative, the
    // AABB lies behind the ray entirely.
    tmin: Mf32,
    tmax: Mf32,

    // The mask can be computed from tmin and tmax, but benchmarks show that it
    // is slightly faster to store it, than to re-compute it when needed.
    mask: Mask,
}

impl Aabb {
    pub fn new(origin: SVector3, far: SVector3) -> Aabb {
        Aabb {
            origin: origin,
            far: far,
        }
    }

    pub fn zero() -> Aabb {
        Aabb {
            origin: SVector3::zero(),
            far: SVector3::zero(),
        }
    }

    /// Returns the smalles axis-aligned bounding box that contains all input
    /// points.
    pub fn enclose_points<'a, I>(points: I) -> Aabb where I: IntoIterator<Item = &'a SVector3> {
        let mut it = points.into_iter();
        let &first = it.next().expect("enclosure must encluse at least one point");

        let mut min = first;
        let mut max = first;

        while let Some(&point) = it.next() {
            min = SVector3::min(min, point);
            max = SVector3::max(max, point);
        }

        Aabb::new(min, max)
    }

    /// Returns the smallest bounding box that contains all input boxes.
    pub fn enclose_aabbs<'a, I>(aabbs: I) -> Aabb where I: IntoIterator<Item = &'a Aabb> {
        let mut it = aabbs.into_iter();
        let first = it.next().expect("enclosure must enclose at least one AABB");

        let mut min = first.origin;
        let mut max = first.far;

        while let Some(aabb) = it.next() {
            min = SVector3::min(min, aabb.origin);
            max = SVector3::max(max, aabb.far);
        }

        Aabb::new(min, max)
    }

    /// Returns the bounding box extended to contain the point.
    pub fn extend_point(&self, point: SVector3) -> Aabb {
        let min = SVector3::min(point, self.origin);
        let max = SVector3::max(point, self.far);
        Aabb::new(min, max)
    }

    /// Returns the center of the bounding box.
    pub fn center(&self) -> SVector3 {
        (self.origin + self.far) * 0.5
    }

    /// Returns the size of the bounding box.
    pub fn size(&self) -> SVector3 {
        self.far - self.origin
    }

    /// Returns the surface area of the bounding box.
    pub fn area(&self) -> f32 {
        let s = self.size();
        let x = s.y * s.z;
        let y = s.z * s.x;
        let z = s.x * s.y;
        2.0 * (x + y + z)
    }

    pub fn intersect(&self, ray: &MRay) -> MAabbIntersection {
        // Note: this method, in combination with `MAabbIntersection::any()`
        // compiles down to ~65 instructions, taking up ~168 bytes of
        // instruction cache; 3 cache lines.

        // Note: the compiler is smart enough to inline this method and compute
        // these reciprocals only once per ray, so there is no need to clutter
        // the code by passing around precomputed values.
        let xinv = ray.direction.x.recip_fast();
        let yinv = ray.direction.y.recip_fast();
        let zinv = ray.direction.z.recip_fast();

        let d1 = MVector3::broadcast(self.origin) - ray.origin;
        let d2 = MVector3::broadcast(self.far) - ray.origin;

        let (tx1, tx2) = (d1.x * xinv, d2.x * xinv);
        let txmin = tx1.min(tx2);
        let txmax = tx1.max(tx2);

        let (ty1, ty2) = (d1.y * yinv, d2.y * yinv);
        let tymin = ty1.min(ty2);
        let tymax = ty1.max(ty2);

        let (tz1, tz2) = (d1.z * zinv, d2.z * zinv);
        let tzmin = tz1.min(tz2);
        let tzmax = tz1.max(tz2);

        // The minimum t in all dimension is the maximum of the per-axis minima.
        let tmin = txmin.max(tymin.max(tzmin));
        let tmax = txmax.min(tymax.min(tzmax));

        MAabbIntersection {
            tmin: tmin,
            tmax: tmax,
            mask: tmax.geq(tmin),
        }
    }
}

impl MAabbIntersection {
    /// Returns whether any of the rays intersected the AABB.
    pub fn any(&self) -> bool {
        // If there was an intersection in front of the ray, then tmax will
        // definitely be positive. The mask is only set for the rays that
        // actually intersected the bounding box.
        self.tmax.any_sign_bit_positive_masked(self.mask)
    }

    /// Returns whether for all rays that intersect the AABB, the given distance
    /// is smaller than the distance to the AABB.
    pub fn is_further_away_than(&self, distance: Mf32) -> bool {
        // If distance < self.tmin (when false should be returned for the ray),
        // the comparison results in positive 0.0. If distance < self.min for
        // any of the values for which the mask is set, then for that ray the
        // AABB is not further away. Hence all sign bits must be negative.
        self.tmin.geq(distance).all_sign_bits_negative_masked(self.mask)
    }
}

#[test]
fn aabb_enclose_aabbs() {
    let a = Aabb::new(SVector3::new(1.0, 2.0, 3.0), SVector3::new(5.0, 7.0, 9.0));
    let b = Aabb::new(SVector3::new(0.0, 3.0, 2.0), SVector3::new(9.0, 6.0, 9.0));
    let ab = Aabb::enclose_aabbs(&[a, b]);
    assert_eq!(ab.origin, SVector3::new(0.0, 2.0, 2.0));
    assert_eq!(ab.far, SVector3::new(9.0, 7.0, 9.0));
}

#[test]
fn aabb_center() {
    let aabb = Aabb::new(SVector3::new(1.0, 2.0, 3.0), SVector3::new(5.0, 7.0, 9.0));
    assert_eq!(aabb.center(), SVector3::new(3.0, 4.5, 6.0));
}

#[test]
fn aabb_area() {
    // Width: 4, height: 5, depth: 6.
    let aabb = Aabb::new(SVector3::new(1.0, 2.0, 3.0), SVector3::new(5.0, 7.0, 9.0));
    assert_eq!(40.0 + 60.0 + 48.0, aabb.area());
}

#[test]
fn aabb_extend_point() {
    let aabb = Aabb::new(SVector3::zero(), SVector3::one());
    let p = SVector3::new(0.5, 0.5, 1.5);
    let aabb_p = aabb.extend_point(p);

    assert_eq!(aabb_p.origin, SVector3::zero());
    assert_eq!(aabb_p.far, SVector3::new(1.0, 1.0, 1.5));

    let q = SVector3::new(-0.2, 1.2, 1.0);
    let aabb_pq = aabb_p.extend_point(q);

    assert_eq!(aabb_pq.origin, SVector3::new(-0.2, 0.0, 0.0));
    assert_eq!(aabb_pq.far, SVector3::new(1.0, 1.2, 1.5));
}

#[test]
fn intersect_aabb() {
    use ray::SRay;

    let aabb = Aabb {
        origin: SVector3::new(0.0, 1.0, 2.0),
        far: SVector3::new(1.0, 3.0, 5.0),
    };

    // Intersects forwards but not backwards.
    let r1 = SRay {
        origin: SVector3::zero(),
        direction: SVector3::new(2.0, 3.0, 5.0).normalized(),
    };
    let mr1 = MRay::broadcast(&r1);
    assert!(aabb.intersect(&mr1).any());
    assert!(!aabb.intersect(&-mr1).any());

    // Intersects forwards but not backwards.
    let r2 = SRay {
        origin: SVector3::zero(),
        direction: SVector3::new(1.0, 4.0, 5.0).normalized(),
    };
    let mr2 = MRay::broadcast(&r2);
    assert!(aabb.intersect(&mr2).any());
    assert!(!aabb.intersect(&-mr2).any());

    // Intersects neither forwards nor backwards.
    let r3 = SRay {
        origin: SVector3::zero(),
        direction: SVector3::new(2.0, 3.0, 0.0).normalized(),
    };
    let mr3 = MRay::broadcast(&r3);
    assert!(!aabb.intersect(&mr3).any());
    assert!(!aabb.intersect(&-mr3).any());

    // Intersects both forwards and backwards (origin is inside the aabb).
    let r4 = SRay {
        origin: SVector3::new(0.2, 1.2, 2.2),
        direction: SVector3::new(1.0, 1.0, 0.0).normalized(),
    };
    let mr4 = MRay::broadcast(&r4);
    assert!(aabb.intersect(&mr4).any());
    assert!(aabb.intersect(&-mr4).any());

    // Intersects both forwards and backwards (origin is inside the aabb).
    let r5 = SRay {
        origin: SVector3::new(0.01, 2.0, 3.5),
        direction: SVector3::new(0.0, 0.0, 1.0).normalized(),
    };
    let mr5 = MRay::broadcast(&r5);
    assert!(aabb.intersect(&mr5).any());
    assert!(aabb.intersect(&-mr5).any());
}

#[bench]
fn bench_intersect_p100(b: &mut test::Bencher) {
    let (aabb, rays) = bench::aabb_with_mrays(4096, 4096);
    let mut rays_it = rays.iter().cycle();
    b.iter(|| {
        let isect = aabb.intersect(rays_it.next().unwrap());
        test::black_box(isect.any());
    });
}

#[bench]
fn bench_intersect_p50(b: &mut test::Bencher) {
    let (aabb, rays) = bench::aabb_with_mrays(4096, 2048);
    let mut rays_it = rays.iter().cycle();
    b.iter(|| {
        let isect = aabb.intersect(rays_it.next().unwrap());
        test::black_box(isect.any());
    });
}

#[bench]
fn bench_intersect_8_mrays_per_aabb(b: &mut test::Bencher) {
    let rays = bench::mrays_inward(4096 / 8);
    let aabbs = bench::aabbs(4096);
    let mut rays_it = rays.iter().cycle();
    let mut aabbs_it = aabbs.iter().cycle();
    b.iter(|| {
        let aabb = aabbs_it.next().unwrap();
        for _ in 0..8 {
            let isect = aabb.intersect(rays_it.next().unwrap());
            test::black_box(isect.any());
        }
    });
}

#[bench]
fn bench_intersect_8_aabbs_per_mray(b: &mut test::Bencher) {
    let rays = bench::mrays_inward(4096 / 8);
    let aabbs = bench::aabbs(4096);
    let mut rays_it = rays.iter().cycle();
    let mut aabbs_it = aabbs.iter().cycle();
    b.iter(|| {
        let ray = rays_it.next().unwrap();
        for _ in 0..8 {
            let aabb = aabbs_it.next().unwrap();
            let isect = aabb.intersect(ray);
            test::black_box(isect.any());
        }
    });
}
