//! Implements quaternion utilities to handle rotation.

use simd::Mf32;
use vector3::MVector3;

#[cfg(test)]
use {bench, test};

#[derive(Copy, Clone, Debug)]
pub struct SQuaternion {
    pub a: f32,
    pub b: f32,
    pub c: f32,
    pub d: f32,
}

pub struct MQuaternion {
    pub a: Mf32,
    pub b: Mf32,
    pub c: Mf32,
    pub d: Mf32,
}

impl SQuaternion {
    pub fn new(a: f32, b: f32, c: f32, d: f32) -> SQuaternion {
        SQuaternion {
            a: a,
            b: b,
            c: c,
            d: d,
        }
    }
}

impl MQuaternion {
    pub fn new(a: Mf32, b: Mf32, c: Mf32, d: Mf32) -> MQuaternion {
        MQuaternion {
            a: a,
            b: b,
            c: c,
            d: d,
        }
    }

    pub fn broadcast(q: SQuaternion) -> MQuaternion {
        MQuaternion {
            a: Mf32::broadcast(q.a),
            b: Mf32::broadcast(q.b),
            c: Mf32::broadcast(q.c),
            d: Mf32::broadcast(q.d),
        }
    }

    /// Interpolates two quaternions and normalizes the result.
    pub fn interpolate(&self, other: &MQuaternion, t: Mf32) -> MQuaternion {
        // The hypersphere of unit quaternions forms a double cover of SO3(R).
        // Every rotation is represented by two antipodal points on the
        // hypersphere. If we naively run over the arc subtended by the two
        // quaternions, then we could make an arc of more than pi/2 radians, but
        // that means that we could make a shorter arc by taking the antipodal
        // point of one of the quaternions. The shortest arc corresponds to the
        // interpolation we want, the longer arc rotates too much. So for
        // correct interpolation, compute the dot product of the two
        // quaternions, and if it is negative, negate one of the two.
        // Fortunately, in my demo I get to pick the quaternions, so I can
        // choose them so they get interpolated correctly, and there is no need
        // to negate anything.

        // Interpolate linearly between the two quaternions, and then project
        // the result onto the unit hypersphere. This is not entirely correct
        // because the rotation will not have a constant angular velocity. For a
        // proper interpolation with constant velocity, a spherical linear
        // interpolation is required, but that is expensive to compute. (It
        // involves an inverse cosine, two sines and two divisions.) For small
        // angles the error is very small, so do the fast thing here.
        let u = Mf32::one() - t;
        let a = self.a.mul_add(u, other.a * t);
        let b = self.b.mul_add(u, other.b * t);
        let c = self.c.mul_add(u, other.c * t);
        let d = self.d.mul_add(u, other.d * t);

        let norm_squared = a.mul_add(a, b * b) + c.mul_add(c, d * d);

        // Using a full square root and division here makes this method about
        // 17% slower in comparison to using an `rsqrt()`. However, this is also
        // more accurate. The `rsqrt()` approach has a relatively big error, and
        // as this code is used to generate camera rays, it had better be
        // accurate. If after a few bounces the ray direction norm is 1.01, then
        // that will result in wrong intersection tests, but the difference is
        // probably not noticeable due to randomness anyway. However, the first
        // intersection should be correct, otherwise the geometry gets
        // distorted. Therefore the camera rays must be accurate.
        let rnorm = Mf32::one() / norm_squared.sqrt();

        MQuaternion {
            a: a * rnorm,
            b: b * rnorm,
            c: c * rnorm,
            d: d * rnorm,
        }
    }
}

pub fn rotate(vector: &MVector3, rotation: &MQuaternion) -> MVector3 {
    let v = vector;
    let q = rotation;

    // For a unit quaternion q and a vector in R3 identified with the subspace
    // of the quaternion algebra spanned by (i, j, k), the rotated vector is
    // given by q * v * q^-1. (And because q is a unit quaternion, its inverse
    // is its conjugate.) This means that we can compute the rotation in two
    // steps: p = v * q^-1, and q * p. The first step is simpler than generic
    // quaternion multiplication because we know that v is pure imaginary. The
    // second step simpler than generc quaternion multiplication because we know
    // that the result is pure imaginary, so the real component does not have to
    // be computed.

    // For q = a + b*i + c*j + d*k and v = x*i + y*j + c*z, v * q^-1 is given
    // by
    //
    //     b*x + c*y + d*z +
    //     ((a - b)*x + (c - d)*(y + z) + b*x - c*y + d*z)*i +
    //     (d*x + a*y - b*z)*j +
    //     (-(c + d)*x + (a + b)*(y + z) + d*x - a*y - b*z)*k
    //
    // I did not bother with using `mul_add` or eliminating common
    // subexpressions below because the code is unreadable enough as it is ...

    let pa = q.b * v.x + q.c * v.y + q.d * v.z;
    let pb = q.b * v.x - q.c * v.y + q.d * v.z + (q.a - q.b) * v.x + (q.c - q.d) * (v.y + v.z);
    let pc = q.d * v.x + q.a * v.y - q.b * v.z;
    let pd = q.d * v.x - q.a * v.y - q.b * v.z - (q.c + q.d) * v.x + (q.a + q.b) * (v.y + v.z);

    // The product of q = qa + qb*i + qc*j + qd*k and
    // p = pa + pb*i + pc*j + pd*k is given by
    //
    //    pa*qa - pb*qb - pc*qc - pd*qd +
    //    ((pa + pb)*(qa + qb) - (pc - pd)*(qc + qd) - pa*qa - pb*qb + pc*qc - pd*qd)*i +
    //    (pc*qa - pd*qb + pa*qc + pb*qd)*j +
    //    ((pc + pd)*(qa + qb) + (pa - pb)*(qc + qd) - pc*qa - pd*qb - pa*qc + pb*qd)*k

    let rb = (pa + pb) * (q.a + q.b) - (pc - pd) * (q.c + q.d) - pa * q.a - pb * q.b + pc * q.c - pd * q.d;
    let rc = pc * q.a - pd * q.b + pa * q.c + pb * q.d;
    let rd = (pc + pd) * (q.a + q.b) + (pa - pb) * (q.c + q.d) - pc * q.a - pd * q.b - pa * q.c + pb * q.d;

    MVector3::new(rb, rc, rd)
}

#[cfg(test)]
fn assert_mvectors_equal(expected: MVector3, computed: MVector3, margin: f32) {
    // Test that the vectors are equal, to within floating point inaccuracy
    // margins.
    let error = (computed - expected).norm_squared();
    assert!((Mf32::broadcast(margin * margin) - error).all_sign_bits_positive(),
            "expected: ({}, {}, {}), computed: ({}, {}, {})",
            expected.x.0, expected.y.0, expected.z.0,
            computed.x.0, computed.y.0, computed.z.0);
}

#[test]
fn rotate_identity() {
    let identity = SQuaternion::new(1.0, 0.0, 0.0, 0.0);
    let vectors = bench::mvectors_on_unit_sphere(32);
    for v in &vectors {
        assert_mvectors_equal(*v, rotate(v, &MQuaternion::broadcast(identity)), 1e-7);
    }
}

#[test]
fn rotate_x() {
    let half_sqrt_2 = 0.5 * 2.0_f32.sqrt();
    let rotation = SQuaternion::new(half_sqrt_2, half_sqrt_2, 0.0, 0.0);
    let vectors = bench::mvectors_on_unit_sphere(32);
    for v in &vectors {
        // Rotate the vector by pi/2 radians around the x-axis. This is
        // equivalent to y <- -z, z <- y, so compute the rotation in two
        // different ways, and verify that the result is the same to within the
        // floating point inaccuracy margin.
        let computed = rotate(v, &MQuaternion::broadcast(rotation));
        let expected = MVector3::new(v.x, -v.z, v.y);
        assert_mvectors_equal(expected, computed, 1e-6);
    }
}

#[test]
fn rotate_y() {
    let half_sqrt_2 = 0.5 * 2.0_f32.sqrt();
    let rotation = SQuaternion::new(half_sqrt_2, 0.0, half_sqrt_2, 0.0);
    let vectors = bench::mvectors_on_unit_sphere(32);
    for v in &vectors {
        // Rotate the vector by pi/2 radians around the y-axis. This is
        // equivalent to x <- z, z <- -x, so compute the rotation in two
        // different ways, and verify that the result is the same to within the
        // floating point inaccuracy margin.
        let computed = rotate(v, &MQuaternion::broadcast(rotation));
        let expected = MVector3::new(v.z, v.y, -v.x);
        assert_mvectors_equal(expected, computed, 1e-6);
    }
}

#[test]
fn rotate_z() {
    let half_sqrt_2 = 0.5 * 2.0_f32.sqrt();
    let rotation = SQuaternion::new(half_sqrt_2, 0.0, 0.0, half_sqrt_2);
    let vectors = bench::mvectors_on_unit_sphere(32);
    for v in &vectors {
        // Rotate the vector by pi/2 radians around the y-axis. This is
        // equivalent to y <- x, x <- -y, so compute the rotation in two
        // different ways, and verify that the result is the same to within the
        // floating point inaccuracy margin.
        let computed = rotate(v, &MQuaternion::broadcast(rotation));
        let expected = MVector3::new(-v.y, v.x, v.z);
        assert_mvectors_equal(expected, computed, 1e-6);
    }
}

#[test]
fn interpolate() {
    use vector3::SVector3;
    let half_sqrt_2 = 0.5 * 2.0_f32.sqrt();
    let identity = MQuaternion::broadcast(SQuaternion::new(1.0, 0.0, 0.0, 0.0));
    let rotate_z = MQuaternion::broadcast(SQuaternion::new(half_sqrt_2, 0.0, 0.0, half_sqrt_2));
    let rotation = identity.interpolate(&rotate_z, Mf32::broadcast(0.5));
    let v = MVector3::broadcast(SVector3::new(1.0, 0.0, 0.0));
    let expected = MVector3::broadcast(SVector3::new(half_sqrt_2, half_sqrt_2, 0.0));
    let computed = rotate(&v, &rotation);
    assert_mvectors_equal(expected, computed, 1e-6);
}

macro_rules! unroll_10 {
    { $x: block } => {
        $x $x $x $x $x $x $x $x $x $x
    }
}

#[bench]
fn bench_rotate_10(b: &mut test::Bencher) {
    let vectors = bench::mvectors_on_unit_sphere(4096 / 8);
    let quaternions = bench::unit_mquaternions(4096 / 8);
    let mut it = vectors.iter().cycle().zip(quaternions.iter().cycle());
    b.iter(|| {
        let (v, q) = it.next().unwrap();
        unroll_10! {{
            test::black_box(rotate(test::black_box(v), test::black_box(q)));
        }};
    });
}

#[bench]
fn bench_interpolate_10(b: &mut test::Bencher) {
    let q0s = bench::unit_mquaternions(4096 / 8);
    let q1s = bench::unit_mquaternions(4096 / 8);
    let ts = bench::mf32_unit(4096 / 8);
    let mut it = q0s.iter().cycle().zip(q1s.iter().cycle()).zip(ts.iter().cycle());
    b.iter(|| {
        let ((q0, q1), &t) = it.next().unwrap();
        unroll_10! {{
            test::black_box(test::black_box(q0).interpolate(test::black_box(q1), test::black_box(t)));
        }};
    });
}
