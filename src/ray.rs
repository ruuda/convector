//! This module implements the ray and related structures.

use simd::OctaF32;
use std::ops::Neg;
use vector3::{OctaVector3, Vector3};

pub struct Ray {
    pub origin: Vector3,
    pub direction: Vector3,
}

pub struct OctaRay {
    pub origin: OctaVector3,
    pub direction: OctaVector3,
}

pub struct Intersection {
    /// The position at which the ray intersected the surface.
    pub position: Vector3,

    /// The surface normal at the intersection point.
    pub normal: Vector3,

    /// This distance between the ray origin and the position.
    pub distance: f32,
}

pub struct OctaIntersection {
    pub position: OctaVector3,
    pub normal: OctaVector3,
    pub distance: OctaF32,
}

/// Returns the nearest of two intersections.
pub fn nearest(i1: Option<Intersection>, i2: Option<Intersection>) -> Option<Intersection> {
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

impl Ray {
    pub fn new(origin: Vector3, direction: Vector3) -> Ray {
        Ray {
            origin: origin,
            direction: direction,
        }
    }

    pub fn advance_epsilon(&self) -> Ray {
        Ray {
            origin: self.origin + self.direction * 1.0e-5,
            direction: self.direction,
        }
    }
}

impl OctaRay {
    pub fn new(origin: OctaVector3, direction: OctaVector3) -> OctaRay {
        OctaRay {
            origin: origin,
            direction: direction,
        }
    }

    pub fn advance_epsilon(&self) -> OctaRay {
        let epsilon = OctaF32::broadcast(1.0e-5);
        OctaRay {
            origin: self.direction.mul_add(epsilon, self.origin),
            direction: self.direction,
        }
    }
}

impl Neg for Ray {
    type Output = Ray;

    fn neg(self) -> Ray {
        Ray {
            origin: self.origin,
            direction: -self.direction,
        }
    }
}
