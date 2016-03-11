//! This module implements the ray and related structures.

use simd::{Mask, Mf32};
use std::ops::Neg;
use vector3::{OctaVector3, SVector3};

#[derive(Clone)]
pub struct Ray {
    pub origin: SVector3,
    pub direction: SVector3,
}

#[derive(Clone)]
pub struct OctaRay {
    pub origin: OctaVector3,
    pub direction: OctaVector3,
}

pub struct Intersection {
    /// The position at which the ray intersected the surface.
    pub position: SVector3,

    /// The surface normal at the intersection point.
    pub normal: SVector3,

    /// This distance between the ray origin and the position.
    pub distance: f32,
}

pub struct OctaIntersection {
    pub position: OctaVector3,
    pub normal: OctaVector3,
    pub distance: Mf32,
}

impl Ray {
    pub fn new(origin: SVector3, direction: SVector3) -> Ray {
        Ray {
            origin: origin,
            direction: direction,
        }
    }
}

impl OctaRay {
    /// Builds an octaray by applying the function to the numbers 0..7.
    ///
    /// Note: this is essentially a transpose, avoid in hot code.
    pub fn generate<F: FnMut(usize) -> Ray>(mut f: F) -> OctaRay {
        OctaRay {
            origin: OctaVector3::generate(|i| f(i).origin),
            direction: OctaVector3::generate(|i| f(i).direction),
        }
    }

    pub fn advance_epsilon(&self) -> OctaRay {
        let epsilon = Mf32::broadcast(1.0e-5);
        OctaRay {
            origin: self.direction.mul_add(epsilon, self.origin),
            direction: self.direction,
        }
    }
}

impl OctaIntersection {
    pub fn pick(&self, other: &OctaIntersection, mask: Mask) -> OctaIntersection {
        OctaIntersection {
            position: self.position.pick(other.position, mask),
            normal: self.normal.pick(other.normal, mask),
            distance: self.distance.pick(other.distance, mask),
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
