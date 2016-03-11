//! This module implements the ray and related structures.

use simd::{Mask, Mf32};
use std::ops::Neg;
use vector3::{MVector3, SVector3};

#[derive(Clone)]
pub struct SRay {
    pub origin: SVector3,
    pub direction: SVector3,
}

#[derive(Clone)]
pub struct MRay {
    pub origin: MVector3,
    pub direction: MVector3,
}

pub struct SIntersection {
    /// The position at which the ray intersected the surface.
    pub position: SVector3,

    /// The surface normal at the intersection point.
    pub normal: SVector3,

    /// This distance between the ray origin and the position.
    pub distance: f32,
}

pub struct OctaIntersection {
    pub position: MVector3,
    pub normal: MVector3,
    pub distance: Mf32,
}

impl SRay {
    pub fn new(origin: SVector3, direction: SVector3) -> SRay {
        SRay {
            origin: origin,
            direction: direction,
        }
    }
}

impl MRay {
    /// Builds an mray by applying the function to the numbers 0..7.
    ///
    /// Note: this is essentially a transpose, avoid in hot code.
    pub fn generate<F>(mut f: F) -> MRay where F: FnMut(usize) -> SRay {
        MRay {
            origin: MVector3::generate(|i| f(i).origin),
            direction: MVector3::generate(|i| f(i).direction),
        }
    }

    pub fn advance_epsilon(&self) -> MRay {
        let epsilon = Mf32::broadcast(1.0e-5);
        MRay {
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

impl Neg for SRay {
    type Output = SRay;

    fn neg(self) -> SRay {
        SRay {
            origin: self.origin,
            direction: -self.direction,
        }
    }
}
