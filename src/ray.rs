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

pub struct MIntersection {
    /// The position at which the ray intersected the surface.
    pub position: MVector3,

    /// The surface normal at the intersection point.
    pub normal: MVector3,

    /// This distance between the ray origin and the position.
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
    pub fn new(origin: MVector3, direction: MVector3) -> MRay {
        MRay {
            origin: origin,
            direction: direction,
        }
    }

    pub fn broadcast(ray: &SRay) -> MRay {
        MRay {
            origin: MVector3::broadcast(ray.origin),
            direction: MVector3::broadcast(ray.direction),
        }
    }

    /// Builds an mray by applying the function to the numbers 0..7.
    ///
    /// Note: this is essentially a transpose, avoid in hot code.
    pub fn generate<F>(mut f: F) -> MRay where F: FnMut(usize) -> SRay {
        MRay {
            origin: MVector3::generate(|i| f(i).origin),
            direction: MVector3::generate(|i| f(i).direction),
        }
    }
}

impl MIntersection {
    pub fn pick(&self, other: &MIntersection, mask: Mask) -> MIntersection {
        MIntersection {
            position: self.position.pick(other.position, mask),
            normal: self.normal.pick(other.normal, mask),
            distance: self.distance.pick(other.distance, mask),
        }
    }
}

impl Neg for MRay {
    type Output = MRay;

    fn neg(self) -> MRay {
        MRay {
            origin: self.origin,
            direction: MVector3::zero() - self.direction,
        }
    }
}
