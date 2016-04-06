//! This module implements the ray and related structures.

use material::MMaterial;
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
    pub active: Mask,
}

pub struct MIntersection {
    /// The position at which the ray intersected the surface.
    pub position: MVector3,

    /// The surface normal at the intersection point.
    pub normal: MVector3,

    /// This distance between the ray origin and the position.
    pub distance: Mf32,

    /// The material at the intersection surface.
    pub material: MMaterial,

    /// Texture coordinates at the intersection point.
    pub tex_coords: (Mf32, Mf32),
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
            active: Mask::ones(),
        }
    }

    pub fn broadcast(ray: &SRay) -> MRay {
        MRay {
            origin: MVector3::broadcast(ray.origin),
            direction: MVector3::broadcast(ray.direction),
            active: Mask::ones(),
        }
    }

    /// Builds an mray by applying the function to the numbers 0..7.
    ///
    /// Note: this is essentially a transpose, avoid in hot code.
    pub fn generate<F>(mut f: F) -> MRay where F: FnMut(usize) -> SRay {
        MRay {
            origin: MVector3::generate(|i| f(i).origin),
            direction: MVector3::generate(|i| f(i).direction),
            active: Mask::ones(),
        }
    }
}

impl MIntersection {
    /// Constructs an empyt intersection with the specified distance and zeroes
    /// in all other fields.
    pub fn with_max_distance(max_dist: f32) -> MIntersection {
        MIntersection {
            position: MVector3::zero(),
            normal: MVector3::zero(),
            distance: Mf32::broadcast(max_dist),
            material: Mf32::zero(),
            tex_coords: (Mf32::zero(), Mf32::zero()),
        }
    }

    pub fn pick(&self, other: &MIntersection, mask: Mask) -> MIntersection {
        let u = self.tex_coords.0.pick(other.tex_coords.0, mask);
        let v = self.tex_coords.1.pick(other.tex_coords.1, mask);
        MIntersection {
            position: self.position.pick(other.position, mask),
            normal: self.normal.pick(other.normal, mask),
            distance: self.distance.pick(other.distance, mask),
            material: self.material.pick(other.material, mask),
            tex_coords: (u, v),
        }
    }
}

impl Neg for MRay {
    type Output = MRay;

    fn neg(self) -> MRay {
        MRay {
            origin: self.origin,
            direction: MVector3::zero() - self.direction,
            active: self.active,
        }
    }
}
