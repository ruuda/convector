//! This module implements the ray and related structures.

use std::ops::Neg;
use vector3::Vector3;

pub struct Ray {
    pub origin: Vector3,
    pub direction: Vector3,
}

impl Ray {
    pub fn advance_epsilon(&self) -> Ray {
        Ray {
            origin: self.origin + self.direction * 1.0e-5,
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

// TODO: Do not derive clone, sort out the BVH traversal mess.
#[derive(Clone)]
pub struct Intersection {
    /// The position at which the ray intersected the surface.
    pub position: Vector3,

    /// The surface normal at the intersection point.
    pub normal: Vector3,

    /// This distance between the ray origin and the position.
    pub distance: f32,
}

