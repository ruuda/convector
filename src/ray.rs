//! This module implements the ray and related structures.

use std::ops::Neg;
use vector3::Vector3;

pub struct Ray {
    pub origin: Vector3,
    pub direction: Vector3,
}

pub struct Intersection {
    /// The position at which the ray intersected the surface.
    pub position: Vector3,

    /// The surface normal at the intersection point.
    pub normal: Vector3,

    /// This distance between the ray origin and the position.
    pub distance: f32,
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
