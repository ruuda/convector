//! Determines how light bounces off a surface.
//!
//! # Material encoding
//!
//! A material is associated with every surface. It is a 32-bit value that
//! consists of the following parts:
//!
//!  * Bit 31 (sign bit): if 1, the material is emissive, if 0,
//!    the material is not.
//!
//!  * Bit 30: if 1, a primitive with this material is eligible for direct
//!    sampling.
//!
//!  * Bit 29: if 1, this material is a glass material.
//!
//!  * Bits 24-37 contain the texture index ranging from 0 to 7.
//!
//!  * Bits 0-23 contain the RGB color of the material, red in the least
//!    significant bits, blue in the most significant bits.
//!
//! # A note on CPU and GPU shading
//!
//! Doing texture lookup and filtering on the CPU is extremely expensive.
//! Looking up four pixels in a bitmap is essentially doing random access into
//! memory. Everything will be a cache miss, and it will trash the cache for
//! other data too. The GPU is optimized for this kind of thing, so it would be
//! nice if we could to texture lookup and filtering there.
//!
//! Without fancy materials, every bounce off a surface multiplies the pixel
//! color by a factor for each channel. To compute the final color of a pixel,
//! we collect a color at every bounce, and multiply all of these together. To
//! avoid texture lookup on the CPU, we could send a buffer with texture
//! coordinates to the GPU per bounce, and do the lookup there. However, that is
//! a lot of data even for a few bounces. Sending a 720p R8G8B8A8 buffer to the
//! GPU takes about 2 ms already, and we don't want to spend the entire frame
//! doing texture upload. So here's an idea:
//!
//!  * If textures are only used to add a bit of detail, not for surfaces that
//!    vary wildly in color, then after one bounce we could simply not sample
//!    the texture, and take an average surface color. For diffuse surfaces
//!    light from all directions is mixed anyway, so the error is very small.
//!
//!  * We can store the average surface color with the material and compute
//!    all the shading on the CPU, except for the contribution of the first
//!    bounce. Then send only one set of texture coordinates to the GPU, plus
//!    the color computed on the CPU.
//!
//!  * For pure specular reflections, the texture lookup can be postponed to the
//!    next bounce. It does not matter for which bounce we do the lookup, but we
//!    can only do one per pixel.

use random::Rng;
use ray::{MIntersection, MRay};
use scene::Scene;
use simd::Mf32;
use std::f32::consts;
use vector3::MVector3;

#[derive(Copy, Clone, Debug)]
pub struct SMaterial(u32);

pub type MMaterial = Mf32;

pub struct MDirectSample {
    pub position: MVector3,
    pub normal: MVector3,
    pub area: Mf32,
}

impl SMaterial {
    pub fn sky() -> SMaterial {
        // Set only the emissive bit.
        // TODO: Disable the direct sampling bit once I have glass windows.
        let mat = 0b11000000_00000000_00000000_00000000_u32;
        SMaterial(mat)
    }

    /// A white diffuse material.
    pub fn white() -> SMaterial {
        SMaterial::diffuse(255, 255, 255)
    }

    /// A diffuse material with the given color.
    pub fn diffuse(r: u8, g: u8, b: u8) -> SMaterial {
        let mat = ((b as u32) << 16) | ((g as u32) << 8) | (r as u32);
        SMaterial(mat)
    }

    /// A transparent and reflective material.
    pub fn glass() -> SMaterial {
        let mat = 0b0110_0000_00000000_00000000_00000000_u32;
        SMaterial(mat)
    }

    /// Returns whether the material is eligible for direct sampling.
    pub fn is_direct_sample(&self) -> bool {
        let ds_mask = 0b01000000_00000000_00000000_00000000;
        let SMaterial(mat) = *self;
        (mat & ds_mask) == ds_mask
    }
}

impl MMaterial {
    pub fn broadcast_material(material: SMaterial) -> MMaterial {
        use std::mem::transmute;
        let SMaterial(mat) = material;
        let matf: f32 = unsafe { transmute(mat) };
        Mf32::broadcast(matf)
    }

    pub fn sky() -> MMaterial {
        MMaterial::broadcast_material(SMaterial::sky())
    }
}

/// Returns the sky color for a ray in the given direction.
pub fn sky_intensity(ray_direction: MVector3) -> MVector3 {
    // TODO: Better sky model.
    let up = MVector3::new(Mf32::zero(), Mf32::zero(), Mf32::one());
    let half = Mf32::broadcast(0.5);
    let two = Mf32::broadcast(2.0);
    let d = ray_direction.dot(up).mul_add(half, half);
    let r = d;
    let g = d * d;
    let b = d * (d * d);
    MVector3::new(r, g, b).mul_add(two, MVector3::new(half, half, half))
}

/// Continues the path of a photon by sampling the BRDF.
///
/// Returns the new ray, the probability density for that ray, and the color
/// modulation for the bounce.
#[inline(always)]
fn continue_path_brdf(ray: &MRay,
                      isect: &MIntersection,
                      rng: &mut Rng)
                      -> (MRay, Mf32, MVector3) {
    // Bounce in a random direction in the hemisphere around the surface
    // normal, with a cosine-weighted distribution, for a diffuse bounce.
    let dir_z = rng.sample_hemisphere_vector();
    let direction = dir_z.rotate_hemisphere(isect.normal);

    // Build a new ray, offset by an epsilon from the intersection so we
    // don't intersect the same surface again.
    let origin = direction.mul_add(Mf32::epsilon(), isect.position);
    let new_ray = MRay {
        origin: origin,
        direction: direction,
        active: Mf32::zero(),
    };

    // The probability density for the ray is dot(normal, direction) divided by
    // the intgral of that over the hemisphere (which happens to be pi).
    let pd = dir_z.z * Mf32::broadcast(1.0 / consts::PI);

    // There is the factor dot(normal, direction) that modulates the
    // incoming contribution. The incoming energy is then radiated evenly in all
    // directions (the diffuse assumption), so the integral over the hemisphere
    // of that factor (excluding the dot, that one was for _incoming_ energy)
    // should be 1. The area of the hemisphere is 2pi, so divide by that.
    let modulation = Mf32::broadcast(0.5 / consts::PI) * dir_z.z;
    let color_mod = MVector3::new(modulation, modulation, modulation);

    (new_ray, pd, color_mod)
}

/// Continues the path of a photon by sampling a point on a surface.
///
/// Returns the new ray, the probability density for that ray, and the color
/// modulation for the bounce.
fn continue_path_direct_sample(scene: &Scene,
                               isect: &MIntersection,
                               rng: &mut Rng)
                               -> (MRay, Mf32, MVector3) {
    // Get two candidate points on a light source to sample. One of those will
    // be picked by resampled importance sampling.
    let ds_0 = scene.get_direct_sample(rng);
    let ds_1 = scene.get_direct_sample(rng);
    let num = scene.direct_sample_num();
    debug_assert!(num > 0);

    let to_surf_0 = ds_0.position - isect.position;
    let to_surf_1 = ds_1.position - isect.position;
    let distance_sqr_0 = to_surf_0.norm_squared();
    let distance_sqr_1 = to_surf_1.norm_squared();
    let direction_0 = to_surf_0 * distance_sqr_0.rsqrt();
    let direction_1 = to_surf_1 * distance_sqr_1.rsqrt();

    // Take the absolute value because the probability density should not be
    // negative. This is equivalent to making direct sampling surfaces
    // two-sided.
    let dot_surface_0 = isect.normal.dot(direction_0).abs();
    let dot_surface_1 = isect.normal.dot(direction_1).abs();

    // Now pick either sample 0 or sample 1 with a probability weighed by the
    // angle with the surface. After all, this angle weighs the contribution,
    // so to lower variance, favor a sample that will contribute more.
    let (w0, w1) = (dot_surface_0, dot_surface_1);
    let w = (w0 + w1).recip_fast();
    let (p0, p1) = (w0 * w, w1 * w);

    // If the sign bit of rr is positive (which happens with probability p0), we
    // take sample 0, otherwise take sample 1.
    let rr = p0 - rng.sample_unit();

    let direction = direction_0.pick(direction_1, rr);
    let distance_sqr = distance_sqr_0.pick(distance_sqr_1, rr);
    let area = ds_0.area.pick(ds_1.area, rr);
    let dot_emissive = ds_0.normal.pick(ds_1.normal, rr).dot(direction).abs();
    let dot_surface = dot_surface_0.pick(dot_surface_1, rr);

    // We need to modulate the results by p0 or p1 based on which one we picked
    // to keep the estimator unbiased.
    let modulation = Mf32::broadcast(0.5) * p0.pick(p1, rr).recip_fast();

    // Build a new ray, offset by an epsilon from the intersection so we
    // don't intersect the same surface again.
    let origin = direction.mul_add(Mf32::epsilon(), isect.position);
    let new_ray = MRay {
        origin: origin,
        direction: direction,
        active: Mf32::zero(),
    };

    // The probability density for the point on the triangle is simply 1/area,
    // but we want to know the probability of the ray direction, not the
    // probability of the point. The conversion factor is cos(phi)/r^2, where
    // phi is the angle between the ray and the surface normal. A hand-waving
    // justification: imagine a small triangle on a unit hemisphere, small
    // enough that its area equals the solid angle it subtends. Then the pdf for
    // the point an the ray will be equal (1/area). Now move the triangle away.
    // The solid angle decreases proportional to r^2, so we must compensate the
    // pdf to keep it normalized. Now rotate the small triangle. When cos(phi)
    // is 0, the projection is a line of zero surface area, but it needs to
    // integrate to 0, so the pdf goes to infinity as cos(phi) goes to 0. So far
    // this is the probability per triangle, but we picked one out of num
    // triangles uniformly, so compensate for that too.
    let numf = Mf32::broadcast(num as f32);
    let pd = distance_sqr * (area * dot_emissive * numf).recip_fast();
    // TODO: Do I need to mix in p0 or p1 here? Because now my original
    // probability is not simply 1/area any more.

    // For modulation, there is only the cosine factor of the diffuse BRDF and
    // again its factor 1/2pi.
    let modulation = modulation * Mf32::broadcast(0.5 / consts::PI) * dot_surface;
    let color_mod = MVector3::new(modulation, modulation, modulation);

    debug_assert!(pd.all_finite());
    debug_assert!(pd.all_sign_bits_positive(), "probability density cannot be negative");
    debug_assert!(modulation.all_finite());
    debug_assert!(modulation.all_sign_bits_positive(), "color modulation cannot be negative");

    (new_ray, pd, color_mod)
}

pub fn pd_brdf(isect: &MIntersection, ray: &MRay) -> Mf32 {
    let dot_surface = isect.normal.dot(ray.direction).abs();
    dot_surface * Mf32::broadcast(1.0 / consts::PI)
}

/// Returns the probability density for the given ray, for the direct sampling
/// distribution.
pub fn pd_direct_sample(scene: &Scene, ray: &MRay) -> Mf32 {
    let mut pd_total = Mf32::zero();
    scene.foreach_direct_sample(|triangle| {
        let isect = triangle.intersect_direct(ray);
        let distance_sqr = isect.distance * isect.distance;
        let dot_emissive = isect.normal.dot(ray.direction).abs();
        let pd = distance_sqr * (isect.area * dot_emissive).recip_fast();

        // Add the probability density if the triangle was intersected.
        pd_total = (pd_total + pd).pick(pd_total, isect.mask);
    });
    pd_total * Mf32::broadcast(1.0 / scene.direct_sample_num() as f32)
}

/// Continues the path of a photon.
///
/// If a ray intersected a surface with a certain material, then this will
/// compute the ray that continues the light path. A factor to multiply the
/// final color by is returned as well.
pub fn continue_path(scene: &Scene,
                     ray: &MRay,
                     isect: &MIntersection,
                     rng: &mut Rng)
                     -> (MRay, MVector3) {
    // Emissive materials have the sign bit set to 1, and a sign bit of 1
    // means that the ray is inactive. So hitting an emissive material
    // deactivates the ray: there is no need for an additional bounce.
    let active = ray.active | isect.material;

    let (brdf_ray, brdf_pd, brdf_mod) = continue_path_brdf(ray, isect, rng);
    let color_mod = brdf_mod * brdf_pd.recip_fast();
    let new_ray = MRay {
        origin: brdf_ray.origin.pick(ray.origin, active),
        direction: brdf_ray.direction.pick(ray.direction, active),
        active: active,
    };

    // let (direct_ray, direct_pd, direct_mod) = continue_path_direct_sample(scene, isect, rng);
    // let color_mod = direct_mod * direct_pd.recip_fast();
    // let new_ray = MRay {
    //     origin: direct_ray.origin.pick(ray.origin, active),
    //     direction: direct_ray.direction.pick(ray.direction, active),
    //     active: active,
    // };

    let white = MVector3::new(Mf32::one(), Mf32::one(), Mf32::one());
    let color_mod = color_mod.pick(white, active);

    debug_assert!(color_mod.x.all_sign_bits_positive(), "red color modulation can never be negative");
    debug_assert!(color_mod.y.all_sign_bits_positive(), "green color modulation can never be negative");
    debug_assert!(color_mod.z.all_sign_bits_positive(), "blue color modulation can never be negative");

    (new_ray, color_mod)
}
