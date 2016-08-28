// Convector -- An interactive CPU path tracer
// Copyright 2016 Ruud van Asseldonk

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License version 3. A copy
// of the License is available in the root of the repository.

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
//!  * Bits 26-28: the 2-log of the exponent for the Blinn-Phong BRDF plus one.
//!    Must be between 0 and 6 (inclusive), so the exponent can be 0, 1, 2, 4,
//!    8, or 16.
//!
//!  * Bits 24-25: the texture index, ranging from 0 to 3.
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
//!
//!  * When the Fresnel factor is taken into account, it must be sent to the GPU
//!    too (because it blends between the surface color and pure white).

use random::Rng;
use ray::{MIntersection, MRay};
use scene::Scene;
use simd::{Mask, Mf32, Mi32};
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
        SMaterial::diffuse(1.0, 1.0, 1.0)
    }

    /// A diffuse material with the given color.
    pub fn diffuse(r: f32, g: f32, b: f32) -> SMaterial {
        let mat = (((b * 255.0) as u32) << 16) | (((g * 255.0) as u32) << 8) | ((r * 255.0) as u32);
        SMaterial(mat)
    }

    /// A transparent and reflective material.
    pub fn glass() -> SMaterial {
        let mat = 0b0110_0000_00000000_00000000_00000000_u32;
        SMaterial(mat)
    }

    /// Sets the glossiness of the material. Valid values are 0 (completely
    /// diffuse) trough 6 (a bit glossy, but not mirror-like).
    pub fn with_glossiness(self, glossiness: u32) -> SMaterial {
        assert!(glossiness <= 6);
        let SMaterial(mat) = self;
        // Mask that resets the glossiness to 0.
        let no_gloss = 0b11100011_11111111_11111111_11111111_u32;
        let mat = (mat & no_gloss) | (glossiness << 26);
        SMaterial(mat)
    }

    /// Sets the texture index of the material. Valid values are 0 through 3.
    pub fn with_texture(self, texture_index: u32) -> SMaterial {
        assert!(texture_index <= 3);
        let SMaterial(mat) = self;
        // Mask that resets the texture index to 0.
        let no_tidx = 0b11111100_11111111_11111111_11111111_u32;
        let mat = (mat & no_tidx) | (texture_index << 24);
        SMaterial(mat)
    }

    /// Returns whether the material is eligible for direct sampling.
    pub fn is_direct_sample(self) -> bool {
        let ds_mask = 0b01000000_00000000_00000000_00000000;
        let SMaterial(mat) = self;
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

    /// Unpacks the color information from the material.
    pub fn get_color(&self) -> MVector3 {
        use std::mem::transmute;
        let mask = Mi32::broadcast(0xff);

        // Shift and mask out the component bytes.
        let mi32: Mi32 = unsafe { transmute(*self) };
        let mir255 = mi32 & mask;
        let mig255 = mi32.map(|x| x >> 8) & mask;
        let mib255 = mi32.map(|x| x >> 16) & mask;

        // Convert bytes into floats in the range [0.0, 255.0].
        let mfr255 = mir255.into_mf32();
        let mfg255 = mig255.into_mf32();
        let mfb255 = mib255.into_mf32();

        // Convert to a color in the range [0.0, 1.0].
        MVector3::new(mfr255, mfg255, mfb255) * Mf32::broadcast(1.0 / 255.0)
    }

    /// Unpacks the Blinn-Phong glossiness exponent.
    pub fn get_glossiness(&self) -> Mi32 {
        use std::mem::transmute;

        // Extract the three glossiness exponent bits.
        let mati: Mi32 = unsafe { transmute(*self) };
        let exponent = mati.map(|x| x >> 26);
        exponent & Mi32::broadcast(0b111)
    }

    /// Unpacks the texture index.
    pub fn get_texture(&self) -> Mi32 {
        use std::mem::transmute;

        // Extract the two texture index bits.
        let mati: Mi32 = unsafe { transmute(*self) };
        let tidx = mati.map(|x| x >> 24);
        tidx & Mi32::broadcast(0b11)
    }

    /// Sets the sign bit to 1 if the surface has a texture, or 0 if the texture
    /// index is 0 (indicating no texture).
    pub fn has_texture(&self) -> Mask {
        use std::mem::transmute;

        // Take the bitwise or of all bits that determine the texture ID. Only
        // if the texture ID was zero will this result in 0.
        let mati: Mi32 = unsafe { transmute(*self) };
        let has_tex = mati.map(|x| x << 6) | mati.map(|x| x << 7);

        unsafe { transmute(has_tex) }
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
/// This samples the hemisphere in a cosine-weighted distribution, it does not
/// sample proportional to the Blinn-Phong BRDF. There is a correction factor
/// when the final color is computed, so this does not introduce bias, it just
/// leads to more variance because the pdf from which samples are drawn is not
/// such a good match for the BRDF as it could be. However, in my case the
/// Blinn-Phong BRDF is only used with low exponents (diffuse, not very glossy),
/// so the cosine distribution still does a decent job, and it is much cheaper
/// to sample from than a distribution specific for the Blinn-Phong BRDF.
#[inline(always)]
fn continue_path_brdf(isect: &MIntersection, rng: &mut Rng) -> MRay {
    // Bounce in a random direction in the hemisphere around the surface
    // normal, with a cosine-weighted distribution, for a diffuse bounce.
    let dir_z = rng.sample_hemisphere_vector();
    let direction = dir_z.rotate_hemisphere(isect.normal);

    // Build a new ray, offset by an epsilon from the intersection so we
    // don't intersect the same surface again.
    let origin = direction.mul_add(Mf32::epsilon(), isect.position);
    MRay {
        origin: origin,
        direction: direction,
        active: Mf32::zero(),
    }
}

/// Returns the probability density for the BRDF sampler at a given ray.
fn pd_brdf(isect: &MIntersection, ray: &MRay) -> Mf32 {
    // The probability density for the ray is dot(normal, direction) divided by
    // the intgral of that over the hemisphere (which happens to be pi).
    let dot_surface = isect.normal.dot(ray.direction).max(Mf32::zero());
    dot_surface * Mf32::broadcast(1.0 / consts::PI)
}

/// Continues the path of a photon by sampling a point on a surface.
fn continue_path_direct_sample(scene: &Scene, isect: &MIntersection, rng: &mut Rng) -> MRay {
    let ds = scene.get_direct_sample(rng);
    let direction = (ds.position - isect.position).normalized();

    // Build a new ray, offset by an epsilon from the intersection so we
    // don't intersect the same surface again.
    let origin = direction.mul_add(Mf32::epsilon(), isect.position);
    MRay {
        origin: origin,
        direction: direction,
        active: Mf32::zero(),
    }
}

/// Asserts that the values where active has sign bit 0 (positive) are nonzero.
fn debug_assert_all_nonzero(x: Mf32, active: Mf32, tag: &str) {
    debug_assert!(x.0 != 0.0 || active.0.is_sign_negative(), "{} {:?} must be nonzero", tag, x);
    debug_assert!(x.1 != 0.0 || active.1.is_sign_negative(), "{} {:?} must be nonzero", tag, x);
    debug_assert!(x.2 != 0.0 || active.2.is_sign_negative(), "{} {:?} must be nonzero", tag, x);
    debug_assert!(x.3 != 0.0 || active.3.is_sign_negative(), "{} {:?} must be nonzero", tag, x);
    debug_assert!(x.4 != 0.0 || active.4.is_sign_negative(), "{} {:?} must be nonzero", tag, x);
    debug_assert!(x.5 != 0.0 || active.5.is_sign_negative(), "{} {:?} must be nonzero", tag, x);
    debug_assert!(x.6 != 0.0 || active.6.is_sign_negative(), "{} {:?} must be nonzero", tag, x);
    debug_assert!(x.7 != 0.0 || active.7.is_sign_negative(), "{} {:?} must be nonzero", tag, x);
}

/// Returns the probability density for the given ray, for the direct sampling
/// distribution.
fn pd_direct_sample(scene: &Scene, ray: &MRay) -> Mf32 {
    let mut pd_total = Mf32::zero();
    scene.foreach_direct_sample(|triangle| {
        // The probability density for the point on the triangle is simply
        // 1/area, but we want to know the probability of the ray direction, not
        // the probability of the point. The conversion factor is cos(phi)/r^2,
        // where phi is the angle between the ray and the surface normal. A
        // hand-waving justification: imagine a small triangle on a unit
        // hemisphere, small enough that its area equals the solid angle it
        // subtends. Then the pdf for the point and the ray will be equal
        // (1/area). Now move the triangle away. The solid angle decreases
        // proportional to r^2, so we must compensate the pdf to keep it
        // normalized. Now rotate the small triangle. When cos(phi) is 0, the
        // projection is a line of zero surface area, but it needs to integrate
        // to 1, so the pdf goes to infinity as cos(phi) goes to 0.
        let sample_isect = triangle.intersect_direct(ray);
        let distance_sqr = sample_isect.distance * sample_isect.distance;
        // Add a small constant to avoid division by zero later on.
        let dot_emissive = sample_isect.normal.dot(ray.direction).abs() + Mf32::broadcast(0.0001);
        let pd = distance_sqr * (sample_isect.area * dot_emissive).recip_fast();

        debug_assert_all_nonzero(sample_isect.area, ray.active, "area");
        debug_assert_all_nonzero(dot_emissive, ray.active, "dot_emissive");

        // Add the probability density if the triangle was intersected. If the
        // triangle was not intersected, the probability of sampling it directly
        // was 0.
        pd_total = (pd_total + pd).pick(pd_total, sample_isect.mask);

        debug_assert!(pd_total.all_finite());
        debug_assert!(pd_total.all_sign_bits_positive(),
                      "probability density must be positive");
    });

    // So far we computed the probability density per triangle, but we picked
    // one triangle uniformly, so compensate for that too.
    // TODO: I can make my pick non-uniform but take the angle into account.
    // That should reduce variance.
    pd_total * Mf32::broadcast(1.0 / scene.direct_sample_num() as f32)
}

/// Continues the path of a photon.
///
/// If a ray intersected a surface with a certain material, then this will
/// compute the ray that continues the light path. A factor to multiply the
/// final color by is returned as well, and the Fresnel factor.
pub fn continue_path(material: MMaterial,
                     scene: &Scene,
                     ray: &MRay,
                     isect: &MIntersection,
                     rng: &mut Rng,
                     ignore_fresnel: bool)
                     -> (MRay, MVector3, Mf32) {

    // Emissive materials have the sign bit set to 1, and a sign bit of 1
    // means that the ray is inactive. So hitting an emissive material
    // deactivates the ray: there is no need for an additional bounce.
    let active = ray.active | isect.material;

    // Generate one ray by sampling the BRDF, and one ray for direct light
    // sampling.
    let ray_brdf = continue_path_brdf(isect, rng);
    let ray_direct = continue_path_direct_sample(scene, isect, rng);

    // Randomly pick one of the two rays to use, then compute the weight for
    // multiple importance sampling. **Cheat Alert** with which probability do
    // we pick BRDF sampling or direct sampling? All my light sources are on one
    // side of the scene, so for surfaces that face the light source, direct
    // sampling is going to work well. For surfaces that do not face the light
    // source, direct sampling is going to return black and the only
    // contribution comes from indirect light. So there we want to sample the
    // BRDF. Solution: pick with a probability proportional to the z-component
    // of the normal.
    let rr = isect.normal.z.mul_add(Mf32::broadcast(0.8), rng.sample_biunit());
    let new_ray = MRay {
        origin: ray_brdf.origin.pick(ray_direct.origin, rr),
        direction: ray_brdf.direction.pick(ray_direct.direction, rr),
        active: Mf32::zero(),
    };
    let pd_brdf = pd_brdf(isect, &new_ray);
    let pd_direct = pd_direct_sample(scene, &new_ray);
    // Add a small constant to avoid division by zero later on.
    let weight_denom = pd_brdf + pd_direct + Mf32::broadcast(0.01);

    debug_assert!(weight_denom.all_finite());
    debug_assert!(pd_brdf.all_sign_bits_positive(), "probability density cannot be negative");
    debug_assert!(pd_direct.all_sign_bits_positive(), "probability density cannot be negative");

    // There is a compensation factor 1 / (probability that sampling method was
    // chosen) in the multiple importance sampler. There is a probability of
    // (0.5 + normal.z * 0.4) of picking BRDF sampling.
    let half = Mf32::broadcast(0.5);
    let p_brdf = isect.normal.z.mul_add(Mf32::broadcast(0.4), half);
    let p_direct = isect.normal.z.neg_mul_add(Mf32::broadcast(0.4), half);
    let p = p_brdf.pick(p_direct, rr);

    // Compute the contribution using the one-sample multiple importance
    // sampler. This is equation 9.15 from section 9.2.4 of Veach, 1997. The
    // probability densities in the weight and denominator cancel, so they have
    // been left out. Finally, there is the correction factor for the incident
    // angle.
    let modulation = (weight_denom * p).recip_fast();
    let cos_theta = isect.normal.dot(new_ray.direction).max(Mf32::zero());
    let (brdf_term, fresnel) = microfacet_brdf(material, &new_ray, ray, isect, ignore_fresnel);
    let color_mod = brdf_term * (modulation * cos_theta);

    debug_assert!(modulation.all_finite());
    debug_assert!(brdf_term.all_finite());
    debug_assert!(modulation.all_sign_bits_positive(), "color modulation can never be negative");
    debug_assert!(brdf_term.x.all_sign_bits_positive(), "red brdf term can never be negative");
    debug_assert!(brdf_term.y.all_sign_bits_positive(), "green brdf term can never be negative");
    debug_assert!(brdf_term.z.all_sign_bits_positive(), "blue brdf term can never be negative");

    // Limit the color modulation to avoid fireflies in the final image.
    let color_mod = MVector3 {
        x: color_mod.x.min(Mf32::broadcast(2.0)),
        y: color_mod.y.min(Mf32::broadcast(2.0)),
        z: color_mod.z.min(Mf32::broadcast(2.0)),
    };

    let new_ray = MRay {
        origin: new_ray.origin.pick(ray.origin, active),
        direction: new_ray.direction.pick(ray.direction, active),
        active: active,
    };

    let white = MVector3::new(Mf32::one(), Mf32::one(), Mf32::one());
    let color_mod = color_mod.pick(white, active);

    (new_ray, color_mod, fresnel)
}

/// Returns the color modulation for the microfacet BRDF and also the raw
/// Fresnel factor.
fn microfacet_brdf(material: MMaterial,
                   ray_in: &MRay,
                   ray_out: &MRay,
                   isect: &MIntersection,
                   ignore_fresnel: bool)
                   -> (MVector3, Mf32) {
    // Compute the half-way vector. The outgoing ray points to the surface,
    // so negate it.
    let color = material.get_color();
    let gloss = material.get_glossiness();
    let h = (ray_in.direction - ray_out.direction).normalized();
    let d = microfacet_normal_dist(h, isect, gloss);
    let (f_color, f_raw) = microfacet_fresnel(ray_in.direction, h, color);

    let white = MVector3::new(Mf32::one(), Mf32::one(), Mf32::one());
    let f_color = if ignore_fresnel {
        // If the material has a texture, pick white instead of the color,
        // because when `ignore_fresnel` is set, the color will be sampled from
        // the texture on the GPU, so we should not take it into account here.
        f_color.pick(white, material.has_texture())
    } else {
        f_color
    };

    debug_assert!(d.all_sign_bits_positive(),
                  "surface normal density must not be negative");

    // Compute the final microfacet transmission. The factor
    // 4 * dot(n, l) * dot(n, v) has been absorbed into the geometry factor,
    // which is set to 1 now. (I tried a Kelemen-Szirmay-Kalos geometry term,
    // but it gave unrealistic results with hemisphere sampling.)
    (f_color * d, f_raw)
}

/// Computes the Fresnel factor using Schlick’s approximation. Also returns the
/// raw interpolation value, where 0.0 means material color, and 1.0 means
/// white.
#[inline(always)]
fn microfacet_fresnel(incoming: MVector3, half_way: MVector3, color: MVector3) -> (MVector3, Mf32) {
    let r0 = color;
    let r1 = MVector3::new(Mf32::one(), Mf32::one(), Mf32::one()) - r0;
    let ct = Mf32::one() - half_way.dot(incoming).abs();
    let ct2 = ct * ct;
    let ct3 = ct2 * ct;
    let ct5 = ct2 * ct3;

    // This value should never be negative. However, due to inaccuracies, values
    // close to 0 can sometimes be very small negative numbers. That triggers
    // one of my debug asserts because the BRDF of which this is a factor,
    // should never be negative. To avoid hitting the assert, take the absolute
    // value. This is very cheap anyway (just one AVX bitwise and).
    let ct5 = ct5.abs();

    (r1.mul_add(ct5, r0), ct5)
}

/// The factor due to the surface normal.
///
/// (This is not related to the statistical distribution called "normal
/// distribution".)
fn microfacet_normal_dist(half_way: MVector3,
                          isect: &MIntersection,
                          glossiness_exponent_index: Mi32)
                          -> Mf32 {
    // Compute powers of the cosine.
    let c1 = half_way.dot(isect.normal);
    let c2 = c1 * c1;
    let c4 = c2 * c2;
    let c8 = c4 * c4;
    let c16 = c8 * c8;

    // The normalization factor for Blinn-Phong with exponent n in (n + 1)/2pi.

    // Blinn-Phong with exponents alpha = 0 (just Lambertian diffuse) through
    // 16 (a bit glossy, but not yet mirror-like).
    let a0 = Mf32::broadcast(0.5 / consts::PI);
    let a1 = c1.max(Mf32::zero()) * Mf32::broadcast(1.0 / consts::PI);
    let a2 = c2 * Mf32::broadcast(1.5 / consts::PI);
    let a4 = c4 * Mf32::broadcast(2.5 / consts::PI);
    let a8 = c8 * Mf32::broadcast(4.5 / consts::PI);
    let a16 = c16 * Mf32::broadcast(8.5 / consts::PI);

    let values = [a0, a1, a2, a4, a8, a16];

    // For simplicity (or rather, to keep myself sane), I stray from the SIMD
    // path here and pick the correct value for every element separately. If it
    // were done with AVX it would take at least 6 AVX steps anyway, so 8 serial
    // steps is not so much worse.

    Mf32::generate(|i| {
        // The exponent index ranges from 0 through 5, so this should be within
        // bounds. If it is not, then that is a bug in how the material was
        // constructed. In release, we don't want to pay the bounds check
        // overhead here.
        let index = glossiness_exponent_index.get_coord(i);
        debug_assert!(0 <= index && index <= 5);
        unsafe { values.get_unchecked(index as usize).get_coord(i) }
    })
}
