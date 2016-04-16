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
use vector3::{MVector3, SVector3};

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

/// Returns the proportion of incoming energy that is transmitted into the
/// outgoing direction.
fn diffuse_brdf(isect: &MIntersection, ray_in: &MRay) -> MVector3 {
    // There is the factor dot(normal, direction) that modulates the incoming
    // contribution. The incoming energy is then radiated evenly in all
    // directions, so the energy density in every direction is 1/2pi, to
    // ensure that energy density integrates to 1.
    let cos_theta = isect.normal.dot(ray_in.direction).max(Mf32::zero());
    let modulation = Mf32::broadcast(0.5 / consts::PI) * cos_theta;

    debug_assert!(modulation.all_finite());
    debug_assert!(modulation.all_sign_bits_positive(), "color modulation cannot be negative");

    MVector3::new(modulation, modulation, modulation)
}

/// Continues the path of a photon by sampling the BRDF.
#[inline(always)]
fn continue_path_brdf(ray: &MRay,
                      isect: &MIntersection,
                      rng: &mut Rng)
                      -> MRay {
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
pub fn pd_brdf(isect: &MIntersection, ray: &MRay) -> Mf32 {
    // The probability density for the ray is dot(normal, direction) divided by
    // the intgral of that over the hemisphere (which happens to be pi).
    let dot_surface = isect.normal.dot(ray.direction).max(Mf32::zero());
    dot_surface * Mf32::broadcast(1.0 / consts::PI)
}

/// Continues the path of a photon by sampling a point on a surface.
fn continue_path_direct_sample(scene: &Scene,
                               isect: &MIntersection,
                               rng: &mut Rng)
                               -> MRay {
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

/// Returns the probability density for the given ray, for the direct sampling
/// distribution.
pub fn pd_direct_sample(scene: &Scene, ray: &MRay) -> Mf32 {
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
        let dot_emissive = sample_isect.normal.dot(ray.direction).abs();
        let pd = distance_sqr * (sample_isect.area * dot_emissive).recip_fast();

        // Add the probability density if the triangle was intersected. If the
        // triangle was not intersected, the probability of sampling it directly
        // was 0.
        pd_total = (pd_total + pd).pick(pd_total, sample_isect.mask);

        debug_assert!(pd_total.all_finite());
        debug_assert!(pd_total.all_sign_bits_positive(), "probability density must be positive");
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

    // Generate one ray by sampling the BRDF, and one ray for direct light
    // sampling.
    let ray_brdf = continue_path_brdf(ray, isect, rng);
    let ray_direct = continue_path_direct_sample(scene, isect, rng);

    // If the direct sampling ray shears the surface, we are likely to get
    // artifacts due to division by almost zero and floating point inprecision.
    // In that case, always pick the BRDF sample.
    let direct_degenerate = ray_direct.direction.dot(isect.normal).abs();
    let ignore_direct = direct_degenerate.geq(Mf32::broadcast(0.00001));

    // Randomly pick one of the two rays to use, then compute the weight for
    // multiple importance sampling.
    let rr = ignore_direct; // rng.sample_sign() & ignore_direct;
    let new_ray = MRay {
        origin: ray_brdf.origin.pick(ray_direct.origin, rr),
        direction: ray_brdf.direction.pick(ray_direct.direction, rr),
        active: Mf32::zero(),
    };
    let pd_brdf = pd_brdf(isect, &new_ray);
    let pd_direct = pd_direct_sample(scene, &new_ray);
    // Add a small constant to avoid division by zero later on.
    let weight_denom = pd_brdf + pd_direct + Mf32::broadcast(0.001);

    debug_assert!(weight_denom.all_finite());
    debug_assert!(pd_brdf.all_sign_bits_positive(), "probability density cannot be negative");
    debug_assert!(pd_direct.all_sign_bits_positive(), "probability density cannot be negative");

    // Compute the contribution using the one-sample multiple importance
    // sampler. This is equation 9.15 from section 9.2.4 of Veach, 1998. The
    // probability densities in the weight and denominator cancel, so they have
    // been left out. The factor 2.0 is because each sampling method has
    // probability 0.5 of being chosen.
    let modulation = weight_denom.recip_fast() * Mf32::broadcast(2.0);
    let modulation = Mf32::zero().pick(modulation, ignore_direct);
    let brdf_term = microfacet_brdf(&new_ray, ray, isect);
    let color_mod = brdf_term * modulation;

    debug_assert!(modulation.all_finite());
    debug_assert!(brdf_term.all_finite());
    debug_assert!(modulation.all_sign_bits_positive(), "color modulation can never be negative");
    debug_assert!(brdf_term.x.all_sign_bits_positive(), "red brdf term can never be negative");
    debug_assert!(brdf_term.y.all_sign_bits_positive(), "green brdf term can never be negative");
    debug_assert!(brdf_term.z.all_sign_bits_positive(), "blue brdf term can never be negative");

    let new_ray = MRay {
        origin: new_ray.origin.pick(ray.origin, active),
        direction: new_ray.direction.pick(ray.direction, active),
        active: active,
    };

    let white = MVector3::new(Mf32::one(), Mf32::one(), Mf32::one());
    let color_mod = color_mod.pick(white, active);

    (new_ray, color_mod)
}

fn microfacet_brdf(ray_in: &MRay, ray_out: &MRay, isect: &MIntersection) -> MVector3 {
    // Compute the half-way vector. The outgoing ray points to the surface,
    // so negate it.
    // TODO: take color from material.
    let color = MVector3::broadcast(SVector3::new(0.9, 0.7, 0.9));
    let h = (ray_in.direction - ray_out.direction).normalized();
    let f = microfacet_fresnel(ray_in.direction, h, color);
    let d = microfacet_normal_dist(h, isect);
    let cosl = isect.normal.dot(ray_out.direction);
    let cosv = isect.normal.dot(ray_in.direction);
    // Add a small constant to avoid division by 0.
    let denom = (cosl * cosv).abs() + Mf32::broadcast(0.01);

    // Compute the final microfacet transmission. There is a factor 4 in the
    // denominator.
    let white = MVector3::new(Mf32::one(), Mf32::one(), Mf32::one());
    // f * (Mf32::broadcast(0.25) * d * denom.recip_fast())
    white * Mf32::broadcast(0.25) * denom.recip_fast()
}

/// Computes the Fresnel term using Schlick’s approximation.
fn microfacet_fresnel(incoming: MVector3, half_way: MVector3, color: MVector3) -> MVector3 {
    let r0 = color;
    let r1 = MVector3::new(Mf32::one(), Mf32::one(), Mf32::one()) - r0;
    let ct = Mf32::one() - half_way.dot(incoming).abs();
    let ct2 = ct * ct;
    let ct3 = ct2 * ct;
    let ct5 = ct2 * ct3;
    r0 + r1 * ct5
}

/// The term due to the surface normal.
///
/// (This is not related to the statistical distribution called "normal
/// distribution".)
fn microfacet_normal_dist(half_way: MVector3, isect: &MIntersection) -> Mf32 {
    let cos = half_way.dot(isect.normal);

    // Blinn-Phong with parameter alpha = 1.
    // cos * Mf32::broadcast(1.5 * consts::PI)

    // Blinn-Phong with parameter alpha = 2.
    // (cos * cos) * Mf32::broadcast(1.0 / consts::PI)

    // Blinn-Phong with parameter alpha = 4;
    // let c2 = cos * cos;
    // let c4 = c2 * c2;
    // c4 * Mf32::broadcast(1.5 / consts::PI)

    // Blinn-Phong with parameter alpha = 8;
    let c2 = cos * cos;
    let c4 = c2 * c2;
    let c8 = c4 * c4;
    c8 * Mf32::broadcast(2.5 / consts::PI)
}
