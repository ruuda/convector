use ray::Ray;
use scene::Scene;
use simd::OctaF32;
use time::PreciseTime;
use util::z_order;
use vector3::{OctaVector3, Vector3};

pub struct Renderer {
    scene: Scene,
    width: u32,
    height: u32,
    epoch: PreciseTime,
}

/// Clamps a float to the unit interval [0, 1].
fn clamp_unit(x: f32) -> f32 {
    if x < 0.0 { 0.0 } else if x > 1.0 { 1.0 } else { x }
}

impl Renderer {
    pub fn new(scene: Scene, width: u32, height: u32) -> Renderer {
        Renderer {
            scene: scene,
            width: width,
            height: height,
            epoch: PreciseTime::now(),
        }
    }

    /// For an interactive scene, updates the scene for the new frame.
    /// TODO: This method does not really belong here.
    pub fn update_scene(&mut self) {
        let t = self.epoch.to(PreciseTime::now()).num_milliseconds() as f32 * 1e-3;

        // Make the light circle around.
        self.scene.lights[0].position = Vector3 {
            x: t.cos() * 5.0,
            y: (t * 0.3).cos() * 7.0,
            z: t.sin() * 5.0,
        };
    }

    /// Returns the screen coordinates of the block 16 pixels where (x, y) is the bottom-left
    /// coordinate. The coordinates are ordered in a z-order.
    fn get_pixel_coords(&self, x: u32, y: u32) -> ([OctaF32; 2], [OctaF32; 2]) {
        // TODO: There is little point in using the z-order here.
        // Perhaps the patching can be sped up by using a normal order,
        // but a z-order for the patches?
        let z_order_x0 = OctaF32(0.0, 1.0, 0.0, 1.0, 2.0, 3.0, 2.0, 3.0);
        let z_order_y0 = OctaF32(0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0);
        let z_order_y1 = OctaF32(2.0, 2.0, 3.0, 3.0, 2.0, 2.0, 3.0, 3.0);
        let xf = OctaF32::broadcast(x as f32 - self.width as f32 * 0.5);
        let yf = OctaF32::broadcast(y as f32 - self.height as f32 * 0.5);
        let scale = OctaF32::broadcast(2.0 / self.width as f32);
        let xs0 = (z_order_x0 + xf) * scale;
        let ys0 = (z_order_y0 + yf) * scale;
        let ys1 = (z_order_y1 + yf) * scale;
        ([xs0, xs0], [ys0, ys1])
    }

    /// Renders part of a frame.
    ///
    /// The (x, y) coordinate is the coordinate of the bottom-left pixel of the
    /// patch. The patch width must be a power of 16.
    pub fn render_patch(&self, patch: &mut [u8], patch_width: u32, x: u32, y: u32) {
        assert_eq!(patch.len(), (3 * patch_width * patch_width) as usize);
        assert_eq!(patch_width & (patch_width - 1), 0); // Patch width must be a power of 2.
        assert_eq!(patch_width & 3, 0); // Patch width must be a multiple of 4.

        let n = patch_width / 4;
        for i in 0..(n * n) {
            let (px, py) = z_order((i * 16) as u16);
            let (xs, ys) = self.get_pixel_coords(x + px as u32, y + py as u32);

            let rgb0 = self.render_pixels(xs[0], ys[0]);
            let rgb1 = self.render_pixels(xs[1], ys[1]);

            // Convert the color to linear RGB, 8 bytes per pixel. The window
            // has been set up so that this will be converted to sRGB when
            // it is displayed.
            let range = OctaF32::broadcast(255.0);
            let rgb0_255 = rgb0.clamp_one() * range;
            let rgb1_255 = rgb1.clamp_one() * range;

            // TODO: SIMD conversion into integers.
            for j in 0..8 {
                let idx0 = ((i * 16 + j) * 3) as usize;
                let idx1 = ((i * 16 + j + 8) * 3) as usize;
                patch[idx0 + 0] = rgb0_255.x.as_slice()[j as usize] as u8;
                patch[idx0 + 1] = rgb0_255.y.as_slice()[j as usize] as u8;
                patch[idx0 + 2] = rgb0_255.z.as_slice()[j as usize] as u8;
                patch[idx1 + 0] = rgb1_255.x.as_slice()[j as usize] as u8;
                patch[idx1 + 1] = rgb1_255.y.as_slice()[j as usize] as u8;
                patch[idx1 + 2] = rgb1_255.z.as_slice()[j as usize] as u8;
            }
        }
    }

    fn render_pixels(&self, x: OctaF32, y: OctaF32) -> OctaVector3 {
        let octa_ray = self.scene.camera.get_octa_ray(x, y);
        let mut color = OctaVector3::zero();
        let isect = self.scene.intersect_nearest(&octa_ray);

        for ref light in &self.scene.lights {
            let light_pos = OctaVector3::broadcast(light.position);
            let to_light = light_pos - isect.position;
            // TODO: shadow rays.
            let dist_cos_alpha = isect.normal.dot(to_light).max(OctaF32::zero());

            // Compensate for the norm factor in dist_cos_alpha, then
            // incorporate the inverse square falloff.
            let rnorm = to_light.rnorm();
            let strength = (dist_cos_alpha * rnorm) * (rnorm * rnorm);

            color = OctaVector3 {
                x: strength * OctaF32::broadcast(10.0),
                y: OctaF32::zero(),
                z: OctaF32::zero(),
            };
        }

        color
    }

    fn render_pixel(&self, x: f32, y: f32) -> Vector3 {
        let ray = self.scene.camera.get_ray(x, y);
        let mut color = Vector3::zero();
        if let Some(isect) = self.scene.intersect(&ray) {
            for ref light in &self.scene.lights {
                let to_light = light.position - isect.position;
                let distance = to_light.norm();
                let shadow_ray = Ray {
                    origin: isect.position,
                    direction: to_light * (1.0 / distance),
                }.advance_epsilon();
                if self.scene.intersect(&shadow_ray)
                    // TODO: Actually, the distance squared would be sufficient in most cases.
                    .map_or(true, |occluder| occluder.distance > distance) {
                    let mut strength = isect.normal.dot(to_light);
                    if strength < 0.0 { strength = 0.0; }
                    strength = strength * (1.0 / (distance * distance));
                    color = color + Vector3::new(strength * 5.0, 0.0, 0.0);
                }
            }
        }
        color
    }
}
