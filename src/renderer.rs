use ray::Ray;
use scene::Scene;
use time::PreciseTime;
use util::z_order;
use vector3::{Vector3, dot};

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

    /// Renders part of a frame.
    ///
    /// The (x, y) coordinate is the coordinate of the bottom-left pixel of the
    /// patch. The patch width must be a power of two.
    pub fn render_patch(&self, patch: &mut [u8], patch_width: u32, x: u32, y: u32) {
        assert_eq!(patch.len(), (3 * patch_width * patch_width) as usize);
        assert_eq!(patch_width & (patch_width - 1), 0);

        let scale = 2.0 / self.width as f32;

        for i in 0..(patch_width * patch_width) {
            let (px, py) = z_order(i as u16);
            let xf = ((x + px as u32) as f32 - self.width as f32 / 2.0) * scale;
            let yf = ((y + py as u32) as f32 - self.height as f32 / 2.0) * scale;

            let rgb = self.render_pixel(xf, yf);

            // Write the color as linear RGB, 8 bytes per pixel. The window
            // has been set up so that this will be converted to sRGB when
            // it is displayed.
            patch[i as usize * 3 + 0] = (255.0 * clamp_unit(rgb.x)) as u8;
            patch[i as usize * 3 + 1] = (255.0 * clamp_unit(rgb.y)) as u8;
            patch[i as usize * 3 + 2] = (255.0 * clamp_unit(rgb.z)) as u8;
        }
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
                    let mut strength = dot(isect.normal, to_light);
                    if strength < 0.0 { strength = 0.0; }
                    strength = strength * (1.0 / (distance * distance));
                    color = color + Vector3::new(strength * 5.0, 0.0, 0.0);
                }
            }
        }
        color
    }
}
