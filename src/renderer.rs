use scene::Scene;
use vector3::{Ray, Vector3, dot};

pub struct Renderer {
    scene: Scene,
    width: u32,
    height: u32,
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
        }
    }

    /// Renders part of a frame.
    pub fn render_frame_slice(&self,
                              backbuffer_slice: &mut [u8],
                              y_from: u32,
                              y_to: u32) {
        assert_eq!(backbuffer_slice.len(), self.width as usize * (y_to - y_from) as usize * 3);

        let scale = 2.0 / self.width as f32;

        for y in y_from..y_to {
            for x in 0..self.width {
                let xf = (x as f32 - self.width as f32 / 2.0) * scale;
                let yf = (y as f32 - self.height as f32 / 2.0) * scale;

                let rgb = self.render_pixel(xf, yf);

                // Write the color as linear RGB, 8 bytes per pixel. The window
                // has been set up so that this will be converted to sRGB when
                // it is displayed.
                let idx = (((y - y_from) * self.width + x) * 3) as usize;
                backbuffer_slice[idx + 0] = (255.0 * clamp_unit(rgb.x)) as u8;
                backbuffer_slice[idx + 1] = (255.0 * clamp_unit(rgb.y)) as u8;
                backbuffer_slice[idx + 2] = (255.0 * clamp_unit(rgb.z)) as u8;
            }
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
                    color = color + Vector3::new(strength, 0.0, 0.0);
                }
            }
        }
        color
    }
}
