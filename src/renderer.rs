use scene::Scene;
use simd::Mf32;
use std::slice::ChunksMut;
use time::PreciseTime;
use util::z_order;
use vector3::{MVector3, SVector3};

pub struct Renderer {
    scene: Scene,
    width: u32,
    height: u32,
    epoch: PreciseTime,
}

/// The buffer that an image is rendered into.
///
/// Memory layout is optimized for ray locality.
pub struct PatchBuffer {
    buffer: Vec<u8>,
    patch_width: u32,
}

impl PatchBuffer {
    /// Allocates a new patch buffer to render into, memory uninitialized.
    pub fn new_uninit(width: u32, height: u32, patch_width: u32) -> PatchBuffer {
        assert_eq!(patch_width & (patch_width - 1), 0); // Patch width must be a power of 2.
        assert_eq!(patch_width & 3, 0);                 // Patch width must be a multiple of 4.
        assert_eq!(width & (patch_width - 1), 0);       // With must be a multiple of the patch width.
        assert_eq!(height & (patch_width - 1), 0);      // Height must be a multiple of the patch width.

        let len = (width as usize) * (height as usize) * 3;

        // Allocate a buffer for the screen. All memory is written to by the
        // renderer, no uninitialized memory is exposed.
        let mut buffer = Vec::with_capacity(len);
        unsafe { buffer.set_len(len); }

        PatchBuffer {
            buffer: buffer,
            patch_width: patch_width
        }
    }

    /// Allocates a new patch buffer to render into, initialized to black.
    pub fn new_black(width: u32, height: u32, patch_width: u32) -> PatchBuffer {
        let mut patchbuffer = PatchBuffer::new_uninit(width, height, patch_width);
        let len = patchbuffer.buffer.len();
        patchbuffer.buffer.clear();
        patchbuffer.buffer.resize(len, 0);
        patchbuffer
    }

    /// Returns slices which are the patches in this buffer.
    ///
    /// The order of the patches is from left to right and from bottom to top.
    /// So the bottom-left pixel of the first patch is at (0, 0) and the
    /// top-right pixel of the last patch is at (width, height).
    ///
    /// For every patch, the memory layout is as follows: the patch is divided
    /// into blocks of 4x2 pixels. The pixels in these blocks are stored RGB
    /// from left to right and from bottom to top. These blocks then fill the
    /// patch in a Z-order starting at the bottom-left block.
    pub fn patches(&mut self) -> ChunksMut<u8> {
        let patch_len = (self.patch_width * self.patch_width * 3) as usize;
        self.buffer.chunks_mut(patch_len)
    }

    fn swap_mem(xs: &mut [u8], ys: &mut [u8]) {
        use std::mem;
        for (x, y) in xs.iter_mut().zip(ys.iter_mut()) {
            mem::swap(x, y);
        }
    }

    /// Converts the patches from z-order memory order to bitmap order.
    fn untangle_patches(&mut self) {
        let width = self.patch_width;
        let patch_len = (width * width * 3) as usize;

        // Create a temporary buffer to untangle the patch into.
        // TODO: It must be possible to do this in-place using a recursive
        // algorithm.
        let mut scratch = Vec::with_capacity(patch_len);
        unsafe { scratch.set_len(patch_len); }

        for patch in self.patches() {
            for i in 0..(width * width) {
                let (px, py) = z_order(i as u16);
                let idx = ((py as u32) * width * 3 + (px as u32) * 3) as usize;
                scratch[idx + 0] = patch[i as usize * 3 + 0];
                scratch[idx + 1] = patch[i as usize * 3 + 1];
                scratch[idx + 2] = patch[i as usize * 3 + 2];
            }

            PatchBuffer::swap_mem(patch, &mut scratch[..]);
        }
    }

    fn stitch_patches(&mut self) {
        // TODO.
    }

    /// Stitches the internal representation into an RGB image.
    pub fn into_bitmap(mut self) -> Vec<u8> {
        self.untangle_patches();
        self.stitch_patches();
        self.buffer
    }
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
        self.scene.lights[0].position = SVector3 {
            x: t.cos() * 5.0,
            y: (t * 0.3).cos() * 7.0,
            z: t.sin() * 5.0,
        };
    }

    /// Returns the screen coordinates of the block 16 pixels where (x, y) is the bottom-left
    /// coordinate. The coordinates are ordered in a z-order.
    fn get_pixel_coords(&self, x: u32, y: u32) -> ([Mf32; 2], [Mf32; 2]) {
        // TODO: There is little point in using the z-order here.
        // Perhaps the patching can be sped up by using a normal order,
        // but a z-order for the patches?
        let z_order_x0 = Mf32(0.0, 1.0, 0.0, 1.0, 2.0, 3.0, 2.0, 3.0);
        let z_order_y0 = Mf32(0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0);
        let z_order_y1 = Mf32(2.0, 2.0, 3.0, 3.0, 2.0, 2.0, 3.0, 3.0);
        let xf = Mf32::broadcast(x as f32 - self.width as f32 * 0.5);
        let yf = Mf32::broadcast(y as f32 - self.height as f32 * 0.5);
        let scale = Mf32::broadcast(2.0 / self.width as f32);
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

        let n = patch_width / 4;
        for i in 0..(n * n) {
            let (px, py) = z_order((i * 16) as u16);
            let (xs, ys) = self.get_pixel_coords(x + px as u32, y + py as u32);

            let rgb0 = self.render_pixels(xs[0], ys[0]);
            let rgb1 = self.render_pixels(xs[1], ys[1]);

            // Convert the color to linear RGB, 8 bytes per pixel. The window
            // has been set up so that this will be converted to sRGB when
            // it is displayed.
            let range = Mf32::broadcast(255.0);
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

    fn render_pixels(&self, x: Mf32, y: Mf32) -> MVector3 {
        let ray = self.scene.camera.get_ray(x, y);
        let mut color = MVector3::zero();
        let isect = self.scene.intersect_nearest(&ray);

        for ref light in &self.scene.lights {
            let light_pos = MVector3::broadcast(light.position);
            let to_light = light_pos - isect.position;
            // TODO: shadow rays.
            let dist_cos_alpha = isect.normal.dot(to_light).max(Mf32::zero());

            // Compensate for the norm factor in dist_cos_alpha, then
            // incorporate the inverse square falloff.
            let rnorm = to_light.rnorm();
            let strength = (dist_cos_alpha * rnorm) * (rnorm * rnorm);

            color = MVector3 {
                x: strength * Mf32::broadcast(10.0),
                y: Mf32::zero(),
                z: Mf32::zero(),
            };
        }

        color
    }
}
