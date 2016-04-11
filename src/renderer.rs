use material::{continue_path, sky_intensity};
use random::Rng;
use scene::Scene;
use simd::{Mf32, Mi32};
use std::cell::UnsafeCell;
use util;
use vector3::{MVector3, SVector3};

pub struct Renderer {
    scene: Scene,
    width: u32,
    height: u32,
    enable_debug_view: bool,

    /// A value that increases at a rate of 1 per second.
    time: f32,

    /// The amount that time increases per frame.
    time_delta: f32,
}

/// The buffer that an image is rendered into.
pub struct RenderBuffer {
    buffer: UnsafeCell<Vec<Mi32>>,
}

impl RenderBuffer {
    /// Allocates a new buffer to render into, memory uninitialized.
    ///
    /// The width and height must be a multiple of 16.
    pub fn new(width: u32, height: u32) -> RenderBuffer {
        assert_eq!(width & 15, 0);  // Width must be a multiple of 16.
        assert_eq!(height & 15, 0); // Height must be a multiple of 16.

        // There are 8 RGBA pixels in one mi32.
        let num_elems = (width as usize) * (height as usize) / 8;

        let mut vec = util::cache_line_aligned_vec(num_elems);
        unsafe { vec.set_len(num_elems); }

        RenderBuffer {
            buffer: UnsafeCell::new(vec),
        }
    }

    /// Zeroes the buffer.
    pub fn fill_black(&mut self) {
        // This is actually safe because self is borrowed mutably.
        for pixels in unsafe { self.get_mut_slice() } {
            *pixels = Mi32::zero();
        }
    }

    /// Returns a mutable view into the buffer.
    ///
    /// This is unsafe because it allows creating multiple mutable borrows of
    /// the buffer, which could result in races. Threads should ensure that
    /// they write to disjoint parts of the buffer.
    pub unsafe fn get_mut_slice(&self) -> &mut [Mi32] {
        (*self.buffer.get()).as_mut_slice()
    }

    /// Returns an RGBA bitmap suitable for display.
    #[cfg(not(windows))]
    pub fn into_bitmap(self) -> Vec<u8> {
        // This is actually safe because self is moved into the method.
        let buffer = unsafe { self.buffer.into_inner() };
        unsafe { util::transmute_vec(buffer) }
    }

    /// Returns an RGBA bitmap suitable for display.
    #[cfg(windows)]
    pub fn into_bitmap(self) -> Vec<u8> {
        use std::mem;

        // This is actually safe because self is moved into the method.
        let buffer = unsafe { self.buffer.into_inner() };

        // On Windows we must make an extra copy; we cannot just transmute the
        // buffer into a buffer of bytes, because the allocator then uses the
        // alignment of a byte to free the buffer, but it asserts that the
        // alignment for deallocation matches the alignment that the buffer was
        // allocated with. I raised this point in the allocator RFC discussion:
        // https://github.com/rust-lang/rfcs/pull/1398#issuecomment-198584430.
        // The extra copy is unfortunate, but the allocator API needs to change
        // before it can be avoided.
        let byte_buffer = buffer.iter().flat_map(|mi32| {
            let bytes: &[u8; 32] = unsafe { mem::transmute(mi32) };
            bytes
        }).cloned().collect();

        util::drop_cache_line_aligned_vec(buffer);
        byte_buffer
    }
}

// The render buffer must be shared among threads, but UnsafeCell is not Sync.
unsafe impl Sync for RenderBuffer { }

/// Builds a fixed-size slice by calling f for every index.
fn generate_slice8<T, F>(mut f: F) -> [T; 8] where F: FnMut(usize) -> T {
    [f(0), f(1), f(2), f(3), f(4), f(5), f(6), f(7)]
}

impl Renderer {
    pub fn new(scene: Scene, width: u32, height: u32) -> Renderer {
        Renderer {
            scene: scene,
            width: width,
            height: height,
            enable_debug_view: false,
            time: 0.0,
            time_delta: 0.0,
        }
    }

    /// Sets the current time and the amount that the time is expected to change
    /// per frame.
    pub fn set_time(&mut self, time: f32, delta: f32) {
        self.time = time;
        self.time_delta = delta;
    }

    /// For an interactive scene, updates the scene for the new frame.
    /// TODO: This method does not really belong here.
    pub fn update_scene(&mut self) {
        let alpha = self.time * -0.05 + 0.5;
        let alpha_delta = self.time_delta * -0.05;
        let cam_position = SVector3::new(-7.8 * alpha.sin(), 1.8, 7.8 * alpha.cos());
        let cam_pos_delta = SVector3::new(-7.8 * alpha.cos(), 0.0, -7.8 * alpha.sin()) * alpha_delta;
        self.scene.camera.set_position(cam_position, cam_pos_delta);
        self.scene.camera.set_rotation(alpha, alpha_delta);
    }

    pub fn toggle_debug_view(&mut self) {
        self.enable_debug_view = !self.enable_debug_view;
    }

    /// Returns the screen coordinates of the block of 16x4 pixels where (x, y)
    /// is the bottom-left coordinate. The order is as follows:
    ///
    ///     0c 0d 0e 0f  1c 1d 1e 1f  2c 2d 2e 2f  3c 3d 3e 3f
    ///     08 09 0a 0b  18 19 1a 1b  28 29 2a 2b  38 39 3a 3b
    ///     04 05 06 07  14 15 16 17  24 25 26 27  34 35 36 37
    ///     00 01 02 03  10 11 12 13  20 21 22 23  30 31 32 33
    ///
    /// Or, in terms of the mf32s:
    ///
    ///     1 1 1 1  3 3 3 3  5 5 5 5  7 7 7 7
    ///     1 1 1 1  3 3 3 3  5 5 5 5  7 7 7 7
    ///     0 0 0 0  2 2 2 2  4 4 4 4  6 6 6 6
    ///     0 0 0 0  2 2 2 2  4 4 4 4  6 6 6 6
    ///
    /// Where inside every mf32 the pixels are ordered from left to right,
    /// bottom to top.
    fn get_pixel_coords_16x4(&self, x: u32, y: u32, rng: &mut Rng) -> ([Mf32; 8], [Mf32; 8]) {
        let scale = Mf32::broadcast(2.0 / self.width as f32);
        let scale_mul = Mf32(2.0, 4.0, 8.0, 12.0, 0.0, 0.0, 0.0, 0.0) * scale;

        let off_x = Mf32(0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0);
        let off_y = Mf32(0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0);

        let base_x = scale * (off_x + Mf32::broadcast(x as f32 - self.width as f32 * 0.5));
        let base_y = scale * (off_y + Mf32::broadcast(y as f32 - self.height as f32 * 0.5));

        let xs = [
            base_x,
            base_x,
            base_x + Mf32::broadcast(scale_mul.1), // 4.0 * scale
            base_x + Mf32::broadcast(scale_mul.1), // 4.0 * scale
            base_x + Mf32::broadcast(scale_mul.2), // 8.0 * scale
            base_x + Mf32::broadcast(scale_mul.2), // 8.0 * scale
            base_x + Mf32::broadcast(scale_mul.3), // 12.0 * scale
            base_x + Mf32::broadcast(scale_mul.3)  // 12.0 * scale
        ];

        let ys = [
            base_y, base_y + Mf32::broadcast(scale_mul.0), // 2.0 * scale
            base_y, base_y + Mf32::broadcast(scale_mul.0), // 2.0 * scale
            base_y, base_y + Mf32::broadcast(scale_mul.0), // 2.0 * scale
            base_y, base_y + Mf32::broadcast(scale_mul.0)  // 2.0 * scale
        ];

        // Add a random offset of at most one pixel, to sample with anti-alias.
        // TODO: If I ever do multiple samples per pixel in one frame, I could
        // do stratified sampling here.
        let xs_aa = generate_slice8(|i| rng.sample_unit().mul_add(scale, xs[i]));
        let ys_aa = generate_slice8(|i| rng.sample_unit().mul_add(scale, ys[i]));

        (xs_aa, ys_aa)
    }

    /// Converts floating-point color values to 24-bit RGB and stores the values
    /// in the bitmap.
    fn store_pixels_16x4(&self, bitmap: &mut [Mi32], x: u32, y: u32, rgbs: &[MVector3; 8]) {
        // Convert f32 colors to i32 colors in the range 0-255.
        let range = Mf32::broadcast(255.0);
        let rgbas = generate_slice8(|i| {
            let rgb_255 = rgbs[i].clamp_one() * range;
            let r = rgb_255.x.into_mi32();
            let g = rgb_255.y.into_mi32().map(|x| x << 8);
            let b = rgb_255.z.into_mi32().map(|x| x << 16);
            let a = Mi32::broadcast(0xff000000_u32 as i32);
            (r | g) | (b | a)
        });

        // Helper functions to shuffle around the pixels from the order as
        // described in `get_pixel_coords_16x4` into four rows of 16 pixels.
        let mk_line0 = |left: Mi32, right: Mi32|
            Mi32(left.0, left.1, left.2, left.3, right.0, right.1, right.2, right.3);
        let mk_line1 = |left: Mi32, right: Mi32|
            Mi32(left.4, left.5, left.6, left.7, right.4, right.5, right.6, right.7);

        // Store the pixels in the bitmap. If the bitmap is aligned to the cache
        // line size, this stores exactly four cache lines, so there is no need
        // to fetch those lines because all bytes are overwritten. This saves a
        // trip to memory, which makes this store fast.
        let idx_line0 = ((y * self.width + 0 * self.width + x) / 8) as usize;
        let idx_line1 = ((y * self.width + 1 * self.width + x) / 8) as usize;
        let idx_line2 = ((y * self.width + 2 * self.width + x) / 8) as usize;
        let idx_line3 = ((y * self.width + 3 * self.width + x) / 8) as usize;
        bitmap[idx_line0 + 0] = mk_line0(rgbas[0], rgbas[2]);
        bitmap[idx_line0 + 1] = mk_line0(rgbas[4], rgbas[6]);
        bitmap[idx_line1 + 0] = mk_line1(rgbas[0], rgbas[2]);
        bitmap[idx_line1 + 1] = mk_line1(rgbas[4], rgbas[6]);
        bitmap[idx_line2 + 0] = mk_line0(rgbas[1], rgbas[3]);
        bitmap[idx_line2 + 1] = mk_line0(rgbas[5], rgbas[7]);
        bitmap[idx_line3 + 0] = mk_line1(rgbas[1], rgbas[3]);
        bitmap[idx_line3 + 1] = mk_line1(rgbas[5], rgbas[7]);
    }

    /// Renders a block of 16x4 pixels, where (x, y) is the coordinate of the
    /// bottom-left pixel. Bitmap must be an array of 8 pixels at once, and it
    /// must be aligned to 64 bytes (a cache line).
    fn render_block_16x4(&self, x: u32, y: u32, rng: &mut Rng) -> [MVector3; 8] {
        let (xs, ys) = self.get_pixel_coords_16x4(x, y, rng);

        if self.enable_debug_view {
            generate_slice8(|i| self.render_pixels_debug(xs[i], ys[i]))
        } else {
            generate_slice8(|i| self.render_pixels(xs[i], ys[i], rng))
        }
    }

    /// Renders a square part of a frame.
    ///
    /// The (x, y) coordinate is the coordinate of the bottom-left pixel of the
    /// patch. The patch width must be a multiple of 16.
    pub fn render_patch_u8(&self,
                           bitmap: &mut [Mi32],
                           patch_width: u32,
                           x: u32,
                           y: u32,
                           frame_number: u32) {
        assert_eq!(patch_width & 15, 0); // Patch width must be a multiple of 16.
        let w = patch_width / 16;
        let h = patch_width / 4;
        let mut rng = Rng::with_seed(x, y, frame_number);

        for i in 0..w {
            for j in 0..h {
                let xb = x + i * 16;
                let yb = y + j * 4;
                let rgbs = self.render_block_16x4(xb, yb, &mut rng);
                self.store_pixels_16x4(bitmap, xb, yb, &rgbs);
            }
        }
    }

    /// Renders a square part of a frame, adds the contribution to the buffer.
    ///
    /// The (x, y) coordinate is the coordinate of the bottom-left pixel of the
    /// patch. The patch width must be a multiple of 16. The memory layout of
    /// the HDR buffer is as a bitmap of 16x4 blocks.
    pub fn accumulate_patch_f32(&self,
                                hdr_buffer: &mut [[MVector3; 8]],
                                patch_width: u32,
                                x: u32,
                                y: u32,
                                frame_number: u32) {
        assert_eq!(patch_width & 15, 0); // Patch width must be a multiple of 16.
        let w = patch_width / 16;
        let h = patch_width / 4;
        let index = ((y / 4) * (self.width / 16) + (x / 16)) as usize;
        let mut rng = Rng::with_seed(x, y, frame_number);

        for i in 0..w {
            for j in 0..h {
                let xb = x + i * 16;
                let yb = y + j * 4;
                let rgbs = self.render_block_16x4(xb, yb, &mut rng);
                let current = hdr_buffer[index];
                hdr_buffer[index] = generate_slice8(|k| current[k] + rgbs[k]);
            }
        }
    }

    pub fn buffer_f32_to_u8(&self, hdr_buffer: &[[MVector3; 8]], num_samples: u32) -> Vec<u8> {
        let render_buffer = RenderBuffer::new(self.width, self.height);
        let w = self.width / 16;
        let h = self.height / 4;
        let factor = Mf32::broadcast(1.0 / (num_samples as f32));

        {
            // This is safe here because there is only one mutable borrow.
            let bitmap = unsafe { render_buffer.get_mut_slice() };

            for j in 0..h {
                for i in 0..w {
                    let rgbs = hdr_buffer[(j * w + i) as usize];
                    let rgbs = generate_slice8(|k| rgbs[k] * factor);
                    self.store_pixels_16x4(bitmap, j * 4, i * 16, &rgbs);
                }
            }
        }

        render_buffer.into_bitmap()
    }

    fn render_pixels(&self, x: Mf32, y: Mf32, rng: &mut Rng) -> MVector3 {
        let t = rng.sample_unit();
        let mut ray = self.scene.camera.get_ray(x, y, t);
        let mut color = MVector3::new(Mf32::one(), Mf32::one(), Mf32::one());
        let mut hit_emissive = Mf32::zero();

        let max_bounces = 8;
        for i in 0..max_bounces {
            let isect = self.scene.intersect_nearest(&ray);
            hit_emissive = isect.material;

            // Do not allow NaNs to creep in.
            debug_assert!(ray.direction.all_finite(), "infinite ray direction at iteration {}", i);
            debug_assert!(isect.position.all_finite(), "infinite intersection at iteration {}", i);
            debug_assert!(isect.distance.all_finite(), "infinite distance at iteration {}", i);

            // Stop when every ray hit a light source.
            if isect.material.all_sign_bits_negative() { break }

            let (factor, new_ray) = continue_path(&ray, &isect, rng);
            ray = new_ray;
            color = color.mul_coords(factor);
        }

        // Compute light contribution.
        let emission = sky_intensity(ray.direction);
        color = color.mul_coords(emission);

        // If the last thing that a ray hit was an emissive material, it has
        // found a light source and the computed color is correct. If the ray
        // did not find a light source but the loop was terminated, the computed
        // color is invalid; it should be black.
        MVector3::zero().pick(color, hit_emissive)
    }

    fn render_pixels_debug(&self, x: Mf32, y: Mf32) -> MVector3 {
        let t = Mf32::zero();
        let ray = self.scene.camera.get_ray(x, y, t);
        let (numi_aabb, numi_tri) = self.scene.intersect_debug(&ray);

        let g = Mf32::broadcast((numi_aabb as f32).log2() * 0.1);
        let b = Mf32::broadcast((numi_tri as f32).log2() * 0.1);

        MVector3::new(Mf32::zero(), g, b)
    }
}

#[test]
fn render_buffer_into_bitmap() {
    let render_buffer = RenderBuffer::new(1280, 736);
    let bitmap = render_buffer.into_bitmap();
    drop(bitmap);
    let render_buffer = RenderBuffer::new(1280, 736);
    let _bitmap = render_buffer.into_bitmap();
    // The render buffer was transmuted or copied into a vector of pixels, and
    // dropping the vector at this point should not result in a crash.
}
