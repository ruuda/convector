pub struct Renderer {
    width: u32,
    height: u32,
}

impl Renderer {
    pub fn new(width: u32, height: u32) -> Renderer {
        Renderer {
            width: width,
            height: height,
        }
    }

    /// Renders one frame, returns the rendered surface.
    pub fn render_frame(&self) -> Vec<u8> {
        // Create an uninitialized buffer to render into. Because the renderer
        // will write to every pixel, no uninitialized memory is exposed.
        let surface_len = self.width as usize * self.height as usize * 3;
        let mut surface: Vec<u8> = Vec::with_capacity(surface_len);
        unsafe { surface.set_len(surface_len); }
        surface
    }

    pub fn render(&self, dest_bitmap: &mut [u8]) {
        assert_eq!(dest_bitmap.len(), self.width as usize * self.height as usize * 3);
        for j in 0..self.height {
            for i in 0..self.width {
                let idx = ((j * self.width + i) * 3) as usize;
                dest_bitmap[idx + 0] = (256 * i / self.width) as u8;
                dest_bitmap[idx + 1] = (256 * j / self.height) as u8;
                dest_bitmap[idx + 2] = 0;
            }
        }
    }
}
