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

    pub fn render(&self, dest_bitmap: &mut [u8]) {
        debug_assert_eq!(dest_bitmap.len(), self.width as usize * self.height as usize * 3);
        // TODO
    }
}
