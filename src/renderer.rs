use scene::Scene;

pub struct Renderer {
    scene: Scene,
    width: u32,
    height: u32,
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

        for j in y_from..y_to {
            for i in 0..self.width {
                let idx = (((j - y_from) * self.width + i) * 3) as usize;
                backbuffer_slice[idx + 0] = (256 * i / self.width) as u8;
                backbuffer_slice[idx + 1] = (256 * j / self.height) as u8;
                backbuffer_slice[idx + 2] = 0;
            }
        }
    }
}
