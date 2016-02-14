#[warn(missing_docs)]

#[macro_use]
extern crate glium;
extern crate time;

mod renderer;
mod ui;

use renderer::Renderer;
use ui::Window;

fn main() {
    let width = 1280;
    let height = 720;

    let mut window = Window::new(width, height, "infomagr interactive tracer");
    let renderer = Renderer::new(width, height);

    while window.handle_events() {
        // Create an uninitialized buffer to render into. Because the renderer
        // will write to every pixel, no uninitialized memory is exposed.
        let screen_len = width as usize * height as usize * 3;
        let mut screen: Vec<u8> = Vec::with_capacity(screen_len);
        unsafe { screen.set_len(screen_len); }
        renderer.render(&mut screen[..]);
        window.render(screen);
    }
}
