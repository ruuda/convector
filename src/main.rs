#![warn(missing_docs)]

#[macro_use]
extern crate glium;
extern crate num_cpus;
extern crate time;

mod renderer;
mod scheduler;
mod stats;
mod ui;
mod vector3;

use renderer::Renderer;
use stats::GlobalStats;
use ui::Window;

fn main() {
    let width = 1280;
    let height = 720;

    let mut window = Window::new(width, height, "infomagr interactive tracer");
    let renderer = Renderer::new(width, height);
    let mut stats = GlobalStats::new();

    while window.handle_events(&stats) {
        // Create an uninitialized buffer to render into. Because the renderer
        // will write to every pixel, no uninitialized memory is exposed.
        let screen_len = width as usize * height as usize * 3;
        let mut screen: Vec<u8> = Vec::with_capacity(screen_len);
        unsafe { screen.set_len(screen_len); }
        renderer.render(&mut screen[..]);
        window.render(screen, &mut stats);
    }
}
