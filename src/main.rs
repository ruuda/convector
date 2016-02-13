extern crate glium;

mod renderer;

use renderer::Renderer;
use std::iter;

fn main() {
    use glium::DisplayBuild;

    // TODO: Proper HiDPI support.
    let width = 1280;
    let height = 720;

    let renderer = Renderer::new(width, height);
    let mut screen: Vec<u8> = iter::repeat(0).take(width as usize * height as usize * 3).collect();

    // TODO: Proper HiDPI support.
    let display = glium::glutin::WindowBuilder::new()
        .with_dimensions(width, height)
        .with_title(format!("Hello world"))
        .build_glium()
        .expect("failed to create gl window");

    loop {
        renderer.render(&mut screen[..]);
        for ev in display.poll_events() {
            match ev {
                // Window was closed by the user.
                glium::glutin::Event::Closed => return,
                _ => ()
            }
        }

        // TODO: Wait for vsync.
    }
}
