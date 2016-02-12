extern crate glium;

fn main() {
    use glium::DisplayBuild;

    // TODO: Proper HiDPI support.
    let display = glium::glutin::WindowBuilder::new()
        .with_dimensions(1280, 720)
        .with_title(format!("Hello world"))
        .build_glium()
        .expect("failed to create gl window");

    loop {
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
