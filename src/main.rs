#![warn(missing_docs)]

#[macro_use]
extern crate glium;
extern crate num_cpus;
extern crate scoped_threadpool;
extern crate time;

mod renderer;
mod stats;
mod ui;
mod vector3;

use renderer::Renderer;
use stats::GlobalStats;
use time::PreciseTime;
use ui::Window;

fn main() {
    let width = 1280;
    let height = 720;

    let mut window = Window::new(width, height, "infomagr interactive tracer");
    let renderer = Renderer::new(width, height);
    let mut stats = GlobalStats::new();

    // Initialize a buffer to black.
    let screen_len = width as usize * height as usize * 3;
    let mut frontbuffer = Vec::with_capacity(screen_len);
    frontbuffer.resize(screen_len, 0);

    let mut threadpool = scoped_threadpool::Pool::new(num_cpus::get() as u32);
    let mut should_continue = true;
    let mut frame_start = PreciseTime::now();

    while should_continue {
        // Create a new uninitialized buffer to render into. Because the
        // renderer will write to every pixel, no uninitialized memory is
        // exposed.
        let mut backbuffer: Vec<u8> = Vec::with_capacity(screen_len);
        unsafe { backbuffer.set_len(screen_len); }

        {
            // TODO: Split up the frame in slices.
            let slice = &mut backbuffer[..];

            threadpool.scoped(|scope| {

                // Start rendering with worker threads.
                scope.execute(|| renderer.render_frame_slice(slice, 0, height));

                // In the mean time, upload previously rendered frame to the GPU and
                // display it, then wait for a vsync.
                window.display_buffer(frontbuffer, &mut stats);

                should_continue = window.handle_events(&mut stats);

                // The scope automatically waits for all tasks to complete before
                // the loop continue.
            });
        }

        frontbuffer = backbuffer;
        let now = PreciseTime::now();
        stats.frame_us.insert_time_us(frame_start.to(now));
        frame_start = now;
    }
}
