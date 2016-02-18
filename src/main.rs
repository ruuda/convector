#![warn(missing_docs)]

#[macro_use]
extern crate filebuffer;
extern crate glium;
extern crate num_cpus;
extern crate scoped_threadpool;
extern crate time;

mod renderer;
mod scene;
mod stats;
mod ui;
mod vector3;
mod wavefront;

use renderer::Renderer;
use scene::Scene;
use stats::GlobalStats;
use std::slice;
use time::PreciseTime;
use ui::Window;
use wavefront::Mesh;

fn build_scene() -> Scene {
    use vector3::Vector3;
    let mut scene = Scene::new();
    scene.camera.position = Vector3::new(0.0, 0.0, -5.0);
    let suzanne = Mesh::load("suzanne.obj");
    scene.add_mesh(&suzanne);
    scene
}

fn main() {
    let width = 1280;
    let height = 720;
    let slice_height = height / 20;

    let mut window = Window::new(width, height, "infomagr interactive tracer");
    let renderer = Renderer::new(build_scene(), width, height);
    let mut stats = GlobalStats::new();

    // Initialize a buffer to black.
    let screen_len = width as usize * height as usize * 3;
    let mut frontbuffer = Vec::with_capacity(screen_len);
    frontbuffer.resize(screen_len, 0);

    let mut threadpool = scoped_threadpool::Pool::new(num_cpus::get() as u32);
    let mut should_continue = true;
    println!("scene and renderer initialized, entering render loop");
    let mut frame_start = PreciseTime::now();

    while should_continue {
        // Create a new uninitialized buffer to render into. Because the
        // renderer will write to every pixel, no uninitialized memory is
        // exposed.
        let mut backbuffer: Vec<u8> = Vec::with_capacity(screen_len);
        unsafe { backbuffer.set_len(screen_len); }

        {
            let slice_len = 3 * width * slice_height;
            let slices = make_slices(&mut backbuffer[..], slice_len as usize);
            let renderer_ref = &renderer;

            threadpool.scoped(|scope| {
                // Queue tasks for the worker threads to render slices.
                let mut y_from = 0;
                let mut y_to = slice_height;
                for slice in slices {
                    scope.execute(move || renderer_ref.render_frame_slice(slice, y_from, y_to));
                    y_from += slice_height;
                    y_to += slice_height;
                }

                // In the mean time, upload previously rendered frame to the GPU
                // and display it, then wait for a vsync.
                window.display_buffer(frontbuffer, &mut stats);

                should_continue = window.handle_events(&mut stats);

                // The scope automatically waits for all tasks to complete
                // before the loop continue.
            });
        }

        frontbuffer = backbuffer;
        let now = PreciseTime::now();
        stats.frame_us.insert_time_us(frame_start.to(now));
        frame_start = now;
    }
}

/// Splits the buffer into slices of length `slice_len`.
///
/// This is similar to `slice::split_at_mut()`.
fn make_slices<'a>(backbuffer: &'a mut [u8], slice_len: usize) -> Vec<&'a mut [u8]> {
    let len = backbuffer.len();
    let mut ptr = backbuffer.as_mut_ptr();
    let end = unsafe { ptr.offset(len as isize) };
    let mut slices = Vec::with_capacity(len / slice_len);
    while ptr < end {
        slices.push(unsafe { slice::from_raw_parts_mut(ptr, slice_len) });
        ptr = unsafe { ptr.offset(slice_len as isize) };
    }
    slices
}
