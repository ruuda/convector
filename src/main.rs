//! An interactive raytracer.

#![warn(missing_docs)]
#![allow(dead_code)] // TODO: Remove before v0.1.

// Note: the following unstable feature is required to run benchmarks, but
// unstable features can only be used with a Rust compiler from the nightly
// channel. If you only want to run the program you can safely comment out this
// line.
#![feature(test)]

extern crate filebuffer;
extern crate glium;
extern crate num_cpus;
extern crate rand;
extern crate scoped_threadpool;
extern crate time;

#[cfg(test)]
extern crate test;

mod aabb;
mod bench;
mod bvh;
mod geometry;
mod ray;
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
    use scene::Light;
    use vector3::Vector3;

    let suzanne = Mesh::load("suzanne.obj");
    let mut scene = Scene::from_mesh(&suzanne);

    scene.camera.position = Vector3::new(0.0, 0.0, 5.0);

    let light = Light {
        position: Vector3::new(5.0, 0.0, 6.0),
    };
    scene.lights.push(light);

    scene
}

fn main() {
    let width = 1280;
    let height = 720;
    let patch_width = 16;

    let mut window = Window::new(width, height, "infomagr interactive raytracer");
    let mut renderer = Renderer::new(build_scene(), width, height);
    let mut stats = GlobalStats::new();

    // Initialize a buffer to black.
    let screen_len = width as usize * height as usize * 3;
    let mut frontbuffer = Vec::with_capacity(screen_len);
    let mut patchbuffer = Vec::with_capacity(screen_len);
    frontbuffer.resize(screen_len, 0);
    patchbuffer.resize(screen_len, 0);

    let mut threadpool = scoped_threadpool::Pool::new(num_cpus::get() as u32);
    let mut should_continue = true;
    println!("scene and renderer initialized, entering render loop");
    let mut frame_start = PreciseTime::now();

    while should_continue {

        renderer.update_scene();
        {
            let mut patches = make_slices(&mut patchbuffer[..], (patch_width * patch_width * 3) as usize);
            let renderer_ref = &renderer;

            threadpool.scoped(|scope| {
                let mut x = 0;
                let mut y = 0;

                // Queue tasks for the worker threads to render patches.
                for patch in &mut patches {
                    scope.execute(move ||
                        renderer_ref.render_patch(patch, patch_width as u32, x as u32, y as u32));

                    x = x + patch_width;
                    if x >= width {
                        x = 0;
                        y = y + patch_width;
                    }
                }

                // In the mean time, upload the previously rendered frame to the
                // GPU and display it.
                window.display_buffer(frontbuffer, &mut stats);

                should_continue = window.handle_events(&mut stats);

                // The scope automatically waits for all tasks to complete
                // before the loop continues.
            });

            // Create a new uninitialized buffer to stitch the patches into.
            // Because this will write to every pixel, no uninitialized memory
            // is exposed.
            let mut backbuffer: Vec<u8> = Vec::with_capacity(screen_len);
            unsafe { backbuffer.set_len(screen_len); }

            // Stitch together the patches into an image.
            let mut x = 0;
            let mut y = 0;
            for patch in &patches {
                for j in 0..patch_width {
                    for i in 0..patch_width {
                        // TODO: Morton copier.
                        let bb_idx = ((y + j) * width * 3 + (x + i) * 3) as usize;
                        let pa_idx = (j * patch_width * 3 + i * 3) as usize;
                        backbuffer[bb_idx] = patch[pa_idx];
                    }
                }
                x = x + patch_width; // TODO: DRY this up.
                if x >= width {
                    x = 0;
                    y = y + patch_width;
                }
            }

            frontbuffer = backbuffer;
        }

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
