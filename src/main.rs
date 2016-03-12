//! An interactive raytracer.

#![warn(missing_docs)]
#![allow(dead_code)] // TODO: Remove before v0.1.

#![feature(platform_intrinsics, repr_simd, test)]

extern crate filebuffer;
extern crate glium;
extern crate num_cpus;
extern crate rand;
extern crate scoped_threadpool;
extern crate test;
extern crate thread_id;
extern crate time;

mod aabb;
mod bvh;
mod geometry;
mod ray;
mod renderer;
mod scene;
mod simd;
mod stats;
mod trace;
mod ui;
mod util;
mod vector3;
mod wavefront;

#[cfg(test)]
mod bench;

use renderer::Renderer;
use scene::Scene;
use stats::GlobalStats;
use std::slice;
use ui::Window;
use util::z_order;
use wavefront::Mesh;

fn build_scene() -> Scene {
    use scene::Light;
    use vector3::SVector3;

    let suzanne = Mesh::load("suzanne.obj");
    let mut scene = Scene::from_mesh(&suzanne);

    scene.camera.position = SVector3::new(0.0, 0.0, 5.0);

    let light = Light {
        position: SVector3::new(5.0, 0.0, 6.0),
    };
    scene.lights.push(light);

    scene
}

fn main() {
    // The patch size has been tuned for 8 cores. With a resolution of 1280x736 there are 920
    // patches to be rendered by the worker pool. Increasing the patch size to 64 results in 230
    // patches, but some patches are very heavy to render and some are practically a no-op, so all
    // threads might stall because one thread did not yet finish the frame. A patch width of 32 is
    // a good balance between troughput and latency.
    let width = 1280;
    let height = 736;
    let patch_width = 32;

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

    let mut trace_log = trace::TraceLog::with_limit(6 * 1024);

    while should_continue {
        let stw_frame = trace_log.scoped("render_frame", 0);

        renderer.update_scene();
        {
            let mut patches = make_slices(&mut patchbuffer[..], (patch_width * patch_width * 3) as usize);
            let renderer_ref = &renderer;
            let trace_log_ref = &trace_log;

            threadpool.scoped(|scope| {
                let mut x = 0;
                let mut y = 0;

                // Queue tasks for the worker threads to render patches.
                let mut i = 0;
                for patch in &mut patches {
                    scope.execute(move || {
                        let _stw = trace_log_ref.scoped("render_patch", i);
                        renderer_ref.render_patch(patch, patch_width as u32, x as u32, y as u32);
                    });

                    i = i + 1;
                    x = x + patch_width;
                    if x >= width {
                        x = 0;
                        y = y + patch_width;
                    }
                }

                {
                    // In the mean time, upload the previously rendered frame to
                    // the GPU and display it.
                    let _stw = trace_log.scoped("display_buffer", 0);
                    window.display_buffer(frontbuffer, &mut stats);
                }

                should_continue = window.handle_events(&mut stats, &trace_log);

                // The scope automatically waits for all tasks to complete
                // before the loop continues.
            });

            let _stw = trace_log.scoped("stitch_patches", 0);

            // Create a new uninitialized buffer to stitch the patches into.
            // Because this will write to every pixel, no uninitialized memory
            // is exposed.
            let mut backbuffer: Vec<u8> = Vec::with_capacity(screen_len);
            unsafe { backbuffer.set_len(screen_len); }

            // Stitch together the patches into an image.
            let mut x = 0;
            let mut y = 0;
            for patch in &patches {
                for i in 0..(patch_width * patch_width) {
                    let (px, py) = z_order(i as u16);
                    let bb_idx = ((y + py as u32) * width * 3 + (x + px as u32) * 3) as usize;
                    backbuffer[bb_idx + 0] = patch[i as usize * 3 + 0];
                    backbuffer[bb_idx + 1] = patch[i as usize * 3 + 1];
                    backbuffer[bb_idx + 2] = patch[i as usize * 3 + 2];
                }
                x = x + patch_width; // TODO: DRY this up.
                if x >= width {
                    x = 0;
                    y = y + patch_width;
                }
            }

            frontbuffer = backbuffer;
        }

        stats.frame_us.insert_time_us(stw_frame.take_duration());
        trace_log.inc_frame_number();
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
