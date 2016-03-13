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

use renderer::{PatchBuffer, Renderer};
use scene::Scene;
use stats::GlobalStats;
use std::mem;
use ui::Window;
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
    let mut trace_log = trace::TraceLog::with_limit(6 * 1024);
    let mut threadpool = scoped_threadpool::Pool::new(num_cpus::get() as u32);
    let mut backbuffer = PatchBuffer::new_black(width, height, patch_width);
    let mut should_continue = true;

    println!("scene and renderer initialized, entering render loop");

    while should_continue {
        trace_log.inc_frame_number();
        let stw_frame = trace_log.scoped("render_frame", 0);

        renderer.update_scene();

        let new_backbuffer = PatchBuffer::new_uninit(width, height, patch_width);
        let frontbuffer = mem::replace(&mut backbuffer, new_backbuffer);
        let patches = backbuffer.patches();
        let renderer_ref = &renderer;
        let trace_log_ref = &trace_log;

        threadpool.scoped(|scope| {
            let mut x = 0;
            let mut y = 0;

            // Queue tasks for the worker threads to render patches.
            for (i, patch) in (0..).zip(patches) {
                scope.execute(move || {
                    let _stw = trace_log_ref.scoped("render_patch", i);
                    renderer_ref.render_patch(patch, patch_width as u32, x as u32, y as u32);
                });

                // TODO: DRY.
                x = x + patch_width;
                if x >= width {
                    x = 0;
                    y = y + patch_width;
                }
            }

            // In the mean time, stitch together the patches from the previous
            // frame, upload the frame to the GPU, and display it.
            let stw_stitch = trace_log.scoped("stitch_patches", 0);
            let frame = frontbuffer.into_bitmap();
            stw_stitch.take_duration();

            let stw_display = trace_log.scoped("display_buffer", 0);
            window.display_buffer(frame, &mut stats);
            stw_display.take_duration();

            should_continue = window.handle_events(&mut stats, &trace_log);

            // The scope automatically waits for all tasks to complete
            // before the loop continues.
        });

        stats.frame_us.insert_time_us(stw_frame.take_duration());
    }
}
