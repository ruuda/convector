//! An interactive raytracer.

#![allow(dead_code)] // TODO: Remove at some point.

#![feature(alloc, cfg_target_feature, heap_api, iter_arith, platform_intrinsics, repr_simd, test)]

extern crate alloc;
extern crate filebuffer;
extern crate glium;
extern crate num_cpus;
extern crate rand;
extern crate rayon;
extern crate scoped_threadpool;
extern crate test;
extern crate thread_id;
extern crate time;

mod aabb;
mod bvh;
mod material;
mod quaternion;
mod random;
mod ray;
mod renderer;
mod scene;
mod simd;
mod stats;
mod trace;
mod triangle;
mod ui;
mod util;
mod vector3;
mod wavefront;

#[cfg(test)]
mod bench;

use renderer::{RenderBuffer, Renderer};
use scene::Scene;
use stats::GlobalStats;
use std::mem;
use ui::{Action, Window};
use wavefront::Mesh;

fn build_scene() -> Scene {
    use vector3::SVector3;

    println!("loading geometry");
    let plane = Mesh::load("models/plane.obj");
    // let suzanne = Mesh::load("models/suzanne.obj");
    // let bunny = Mesh::load("models/stanford_bunny.obj");
    let dragon = Mesh::load("models/stanford_dragon.obj");
    let meshes = [dragon, plane];

    println!("building bvh");
    let mut scene = Scene::from_meshes(&meshes);

    scene.bvh.print_stats();

    scene.camera.position = SVector3::new(0.0, 5.0, 25.0);
    scene.camera.set_fov(0.9);

    scene
}

fn main() {
    // The patch size has been tuned for 8 cores. With a resolution of 1280x736 there are 920
    // patches to be rendered by the worker pool. Increasing the patch size to 64 results in 230
    // patches, but some patches are very heavy to render and some are practically a no-op, so all
    // threads might stall because one thread did not yet finish the frame. A patch width of 32 is
    // a good balance between throughput and latency.
    let width = 1280;
    let height = 736;
    let patch_width = 32;

    let mut window = Window::new(width, height, "infomagr interactive raytracer");
    let mut renderer = Renderer::new(build_scene(), width, height);
    let mut stats = GlobalStats::new();
    let mut trace_log = trace::TraceLog::with_limit(6 * 1024);
    let mut threadpool = scoped_threadpool::Pool::new(num_cpus::get() as u32);
    let mut backbuffer = RenderBuffer::new(width, height);
    let mut should_continue = true;

    backbuffer.fill_black();

    println!("scene and renderer initialized, entering render loop");

    while should_continue {
        trace_log.inc_frame_number();
        let stw_frame = trace_log.scoped("render_frame", 0);

        renderer.update_scene();

        match window.handle_events(&mut stats, &trace_log) {
            Action::Quit => should_continue = false,
            Action::ToggleDebugView => renderer.toggle_debug_view(),
            Action::None => { }
        }

        let new_backbuffer = RenderBuffer::new(width, height);
        let frontbuffer = mem::replace(&mut backbuffer, new_backbuffer);
        let renderer_ref = &renderer;
        let trace_log_ref = &trace_log;
        let backbuffer_ref = &backbuffer;

        threadpool.scoped(|scope| {

            let w = width / patch_width;
            let h = height / patch_width;

            // Queue tasks for the worker threads to render patches.
            for i in 0..w {
                for j in 0..h {
                    scope.execute(move || {
                        let _stw = trace_log_ref.scoped("render_patch", j * w + i);
                        let x = i * patch_width;
                        let y = j * patch_width;

                        // Multiple threads mutably borrow the buffer, which
                        // could cause races, but all of the patches are
                        // disjoint, hence this is safe.
                        let bitmap = unsafe { backbuffer_ref.get_mut_slice() };
                        renderer_ref.render_patch(bitmap, patch_width, x, y);
                    });
                }
            }

            // In the mean time upload the previous frame to the GPU
            // and display it.
            let _stw_display = trace_log.scoped("display_buffer", 0);
            window.display_buffer(frontbuffer.into_bitmap(), &mut stats);

            // The scope automatically waits for all tasks to complete
            // before the loop continues.
        });

        stats.frame_us.insert_time_us(stw_frame.take_duration());
    }
}
