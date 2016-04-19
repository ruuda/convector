//! An interactive path tracer.

#![allow(dead_code)] // TODO: Remove at some point.

#![feature(alloc, cfg_target_feature, heap_api, iter_arith, platform_intrinsics, repr_simd, test)]

extern crate alloc;
extern crate filebuffer;
extern crate imagefmt;
extern crate num_cpus;
extern crate rand;
extern crate rayon;
extern crate scoped_threadpool;
extern crate test;
extern crate thread_id;
extern crate time;

#[macro_use]
extern crate glium;

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

use material::SMaterial;
use renderer::{RenderBuffer, Renderer};
use scene::Scene;
use stats::GlobalStats;
use std::collections::HashMap;
use std::mem;
use time::PreciseTime;
use ui::{Action, Window};
use wavefront::Mesh;

fn load_textures() -> Vec<Vec<u8>> {
    use imagefmt::ColFmt;

    println!("loading textures");
    let tex_floor = imagefmt::read("textures/floor.jpg", ColFmt::RGB);
    let tex_wood = imagefmt::read("textures/wood_light.jpg", ColFmt::RGB);
    let mut textures = Vec::with_capacity(2);
    textures.push(tex_floor.expect("failed to read floor.jpeg").buf);
    textures.push(tex_wood.expect("failed to read wood_light.jpg").buf);
    textures
}

fn build_scene() -> Scene {
    println!("loading geometry");
    let mut materials = HashMap::new();
    materials.insert("baseboard", SMaterial::white().with_glossiness(4));
    materials.insert("ceiling", SMaterial::white().with_glossiness(1));
    materials.insert("fauteuil", SMaterial::diffuse(1.0, 0.1, 0.4));
    materials.insert("floor", SMaterial::diffuse(0.569, 0.494, 0.345).with_glossiness(4).with_texture(1));
    materials.insert("glass", SMaterial::sky());
    materials.insert("wall", SMaterial::diffuse(0.65, 0.7, 0.9).with_glossiness(1));
    materials.insert("wood_light", SMaterial::diffuse(0.6, 0.533, 0.455).with_glossiness(3).with_texture(2));
    let indoor = Mesh::load_with_materials("models/indoor.obj", &materials);
    let meshes = [indoor];

    println!("building bvh");
    let scene = Scene::from_meshes(&meshes);
    scene.print_stats();

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

    let mut window = Window::new(width, height, "infomagr interactive path tracer");
    let mut renderer = Renderer::new(build_scene(), width, height);
    let mut stats = GlobalStats::new();
    let mut trace_log = trace::TraceLog::with_limit(6 * 1024);
    let mut threadpool = scoped_threadpool::Pool::new(num_cpus::get() as u32);
    let mut backbuffer = RenderBuffer::new(width, height);
    let mut backbuffer_g = RenderBuffer::new(width, height);
    let mut f32_buffer = renderer.new_buffer_f32(); // TODO: Consistency.
    let mut f32_buffer_samples = 0;
    let mut should_continue = true;
    let mut render_realtime = true;

    for texture in load_textures() {
        window.upload_texture(texture);
    }

    backbuffer.fill_black();
    let epoch = PreciseTime::now();

    // Insert one fake value so we have an initial guess for the time delta.
    stats.frame_us.insert(16_667);

    println!("scene and renderer initialized, entering render loop");

    while should_continue {
        let frame_number = trace_log.inc_frame_number();
        let stw_frame = trace_log.scoped("render_frame", 0);

        let time = epoch.to(PreciseTime::now()).num_milliseconds() as f32 * 1e-3;
        let time_delta = (stats.frame_us.median() as f32) * 1e-6;

        match window.handle_events() {
            Action::DumpTrace => {
                trace_log.export_to_file("trace.json").expect("failed to write trace");
                println!("wrote trace to trace.json");
            }
            Action::Quit => should_continue = false,
            Action::PrintStats => stats.print(),
            Action::ToggleDebugView => renderer.toggle_debug_view(),
            Action::ToggleRealtime => {
                render_realtime = !render_realtime;
                f32_buffer = renderer.new_buffer_f32();
                f32_buffer_samples = 0;
                // In accumulative mode the time is fixed and there is no motion
                // blur.
                renderer.set_time(time, 0.0);
            }
            Action::None => { }
        }

        if render_realtime { renderer.set_time(time, time_delta); }
        renderer.update_scene();

        // When rendering in accumulation mode, first copy the current state
        // into the backbuffer (which will immediately after this become the new
        // front buffer) so we can display it later.
        if !render_realtime {
            let n = if f32_buffer_samples > 0 { f32_buffer_samples } else { 1 };
            renderer.buffer_f32_into_render_buffer(&f32_buffer, &mut backbuffer, n);
            f32_buffer_samples += 1;
        }

        let new_backbuffer = RenderBuffer::new(width, height);
        let new_backbuffer_g = RenderBuffer::new(width, height);
        let frontbuffer = mem::replace(&mut backbuffer, new_backbuffer);
        let frontbuffer_g = mem::replace(&mut backbuffer_g, new_backbuffer_g);
        let renderer_ref = &renderer;
        let trace_log_ref = &trace_log;
        let backbuffer_ref = &backbuffer;
        let backbuffer_g_ref = &backbuffer_g;
        let f32_buffer_ref = &f32_buffer[..];

        threadpool.scoped(|scope| {

            let w = width / patch_width;
            let h = height / patch_width;

            // Queue tasks for the worker threads to render patches.
            for i in 0..w {
                for j in 0..h {
                    scope.execute(move || {
                        let x = i * patch_width;
                        let y = j * patch_width;

                        // Multiple threads mutably borrow the buffer below,
                        // which could cause races, but all of the patches are
                        // disjoint, hence it is safe.

                        if render_realtime {
                            let _stw = trace_log_ref.scoped("render_patch_u8", j * w + i);
                            let bitmap = unsafe { backbuffer_ref.get_mut_slice() };
                            let gbuffer = unsafe { backbuffer_g_ref.get_mut_slice() };
                            renderer_ref.render_patch_u8(bitmap, gbuffer, patch_width, x, y, frame_number);
                        } else {
                            let _stw = trace_log_ref.scoped("accumulate_patch_f32", j * w + i);
                            let buffer = unsafe { util::make_mutable(f32_buffer_ref) };
                            let gbuffer = unsafe { backbuffer_g_ref.get_mut_slice() };
                            renderer_ref.accumulate_patch_f32(buffer, gbuffer, patch_width, x, y, frame_number);
                            let bitmap = unsafe { backbuffer_ref.get_mut_slice() };
                            renderer_ref.render_patch_u8(bitmap, gbuffer, patch_width, x, y, frame_number);
                        }
                    });
                }
            }

            // In the mean time upload the previous frame to the GPU
            // and display it.
            let _stw_display = trace_log.scoped("display_buffer", 0);
            window.display_buffer(frontbuffer.into_bitmap(),
                                  frontbuffer_g.into_bitmap(),
                                  &mut stats);

            // The scope automatically waits for all tasks to complete
            // before the loop continues.
        });

        stats.frame_us.insert_time_us(stw_frame.take_duration());
    }
}
