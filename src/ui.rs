//! This module handles user input and getting pixels onto the screen. It uses
//! the Glium library, a safe wrapper around OpenGL.

use glium::{DisplayBuild, Surface};
use glium::backend::glutin_backend::GlutinFacade;
use glium::glutin::{Event, WindowBuilder};
use glium::texture::{MipmapsOption, RawImage2d, Texture2d};
use glium::uniforms::MagnifySamplerFilter;
use stats::GlobalStats;
use time::PreciseTime;
use trace::TraceLog;

pub struct Window {
    display: GlutinFacade,
    width: u32,
    height: u32,
}

impl Window {
    /// Opens a new window using Glutin.
    pub fn new(width: u32, height: u32, title: &str) -> Window {
        // TODO: Proper HiDPI support.
        let display = WindowBuilder::new()
            .with_dimensions(width, height)
            .with_title(From::from(title))
            .with_srgb(Some(true)) // Automatically convert RGB -> sRGB.
            .build_glium()
            .expect("failed to create gl window");

        Window {
            display: display,
            width: width,
            height: height,
        }
    }

    pub fn display_buffer(&mut self, rgb_buffer: Vec<u8>, stats: &mut GlobalStats) {
        assert_eq!(rgb_buffer.len(), self.width as usize * self.height as usize * 3);

        let begin_texture = PreciseTime::now();

        // Create a texture from the rgb data and upload it to the GPU.
        let dimensions = (self.width, self.height);
        let texture_data = RawImage2d::from_raw_rgb(rgb_buffer, dimensions);
        let texture = Texture2d::with_mipmaps(&self.display,
                                              texture_data,
                                              MipmapsOption::NoMipmap)
            .expect("failed to create texture");

        let begin_draw = PreciseTime::now();

        // Draw a full-screen quad with the texture. Finishing drawing will swap
        // the buffers and wait for a vsync.
        let target = self.display.draw();
        texture.as_surface().fill(&target, MagnifySamplerFilter::Linear);
        target.finish().expect("failed to swap buffers");

        let end_draw = PreciseTime::now();
        stats.tex_upload_us.insert_time_us(begin_texture.to(begin_draw));
        stats.draw_vsync_us.insert_time_us(begin_draw.to(end_draw));
    }

    /// Handles all window events and returns whether the app should continue to
    /// run.
    pub fn handle_events(&mut self, stats: &GlobalStats, trace_log: &TraceLog) -> bool {
        for ev in self.display.poll_events() {
            match ev {
                // Window was closed by the user.
                Event::Closed => return false,
                // The user pressed 'q' for quit.
                Event::ReceivedCharacter('q') => return false,
                // The user pressed 's' for stats.
                // TODO: Invert dependency, handle_events should not take stats.
                Event::ReceivedCharacter('s') => stats.print(),
                // The user pressed 't' for trace.
                // TODO: Invert dependency, handle_events should not take the trace log.
                Event::ReceivedCharacter('t') => {
                    trace_log.export_to_file("trace.json").expect("failed to write trace");
                    println!("wrote trace to trace.json");
                },
                // Something else.
                _ => ()
            }
        }
        true
    }
}
