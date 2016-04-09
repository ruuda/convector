//! This module handles user input and getting pixels onto the screen. It uses
//! the Glium library, a safe wrapper around OpenGL.

use filebuffer::FileBuffer;
use glium::{DisplayBuild, Program, Surface, VertexBuffer};
use glium::backend::Facade;
use glium::backend::glutin_backend::GlutinFacade;
use glium::glutin::{Event, WindowBuilder};
use glium::index::{NoIndices, PrimitiveType};
use glium::texture::{MipmapsOption, RawImage2d, Texture2d};
use stats::GlobalStats;
use std::str;
use time::PreciseTime;
use trace::TraceLog;

/// Vertex for the full-screen quad.
#[derive(Copy, Clone)]
struct Vertex {
    position: [f32; 2],
    tex_coords: [f32; 2],
}

implement_vertex!(Vertex, position, tex_coords);

/// A full-screen quad that can be rendered by OpenGL.
struct FullScreenQuad {
    vertex_buffer: VertexBuffer<Vertex>,
    indices: NoIndices,
    program: Program,
}

impl FullScreenQuad {
    /// Sets up the vertex buffer and shader for a full-screen quad.
    pub fn new<F: Facade>(facade: &F) -> FullScreenQuad {
        let vertex1 = Vertex { position: [-1.0, -1.0], tex_coords: [0.0, 0.0] };
        let vertex2 = Vertex { position: [ 1.0, -1.0], tex_coords: [1.0, 0.0] };
        let vertex3 = Vertex { position: [-1.0,  1.0], tex_coords: [0.0, 1.0] };
        let vertex4 = Vertex { position: [ 1.0,  1.0], tex_coords: [1.0, 1.0] };
        let quad = vec![vertex1, vertex2, vertex3, vertex4];
        let vertex_buffer = VertexBuffer::new(facade, &quad).unwrap();
        let indices = NoIndices(PrimitiveType::TriangleStrip);

        let vertex_shader = FileBuffer::open("src/vertex.glsl")
            .expect("failed to load vertex shader source");
        let fragment_shader = FileBuffer::open("src/fragment.glsl")
            .expect("failed to load fragment shader source");

        // TODO: Add `as_str()` method to Filebuffer for convenience.
        let program = Program::from_source(facade,
                                           str::from_utf8(&vertex_shader[..]).unwrap(),
                                           str::from_utf8(&fragment_shader[..]).unwrap(),
                                           None).unwrap();
        FullScreenQuad {
            vertex_buffer: vertex_buffer,
            indices: indices,
            program: program,
        }
    }

    /// Renders the texture to the target surface.
    pub fn draw_to_surface<S: Surface>(&self,
                                       target: &mut S,
                                       frames: &[Texture2d]) {
        let uniforms = uniform! {
            frame0: &frames[0],
            frame1: &frames[1],
            frame2: &frames[2],
            frame3: &frames[3],
            frame4: &frames[4],
            frame5: &frames[5],
            frame6: &frames[6],
            frame7: &frames[7],
        };
        target.draw(&self.vertex_buffer,
                    &self.indices,
                    &self.program,
                    &uniforms,
                    &Default::default()).expect("failed to draw quad");
    }
}

pub struct Window {
    display: GlutinFacade,
    quad: FullScreenQuad,
    frames: [Texture2d; 8],
    frame_index: u32,
    width: u32,
    height: u32,
}

pub enum Action {
    None,
    Quit,
    ToggleDebugView,
}

fn black_bitmap(width: u32, height: u32) -> Vec<u8> {
    let size = width * height * 4;
    let mut bitmap = Vec::with_capacity(size as usize);
    for _ in 0..size {
        bitmap.push(0);
    }
    bitmap
}

impl Window {
    /// Opens a new window using Glutin.
    pub fn new(width: u32, height: u32, title: &str) -> Window {
        use std::mem;

        // TODO: Proper HiDPI support.
        let display = WindowBuilder::new()
            .with_dimensions(width, height)
            .with_title(From::from(title))
            .with_srgb(Some(true)) // Automatically convert RGB -> sRGB.
            .with_vsync()
            .build_glium()
            .expect("failed to create gl window");

        let quad = FullScreenQuad::new(&display);

        let mut window = Window {
            display: display,
            quad: quad,
            frames: unsafe { mem::uninitialized() },
            frame_index: 0,
            width: width,
            height: height,
        };

        let f0 = window.upload_texture(black_bitmap(width, height));
        let f1 = window.upload_texture(black_bitmap(width, height));
        let f2 = window.upload_texture(black_bitmap(width, height));
        let f3 = window.upload_texture(black_bitmap(width, height));
        let f4 = window.upload_texture(black_bitmap(width, height));
        let f5 = window.upload_texture(black_bitmap(width, height));
        let f6 = window.upload_texture(black_bitmap(width, height));
        let f7 = window.upload_texture(black_bitmap(width, height));
        let frames = [f0, f1, f2, f3, f4, f5, f6, f7];

        // Put the frames in place and avoid deallocating uninitialized memory.
        mem::forget(mem::replace(&mut window.frames, frames));

        window
    }

    fn upload_texture(&mut self, bitmap: Vec<u8>) -> Texture2d {
        let dimensions = (self.width, self.height);
        let texture_data = RawImage2d::from_raw_rgba(bitmap, dimensions);
        let texture = Texture2d::with_mipmaps(&self.display,
                                              texture_data,
                                              MipmapsOption::NoMipmap)
            .expect("failed to create texture");
        texture
    }

    pub fn display_buffer(&mut self, rgba_buffer: Vec<u8>, stats: &mut GlobalStats) {
        assert_eq!(rgba_buffer.len(), self.width as usize * self.height as usize * 4);

        let begin_texture = PreciseTime::now();

        self.frames[self.frame_index as usize] = self.upload_texture(rgba_buffer);
        self.frame_index = (self.frame_index + 1) % 8;

        let begin_draw = PreciseTime::now();

        // Draw a full-screen quad with the texture. Finishing drawing will swap
        // the buffers and wait for a vsync.
        let mut target = self.display.draw();
        self.quad.draw_to_surface(&mut target, &self.frames[..]);
        target.finish().expect("failed to swap buffers");

        let end_draw = PreciseTime::now();
        stats.tex_upload_us.insert_time_us(begin_texture.to(begin_draw));
        stats.draw_vsync_us.insert_time_us(begin_draw.to(end_draw));
    }

    /// Handles all window events and returns an action to be performed.
    pub fn handle_events(&mut self, stats: &GlobalStats, trace_log: &TraceLog) -> Action {
        for ev in self.display.poll_events() {
            match ev {
                // Window was closed by the user.
                Event::Closed => return Action::Quit,
                // The user pressed 'd' to toggle debug view.
                Event::ReceivedCharacter('d') => return Action::ToggleDebugView,
                // The user pressed 'q' for quit.
                Event::ReceivedCharacter('q') => return Action::Quit,
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
        Action::None
    }
}
