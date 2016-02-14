//! This module handles user input and getting pixels onto the screen. It uses
//! the Glium library, a safe wrapper around OpenGL.

use glium::{DisplayBuild, Program, Surface, VertexBuffer};
use glium::backend::Facade;
use glium::backend::glutin_backend::GlutinFacade;
use glium::glutin::{Event, WindowBuilder};
use glium::index::{NoIndices, PrimitiveType};
use glium::texture::{MipmapsOption, RawImage2d, Texture2d};
use stats::Stats;
use time::PreciseTime;

/// Vertex for the full-screen quad.
#[derive(Copy, Clone)]
struct Vertex {
    position: [f32; 2],
    tex_coords: [f32; 2],
}

implement_vertex!(Vertex, position, tex_coords);

/// Vertex shader for the full-screen quad.
static VERTEX_SHADER: &'static str = r#"
    #version 140
    in vec2 position;
    in vec2 tex_coords;
    out vec2 v_tex_coords;
    void main() {
        gl_Position = vec4(position, 0.0, 1.0);
        v_tex_coords = tex_coords;
    }
"#;

/// Fragment shader for the full-screen quad.
static FRAGMENT_SHADER: &'static str = r#"
    #version 140
    in vec2 v_tex_coords;
    out vec4 color;
    uniform sampler2D tex;
    void main() {
        color = texture(tex, v_tex_coords);
    }
"#;

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
        let program = Program::from_source(facade,
                                           VERTEX_SHADER,
                                           FRAGMENT_SHADER,
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
                                       texture: &Texture2d) {
        let uniforms = uniform! { tex: texture };
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
    width: u32,
    height: u32,
    tex_upload_stats: Stats<u32>,
    draw_vsync_stats: Stats<u32>,
}

impl Window {
    /// Opens a new window using Glutin.
    pub fn new(width: u32, height: u32, title: &str) -> Window {
        // TODO: Proper HiDPI support.
        let display = WindowBuilder::new()
            .with_dimensions(width, height)
            .with_title(From::from(title))
            .with_vsync()
            .with_srgb(Some(true)) // Automatically convert RGB -> sRGB.
            .build_glium()
            .expect("failed to create gl window");

        let quad = FullScreenQuad::new(&display);

        Window {
            display: display,
            quad: quad,
            width: width,
            height: height,
            tex_upload_stats: Stats::new(),
            draw_vsync_stats: Stats::new(),
        }
    }

    pub fn render(&mut self, rgb_buffer: Vec<u8>) {
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
        let mut target = self.display.draw();
        self.quad.draw_to_surface(&mut target, &texture);
        target.finish().expect("failed to swap buffers");

        let end_draw = PreciseTime::now();
        let tex_upload_ns = begin_texture.to(begin_draw).num_nanoseconds();
        let draw_vsync_ns = begin_draw.to(end_draw).num_nanoseconds();
        let tex_upload_us = (tex_upload_ns.unwrap() + 500) / 1000;
        let draw_vsync_us = (draw_vsync_ns.unwrap() + 500) / 1000;
        self.tex_upload_stats.insert(tex_upload_us as u32);
        self.draw_vsync_stats.insert(draw_vsync_us as u32);
        println!("texture upload min: {} us, median: {} us",
                 self.tex_upload_stats.min(),
                 self.tex_upload_stats.median());
        println!("draw and vsync min: {} us, median: {} us",
                 self.draw_vsync_stats.min(),
                 self.draw_vsync_stats.median());
    }

    /// Handles all window events and returns whether the app should continue to
    /// run.
    pub fn handle_events(&mut self) -> bool {
        for ev in self.display.poll_events() {
            match ev {
                // Window was closed by the user.
                Event::Closed => return false,
                // The user pressed 'q'.
                Event::ReceivedCharacter('q') => return false,
                // Something else.
                _ => ()
            }
        }
        true
    }
}
