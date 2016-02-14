#[warn(missing_docs)]

#[macro_use]
extern crate glium;
extern crate time;

mod renderer;

use glium::Surface;
use renderer::Renderer;
use std::iter;

mod window {
    use glium;

    #[derive(Copy, Clone)]
    struct Vertex {
        position: [f32; 2],
        tex_coords: [f32; 2],
    }

    implement_vertex!(Vertex, position, tex_coords);

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

    static FRAGMENT_SHADER: &'static str = r#"
        #version 140
        in vec2 v_tex_coords;
        out vec4 color;
        uniform sampler2D tex;
        void main() {
            color = texture(tex, v_tex_coords);
        }
    "#;

    pub struct Quad {
        vertex_buffer: glium::VertexBuffer<Vertex>,
        indices: glium::index::NoIndices,
        program: glium::Program,
    }

    impl Quad {
        pub fn new<F: glium::backend::Facade>(facade: &F) -> Quad {
            let vertex1 = Vertex { position: [-1.0, -1.0], tex_coords: [0.0, 0.0] };
            let vertex2 = Vertex { position: [ 1.0, -1.0], tex_coords: [1.0, 0.0] };
            let vertex3 = Vertex { position: [-1.0,  1.0], tex_coords: [0.0, 1.0] };
            let vertex4 = Vertex { position: [ 1.0,  1.0], tex_coords: [1.0, 1.0] };
            let shape = vec![vertex1, vertex2, vertex3, vertex4];
            let vertex_buffer = glium::VertexBuffer::new(facade, &shape).unwrap();
            let indices = glium::index::NoIndices(glium::index::PrimitiveType::TriangleStrip);
            let program = glium::Program::from_source(facade, VERTEX_SHADER, FRAGMENT_SHADER,
                                                      None).unwrap();
            Quad {
                vertex_buffer: vertex_buffer,
                indices: indices,
                program: program,
            }
        }

        pub fn draw_to_surface<S: glium::Surface>(&self, target: &mut S,
                                                  texture: &glium::texture::Texture2d) {
            let uniform = uniform! { tex: texture };
            target.draw(&self.vertex_buffer,
                        &self.indices,
                        &self.program,
                        &uniform,
                        &Default::default())
                  .expect("failed to draw quad");
        }
    }
}

fn main() {
    use window::Quad;
    use glium::DisplayBuild;

    // TODO: Proper HiDPI support.
    let width = 1280;
    let height = 720;

    let renderer = Renderer::new(width, height);

    // TODO: Proper HiDPI support.
    let display = glium::glutin::WindowBuilder::new()
        .with_dimensions(width, height)
        .with_title(format!("Hello world"))
        .with_vsync()
        .with_srgb(Some(true)) // Automatically convert RGB -> sRGB.
        .build_glium()
        .expect("failed to create gl window");

    let quad = Quad::new(&display);

    loop {
        let begin = time::PreciseTime::now();
        for ev in display.poll_events() {
            match ev {
                // Window was closed by the user.
                glium::glutin::Event::Closed => return,
                _ => ()
            }
        }

        // TODO: Use uninitialized vector.
        let mut screen: Vec<u8> = iter::repeat(128)
                                       .take(width as usize * height as usize * 3)
                                       .collect();
        renderer.render(&mut screen[..]);
        let texture_data = glium::texture::RawImage2d::from_raw_rgb(screen, (width, height));
        // TODO: Do not generate mipmaps.
        let texture = glium::texture::Texture2d::with_mipmaps(&display,
                                                              texture_data,
                                                              glium::texture::MipmapsOption::NoMipmap)
                                                .expect("failed to create texture");
        let duration = begin.to(time::PreciseTime::now());
        println!("frame without draw took {:?}", duration);

        let mut target = display.draw();
        quad.draw_to_surface(&mut target, &texture);

        // Finishing drawing will swap the buffers and wait for a vsync.
        target.finish().expect("failed to swap buffers");
        let duration = begin.to(time::PreciseTime::now());
        println!("frame took {:?}", duration);
    }
}
