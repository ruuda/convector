#[warn(missing_docs)]

#[macro_use]
extern crate glium;

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

    implement_vertex!(Vertex, position);

    static VERTEX_SHADER: &'static str = r#"
        #version 140
        in vec2 position;
        in vec2 tex_coords;
        out vec2 v_tex_coords;
        void main() {
            gl_Position = vec4(position, 0.0, 1.0);
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
            let vertex1 = Vertex { position: [-0.5, -0.5], tex_coords: [0.0, 0.0] };
            let vertex2 = Vertex { position: [ 0.0,  0.5], tex_coords: [1.0, 0.0] };
            let vertex3 = Vertex { position: [ 0.5, -0.25], tex_coords: [1.0, 1.0] };
            let shape = vec![vertex1, vertex2, vertex3];
            let vertex_buffer = glium::VertexBuffer::new(facade, &shape).unwrap();
            let indices = glium::index::NoIndices(glium::index::PrimitiveType::TrianglesList);
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
    let width = 503; // 1280;
    let height = 521; //720;

    let renderer = Renderer::new(width, height);

    // TODO: Proper HiDPI support.
    let display = glium::glutin::WindowBuilder::new()
        //.with_dimensions(width, height)
        //.with_title(format!("Hello world"))
        .build_glium()
        .expect("failed to create gl window");

    let mut screen: Vec<u8> = iter::repeat(128).take(width as usize * height as usize * 4).collect();
    renderer.render(&mut screen[..]);
    // TODO: Why do I see garbage?
    let mut screen: Vec<u8> = iter::repeat(128).take((width * height * 4) as usize).collect();
    for y in (0..height) {
        for x in (0..width) {
            let i = ((y * width + x) * 4) as usize;
            screen[i + 0] = (x * 256 / width) as u8;
            screen[i + 1] = (y * 256 / height) as u8;
            screen[i + 2] = 0;
            screen[i + 3] = 255;
        }
    }
    let texture_data = glium::texture::RawImage2d::from_raw_rgba_reversed(screen, (width, height));
    let texture = glium::texture::Texture2d::new(&display, texture_data)
                                            .expect("failed to create texture");

    let quad = Quad::new(&display);

    loop {
        let mut target = display.draw();
        target.clear_color(0.0, 0.0, 1.0, 1.0);
        quad.draw_to_surface(&mut target, &texture);
        target.finish().expect("failed to swap buffers");

        for ev in display.poll_events() {
            match ev {
                // Window was closed by the user.
                glium::glutin::Event::Closed => return,
                _ => ()
            }
        }

        // TODO: Wait for vsync.
    }
}
