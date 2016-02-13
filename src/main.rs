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
    }

    implement_vertex!(Vertex, position);

    static vertex_shader: &'static str = r#"
        #version 140
        in vec2 position;
        void main() {
            gl_Position = vec4(position, 0.0, 1.0);
        }
    "#;

    static fragment_shader: &'static str = r#"
        #version 140
        out vec4 color;
        void main() {
            color = vec4(1.0, 0.0, 0.0, 1.0);
        }
    "#;

    pub struct Quad {
        vertex_buffer: glium::VertexBuffer<Vertex>,
        indices: glium::index::NoIndices,
        program: glium::Program,
    }

    impl Quad {
        pub fn new<F: glium::backend::Facade>(facade: &F) -> Quad {
            let vertex1 = Vertex { position: [-0.5, -0.5] };
            let vertex2 = Vertex { position: [ 0.0,  0.5] };
            let vertex3 = Vertex { position: [ 0.5, -0.25] };
            let shape = vec![vertex1, vertex2, vertex3];
            let vertex_buffer = glium::VertexBuffer::new(facade, &shape).unwrap();
            let indices = glium::index::NoIndices(glium::index::PrimitiveType::TrianglesList);
            let program = glium::Program::from_source(facade, vertex_shader, fragment_shader,
                                                      None).unwrap();
            Quad {
                vertex_buffer: vertex_buffer,
                indices: indices,
                program: program,
            }
        }

        pub fn draw_to_surface<S: glium::Surface>(&self, target: &mut S) {
            target.draw(&self.vertex_buffer,
                        &self.indices,
                        &self.program,
                        &glium::uniforms::EmptyUniforms,
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
    let mut screen: Vec<u8> = iter::repeat(0).take(width as usize * height as usize * 3).collect();

    // TODO: Proper HiDPI support.
    let display = glium::glutin::WindowBuilder::new()
        .with_dimensions(width, height)
        .with_title(format!("Hello world"))
        .build_glium()
        .expect("failed to create gl window");

    let quad = Quad::new(&display);

    loop {
        renderer.render(&mut screen[..]);
        let mut target = display.draw();
        target.clear_color(1.0, 1.0, 0.0, 1.0);
        quad.draw_to_surface(&mut target);
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
