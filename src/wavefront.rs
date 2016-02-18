//! This module reads Wavefront OBJ files. There are crates for that, but
//! reinventing the wheel is much more fun.

use filebuffer::FileBuffer;
use std::io;
use std::path::Path;
use std::str::from_utf8;
use vector3::Vector3;

pub struct Mesh {
    vertices: Vec<Vector3>,
    triangles: Vec<(u32, u32, u32)>,
}

impl Mesh {
    pub fn load<P: AsRef<Path>>(path: P) -> io::Result<Mesh> {
        let fbuffer = try!(FileBuffer::open(path));
        let input = from_utf8(&fbuffer[..]).expect("obj must be valid utf-8");
        let mut vertices = Vec::new();
        let mut triangles = Vec::new();
        for line in input.lines() {
            if line.is_empty() { continue }
            match line.as_bytes()[0] {
                b'v' => { /* TODO */ }
                b'f' => { /* TODO */ }
                _ => { /* Anything else is not supported, ignore. */ }
            }
        }
        let mesh = Mesh {
            vertices: vertices,
            triangles: triangles,
        };
        Ok(mesh)
    }
}