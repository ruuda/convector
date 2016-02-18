//! This module reads Wavefront OBJ files. There are crates for that, but
//! reinventing the wheel is much more fun.

use filebuffer::FileBuffer;
use std::path::Path;
use std::str::{FromStr, from_utf8};
use vector3::Vector3;

pub struct Mesh {
    pub vertices: Vec<Vector3>,
    pub triangles: Vec<(u32, u32, u32)>,
}

impl Mesh {
    pub fn load<P: AsRef<Path>>(path: P) -> Mesh {
        let fbuffer = FileBuffer::open(path).expect("failed to open file");
        let input = from_utf8(&fbuffer[..]).expect("obj must be valid utf-8");
        let mut vertices = Vec::new();
        let mut triangles = Vec::new();
        for line in input.lines() {
            if line.is_empty() { continue }
            let mut pieces = line.split_whitespace();
            match pieces.next() {
                Some("v") => {
                    let mut coords = pieces.map(|v| f32::from_str(v).unwrap());
                    let vertex = Vector3 {
                        x: coords.next().expect("missing x coordinate"),
                        y: coords.next().expect("missing y coordinate"),
                        z: coords.next().expect("missing z coordinate"),
                    };
                    vertices.push(vertex);
                }
                Some("f") => {
                    // Indices stored are 1-based, convert to 0-based.
                    let mut indices = pieces.map(|i| u32::from_str(i).unwrap() - 1);
                    let i1 = indices.next().expect("missing triangle index");
                    let i2 = indices.next().expect("missing triangle index");
                    let i3 = indices.next().expect("missing triangle index");
                    triangles.push((i1, i2, i3));

                    // There might be a quad; triangulate it at import time.
                    if let Some(i4) = indices.next() {
                        triangles.push((i1, i3, i4))
                    }
                },
                _ => { /* Anything else is not supported. */ }
            }
        }

        Mesh {
            vertices: vertices,
            triangles: triangles,
        }
    }
}
