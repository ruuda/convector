//! This module reads Wavefront OBJ files. There are crates for that, but
//! reinventing the wheel is much more fun.

use filebuffer::FileBuffer;
use material::SMaterial;
use std::path::Path;
use std::str::{FromStr, from_utf8};
use vector3::SVector3;

pub struct Mesh {
    pub vertices: Vec<SVector3>,
    pub tex_coords: Vec<(f32, f32)>,
    pub triangles: Vec<(u32, u32, u32)>,
    pub material: SMaterial,
}

fn assert_nondegenerate(vertices: &[SVector3], line: u32, i0: u32, i1: u32, i2: u32) {
    let v0 = vertices[i0 as usize];
    let v1 = vertices[i1 as usize];
    let v2 = vertices[i2 as usize];

    // The cross product of two edges must not be zero. If it is, the three
    // vertices are collinear.
    let e1 = v0 - v2;
    let e2 = v1 - v0;
    if e1.cross(e2).norm_squared() == 0.0 {
        println!("encountered degenerate triangle while loading mesh");
        println!("  line:     {}", line);
        println!("  vertices: {}, {}, {}", v0, v1, v2);
        println!("  indices:  {}, {}, {}", i0 + 1, i1 + 1, i2 + 1);
        panic!("go clean your geometry");
    }
}

impl Mesh {
    pub fn load<P: AsRef<Path>>(path: P) -> Mesh {
        let fbuffer = FileBuffer::open(path).expect("failed to open file");
        let input = from_utf8(&fbuffer[..]).expect("obj must be valid utf-8");

        let mut vertices = Vec::new();
        let mut tex_coords = Vec::new();
        let mut triangles = Vec::new();

        for (line, line_nr) in input.lines().zip(1u32..) {
            if line.is_empty() { continue }

            let mut pieces = line.split_whitespace();
            match pieces.next() {
                Some("v") => {
                    let mut coords = pieces.map(|v| f32::from_str(v).unwrap());
                    let vertex = SVector3 {
                        x: coords.next().expect("missing x coordinate"),
                        y: coords.next().expect("missing y coordinate"),
                        z: coords.next().expect("missing z coordinate"),
                    };
                    vertices.push(vertex);
                }
                Some("vt") => {
                    let mut coords = pieces.map(|v| f32::from_str(v).unwrap());
                    let u = coords.next().expect("missing u coordinate");
                    let v = coords.next().expect("missing v coordinate");
                    tex_coords.push((u, v));
                }
                Some("f") => {
                    // Indices stored are 1-based, convert to 0-based.
                    let mut indices = pieces.map(|i| u32::from_str(i).unwrap() - 1);
                    let i0 = indices.next().expect("missing triangle index");
                    let i1 = indices.next().expect("missing triangle index");
                    let mut i2 = indices.next().expect("missing triangle index");

                    assert_nondegenerate(&vertices, line_nr, i0, i1, i2);
                    triangles.push((i0, i1, i2));

                    // There might be a quad or n-gon. Assuming it is convex, we
                    // can triangulate it at import time.
                    while let Some(i3) = indices.next() {
                        assert_nondegenerate(&vertices, line_nr, i0, i2, i3);
                        triangles.push((i0, i2, i3));
                        i2 = i3;
                    }
                },
                _ => { /* Anything else is not supported. */ }
            }
        }

        Mesh {
            vertices: vertices,
            triangles: triangles,
            tex_coords: tex_coords,
            material: SMaterial::white(), // TODO: Allow picking the material.
        }
    }
}
