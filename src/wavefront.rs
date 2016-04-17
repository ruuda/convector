//! This module reads Wavefront OBJ files. There are crates for that, but
//! reinventing the wheel is much more fun.

use filebuffer::FileBuffer;
use material::SMaterial;
use std::collections::HashMap;
use std::path::Path;
use std::str::{FromStr, from_utf8};
use vector3::SVector3;

pub struct Triangle {
    pub vertices: (u32, u32, u32),
    pub tex_coords: Option<(u32, u32, u32)>,
    pub material: SMaterial,
}

pub struct Mesh {
    pub vertices: Vec<SVector3>,
    pub tex_coords: Vec<(f32, f32)>,
    pub triangles: Vec<Triangle>,
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

/// Returns the vertex index, and the texture coordinate index if there is one.
fn parse_vertex_index(index: &str) -> (u32, Option<u32>) {
    let mut parts = index.split('/').map(|i| u32::from_str(i).unwrap());
    let vidx = parts.next().expect("missing vertex index");
    let tidx = parts.next();
    // Indices in the obj file are 1-based, but Rust is 0-based.
    (vidx - 1, tidx.map(|i| i - 1))
}

pub fn push_triangle(vertices: &[SVector3],
                     triangles: &mut Vec<Triangle>,
                     i0: (u32, Option<u32>),
                     i1: (u32, Option<u32>),
                     i2: (u32, Option<u32>),
                     material: SMaterial,
                     line_nr: u32) {
    assert_nondegenerate(&vertices, line_nr, i0.0, i1.0, i2.0);
    let vidxs = (i0.0, i1.0, i2.0);
    let tidxs = match (i0.1, i1.1, i2.1) {
        (Some(t0), Some(t1), Some(t2)) => Some((t0, t1, t2)),
        _ => None,
    };
    let triangle = Triangle {
        vertices: vidxs,
        tex_coords: tidxs,
        material: material,
    };
    triangles.push(triangle);
}

impl Mesh {
    pub fn load<P: AsRef<Path>>(path: P) -> Mesh {
        Mesh::load_with_materials(path, &HashMap::new())
    }

    pub fn load_with_materials<P: AsRef<Path>>(path: P,
                                               materials: &HashMap<&str, SMaterial>)
                                               -> Mesh {
        let fbuffer = FileBuffer::open(path).expect("failed to open file");
        let input = from_utf8(&fbuffer[..]).expect("obj must be valid utf-8");

        let mut vertices = Vec::new();
        let mut tex_coords = Vec::new();
        let mut triangles = Vec::new();
        let mut material = SMaterial::white(); // The default material.

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
                Some("usemtl") => {
                    let material_name = pieces.next().expect("missing material name");
                    if let Some(&new_mat) = materials.get(material_name) {
                        material = new_mat;
                    } else {
                        panic!("material '{}' not present in material dictionary",
                                material_name);
                    }
                }
                Some("f") => {
                    // Indices stored are 1-based, convert to 0-based.
                    let mut indices = pieces.map(parse_vertex_index);
                    let i0 = indices.next().expect("missing triangle index");
                    let i1 = indices.next().expect("missing triangle index");
                    let mut i2 = indices.next().expect("missing triangle index");

                    push_triangle(&vertices, &mut triangles, i0, i1, i2, material, line_nr);

                    // There might be a quad or n-gon. Assuming it is convex, we
                    // can triangulate it at import time.
                    while let Some(i3) = indices.next() {
                        push_triangle(&vertices, &mut triangles, i0, i2, i3, material, line_nr);
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
        }
    }
}

// The loader should be able to load all of these files without crashing. The
// files are known to be well-formed and without degenerate faces.

#[test]
fn read_indoor() {
    let mut materials = HashMap::new();
    materials.insert("wall", SMaterial::white());
    materials.insert("glass", SMaterial::sky());
    Mesh::load_with_materials("models/box_walls.obj", &materials);
}

#[test]
fn read_stanford_bunny() {
    Mesh::load("models/stanford_bunny.obj");
}

#[test]
fn read_stanford_dragon() {
    Mesh::load("models/stanford_dragon.obj");
}

#[test]
fn read_suzanne() {
    Mesh::load("models/suzanne.obj");
}
