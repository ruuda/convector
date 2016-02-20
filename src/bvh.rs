//! Implements a bounding volume hierarchy.

use aabb::Aabb;
use geometry::Triangle;
use ray::Ray;
use std::cmp::PartialOrd;
use vector3::{Axis, Vector3};
use wavefront::Mesh;

/// One node in a bounding volume hierarchy.
struct BvhNode {
    aabb: Aabb,
    children: Vec<BvhNode>,
    geometry: Vec<Triangle>,
}

/// A bounding volume hierarchy.
pub struct Bvh {
    root: BvhNode,
}

fn build_bvh_node(triangles: &mut [Triangle]) -> BvhNode {
    let mut aabb = Aabb::new(Vector3::zero(), Vector3::zero());

    // Compute the bounding box that encloses all triangles.
    for triangle in triangles.iter() {
        aabb = Aabb::enclose_aabbs(&aabb, &triangle.aabb);
    }

    // TODO: My root AABB for Suzanne has infinities for the z coordinate. How
    // is that possible?

    // Ideally every node would contain two triangles, so splitting less than
    // four triangles does not make sense; make a leaf node in that case.
    if triangles.len() < 4 {
        return BvhNode {
            aabb: aabb,
            children: Vec::new(),
            geometry: triangles.iter().cloned().collect(),
        }
    }

    // Split along the axis in which the box is largest.
    let mut size = aabb.size.x;
    let mut axis = Axis::X;

    if aabb.size.y > size {
        size = aabb.size.y;
        axis = Axis::Y;
    }

    if aabb.size.z > size {
        size = aabb.size.z;
        axis = Axis::Z;
    }

    // Sort the  triangles along that axis (panic on NaN).
    triangles.sort_by(|a, b| PartialOrd::partial_cmp(
        &a.barycenter().get_coord(axis),
        &b.barycenter().get_coord(axis)).unwrap());

    // TODO: Split half-way geometrically, not by index.
    let split_point = triangles.len() / 2;
    let (left_triangles, right_triangles) = triangles.split_at_mut(split_point);
    let left_node = build_bvh_node(left_triangles);
    let right_node = build_bvh_node(right_triangles);
    BvhNode {
        aabb: aabb,
        children: vec![left_node, right_node],
        geometry: Vec::new(),
    }
}

impl Bvh {
    pub fn build(mut triangles: Vec<Triangle>) -> Bvh {
        // TODO: Use rayon for data parallelism here.
        let root = build_bvh_node(&mut triangles);
        Bvh {
            root: root,
        }
    }

    pub fn from_mesh(mesh: &Mesh) -> Bvh {
        let triangles: Vec<Triangle> = mesh.triangles.iter().map(
            |&(i1, i2, i3)| {
                let v1 = mesh.vertices[i1 as usize];
                let v2 = mesh.vertices[i2 as usize];
                let v3 = mesh.vertices[i3 as usize];
                Triangle::new(v1, v2, v3)
            }).collect();
        Bvh::build(triangles)
    }

    /// Traverses the BVH, calls a callback when a node is intersected and for
    /// every leaf triangle that might insersect the ray.
    pub fn traverse_with_nodes<OnNode, OnTriangle>(&self,
                                                   ray: &Ray,
                                                   mut on_node: OnNode,
                                                   mut on_triangle: OnTriangle)
                                             where OnNode: FnMut(),
                                                   OnTriangle: FnMut(&Triangle) {
        let mut nodes = Vec::new();

        if self.root.aabb.intersect(ray) {
            // TODO: on_node_tested and on_node_hit.
            on_node();
            nodes.push(&self.root);
        }

        while let Some(node) = nodes.pop() {
            if node.geometry.is_empty() {
                for child in &node.children {
                    if child.aabb.intersect(ray) {
                        on_node();
                        nodes.push(child);
                    }
                }
            } else {
                for triangle in &node.geometry {
                    on_triangle(triangle);
                }
            }
        }
    }

    /// Traverses the BVH, calls a callback for every leaf triangle that might
    /// intersect the ray.
    pub fn traverse<OnTriangle>(&self, ray: &Ray, on_triangle: OnTriangle)
        where OnTriangle: FnMut(&Triangle) {
        self.traverse_with_nodes(ray, || {}, on_triangle);
    }
}
