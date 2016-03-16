//! Implements a bounding volume hierarchy.

use aabb::Aabb;
use ray::{MIntersection, MRay};
use simd::{Mask, Mf32};
use std::cmp::PartialOrd;
use triangle::Triangle;
use vector3::{Axis, SVector3};
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

/// Reference to a triangle used during BVH construction.
struct TriangleRef {
    aabb: Aabb,
    barycenter: SVector3,
    index: usize,
}

/// A node used during BVH construction.
struct InterimNode {
    /// Bounding box of the triangles in the node.
    outer_aabb: Aabb,

    /// Bounding box of the barycenters of the triangles in the node.
    inner_aabb: Aabb,

    children: Vec<InterimNode>,
    triangles: Vec<TriangleRef>,
}

struct Bin<'a> {
    triangles: Vec<&'a TriangleRef>,
    aabb: Option<Aabb>,
}

trait Heuristic {
    /// Given that a ray has intersected the parent bounding box, estimates the
    /// cost of intersecting the child bounding box and the triangles in it.
    fn aabb_cost(&self, parent_aabb: &Aabb, aabb: &Aabb, num_tris: usize) -> f32;

    /// Estimates the cost of intersecting the given number of triangles.
    fn tris_cost(&self, num_tris: usize) -> f32;
}

impl<'a> Bin<'a> {
    fn new() -> Bin<'a> {
        Bin {
            triangles: Vec::new(),
            aabb: None,
        }
    }

    pub fn push(&mut self, tri: &'a TriangleRef) {
        self.triangles.push(tri);
        self.aabb = match self.aabb {
            Some(ref aabb) => Some(Aabb::enclose_aabbs(&[aabb.clone(), tri.aabb.clone()])),
            None => Some(tri.aabb.clone()),
        };
    }
}

impl InterimNode {
    fn from_triangle_refs(trirefs: Vec<TriangleRef>) -> InterimNode {
        InterimNode {
            outer_aabb: Aabb::enclose_aabbs(trirefs.iter().map(|tr| &tr.aabb)),
            inner_aabb: Aabb::enclose_points(trirefs.iter().map(|tr| &tr.barycenter)),
            children: Vec::new(),
            triangles: trirefs,
        }
    }

    fn inner_aabb_origin_and_size(&self, axis: Axis) -> (f32, f32) {
        let min = self.inner_aabb.origin.get_coord(axis);
        let max = self.inner_aabb.far.get_coord(axis);
        let size = max - min;
        (min, size)
    }

    /// Puts triangles into bins along the specified axis.
    fn bin_triangles<'a>(&'a self, bins: &mut [Bin<'a>], axis: Axis) {
        // Compute the bounds of the bins.
        let (min, size) = self.inner_aabb_origin_and_size(axis);

        // Put the triangles in bins.
        for tri in &self.triangles {
            let coord = tri.barycenter.get_coord(axis);
            let index = ((bins.len() as f32) * (coord - min) / size).floor() as usize;
            bins[index].push(tri);

            // If a lot of geometry ends up in one bin, binning is
            // apparently not effective.
            if bins[index].triangles.len() > self.triangles.len() / 8 {
                println!("warning: triangle distribution is very non-uniform");
                println!("         binning will not be effective");
            }
        }
    }

    /// Returs the bounding box enclosing the bin bounding boxes.
    fn enclose_bins(bins: &[Bin]) -> Aabb {
        let aabbs = bins.iter()
                        .filter(|bin| bin.triangles.len() > 0)
                        .map(|bin| bin.aabb.as_ref().unwrap());

        Aabb::enclose_aabbs(aabbs)
    }

    /// Returns the bin index such that for the cheapest split, all bins with a
    /// lower index should go into one node. Also returns the cost of the split.
    fn find_cheapest_split<H>(&self, heuristic: &H, bins: &[Bin]) -> (usize, f32) where H: Heuristic {
        let mut best_split_at = 0;
        let mut best_split_cost = 0.0;
        let mut is_first = false;

        for i in 1..bins.len() - 1 {
            let left_bins = &bins[..i];
            let left_aabb = InterimNode::enclose_bins(left_bins);
            let left_count = left_bins.iter().map(|b| b.triangles.len()).sum();

            let right_bins = &bins[i..];
            let right_aabb = InterimNode::enclose_bins(right_bins);
            let right_count = left_bins.iter().map(|b| b.triangles.len()).sum();

            let left_cost = heuristic.aabb_cost(&self.outer_aabb, &left_aabb, left_count);
            let right_cost = heuristic.aabb_cost(&self.outer_aabb, &right_aabb, right_count);
            let cost = left_cost + right_cost;

            if cost < best_split_cost || is_first {
                best_split_cost = cost;
                best_split_at = i;
                is_first = false;
            }
        }

        (best_split_at, best_split_cost)
    }

    /// Splits the node if that is would be beneficial according to the
    /// heuristic.
    fn split<H>(&mut self, heuristic: &H) where H: Heuristic {
        let mut best_split_axis = Axis::X;
        let mut best_split_at = 0.0;
        let mut best_split_cost = 0.0;
        let mut is_first = false;

        // Find the cheapest split.
        for &axis in &[Axis::X, Axis::Y, Axis::Z] {
            let mut bins: Vec<Bin> = (0..64).map(|_| Bin::new()).collect();

            self.bin_triangles(&mut bins, axis);
            let (index, cost) = self.find_cheapest_split(heuristic, &bins);

            if cost < best_split_cost || is_first {
                let (min, size) = self.inner_aabb_origin_and_size(axis);
                best_split_axis = axis;
                best_split_at = min + size / (bins.len() as f32) * (index as f32);
                best_split_cost = cost;
                is_first = false;
            }
        }

        // Do not split if the split node is more expensive than the unsplit
        // one.
        let no_split_cost = heuristic.tris_cost(self.triangles.len());
        if no_split_cost < best_split_cost {
            return
        }

        // Partition the triangles into two child nodes.
        let pred = |tri: &TriangleRef| tri.barycenter.get_coord(best_split_axis) < best_split_at;
        let (left_tris, right_tris) = self.triangles.drain(..).partition(pred);

        let left = InterimNode::from_triangle_refs(left_tris);
        let right = InterimNode::from_triangle_refs(right_tris);

        // TODO: Perhaps make child with biggest surface area go first.
        self.children.push(left);
        self.children.push(right);
    }
}

fn build_bvh_node(triangles: &mut [Triangle]) -> BvhNode {
    // Compute the bounding box that encloses all triangles.
    let aabb = Aabb::enclose_aabbs(triangles.iter().map(|tri| &tri.aabb));

    let centroids: Vec<SVector3> = triangles.iter().map(|tri| tri.aabb.center()).collect();
    let centroid_aabb = Aabb::enclose_points(&centroids[..]);

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
    let mut size = centroid_aabb.size().x;
    let mut axis = Axis::X;

    if centroid_aabb.size().y > size {
        size = centroid_aabb.size().y;
        axis = Axis::Y;
    }

    if centroid_aabb.size().z > size {
        size = centroid_aabb.size().z;
        axis = Axis::Z;
    }

    // Sort the  triangles along that axis (panic on NaN).
    triangles.sort_by(|a, b| PartialOrd::partial_cmp(
        &a.barycenter().get_coord(axis),
        &b.barycenter().get_coord(axis)).unwrap());

    let half_way = centroid_aabb.origin.get_coord(axis) + size * 0.5;

    // Find the index to split at so that everything before the split has
    // coordinate less than `half_way` and everything after has larger or equal
    // coordinates.
    let mut split_point = triangles.binary_search_by(|tri| {
        PartialOrd::partial_cmp(&tri.aabb.center().get_coord(axis), &half_way).unwrap()
    }).unwrap_or_else(|idx| idx);

    // Ensure a balanced tree at the leaves.
    // (This also ensures that the recursion terminates.)
    if split_point > triangles.len() - 2 {
        split_point = triangles.len() - 2;
    }
    if split_point < 2 {
        split_point = 2;
    }

    let (left_triangles, right_triangles) = triangles.split_at_mut(split_point);
    let left_node = build_bvh_node(left_triangles);
    let right_node = build_bvh_node(right_triangles);
    BvhNode {
        aabb: aabb,
        children: vec![left_node, right_node],
        geometry: Vec::new(),
    }
}

impl TriangleRef {
    fn from_triangle(index: usize, tri: &Triangle) -> TriangleRef {
        TriangleRef {
            aabb: Aabb::enclose_points(&[tri.v0, tri.v1, tri.v2]),
            barycenter: tri.barycenter(),
            index: index,
        }
    }
}

impl Bvh {
    pub fn build(mut triangles: Vec<Triangle>) -> Bvh {
        // Actual triangles are not important to the BVH, convert them to AABBs.
        // let trirefs = (0..).zip(triangles.iter())
        //                    .map(|(i, tri)| TriangleRef::from_triangle(i, tri))
        //                    .collect();

        // TODO: Use rayon for data parallelism here.
        let root = build_bvh_node(&mut triangles);
        Bvh {
            root: root,
        }
    }

    pub fn from_meshes(meshes: &[Mesh]) -> Bvh {
        let mut triangles = Vec::new();

        for mesh in meshes {
            let mesh_triangles = mesh.triangles.iter().map(
                |&(i1, i2, i3)| {
                    let v1 = mesh.vertices[i1 as usize];
                    let v2 = mesh.vertices[i2 as usize];
                    let v3 = mesh.vertices[i3 as usize];
                    Triangle::new(v1, v2, v3)
                });
            triangles.extend(mesh_triangles);
        }

        Bvh::build(triangles)
    }

    pub fn intersect_nearest(&self, ray: &MRay, mut isect: MIntersection) -> MIntersection {
        // Keep a stack of nodes that still need to be intersected. This does
        // involve a heap allocation, but that is not so bad. Using a small
        // on-stack vector from the smallvec crate (which falls back to heap
        // allocation if it grows) actually reduced performance by about 5 fps.
        // If there is an upper bound on the BVH depth, then perhaps manually
        // rolling an on-stack (memory) stack (data structure) could squeeze out
        // a few more fps.
        let mut nodes = Vec::with_capacity(10);

        let root_isect = self.root.aabb.intersect(ray);
        if root_isect.any() {
            nodes.push((root_isect, &self.root));
        }

        while let Some((aabb_isect, node)) = nodes.pop() {
            // If the AABB is further away than the current nearest
            // intersection, then nothing inside the node can yield
            // a closer intersection, so we can skip the node.
            if aabb_isect.is_further_away_than(isect.distance) {
                continue
            }

            if node.geometry.is_empty() {
                for child in &node.children {
                    let child_isect = child.aabb.intersect(ray);
                    if child_isect.any() {
                        nodes.push((child_isect, child));
                    }
                }
            } else {
                for triangle in &node.geometry {
                    isect = triangle.intersect(ray, isect);
                }
            }
        }

        isect
    }

    pub fn intersect_any(&self, ray: &MRay, max_dist: Mf32) -> Mask {
        let isect = MIntersection {
            position: ray.direction.mul_add(max_dist, ray.origin),
            normal: ray.direction,
            distance: max_dist,
        };
        let isect = self.intersect_nearest(ray, isect);
        isect.distance.geq(max_dist - Mf32::epsilon())
    }
}
