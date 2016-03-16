//! Implements a bounding volume hierarchy.

use aabb::Aabb;
use ray::{MIntersection, MRay};
use simd::{Mask, Mf32};
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

struct SurfaceAreaHeuristic {
    aabb_intersection_cost: f32,
    triangle_intersection_cost: f32,
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
    /// Create a single node containing all of the triangles.
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
            let index = if index < bins.len() { index } else { bins.len() - 1 };
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

    /// Returns whether there is more than one non-empty bin.
    fn are_bins_valid(bins: &[Bin]) -> bool {
        1 < bins.iter().filter(|bin| !bin.triangles.is_empty()).count()
    }

    /// Returns the bin index such that for the cheapest split, all bins with a
    /// lower index should go into one node. Also returns the cost of the split.
    fn find_cheapest_split<H>(&self, heuristic: &H, bins: &[Bin]) -> (usize, f32) where H: Heuristic {
        let mut best_split_at = 0;
        let mut best_split_cost = 0.0;
        let mut is_first = true;

        // Consiter every split position after the first non-empty bin, until
        // right before the last non-empty bin.
        let first = bins.iter().position(|bin| !bin.triangles.is_empty()).unwrap() + 1;
        let last = bins.iter().rposition(|bin| !bin.triangles.is_empty()).unwrap();

        for i in first..last {
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
        // If there is only one triangle, splitting does not make sense.
        if self.triangles.len() <= 1 {
            println!("warning: attempted to split single-triangle node");
            return
        }

        let mut best_split_axis = Axis::X;
        let mut best_split_at = 0.0;
        let mut best_split_cost = 0.0;
        let mut is_first = true;

        // Find the cheapest split.
        for &axis in &[Axis::X, Axis::Y, Axis::Z] {
            let mut bins: Vec<Bin> = (0..64).map(|_| Bin::new()).collect();

            self.bin_triangles(&mut bins, axis);

            if InterimNode::are_bins_valid(&bins) {
                let (index, cost) = self.find_cheapest_split(heuristic, &bins);

                if cost < best_split_cost || is_first {
                    println!("replaced previous split cost {} with {}", best_split_cost, cost);
                    let (min, size) = self.inner_aabb_origin_and_size(axis);
                    best_split_axis = axis;
                    best_split_at = min + size / (bins.len() as f32) * (index as f32);
                    best_split_cost = cost;
                    is_first = false;
                }
            } else {
                // Consider a different splitting strategy?
            }
        }

        // Something must have set the cost.
        assert!(!is_first);

        // Do not split if the split node is more expensive than the unsplit
        // one.
        let no_split_cost = heuristic.tris_cost(self.triangles.len());
        if no_split_cost < best_split_cost {
            return
        }

        // Partition the triangles into two child nodes.
        let pred = |tri: &TriangleRef| tri.barycenter.get_coord(best_split_axis) < best_split_at;
        // TODO: remove type annotation.
        let (left_tris, right_tris): (Vec<_>, Vec<_>) = self.triangles.drain(..).partition(pred);

        // It can happen that the best split is not to split at all ... BUT in
        // that case the no split cost should be lower than the all-in-one-side
        // cost ... so this should not occur.
        if left_tris.is_empty() || right_tris.is_empty() {
            println!("one of the sides was empty!");
            println!("no split cost: {}, best split cost: {}, left tris: {}, right tris: {}",
                     no_split_cost, best_split_cost, left_tris.len(), right_tris.len());
        }

        let left = InterimNode::from_triangle_refs(left_tris);
        let right = InterimNode::from_triangle_refs(right_tris);

        // TODO: Perhaps make child with biggest surface area go first.
        self.children.push(left);
        self.children.push(right);
    }

    /// Recursively splits the node, constructing the BVH.
    fn split_recursive<H>(&mut self, heuristic: &H) where H: Heuristic {
        // TODO: This would be an excellent candidate for Rayon I think.
        self.split(heuristic);
        for child_node in &mut self.children {
            child_node.split_recursive(heuristic);
        }
    }

    /// Converts the interim representation that was useful for building the BVH
    /// into a representation that is optimized for traversing the BVH.
    fn crystallize(mut self, triangles: &[Triangle]) -> BvhNode {
        BvhNode {
            aabb: self.outer_aabb,
            children: self.children.drain(..).map(|x| x.crystallize(triangles)).collect(),
            geometry: self.triangles.drain(..).map(|t| triangles[t.index].clone()).collect(),
        }
    }
}

impl Heuristic for SurfaceAreaHeuristic {
    fn aabb_cost(&self, parent_aabb: &Aabb, aabb: &Aabb, num_tris: usize) -> f32 {
        // We are certainly going to intersect the child AABB, so pay the full
        // price for that.
        let fixed_cost = self.aabb_intersection_cost;

        // Without further information, the best guess for the probability
        // that the bounding box was hit, given that the parent was already
        // intersected, is the ratio of their areas.
        let p = aabb.area() / parent_aabb.area();

        // We have to test all of the triangles, but only if the bounding box
        // was intersected, so weigh with the probability.
        fixed_cost + p * self.tris_cost(num_tris)
    }

    fn tris_cost(&self, num_tris: usize) -> f32 {
        num_tris as f32 * self.triangle_intersection_cost
    }
}

impl Bvh {
    pub fn build(triangles: Vec<Triangle>) -> Bvh {
        // Actual triangles are not important to the BVH, convert them to AABBs.
        let trirefs = (0..).zip(triangles.iter())
                           .map(|(i, tri)| TriangleRef::from_triangle(i, tri))
                           .collect();

        let mut root = InterimNode::from_triangle_refs(trirefs);

        // TODO: Get the values from benchmarks.
        let heuristic = SurfaceAreaHeuristic {
            aabb_intersection_cost: 1.0,
            triangle_intersection_cost: 1.0,
        };

        // Build the BVH of interim nodes.
        root.split_recursive(&heuristic);

        Bvh {
            root: root.crystallize(&triangles)
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
