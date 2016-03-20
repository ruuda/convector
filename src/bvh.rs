//! Implements a bounding volume hierarchy.

use aabb::Aabb;
use ray::{MIntersection, MRay};
use simd::{Mask, Mf32};
use triangle::Triangle;
use util;
use vector3::{Axis, SVector3};
use wavefront::Mesh;

#[cfg(test)]
use {bench, test};

/// One node in a bounding volume hierarchy.
struct BvhNode {
    aabb: Aabb,

    /// For leaf nodes, the index of the first triangle, for internal nodes, the
    /// index of the first child. The second child is at `index + 1`.
    index: u32,

    /// For leaf nodes, the number of triangles, zero for internal nodes.
    len: u32,
}

/// A bounding volume hierarchy.
pub struct Bvh {
    nodes: Vec<BvhNode>,
    triangles: Vec<Triangle>,

    /// Average ratio of bounding box surface area to parent surface area.
    avg_area_ratio: f32,

    /// Average number of triangles per leaf.
    avg_tris_per_leaf: f32,
}

/// Reference to a triangle used during BVH construction.
#[derive(Clone, Debug)]
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

trait Heuristic: Sync {
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

/// My own improvement over the classic surface area heuristic. See `aabb_cost`
/// implementation for more details.
struct TreeSurfaceAreaHeuristic {
    aabb_intersection_cost: f32,
    triangle_intersection_cost: f32,
    intersection_probability: f32,
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

    pub fn clear(&mut self) {
        self.triangles.clear();
        self.aabb = None;
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
        // If there are not so many triangles, we might as well try all of the
        // split positions.
        if self.triangles.len() <= bins.len() / 2 {
            return self.bin_triangles_uniform(bins, axis);
        }

        // Compute the bounds of the bins.
        let (min, size) = self.inner_aabb_origin_and_size(axis);

        // Put the triangles in bins.
        for tri in &self.triangles {
            let coord = tri.barycenter.get_coord(axis);
            let index = ((bins.len() as f32) * (coord - min) / size).floor() as usize;
            let index = if index < bins.len() { index } else { bins.len() - 1 };
            bins[index].push(tri);

            // If a lot of geometry ends up in one bin, binning is apparently
            // not effective. In that case, fall back to sorting and try splits
            // based on percentiles.
            let num_tris = self.triangles.len();
            if bins[index].triangles.len() > num_tris / 8 && num_tris > bins.len() {
                // Clear the bins before trying again.
                for bin in &mut bins[..] { bin.clear(); }
                return self.bin_triangles_uniform(bins, axis);
            }
        }
    }

    /// Puts roughly the same number of triangles in every bin, sorted by the
    /// coordinate along the specified axis. This is O(n log n) instead of O(n)
    /// due to sorting.
    fn bin_triangles_uniform<'a>(&'a self, bins: &mut [Bin<'a>], axis: Axis) {
        // Create a vector of pointers to the triangle refs and sort them on
        // coordinate.
        let mut triptrs: Vec<&TriangleRef> = self.triangles.iter().map(|tri| tri).collect();
        triptrs.sort_by(|t1, t2| {
            let a = t1.barycenter.get_coord(axis);
            let b = t2.barycenter.get_coord(axis);
            // Rust is very pedantic about the fact that there is no total order
            // on f32, but NaNs in this data would be a bug so jump through the
            // hoop.
            a.partial_cmp(&b).unwrap()
        });

        let tris_per_bin = (triptrs.len() + bins.len() - 1) / bins.len();
        let tris_per_bin = if tris_per_bin == 0 { 1 } else { tris_per_bin };

        // Be sure not to lose any triangles.
        assert!(tris_per_bin * bins.len() >= triptrs.len());

        for (bin, tris) in bins.iter_mut().zip(triptrs.chunks(tris_per_bin)) {
            assert!(bin.triangles.is_empty());
            for tri in tris {
                bin.push(tri)
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

    /// Returns the cheapest split and its cost.
    fn find_cheapest_split<'a, H>(&self, heuristic: &H, bins: &[Bin<'a>])
                              -> (f32, Vec<&'a TriangleRef>, Vec<&'a TriangleRef>)
                              where H: Heuristic {
        let mut best_split_at = 0;
        let mut best_split_cost = 0.0;
        let mut is_first = true;

        // Consider every split position after the first non-empty bin, until
        // right before the last non-empty bin.
        let first = bins.iter().position(|bin| !bin.triangles.is_empty()).unwrap() + 1;
        let last = bins.iter().rposition(|bin| !bin.triangles.is_empty()).unwrap() + 1;

        assert!(first != last);

        for i in first..last {
            let left_bins = &bins[..i];
            let left_aabb = InterimNode::enclose_bins(left_bins);
            let left_count = left_bins.iter().map(|b| b.triangles.len()).sum();

            let right_bins = &bins[i..];
            let right_aabb = InterimNode::enclose_bins(right_bins);
            let right_count = right_bins.iter().map(|b| b.triangles.len()).sum();

            let left_cost = heuristic.aabb_cost(&self.outer_aabb, &left_aabb, left_count);
            let right_cost = heuristic.aabb_cost(&self.outer_aabb, &right_aabb, right_count);
            let cost = left_cost + right_cost;

            if cost < best_split_cost || is_first {
                best_split_cost = cost;
                best_split_at = i;
                is_first = false;
            }
        }

        assert!(!is_first);

        let left = bins[..best_split_at].iter().flat_map(|b| b.triangles.iter().cloned()).collect();
        let right = bins[best_split_at..].iter().flat_map(|b| b.triangles.iter().cloned()).collect();

        (best_split_cost, left, right)
    }

    /// Splits the node if that is would be beneficial according to the
    /// heuristic.
    fn split<H>(&mut self, heuristic: &H) where H: Heuristic {
        // If there is only one triangle, splitting does not make sense.
        if self.triangles.len() <= 1 {
            return
        }

        let (best_split_cost, left_tris, right_tris) = {
            let mut bins: Vec<Bin> = (0..64).map(|_| Bin::new()).collect();
            let mut best_split = (Vec::new(), Vec::new());
            let mut best_split_cost = 0.0;
            let mut is_first = true;

            // Find the cheapest split.
            for &axis in &[Axis::X, Axis::Y, Axis::Z] {
                self.bin_triangles(&mut bins, axis);

                if InterimNode::are_bins_valid(&bins) {
                    let (cost, left, right) = self.find_cheapest_split(heuristic, &bins);

                    assert!(!left.is_empty());
                    assert!(!right.is_empty());

                    if cost < best_split_cost || is_first {
                        best_split = (left, right);
                        best_split_cost = cost;
                        is_first = false;
                    }
                }

                for bin in &mut bins[..] { bin.clear(); }
            }

            // Something must have set the cost.
            assert!(!is_first);

            let left_tris = best_split.0.drain(..).cloned().collect();
            let right_tris = best_split.1.drain(..).cloned().collect();
            (best_split_cost, left_tris, right_tris)
        };

        // Do not split if the split node is more expensive than the unsplit
        // one.
        let no_split_cost = heuristic.tris_cost(self.triangles.len());
        if no_split_cost < best_split_cost {
            return
        }

        let left_node = InterimNode::from_triangle_refs(left_tris);
        let right_node = InterimNode::from_triangle_refs(right_tris);

        self.triangles.clear();
        self.children.push(left_node);
        self.children.push(right_node);
    }

    /// Recursively splits the node, constructing the BVH.
    fn split_recursive<H>(&mut self, heuristic: &H) where H: Heuristic {
        use rayon;
        self.split(heuristic);

        if !self.children.is_empty() {
            assert_eq!(2, self.children.len());

            // Make sure that the node with the smallest surface area is the
            // first child. The first child will be tested first, and if an
            // intersection is found, the second child might not be intersected
            // at all. This also ensures that the least-probable path is
            // consecutive in memory. It is counter-intuitive: I would expect
            // performance to be better if the most probably child was tested
            // first. Nevertheless, the benchmarks don't lie. Flip the
            // comparison and observe how intersection times increase by 200 ns
            // (10-3% depending on the scene size).
            if self.children[0].outer_aabb.area() > self.children[1].outer_aabb.area() {
                self.children.swap(0, 1);
            }

            let (left, right) = self.children.split_at_mut(1);

            // Recursively split the children. Use Rayon to put the work up
            // for grabs with work stealing, for parallel BVH construction.
            rayon::join(
                || left[0].split_recursive(heuristic),
                || right[0].split_recursive(heuristic)
            );
        }
    }

    /// Returns the number of triangle refs in the leaves.
    fn count_triangles(&self) -> usize {
        let child_tris: usize = self.children.iter().map(|ch| ch.count_triangles()).sum();
        let self_tris = self.triangles.len();
        child_tris + self_tris
    }

    /// Returns the number of nodes in the BVH, including self.
    fn count_nodes(&self) -> usize {
        let child_count: usize = self.children.iter().map(|ch| ch.count_nodes()).sum();
        1 + child_count
    }

    /// Returns the number of leaf nodes in the BVH.
    fn count_leaves(&self) -> usize {
        let leaf_count: usize = self.children.iter().map(|ch| ch.count_leaves()).sum();
        let self_leaf = if self.children.is_empty() { 1 } else { 0 };
        self_leaf + leaf_count
    }

    /// Returns the ratio of the parent area to the child node area,
    /// summed over all the nodes.
    fn summed_area_ratio(&self) -> f32 {
        let child_contribution: f32 = self.children.iter().map(|ch| ch.summed_area_ratio()).sum();
        let self_area = self.outer_aabb.area();
        let child_sum: f32 = self.children.iter().map(|ch| ch.outer_aabb.area() / self_area).sum();
        child_sum + child_contribution
    }

    /// Converts the interim representation that was useful for building the BVH
    /// into a representation that is optimized for traversing the BVH.
    fn crystallize(&self,
                   source_triangles: &[Triangle],
                   nodes: &mut Vec<BvhNode>,
                   sorted_triangles: &mut Vec<Triangle>,
                   into_index: usize) {
        // Nodes must always be pushed in pairs to keep siblings on the same
        // cache line.
        assert_eq!(0, nodes.len() % 2);

        nodes[into_index].aabb = self.outer_aabb.clone();

        if self.triangles.is_empty() {
            // This is an internal node.
            assert_eq!(2, self.children.len());

            // Allocate two new nodes for the children.
            let child_index = nodes.len();
            nodes.push(BvhNode::new());
            nodes.push(BvhNode::new());

            // Recursively crystallize the child nodes.
            self.children[0].crystallize(source_triangles, nodes, sorted_triangles, child_index + 0);
            self.children[1].crystallize(source_triangles, nodes, sorted_triangles, child_index + 1);

            nodes[into_index].index = child_index as u32;
            nodes[into_index].len = 0;
        } else {
            // This is a leaf node.
            assert_eq!(0, self.children.len());

            nodes[into_index].index = sorted_triangles.len() as u32;
            nodes[into_index].len = self.triangles.len() as u32;

            // Copy the triangles into the triangle buffer.
            let tris = self.triangles.iter().map(|triref| source_triangles[triref.index].clone());
            sorted_triangles.extend(tris);
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
        let ac_ap = aabb.area() / parent_aabb.area();

        // We have to test all of the triangles, but only if the bounding box
        // was intersected, so weigh with the probability.
        fixed_cost + ac_ap * self.tris_cost(num_tris)
    }

    fn tris_cost(&self, num_tris: usize) -> f32 {
        (num_tris as f32) * self.triangle_intersection_cost
    }
}

impl Heuristic for TreeSurfaceAreaHeuristic {
    fn aabb_cost(&self, parent_aabb: &Aabb, aabb: &Aabb, num_tris: usize) -> f32 {
        // The SAH adds the cost of intersecting all the triangles, but for a
        // non-leaf node, it is rarely the case that they all will be
        // intersected. Instead, assume that the triangles are organized into a
        // balanced BVH with two triangles per leaf. If you work out the math
        // (see pdf), the following expression is what comes out:

        let ac_ap = aabb.area() / parent_aabb.area();
        let p = self.intersection_probability;
        let n = num_tris as f32;
        let m = n.log2();

        let aabb_term = 1.0 + ac_ap * (2.0 * p - n * p.powf(m)) / (p - 2.0 * p * p);
        let tri_term = n * p.powf(m - 1.0) * ac_ap;

        aabb_term * self.aabb_intersection_cost + tri_term * self.triangle_intersection_cost
    }

    fn tris_cost(&self, num_tris: usize) -> f32 {
        (num_tris as f32) * self.triangle_intersection_cost
    }
}

impl BvhNode {
    /// Returns a zeroed node, to be filled later.
    fn new() -> BvhNode {
        BvhNode {
            aabb: Aabb::zero(),
            index: 0,
            len: 0,
        }
    }
}

impl Bvh {
    pub fn build(source_triangles: &[Triangle]) -> Bvh {
        // Actual triangles are not important to the BVH, convert them to AABBs.
        let trirefs = (0..).zip(source_triangles.iter())
                           .map(|(i, tri)| TriangleRef::from_triangle(i, tri))
                           .collect();

        let mut root = InterimNode::from_triangle_refs(trirefs);

        // The values here are based on benchmarks. You can run `make bench` to
        // run these benchmarks. By plugging in the results for your rig you
        // might be able to achieve slightly better performance.
        let heuristic = TreeSurfaceAreaHeuristic {
            aabb_intersection_cost: 40.0,
            triangle_intersection_cost: 120.0,
            intersection_probability: 0.8,
        };

        // Build the BVH of interim nodes.
        root.split_recursive(&heuristic);

        // There should be at least one split, because crystallized nodes are
        // stored in pairs. There is no single root, there are two roots. (Or,
        // the root is implicit and its bounding box is infinite, if you like.)
        assert_eq!(2, root.children.len());

        // Allocate one buffer for the BVH nodes and one for the triangles. For
        // better data locality, the source triangles are reordered. Also, a
        // triangle might be included in multiple nodes. In that case it is
        // simply duplicated in the new buffer. The node buffer is aligned to a
        // cache line: nodes are always accessed in pairs, and one pair fits
        // exactly in one cache line.
        let num_tris = root.count_triangles();
        let num_nodes = root.count_nodes();
        let mut nodes = util::cache_line_aligned_vec(num_nodes);
        let mut sorted_triangles = Vec::with_capacity(num_tris);

        // Write the tree of interim nodes that is all over the heap currently,
        // neatly packed into the buffers that we just allocated.
        let left = &root.children[0];
        let right = &root.children[1];
        nodes.push(BvhNode::new());
        nodes.push(BvhNode::new());

        left.crystallize(&source_triangles, &mut nodes, &mut sorted_triangles, 0);
        right.crystallize(&source_triangles, &mut nodes, &mut sorted_triangles, 1);

        // Gather some statistics.
        let num_leaves = root.count_leaves();
        let area_ratio_sum = root.summed_area_ratio();

        Bvh {
            nodes: nodes,
            triangles: sorted_triangles,
            avg_area_ratio: area_ratio_sum / (num_nodes as f32),
            avg_tris_per_leaf: (num_tris as f32) / (num_leaves as f32),
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

        Bvh::build(&triangles)
    }

    pub fn print_stats(&self) {
        use std::mem;
        println!("bvh statistics:");
        println!("  average triangles per leaf: {:0.2}", self.avg_tris_per_leaf);
        println!("  average child area / parent area: {:0.2}", self.avg_area_ratio);

        let tris_size = self.triangles.len() * mem::size_of::<Triangle>();
        let nodes_size = self.nodes.len() * mem::size_of::<BvhNode>();
        let tris_kib = (tris_size as f32) / 1024.0;
        let nodes_kib = (nodes_size as f32) / 1024.0;

        println!("  triangle data size: {:0.1} KiB", tris_kib);
        println!("  node data size: {:0.1} KiB", nodes_kib);
    }

    /// Returns the nearest intersection closer than the provided intersection.
    /// Also returns the number of AABBs intersected and the number of triangles
    /// intersected.
    #[inline(always)]
    pub fn intersect_nearest_impl(&self, ray: &MRay, mut isect: MIntersection) -> (MIntersection, u32, u32) {
        // Keep a stack of nodes that still need to be intersected. This does
        // involve a heap allocation, but that is not so bad. Using a small
        // on-stack vector from the smallvec crate (which falls back to heap
        // allocation if it grows) actually reduced performance by about 5 fps.
        // If there is an upper bound on the BVH depth, then perhaps manually
        // rolling an on-stack (memory) stack (data structure) could squeeze out
        // a few more fps.
        let mut stack = Vec::with_capacity(10);

        // Counters for debug view. In normal code these are not used, so LLVM
        // will eliminate them I hope.
        let mut numi_aabb = 2;
        let mut numi_tri = 0;

        // A note about `get_unchecked`: array indexing in Rust is checked by
        // default, but by construction all indices in the BVH are valid, so
        // let's not waste instructions on those bounds checks.

        let root_0 = unsafe { self.nodes.get_unchecked(0) };
        let root_1 = unsafe { self.nodes.get_unchecked(1) };
        let root_isect_0 = root_0.aabb.intersect(ray);
        let root_isect_1 = root_1.aabb.intersect(ray);

        if root_isect_0.any() {
            stack.push((root_isect_0, root_0));
        }
        if root_isect_1.any() {
            stack.push((root_isect_1, root_1));
        }

        while let Some((aabb_isect, node)) = stack.pop() {
            // If the AABB is further away than the current nearest
            // intersection, then nothing inside the node can yield
            // a closer intersection, so we can skip the node.
            if aabb_isect.is_further_away_than(isect.distance) {
                continue
            }

            if node.len == 0 {
                // This is an internal node.
                numi_aabb += 2;
                let child_0 = unsafe { self.nodes.get_unchecked(node.index as usize + 0) };
                let child_1 = unsafe { self.nodes.get_unchecked(node.index as usize + 1) };
                let child_isect_0 = child_0.aabb.intersect(ray);
                let child_isect_1 = child_1.aabb.intersect(ray);

                if child_isect_0.any() {
                    stack.push((child_isect_0, child_0));
                }
                if child_isect_1.any() {
                    stack.push((child_isect_1, child_1));
                }
            } else {
                for i in node.index..node.index + node.len {
                    let triangle = unsafe { self.triangles.get_unchecked(i as usize) };
                    isect = triangle.intersect(ray, isect);
                    numi_tri += 1;
                }
            }
        }

        (isect, numi_aabb, numi_tri)
    }

    pub fn intersect_nearest(&self, ray: &MRay, isect: MIntersection) -> MIntersection {
        let (isect, _, _) = self.intersect_nearest_impl(ray, isect);
        isect
    }

    /// Returns the number of AABBs and the number of triangles intersected to
    /// find the closest intersection.
    pub fn intersect_debug(&self, ray: &MRay, isect: MIntersection) -> (u32, u32) {
        let (_, numi_aabb, numi_tri) = self.intersect_nearest_impl(ray, isect);
        (numi_aabb, numi_tri)
    }

    pub fn intersect_any(&self, ray: &MRay, max_dist: Mf32) -> Mask {
        // This is actually just doing a full BVH intersection. I tried to do an
        // early out here; stop when all rays intersect at least something,
        // instead of finding the nearest intersection, but I could not measure
        // a performance improvement. `intersect_nearest` does try very hard not
        // to intersect more than necessary, and apparently that is good enough
        // already.
        let isect = MIntersection {
            position: ray.direction.mul_add(max_dist, ray.origin),
            normal: ray.direction,
            distance: max_dist,
        };
        let isect = self.intersect_nearest(ray, isect);
        isect.distance.geq(max_dist - Mf32::epsilon())
    }
}

#[bench]
fn bench_intersect_decoherent_mray_suzanne(b: &mut test::Bencher) {
    use wavefront::Mesh;
    let suzanne = Mesh::load("models/suzanne.obj");
    let bvh = Bvh::from_meshes(&[suzanne]);
    let rays = bench::mrays_inward(4096 / 8);
    let mut rays_it = rays.iter().cycle();
    b.iter(|| {
        let ray = rays_it.next().unwrap();
        let far = Mf32::broadcast(1e5);
        let isect = bvh.intersect_any(ray, far);
        test::black_box(isect);
    });
}

#[bench]
fn bench_intersect_coherent_mray_suzanne(b: &mut test::Bencher) {
    use wavefront::Mesh;
    let suzanne = Mesh::load("models/suzanne.obj");
    let bvh = Bvh::from_meshes(&[suzanne]);
    let rays = bench::mrays_inward_coherent(4096 / 8);
    let mut rays_it = rays.iter().cycle();
    b.iter(|| {
        let ray = rays_it.next().unwrap();
        let far = Mf32::broadcast(1e5);
        let isect = bvh.intersect_any(ray, far);
        test::black_box(isect);
    });
}

#[bench]
fn bench_intersect_decoherent_mray_bunny(b: &mut test::Bencher) {
    use wavefront::Mesh;
    let bunny = Mesh::load("models/stanford_bunny.obj");
    let bvh = Bvh::from_meshes(&[bunny]);
    let rays = bench::mrays_inward(4096 / 8);
    let mut rays_it = rays.iter().cycle();
    b.iter(|| {
        let ray = rays_it.next().unwrap();
        let far = Mf32::broadcast(1e5);
        let isect = bvh.intersect_any(ray, far);
        test::black_box(isect);
    });
}

#[bench]
fn bench_intersect_coherent_mray_bunny(b: &mut test::Bencher) {
    use wavefront::Mesh;
    let bunny = Mesh::load("models/stanford_bunny.obj");
    let bvh = Bvh::from_meshes(&[bunny]);
    let rays = bench::mrays_inward_coherent(4096 / 8);
    let mut rays_it = rays.iter().cycle();
    b.iter(|| {
        let ray = rays_it.next().unwrap();
        let far = Mf32::broadcast(1e5);
        let isect = bvh.intersect_any(ray, far);
        test::black_box(isect);
    });
}
