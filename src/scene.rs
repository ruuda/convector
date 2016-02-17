use vector3::Vector3;

pub struct Triangle {
    pub v1: Vector3,
    pub v2: Vector3,
    pub v3: Vector3,
}

pub struct Camera {
    pub origin: Vector3,
    // TODO: Add quaternion orientation.
}

pub struct Light {
    pub position: Vector3,
}

pub struct Scene {
    // The only primitive is the triangle, there are no spheres or other shapes.
    // This avoids having to dispatch on the primitive type to intersect an
    // object. It avoids a virtual method call. This in turn enables the
    // triangle intersection code to be inlined.
    pub geometry: Vec<Triangle>,
    pub lights: Vec<Light>,
    pub camera: Camera,
}

impl Scene {
    pub fn new() -> Scene {
        Scene {
            geometry: Vec::new(),
            lights: Vec::new(),
            camera: Camera {
                origin: Vector3::zero(),
            }
        }
    }
}
