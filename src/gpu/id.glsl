// Convector -- An interactive CPU path tracer
// Copyright 2016 Ruud van Asseldonk

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License version 3. A copy
// of the License is available in the root of the repository.

#version 140

in vec2 v_tex_coords;
out vec4 color;

uniform sampler2D frame;

void main() {
    color = texture(frame, v_tex_coords);
}
