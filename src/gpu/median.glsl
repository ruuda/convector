// Convector -- An interactive CPU path tracer
// Copyright 2016 Ruud van Asseldonk

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License version 3. A copy
// of the License is available in the root of the repository.

#version 140

in vec2 v_tex_coords;
out vec4 color;

uniform sampler2D frame;
uniform vec2 pixel_size;

void sort2(inout vec4 a0, inout vec4 a1) {
    vec4 b0 = min(a0, a1);
    vec4 b1 = max(a0, a1);
    a0 = b0;
    a1 = b1;
}

void sort(inout vec4 a0, inout vec4 a1, inout vec4 a2, inout vec4 a3, inout vec4 a4) {
    sort2(a0, a1);
    sort2(a3, a4);
    sort2(a0, a2);
    sort2(a1, a2);
    sort2(a0, a3);
    sort2(a2, a3);
    sort2(a1, a4);
    sort2(a1, a2);
    sort2(a3, a4);
}

vec4 rgb_to_xyz(vec4 c) {
    mat3 conv = mat3(0.49f, 0.17697f, 0.0f,
                     0.31f, 0.81240f, 0.01f,
                     0.20f, 0.01063f, 0.99f) * (1.0f / 0.17697f);
    c.xyz = conv * c.rgb;
    return c;
}

vec4 xyz_to_rgb(vec4 c) {
    mat3 conv = mat3(0.41847f, -0.091169f, 0.00092090f,
                     -0.15866f, 0.25243f, -0.0025498f,
                     -0.082835, 0.015708, 0.17860);
    c.rgb = conv * c.xyz;
    return c;
}

void main() {
    // Sample 5 pixels in a "+" shape.
    vec4 c0 = texture(frame, v_tex_coords);
    vec4 c1 = texture(frame, v_tex_coords + vec2(pixel_size.x, 0.0f));
    vec4 c2 = texture(frame, v_tex_coords + vec2(0.0f, pixel_size.y));
    vec4 c3 = texture(frame, v_tex_coords - vec2(pixel_size.x, 0.0f));
    vec4 c4 = texture(frame, v_tex_coords - vec2(0.0f, pixel_size.y));

    // Convert all the pixels from CIE 1931 to the CIE XYZ color space before
    // taking the median. This ensures that lightness is better preserved.
    c0 = rgb_to_xyz(c0);
    c1 = rgb_to_xyz(c1);
    c2 = rgb_to_xyz(c2);
    c3 = rgb_to_xyz(c3);
    c4 = rgb_to_xyz(c4);

    // Take the sort-of-median of those pixels. The true median is c2, but do
    // weigh in a bit of the other pixels as well for a more balanced result.
    sort(c0, c1, c2, c3, c4);
    vec4 median = c2 * 0.667f + c1 * 0.1666f + c3 * 0.1666f;

    // Convert back from CIE XYZ to CIE 1931 (which is a linear RGB color
    // space).
    color = xyz_to_rgb(median);
}
