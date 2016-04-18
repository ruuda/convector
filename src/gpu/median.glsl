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

void main() {
    // Sample 5 pixels in a "+" shape.
    vec4 c0 = texture(frame, v_tex_coords);
    vec4 c1 = texture(frame, v_tex_coords + vec2(pixel_size.x, 0.0f));
    vec4 c2 = texture(frame, v_tex_coords + vec2(0.0f, pixel_size.y));
    vec4 c3 = texture(frame, v_tex_coords - vec2(pixel_size.x, 0.0f));
    vec4 c4 = texture(frame, v_tex_coords - vec2(0.0f, pixel_size.y));

    // Take the median of those pixels.
    sort(c0, c1, c2, c3, c4);
    vec4 median = c2;

    color = median;
}
