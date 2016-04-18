#version 140

in vec2 v_tex_coords;
out vec4 color;

uniform sampler2D frame0;
uniform sampler2D frame1;
uniform sampler2D frame2;
uniform sampler2D frame3;
uniform sampler2D frame4;
uniform sampler2D frame5;
uniform sampler2D frame6;
uniform sampler2D frame7;

void main() {
    vec4 c0 = texture(frame0, v_tex_coords);
    vec4 c1 = texture(frame1, v_tex_coords);
    vec4 c2 = texture(frame2, v_tex_coords);
    vec4 c3 = texture(frame3, v_tex_coords);
    vec4 c4 = texture(frame4, v_tex_coords);
    vec4 c5 = texture(frame5, v_tex_coords);
    vec4 c6 = texture(frame6, v_tex_coords);
    vec4 c7 = texture(frame7, v_tex_coords);

    // Take the mean of the eight frames.
    color = (c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7) * 0.125f;
}
