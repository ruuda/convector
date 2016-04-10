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

void sort2(inout vec4 a0, inout vec4 a1) {
    vec4 b0 = min(a0, a1);
    vec4 b1 = max(a0, a1);
    a0 = b0;
    a1 = b1;
}

void sort(inout vec4 a0, inout vec4 a1,
          inout vec4 a2, inout vec4 a3,
          inout vec4 a4, inout vec4 a5,
          inout vec4 a6, inout vec4 a7) {
    sort2(a0, a7);
    sort2(a1, a6);
    sort2(a2, a5);
    sort2(a3, a4);

    sort2(a0, a3);
    sort2(a4, a7);
    sort2(a1, a2);
    sort2(a5, a6);

    sort2(a0, a1);
    sort2(a2, a3);
    sort2(a4, a5);
    sort2(a6, a7);

    sort2(a2, a4);
    sort2(a3, a5);

    sort2(a1, a2);
    sort2(a3, a4);
    sort2(a5, a6);

    sort2(a2, a3);
    sort2(a3, a5);
}

void main() {
    vec4 c0 = texture(frame0, v_tex_coords);
    vec4 c1 = texture(frame1, v_tex_coords);
    vec4 c2 = texture(frame2, v_tex_coords);
    vec4 c3 = texture(frame3, v_tex_coords);
    vec4 c4 = texture(frame4, v_tex_coords);
    vec4 c5 = texture(frame5, v_tex_coords);
    vec4 c6 = texture(frame6, v_tex_coords);
    vec4 c7 = texture(frame7, v_tex_coords);
    sort(c0, c1, c2, c3, c4, c5, c6, c7);
    // color = (c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7) * 0.125f;
    if (v_tex_coords.x > 0.5) {
        color = (c3 + c4) * 0.5;
    } else {
        color = (c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7) * 0.125f;
    }
}
