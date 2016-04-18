#version 140

in vec2 v_tex_coords;
out vec4 color;

uniform sampler2D frame;

void main() {
    color = texture(frame, v_tex_coords);
}
