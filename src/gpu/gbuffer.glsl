// Convector -- An interactive CPU path tracer
// Copyright 2016 Ruud van Asseldonk

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License version 3. A copy
// of the License is available in the root of the repository.

#version 140

in vec2 v_tex_coords;
out vec4 color;

uniform sampler2D frame;
uniform sampler2D gbuffer;
uniform sampler2D texture1;
uniform sampler2D texture2;

void main() {
    color = texture(frame, v_tex_coords);
    vec4 data = texture(gbuffer, v_tex_coords);

    float fresnel = data.b;
    vec4 white = vec4(1.0f, 1.0f, 1.0f, 1.0f);

    // The alpha channel contains the texture index. Texture index 0 indicates
    // that the texture is not used, so the pixel is already correct. For the
    // other textures, sample them and blend according to the Fresnel factor.

    if (data.a == 1.0f / 255.0f) {
        vec4 tex_color = texture(texture1, data.xy);
        vec4 surface_color = white * fresnel + tex_color * (1.0f - fresnel);
        color = color * surface_color;
    }

    if (data.a == 2.0f / 255.0f) {
        vec4 tex_color = texture(texture2, data.xy);
        vec4 surface_color = white * fresnel + tex_color * (1.0f - fresnel);
        color = color * surface_color;
    }

    // Texture index 3 is currently not used.
}
