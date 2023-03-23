#version 330 core

// receiving interpolated color and texture coordinate for fragment shader
in vec3 fragment_color;
in vec2 fragment_texcoord;

// texture uniform variable
uniform sampler2D texture_sampler;

// output fragment color for OpenGL
out vec4 out_color;

void main() {
    // sample texture using texture coordinates
    vec4 tex_color = texture(texture_sampler, fragment_texcoord);

    // output the final color as the texture color multiplied by the interpolated color
    out_color = vec4(tex_color.rgb * fragment_color, 1.0);
}
