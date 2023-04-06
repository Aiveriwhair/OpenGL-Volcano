#version 330 core

in vec3 position;
in vec2 tex_coord;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform float n_repeat_texture;

out vec4 clip_space;
out vec3 w_position;
out vec2 frag_tex_coords;


void main() {
    vec4 w_position4 = (model * vec4(position, 1));
    w_position = w_position4.xyz / w_position4.w;
    frag_tex_coords = tex_coord * n_repeat_texture;

    clip_space = projection * view * model * vec4(position, 1);
    gl_Position = clip_space;
}
