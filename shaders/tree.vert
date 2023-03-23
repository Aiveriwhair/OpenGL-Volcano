#version 330 core

// input attribute variable, given per vertex
in vec3 position;
in vec3 normal;
in vec2 texcoord;

// global matrix variables
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform float red_tint_factor;

// interpolated variables for fragment shader
out vec2 fragment_texcoord;
out vec3 w_position;
out vec3 w_normal;

void main() {
// initialize interpolated colors and texture coordinates at vertices
fragment_texcoord = texcoord;

// transform vertex to clip coordinates
gl_Position = projection * view * model * vec4(position, 1.0);

// transform normal to world coordinates
w_normal = mat3(transpose(inverse(model))) * normal;

// transform vertex position to world coordinates
w_position = vec3(model * vec4(position, 1.0));
}