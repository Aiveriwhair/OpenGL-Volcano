#version 330 core

// global color
uniform vec3 global_color;

// input attribute variable, given per vertex
in vec3 position;
in vec3 color;
in vec3 normal;
in vec2 texcoord;

// global matrix variables
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

// interpolated variables for fragment shader
out vec3 fragment_color;
out vec2 fragment_texcoord;

void main() {
    // initialize interpolated colors and texture coordinates at vertices
    fragment_color = color + normal + global_color;
    fragment_texcoord = texcoord;

    // tell OpenGL how to transform the vertex to clip coordinates
    gl_Position = projection * view * model * vec4(position, 1);
}
