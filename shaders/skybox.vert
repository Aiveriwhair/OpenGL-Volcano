#version 330 core

in vec3 position;

uniform mat4 projection;
uniform mat4 view;

out vec3 fragPos;

void main(){
    fragPos = position;

    vec4 pos = projection * mat4(mat3(view))* vec4(position, 1.0);
    gl_Position = pos.xyww;
}