#version 330 core
out vec4 FragColor;

in vec3 fragPos;

uniform samplerCube skybox;

void main(){
    vec4 texture_skybox = texture(skybox, fragPos);
    FragColor = texture_skybox;
}