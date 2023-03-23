#version 330 core

layout (location = 0) in vec3 position;
layout (location = 1) in vec2 texCoord;

out vec2 TexCoord;
out vec3 WorldPos;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform float time;

void main()
{
    gl_Position = projection * view * model * vec4(position, 1.0);
    TexCoord = texCoord;
    
    // Calculate the displacement based on time and texture coordinates
    float displacement = texture(displacementMap, vec2(texCoord.x + time, texCoord.y)).r;
    WorldPos = position + vec3(0.0, displacement, 0.0);
}
