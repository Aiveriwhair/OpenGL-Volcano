#version 330 core

in vec2 TexCoord;
in vec3 WorldPos;

out vec4 FragColor;

uniform sampler2D reflectionMap;
uniform sampler2D refractionMap;
uniform sampler2D normalMap;

void main()
{
    // Calculate the reflection and refraction coordinates
    vec3 I = normalize(WorldPos - cameraPos);
    vec3 R = reflect(I, normalize(normal));
    vec3 refractDir = refract(I, normalize(normal), refractionIndex);
    vec3 refractCoords = WorldPos + refractDir * depth;
    
    // Sample the reflection, refraction, and normal maps
    vec4 reflectionColor = texture(reflectionMap, R.xy);
    vec4 refractionColor = texture(refractionMap, refractCoords.xy);
    vec3 normalColor = texture(normalMap, TexCoord).xyz;
    
    // Calculate the water color using the reflection and refraction colors
    vec4 waterColor = mix(reflectionColor, refractionColor, waterFactor);
    
    // Calculate the final color using the water color and the normal map
    vec3 reflected = reflect(I, normalColor);
    float spec = pow(max(dot(reflected, normalize(cameraPos - WorldPos)), 0.0), shininess);
    FragColor = vec4(waterColor.rgb + spec * specularColor, waterColor.a);
}
