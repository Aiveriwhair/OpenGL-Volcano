#version 330 core

// receiving interpolated color and texture coordinate for fragment shader
in vec2 fragment_texcoord;
in vec3 w_position;
in vec3 w_normal;

// texture uniform variable
uniform sampler2D texture_sampler;
uniform vec3 light_dir;

// material properties
uniform vec3 k_d, k_a, k_s;
uniform float s;

// world camera position
uniform vec3 w_camera_position;

// fog parameters
uniform vec3 fog_color;
uniform float fog_density;
uniform float red_tint_factor;

out vec4 out_color;

vec3 calculate_lighting(vec3 normal, vec3 view_direction) {
    // Compute all vectors, oriented outwards from the fragment
    vec3 n = normalize(normal);
    vec3 l = normalize(-light_dir);
    vec3 r = reflect(-l, n);
    vec3 v = normalize(view_direction);
    // Compute diffuse lighting
    float diffuse = max(dot(n, l), 0.0);

    // Compute specular lighting
    float specular = pow(max(dot(r, v), 0.0), s);


    // Combine all lighting components and return final color
    vec3 red_tint = vec3(1.0, 0.0, 0.0);
    vec3 color = mix(k_a + k_d * diffuse + k_s * specular, red_tint, red_tint_factor);
    return color;
}

void main() {
    // Sample texture using texture coordinates
    vec4 tex_color = texture(texture_sampler, fragment_texcoord);
    // Compute lighting
    vec3 view_direction = w_camera_position - w_position;
    vec3 lighting = calculate_lighting(w_normal, view_direction);

    // Compute fog factor
    float dist = length(view_direction);
    float fog_factor = exp(-fog_density * dist);

    // Blend fog color with fragment color
    vec3 blended_color = mix(fog_color, tex_color.rgb * lighting, fog_factor);

    // Output the final color
    out_color = vec4(blended_color, tex_color.a);
}