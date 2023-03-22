#version 330 core

// fragment position and normal of the fragment, in WORLD coordinates
in vec3 w_position, w_normal;

// light dir, in world coordinates
uniform vec3 light_dir;

// material properties
uniform vec3 k_d, k_a, k_s;
uniform float s;

// world camera position
uniform vec3 w_camera_position;

// fog parameters
uniform vec3 fog_color;
uniform float fog_density;

out vec4 out_color;

void main() {
    // Compute all vectors, oriented outwards from the fragment
    vec3 n = normalize(w_normal);
    vec3 l = normalize(-light_dir);
    vec3 r = reflect(-l, n);
    vec3 v = normalize(w_camera_position - w_position);
    float dist = distance(w_camera_position,w_position);

    vec3 diffuse_color = k_d * max(dot(n, l), 0);
    vec3 specular_color = k_s * pow(max(dot(r, v), 0), s);
    float fog_density = 0.0001;
    // Compute fog factor
    float fog_factor = exp(-fog_density * dist );
    vec3 fog_color = vec3(0, 0, 0);
    // Blend fog color with fragment color

    out_color = vec4(k_a, 1) + vec4(diffuse_color, 1) + vec4(specular_color, 1);
    vec3 blended_color = mix(fog_color,vec3(out_color), fog_factor);
    out_color.rgb = blended_color ;
}