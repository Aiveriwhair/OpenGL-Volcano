#version 330 core

uniform sampler2D dudv_map;
uniform sampler2D normal_map;

in vec4 clip_space;
in vec3 w_position;
in vec2 frag_tex_coords;

uniform vec3 light_dir;

// material properties
uniform vec3 k_ambient;
uniform vec3 k_shadow;
uniform vec3 k_a;
uniform vec3 k_d;
uniform vec3 k_s;
uniform float s;
uniform float time;

uniform vec3 w_camera_position;

out vec4 out_color;

uniform float displacement_speed;

vec3 calculate_lighting(vec3 normal, vec3 view_direction, vec3 shadow_color) {
    vec3 n = normalize(normal);
    vec3 l = normalize(-light_dir);
    vec3 r = reflect(-l, n);
    vec3 v = normalize(view_direction);
    float diffuse = max(dot(n, l), 0.0);
    float specular = pow(max(dot(r, v), 0.0), s);

    float tint_factor = 1.0 - max(dot(n, l), 0.0);
    vec3 color = mix(k_a + k_d * diffuse + k_s * specular, shadow_color, tint_factor);
    return color;
}

void main()
{
    vec2 screen_space_coords = (clip_space.xy / clip_space.w) / 2.0 + 0.5;

    vec2 dudv_distortion_coords = texture(dudv_map, vec2(frag_tex_coords.x, frag_tex_coords.y - 0.5)).rg * 0.12;
    dudv_distortion_coords = frag_tex_coords + vec2(dudv_distortion_coords.x + time, dudv_distortion_coords.y + time);
    vec2 dudv_distortion = (texture(dudv_map, dudv_distortion_coords).rg * 2.0 - 1.0);

    vec4 normal_map_color = texture(normal_map, dudv_distortion_coords);
    vec3 normal = vec3(normal_map_color.r * 2.0 - 1.0 , normal_map_color.b * 2, normal_map_color.g * 2.0 - 1.0);
    normal = normalize(normal);

    vec3 view_vector = normalize(w_camera_position - w_position);
    float dist = distance(w_camera_position,w_position);

    float fog_density = 0.0000003;
    float fog_factor = exp(-fog_density * dist);
    vec3 fog_color = vec3(190, 190, 190);

    //Phong illumination
    vec3 light = calculate_lighting(normal, view_vector, k_shadow);
    vec3 I = k_ambient + mix(fog_color,light, fog_factor);

    out_color = vec4(I,1.0) * vec4(1.0, 1.0, 1.0, 1.0); 
}