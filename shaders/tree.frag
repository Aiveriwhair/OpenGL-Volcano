#version 330 core

in vec2 fragment_texcoord;
in vec3 w_position;
in vec3 w_normal;

uniform sampler2D texture_sampler;
uniform sampler2D texture_sampler1;
uniform sampler2D texture_sampler2;
uniform float height_threshold1;
uniform float height_threshold2;
uniform int use_texture2;
uniform int use_texture3;
uniform float mix_range;

uniform vec3 light_dir;

uniform vec3 k_d, k_a, k_s;
uniform float s;

uniform vec3 w_camera_position;

uniform vec3 fog_color;
uniform float fog_density;
uniform float red_tint_factor;

out vec4 out_color;

vec3 calculate_lighting(vec3 normal, vec3 view_direction) {
    vec3 n = normalize(normal);
    vec3 l = normalize(-light_dir);
    vec3 r = reflect(-l, n);
    vec3 v = normalize(view_direction);
    float diffuse = max(dot(n, l), 0.0);
    float specular = pow(max(dot(r, v), 0.0), s);

    vec3 red_tint = vec3(1.0, 0.0, 0.0);
    vec3 color = mix(k_a + k_d * diffuse + k_s * specular, red_tint, red_tint_factor);
    return color;
}

void main() {
    vec4 tex_color1 = texture(texture_sampler, fragment_texcoord);
    vec4 tex_color2 = texture(texture_sampler1, fragment_texcoord);
    vec4 tex_color3 = texture(texture_sampler2, fragment_texcoord);

    float height = w_position.y;
    float mix_grass_rock = smoothstep(height_threshold1-mix_range, height_threshold1 + mix_range, height);
    float mix_rock_snow = smoothstep(height_threshold2-mix_range, height_threshold2 + mix_range, height);

    vec4 mixed_tex_color = tex_color1;
    if (use_texture2 == 1) {
        mixed_tex_color = mix(mixed_tex_color, tex_color2, mix_grass_rock);
    }
    if (use_texture3 == 1) {
        mixed_tex_color = mix(mixed_tex_color, tex_color3, mix_rock_snow);
    }
    mixed_tex_color = mix(mixed_tex_color, tex_color3, mix_rock_snow);

    vec3 view_direction = w_camera_position - w_position;
    vec3 lighting = calculate_lighting(w_normal, view_direction);
    float dist = distance(w_camera_position,w_position);


    float fog_density = 0.000003;
    float fog_factor = exp(-fog_density * dist);
    vec3 fog_color = vec3(128, 128, 128);

    vec3 blended_color = mix(fog_color, mixed_tex_color.rgb * lighting, fog_factor);

    out_color = vec4(blended_color, mixed_tex_color.a);
    out_color.rgb=blended_color;
}
