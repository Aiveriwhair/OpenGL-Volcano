#version 330 core
out vec4 out_color;

uniform sampler2D dudv_map;
uniform sampler2D normal_map;

in vec4 clip_space;
in vec3 w_position;   // in world coodinates
in vec2 frag_tex_coords;

// light dir, in world coordinates
uniform vec3 light_dir;

// material properties
uniform vec3 k_ambient;
uniform vec3 k_a;
uniform vec3 k_d;
uniform vec3 k_s;
uniform float s;
uniform float time;

uniform float displacement_speed;

// world camera position
uniform vec3 w_camera_position;

const float DISTORSION_STRENGTH = 0.02;

vec3 calculate_lighting(vec3 normal, vec3 view_direction) {
    vec3 n = normalize(normal);
    vec3 l = normalize(-light_dir);
    vec3 r = reflect(-l, n);
    vec3 v = normalize(view_direction);
    float diffuse = max(dot(n, l), 0.0);
    float specular = pow(max(dot(r, v), 0.0), s);

    // we tint the object red when it is in the shadow

    vec3 red_tint = vec3(1.0, 0.0, 0.0);
    float red_tint_factor = 1.0 - max(dot(n, l), 0.0);
    vec3 color = mix(k_a + k_d * diffuse + k_s * specular, red_tint, red_tint_factor);
    return color;
}

void main()
{
    //change coordinate system (projection mapping of the refle/refra texture)
    vec2 screen_space_coords = (clip_space.xy / clip_space.w) / 2.0 + 0.5;

    //compute the dudv map distorsion value for the rippling effect
    vec2 dudv_distortion_coords = texture(dudv_map, vec2(frag_tex_coords.x, frag_tex_coords.y - 0.5* displacement_speed)).rg * 0.2 ;
    dudv_distortion_coords = frag_tex_coords + vec2(dudv_distortion_coords.x + time, dudv_distortion_coords.y + time);
    vec2 dudv_distortion = (texture(dudv_map, dudv_distortion_coords).rg * 2.0 - 1.0) * DISTORSION_STRENGTH ;

    // compute the final coords of the fragment after the displacement (+ we flip the reflection texture)
    vec2 reflection_coords = vec2(screen_space_coords.x, -screen_space_coords.y) + dudv_distortion;
    reflection_coords = vec2(clamp(reflection_coords.x, 0.001, 0.999), clamp(reflection_coords.y, -0.999, -0.001));
    vec2 refraction_coords = vec2(screen_space_coords.x, screen_space_coords.y) + dudv_distortion;
    refraction_coords = clamp(refraction_coords, 0.001, 0.999);

    //normal calculation from normal map
    vec4 normal_map_color = texture(normal_map, dudv_distortion_coords);
    // we extract the normal of the distorted fragment by taking the value pointing up and clamping the others to (-1;1)
    vec3 normal = vec3(normal_map_color.r *2.0 - 1.0 , normal_map_color.b * 2, normal_map_color.g * 2.0 - 1.0);
    normal = normalize(normal);

    vec3 view_vector = normalize(w_camera_position - w_position);

    //Phong illumination
    vec3 I = k_ambient + calculate_lighting(normal, view_vector);

    //finalize the lighting, reflection/refraction and compute the final color of the frag 
    float refractivity_factor = dot(view_vector, normal); //fresnel effect

    vec4 final_reflection = mix(vec4(1.0,1.0,1.0,1.0),vec4(1.0,1.0,1.0,1.0), refractivity_factor);
    vec4 final_color = mix(final_reflection, vec4(0.0,0.4,0.7,1.0), 0.1); //we add a bit of blue to the final texture
    out_color = vec4(I,1.0) * final_color; 


}