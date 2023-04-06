import numpy as np
import OpenGL.GL as GL
from texture import Textured, Texture
from core import Mesh
import glfw

class FluidTerrain(Textured):
    def __init__(self, shader, dudv_path, normal_path, size, world_height=0, **params):
        self.shader = shader
        self.height = world_height
        width, height = size

        plane_positions = np.array([[-1, 0, -1], [1, 0, -1], [1, 0, 1], [-1, 0, 1]])
        plane_positions[:,0] = plane_positions[:,0] * (width/2)
        plane_positions[:,1] = plane_positions[:,1] + world_height
        plane_positions[:,2] = plane_positions[:,2] * (height/2)

        indices = np.array((1, 0, 3, 1 , 3 , 2), np.uint32)
        texcoords = ([0,0], [1, 0], [1, 1], [0, 1])
        
        mesh = Mesh(shader, attributes=dict(position=plane_positions, tex_coord=texcoords), index=indices, **params)

        dudv_tex = Texture(dudv_path, GL.GL_MIRRORED_REPEAT, *(GL.GL_LINEAR, GL.GL_LINEAR_MIPMAP_LINEAR))
        normal_tex = Texture(normal_path, GL.GL_MIRRORED_REPEAT, *(GL.GL_LINEAR, GL.GL_LINEAR_MIPMAP_LINEAR))
        super().__init__(mesh, dudv_map=dudv_tex, normal_map=normal_tex)

    def draw(self, **params):
        uniforms = dict(
            time = glfw.get_time() / 50
        )
        super().draw(**{**uniforms, **params})




#    USAGE :
#    waterShader = Shader("shaders/fluid.vert", "shaders/fluid.frag")

#     RECOMMENDED VALUES FOR LAVA
#     lava_uniforms = dict(
#            k_ambient=(0.2, 0.2, 0.2),
#             k_shadow=(1,0,0),
#             k_a=(1.0, 0.2, 0.0),
#             k_d=(1.0, 0.6, 0.0),
#             k_s=(1.0, 0.8, 0.0),
#             s=200,
#             light_dir=(0, -3, -6),
#             n_repeat_texture=1,
#         )
#     viewer.add(FluidTerrain(waterShader, dudv_path="ress/watermaps/dudv.png", normal_path="ress/lava/lavanormal.jpg", size=(8, 8),**lava_uniforms))
#     
#     RECOMMENDED VALUES FOR WATER
#     water_uniforms = dict(
#         k_ambient=(0., 0.1, 0.2),
#         k_shadow=(1,1,1),
#         k_a=(0., 0.3, 0.6),
#         k_d=(0., 0.3, 0.7),
#         k_s=(1.0, 1.0, 1.0),
#         s=30,
#         light_dir=(0, -1, -1),
#         n_repeat_texture=2,
#     )
#     viewer.add(FluidTerrain(waterShader, dudv_path="ress/watermaps/dudv.png", normal_path="ress/watermaps/normalmap.png", world_height=-2, size=(8, 8),**water_uniforms))
