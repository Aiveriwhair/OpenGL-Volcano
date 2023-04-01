import numpy as np
import OpenGL.GL as GL
from texture import Textured, Texture
from core import Mesh
import glfw

class WaterTerrain(Textured):
    def __init__(self, shader, w_height=0, size=(512, 512), **params):
        self.shader = shader
        self.height = w_height
        width, height = size

        uniforms = dict(
            k_ambient=(.3,.4,.35), k_a=(0.,0.,0.), k_d=(0.2,0.2,0.2), k_s=(0.9,0.9,1.0), s=20, light_dir=(0., -3, 0.),
            displacement_speed=1
        )
        
        base_coords = np.array([[-1, 0, -1], [1, 0, -1], [1, 0, 1], [-1, 0, 1]])
        scaled_coords = base_coords
        scaled_coords[:,0] = base_coords[:,0] * (width/2)
        scaled_coords[:,1] = base_coords[:,1] + w_height
        scaled_coords[:,2] = base_coords[:,2] * (height/2)
        indices = np.array((1, 0, 3, 1 , 3 , 2), np.uint32)
        texcoords = ([0,0], [1, 0], [1, 1], [0, 1])
        
        mesh = Mesh(shader, attributes=dict(position=scaled_coords, tex_coord=texcoords), index=indices, **{**uniforms, **params})

        dudv_tex = Texture("ress/watermaps/dudv.png", GL.GL_MIRRORED_REPEAT, *(GL.GL_LINEAR, GL.GL_LINEAR_MIPMAP_LINEAR))
        normal_tex = Texture("ress/watermaps/normalmap.png", GL.GL_MIRRORED_REPEAT, *(GL.GL_LINEAR, GL.GL_LINEAR_MIPMAP_LINEAR))
        super().__init__(mesh, dudv_map=dudv_tex, normal_map=normal_tex)

    def draw(self, **params):
        uniforms = dict(
            time = glfw.get_time() / 50
        )
        super().draw(**{**uniforms, **params})
