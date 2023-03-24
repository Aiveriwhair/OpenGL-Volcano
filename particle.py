# -------------- Simple demo of a point animation -----------------------------
#!/usr/bin/env python3
import random
import OpenGL.GL as GL
import glfw
import numpy as np
from core import Shader, Viewer, Mesh, Texture
from math import cos, pi, sin
from PIL import Image

from texture import *  # import Pillow for image loading


# -------------- Simple demo of a point animation -----------------------------
#!/usr/bin/env python3
import random
import OpenGL.GL as GL
import glfw
import numpy as np
from core import Shader, Viewer, Mesh, Texture
from math import cos, pi, sin
from PIL import Image

from texture import *  # import Pillow for image loading


class PointAnimation(Textured):
    """ Animated particle set with texture that simulates lava """
    def __init__(self, shader, **params):
        GL.glPointSize(params['point_size'])

        # initialize particle positions and texture coordinates
        self.coords = [(random.uniform(-1, 1), random.uniform(-1, 1), 0.5) for i in range(params['num_particles'])]
        self.tex_coords = [(0, 0) for i in range(params['num_particles'])]

        # create vertex array object with position and texture attributes
        attributes = dict(position=self.coords, tex_coord=self.tex_coords)
        mesh = Mesh(shader, attributes=attributes, usage=GL.GL_STREAM_DRAW)
        texture = Texture(params['texture_path'], GL.GL_REPEAT, *(GL.GL_NEAREST, GL.GL_NEAREST_MIPMAP_LINEAR))
        super().__init__(mesh, diffuse_map=texture)

        # initialize particle velocities
        self.velocities = [(random.uniform(-0.01, 0.01), random.uniform(0.02, 0.04), 0) for i in range(params['num_particles'])]
        
        # initialize particle base heights
        self.base_heights = [coord[1] for coord in self.coords]

    def draw(self, primitives=GL.GL_POINTS, attributes=None, **uniforms):
        # update particle positions based on time and speed
        num_particles_to_draw = min(len(self.coords), 10)
        for i in range(num_particles_to_draw):
            x, y, z = self.coords[i]
            vx, vy, vz = self.velocities[i]
            # apply gravity to y-velocity
            vy -= 0.0005
            # update particle position
            x += vx
            y += vy
            z += vz
            # wrap particles around the screen if they go out of bounds
            if abs(x) > 1 or abs(y) > 1 or z < 0:
                x = random.uniform(-1, 1)
                y = random.uniform(-1, 1)
                z = 0.5
                vy = random.uniform(0.02, 0.04)
                angle = random.uniform(-np.pi/4, np.pi/4)
                vx = vy * np.tan(angle)
                
                # reset base height for new particle position
                self.base_heights[i] = y
            
            # update particle position and velocity
            self.coords[i] = (x, y, z)
            self.velocities[i] = (vx, vy, vz)
            
            # decide whether to draw particle based on position of previous particle
            if i > 0 and y < self.base_heights[i-1]:
                continue    # skip this particle
            
            # update texture coordinates based on particle positions
            u = (x + 1) / 2  # map x coordinate from [-1, 1] to [0, 1]
            v = (y + 1) / 2  # map y coordinate from [-1, 1] to [0, 1]
            self.tex_coords[i] = (u, v)

        # update position and texture coordinate buffers on CPU
        coords = np.array(self.coords, 'f')
        tex_coords = np.array(self.tex_coords, 'f')
        super().draw(primitives, attributes=dict(position=coords), **uniforms)


