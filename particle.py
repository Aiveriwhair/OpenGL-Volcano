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

    def __init__(self, shader, x, y, z, texturepath, **params):
        # GL.glPointSize(params['point_size'])

        # initialize particle positions and texture coordinates
        self.coords = [(x, y, z) for i in range(params['num_particles'])]
        self.tex_coords = [(0, 0) for i in range(params['num_particles'])]
        self.normals = [(0, 1, 0) for i in range(params['num_particles'])]
        self.point_size = params['point_size']
        uniforms = dict(
            k_d=np.array((0.8, 0.8, 0.8), dtype=np.float32),
            k_s=np.array((0.5673, 0.5673, 0.5673), dtype=np.float32),
            k_a=np.array((0.5, 0.5, 0.5), dtype=np.float32),
            s=60,
        )

        # initialize particle velocities
        self.velocities = [(random.uniform(-0.01, 0.01), random.uniform(0.02, 0.04), 0)
                           for i in range(params['num_particles'])]

        # initialize particle base heights
        self.base_heights = [coord[1] for coord in self.coords]

        # create vertex array object with position and texture attributes
        attributes = dict(position=self.coords,
                          texcoord=self.tex_coords, normal=self.normals)
        mesh = Mesh(shader, attributes=attributes,
                    usage=GL.GL_STREAM_DRAW, **{**uniforms, **params})
        texture = Texture(texturepath)
        super().__init__(mesh, texture_sampler=texture)

    def draw(self, primitives=GL.GL_POINTS, attributes=None, **uniforms):
        # update particle positions based on time and speed

        view_matrix = uniforms['view']
        camera_position = np.linalg.inv(view_matrix)[:, 3]
        num_particles_to_draw = min(len(self.coords), 100)
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
            if y < -2.:
                x = 0.
                y = 0.
                z = 0.
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
            # calculate the distance between the particle and the camera
            camera_position = np.linalg.inv(
                view_matrix[:3, :3]) @ (-view_matrix[:3, 3])

            distance = np.linalg.norm(
                np.array(camera_position) - np.array([x, y, z]))

            # calculate a scaling factor for the particle size
            max_distance = 100.0  # example maximum distance at which particle is visible
            scaling_factor = max(0, (max_distance - distance) / max_distance)

            # calculate the final size of the particle
            size = scaling_factor * self.point_size
            min_size = 0.01  # example minimum size for the particle
            size = max(size, min_size)

            # update the OpenGL point size parameter
            GL.glPointSize(size)

            # update texture coordinates based on particle positions
            u = (x + 1) / 2  # map x coordinate from [-1, 1] to [0, 1]
            v = (y + 1) / 2  # map y coordinate from [-1, 1] to [0, 1]
            self.tex_coords[i] = (u, v)

        # update position and texture coordinate buffers on CPU
        coords = np.array(self.coords, 'f')
        tex_coords = np.array(self.tex_coords, 'f')
        normals = np.array(self.normals, 'f')  # add normals

        super().draw(primitives=primitives, attributes=dict(
            position=coords, texcoord=tex_coords, normal=normals), **uniforms)
