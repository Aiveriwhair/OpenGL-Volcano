from gettext import npgettext
import random
import OpenGL.GL as GL
import numpy as np
from core import Shader, Viewer, Mesh, Texture
from math import cos, pi, sin
import copy

from texture import *  # import Pillow for image loading


class BillboardAnimation(Textured):
    """ Animated particle system with textured billboards """

    def __init__(self, shader, x, y, z, texturepath, **params):
        self.number_of_particles = params['num_particles']
        self.point_size = params['point_size']
        uniforms = dict(
            k_d=np.array((0.8, 0.8, 0.8), dtype=np.float32),
            k_s=np.array((0.5673, 0.5673, 0.5673), dtype=np.float32),
            k_a=np.array((0.5, 0.5, 0.5), dtype=np.float32),
            s=60,
        )

        self.coords = [(x, y, z) for i in range(params['num_particles'])]
        self.initial_coords = copy.copy(self.coords)
        self.velocities = [(random.uniform(-0.01, 0.01), random.uniform(0.02, 0.04), 0)
                           for i in range(params['num_particles'])]
        self.base_heights = [coord[1] for coord in self.coords]

        self.vertices = np.zeros((self.number_of_particles, 4, 3), dtype=np.float32)
        self.tex_coords = np.zeros((self.number_of_particles, 4, 2), dtype=np.float32)
        self.indices = np.zeros((self.number_of_particles, 6), dtype=np.uint32)

        for i in range(self.number_of_particles):
            self.indices[i] = [i * 4, i * 4 + 1, i * 4 + 2, i * 4 + 2, i * 4 + 1, i * 4 + 3]

        # Define the billboard vertices based on the particle positions and size
        for i in range(self.number_of_particles):
            billboard_pos = np.array(self.coords[i], dtype=np.float32)
            billboard_right = np.array((1, 0, 0), dtype=np.float32)
            billboard_up = np.array((0, 1, 0), dtype=np.float32)
            self.vertices[i, 0] = billboard_pos - billboard_right * self.point_size + billboard_up * self.point_size
            self.vertices[i, 1] = billboard_pos + billboard_right * self.point_size + billboard_up * self.point_size
            self.vertices[i, 2] = billboard_pos - billboard_right * self.point_size - billboard_up * self.point_size
            self.vertices[i, 3] = billboard_pos + billboard_right * self.point_size - billboard_up * self.point_size

            self.tex_coords[i] = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])

        attributes = dict(position=self.vertices.reshape(-1, 3), texcoord=self.tex_coords.reshape(-1, 2))
        self.mesh = Mesh(shader, attributes=attributes, index=self.indices.reshape(-1),
                    usage=GL.GL_STREAM_DRAW, **{**uniforms, **params})
        texture = Texture(texturepath)
        super().__init__(self.mesh, texture_sampler=texture)

    def draw(self, primitives=GL.GL_TRIANGLES, attributes=None, **uniforms):
        view_matrix = uniforms['view']
        modelview = uniforms.get('modelview', np.identity(4)) # Add default value for modelview
        
        prev_camera_position = None  #  variable pour stocker la position de la caméra
        for i in range(self.number_of_particles):
            x, y, z = self.coords[i]
            vx, vy, vz = self.velocities[i]

            vy -= 0.0005

            x += vx
            y += vy
            z += vz

            if y < self.initial_coords[i][1] - 8:
                x, y, z = self.initial_coords[i]
                vy = random.uniform(0.02, 0.04)
                angle = random.uniform(-np.pi / 4, np.pi / 4)
                vx = vy * np.tan(angle)

            # Update the coordinates and velocities in the lists
                self.coords[i] = (x, y, z)
                self.velocities[i] = (vx, vy, vz)

            # Reset the base height for new particle position
                self.base_heights[i] = y

            else:
            # Update the particle position and velocity
                self.coords[i] = (x, y, z)
                self.velocities[i] = (vx, vy, vz)

            camera_position = np.linalg.inv(view_matrix[:3, :3]) @ (-view_matrix[:3, 3])
            if not np.array_equal(prev_camera_position, camera_position):
                prev_camera_position = camera_position
                distance = np.linalg.norm(np.array(camera_position) - np.array([x, y, z]))
            if distance is not None:
             # Calculate the scaling factor for the particle size
                max_distance = 100.0  # Example maximum distance at which particle is visible
                scaling_factor = max(0, (max_distance - distance) / max_distance)
            else:
                scaling_factor = 1.0  # Use default value if distance is not defined
        # Calculate the model matrix for the current particle
            billboard_up = np.array([0, 1, 0], dtype=np.float32)
            billboard_right = np.cross(billboard_up, view_matrix[:3, 0:3].T @ np.array([0, 0, -1], dtype=np.float32))
            billboard_right /= np.linalg.norm(billboard_right)
            billboard_forward = np.cross(billboard_right, billboard_up)

            billboard_matrix = np.eye(4, dtype=np.float32)
            billboard_matrix[:3, :3] = np.column_stack((billboard_right, billboard_up, -billboard_forward))
            billboard_pos = np.array([x, y, z], dtype=np.float32)
            billboard_matrix[:3, 3] = billboard_pos

        # Calculate the final size and texture coordinates of the particle
            size = scaling_factor * self.point_size
            min_size = 0.01  # Example minimum size for the particle
            size = max(size, min_size)

            self.vertices[i, 0] = billboard_pos - billboard_right * size + billboard_up * size
            self.vertices[i, 1] = billboard_pos + billboard_right * size + billboard_up * size
            self.vertices[i, 2] = billboard_pos - billboard_right * size - billboard_up * size
            self.vertices[i, 3] = billboard_pos + billboard_right * size - billboard_up * size

            self.tex_coords[i] = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])

           # update position and texture coordinate buffers on CPU

        attributes = dict(position=self.vertices.reshape(-1, 3), texcoord=self.tex_coords.reshape(-1, 2))
        

        super().draw(primitives=GL.GL_TRIANGLES, attributes=attributes, **uniforms)


        