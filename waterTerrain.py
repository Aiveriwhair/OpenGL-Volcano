import numpy as np
import OpenGL.GL as GL
from PIL import Image
from texture import Textured
from core import Mesh
from transform import calculate_normals

class WaterTerrain(Textured):
    def __init__(self, shader, height=0, size=(512, 512), **params):
        self.shader = shader
        self.height = height
        self.size = size

        (attributes, index) = self.generateTerrain()

        self.color = (1, 1, 1)
        uniforms = dict(
            k_d=np.array((0., .5, .5), dtype=np.float32),
            k_s=np.array((0.5673, 0.5673, 0.5673), dtype=np.float32),
            k_a=np.array((0., 0.4, 0.4), dtype=np.float32),
            s=60,
        )

        mesh = Mesh(shader, attributes=attributes, index=index, **{**uniforms, **params})
        textures = dict()

        super().__init__(mesh, textures)

    def draw(self, primitives=GL.GL_TRIANGLES, **uniforms):
        super().draw(primitives=primitives, global_color=self.color, **uniforms)

    def generateTerrain(self):
        w, h = self.size

        # Compute the number of vertices and indices needed
        num_vertices = w * h
        num_indices = (w - 1) * (h - 1) * 6

        # Create arrays to hold the vertices and indices
        position = np.zeros((num_vertices, 3), dtype=np.float32)
        indices = np.zeros(num_indices, dtype=np.uint32)

        # color = np.zeros((num_vertices, 3), dtype=np.float32)

        # Fill in the vertex positions
        for y in range(h):
            for x in range(w):
                i = y * w + x
                z = self.height
                position[i, 0] = x
                position[i, 1] = z
                position[i, 2] = y

        # Fill in the indices to draw triangles
        idx = 0
        for y in range(h - 1):
            for x in range(w - 1):
                i = y * w + x
                indices[idx + 2] = i
                indices[idx + 1] = i + 1
                indices[idx] = i + w
                indices[idx + 5] = i + 1
                indices[idx + 4] = i + w + 1
                indices[idx + 3] = i + w
                idx += 6

        return (dict(position=position, normal=calculate_normals(w, h, position)), indices)
