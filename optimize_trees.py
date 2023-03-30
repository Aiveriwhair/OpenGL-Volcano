import OpenGL.GL as GL              # standard Python OpenGL wrapper
import numpy as np                  # all matrix manipulations & OpenGL args
from core import Node, Mesh
from texture import *
from transform import *
from PIL import Image
from math import radians


class PineTrees(Node):
    def __init__(self, shader, position, trunk_texture_path, leave_texture_path):
        super().__init__()
        leave_tex = Texture(leave_texture_path)
        trunk_tex = Texture(trunk_texture_path)

        nb_trees = len(position)
        trunk = self.generate_tree(
            nb_trees, position, shader, leave_tex, trunk_tex)
        self.add(trunk)

    def generate_tree(self, nb_trees, position, shader, leave_tex, trunk_tex):
        trunks = Generate_Trunk(nb_trees, position, shader, trunk_tex)
        return trunks


class Generate_Trunk(Textured):
    def __init__(self, nb_trees, position, shader, trunk_tex, min_height=10, height=7, division=10, r=0.5, **params):
        self.height = height
        self.division = division
        self.r = r
        self.min_height = min_height

        uniforms = dict(
            k_d=np.array((0.8, 0.8, 0.8), dtype=np.float32),
            k_s=np.array((0.5673, 0.5673, 0.5673), dtype=np.float32),
            k_a=np.array((0.5, 0.5, 0.5), dtype=np.float32),
            s=60,
            use_texture2=False,
            use_texture3=False,
        )

        trunk_vert = self.division*2+2
        trunk_vertices = self.generate_trunk_vertices(trunk_vert)
        tex_coords_trunk = self.generate_trunk_tex_coords(trunk_vert)
        indices_trunk = self.generate_trunk_indices()

        vertices = np.empty(
            shape=(nb_trees * trunk_vert, 3), dtype=np.float32)
        indices = np.empty(
            shape=(nb_trees * (6 * self.division), 3), dtype=np.uint32)
        tex_coords = np.empty(
            shape=(nb_trees * trunk_vert, 2), dtype=np.float32)

        for i in range(0, nb_trees):
            start_vert = i * trunk_vert
            start_ind = i * (6 * self.division)

            vertices[start_vert: start_vert +
                     trunk_vert] = trunk_vertices + position[i]
            indices[start_ind: start_ind + (6 * self.division)
                    ] = indices_trunk.reshape(-1, 3) + i * trunk_vert
            tex_coords[start_vert: start_vert + trunk_vert] = tex_coords_trunk

        # normals = calculate_normals3(vertices, indices)
        mesh = Mesh(shader, attributes=dict(position=vertices, texcoord=tex_coords,
                                            ), index=indices, **{**uniforms, **params})
        super().__init__(mesh, texture_sampler=trunk_tex)

    def generate_trunk_vertices(self, trunk_vert):
        res = np.empty(shape=(trunk_vert, 3), dtype=np.float32)

        count = 0
        for i in np.linspace(0, 2 * np.pi, self.division, endpoint=False):
            res[count] = [self.r * np.cos(i), self.height, self.r * np.sin(i)]
            count += 1
            res[count] = [self.r * np.cos(i), 0, self.r * np.sin(i)]
            count += 1

        return res

    def generate_trunk_tex_coords(self, trunk_vert):
        res = np.empty(shape=(trunk_vert, 2), dtype=np.float32)

        count = 0
        for i in range(self.division):
            res[count] = [i / self.division, 1]
            count += 1
            res[count] = [i / self.division, 0]
            count += 1

        return res

    def generate_trunk_indices(self):
        res = np.empty(shape=(6 * self.division, 3), dtype=np.uint32)

        count = 0
        for i in range(0, self.division):
            if i != self.division - 1:
                res[count] = [2 * i, 2 * i + 1,
                              (2 * i + 2) % (2 * self.division)]
                res[count + 1] = [2 * i + 1, (2 * i + 3) % (2 * self.division),
                                  (2 * i + 2) % (2 * self.division)]
            else:
                res[count] = [2 * i, 2 * i + 1, 0]
                res[count + 1] = [2 * i + 1, 1, 0]
            count += 2

        return res.flatten()
