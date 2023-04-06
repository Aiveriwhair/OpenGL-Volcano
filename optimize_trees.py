import OpenGL.GL as GL              # standard Python OpenGL wrapper
import numpy as np                  # all matrix manipulations & OpenGL args
from core import Node, Mesh
from texture import *
from transform import *
from PIL import Image
from math import radians


class PineTrees(Node):
    def __init__(self, shader, position, trunk_texture_path, leave_texture_path, **params):
        super().__init__()
        leave_tex = Texture(leave_texture_path)
        trunk_tex = Texture(trunk_texture_path)

        nb_trees = len(position)
        trees = self.generate_tree(
            nb_trees, position, shader, leave_tex, trunk_tex, **params)
        self.add(trees)

    def generate_tree(self, nb_trees, position, shader, leave_tex, trunk_tex, **params):
        trunks = Generate_Trunk(
            nb_trees, position, shader, trunk_tex, **params)
        leaves = Generate_Leaves(
            nb_trees, position, shader, leave_tex, **params)
        tree = Node([trunks, leaves])
        return tree


class Generate_Trunk(Textured):
    def __init__(self, nb_trees, position, shader, trunk_tex, min_height=10, height=7, division=6, r=0.5, **params):
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

        vertices = self.generate_trunk_vertices(nb_trees, position)
        tex_coords = self.generate_trunk_tex_coords(nb_trees)
        indices = self.generate_trunk_indices(nb_trees)

        normals = calculate_normals2(vertices, indices)

        mesh = Mesh(shader, attributes=dict(position=vertices, texcoord=tex_coords,
                                            normal=normals,), index=indices, **{**uniforms, **params})
        super().__init__(mesh, texture_sampler=trunk_tex)

    def generate_trunk_vertices(self, nb_trees, position):
        vertices = []
        for i in range(nb_trees):
            x, y, z = position[i]
            for j in range(self.division + 1):
                theta = 2 * np.pi * j / self.division
                vertices.append([x + self.r * np.cos(theta),
                                y, z + self.r * np.sin(theta)])
                vertices.append([x + self.r * np.cos(theta),
                                y + self.height, z + self.r * np.sin(theta)])

            # Add top and bottom center vertices
            vertices.append([x, y, z])
            vertices.append([x, y + self.height, z])

        return np.array(vertices, dtype=np.float32)

    def generate_trunk_tex_coords(self, nb_trees):
        tex_coords = []
        for i in range(nb_trees):
            for j in range(self.division + 1):
                tex_coords.append([j / self.division, 0])
                tex_coords.append([j / self.division, 1])

            # Add texture coordinates for top and bottom center vertices
            tex_coords.append([0.5, 0])
            tex_coords.append([0.5, 1])

        return np.array(tex_coords, dtype=np.float32)

    def generate_trunk_indices(self, nb_trees):
        indices = []
        for i in range(nb_trees):
            base = i * (2 * (self.division + 1) + 2)
            top_center = base + 2 * (self.division + 1)
            bottom_center = top_center + 1

            for j in range(self.division):
                indices.append([base + 2 * j, base + 2 * j + 1,
                               base + 2 * ((j + 1) % (self.division + 1))])
                indices.append([base + 2 * j + 1, base + 2 * ((j + 1) %
                               (self.division + 1)) + 1, base + 2 * ((j + 1) % (self.division + 1))])

                # Add indices for top and bottom faces
                indices.append([top_center, base + 2 * j, base + 2 * (j + 1) %
                               (2 * (self.division + 1))])  # Correct order for the top face
                # Inverted order for the bottom face
                indices.append([bottom_center, base + 2 * (j + 1) %
                               (2 * (self.division + 1)) + 1, base + 2 * j + 1])

        return np.array(indices, dtype=np.uint32)


class Generate_Leaves(Textured):
    def __init__(self, nb_trees, position, shader, leave_tex, heights=(8, 7, 6), widths=(4, 5, 7), subdivisions=40, **params):
        self.heights = heights
        self.widths = widths
        self.subdivisions = subdivisions

        uniforms = dict(
            k_d=np.array((0.8, 0.8, 0.8), dtype=np.float32),
            k_s=np.array((0.5673, 0.5673, 0.5673), dtype=np.float32),
            k_a=np.array((0.5, 0.5, 0.5), dtype=np.float32),
            s=60,
            use_texture2=False,
            use_texture3=False,
        )

        vertices = self.generate_leaves_vertices(nb_trees, position)
        tex_coords = self.generate_leaves_tex_coords(nb_trees)
        indices = self.generate_leaves_indices(nb_trees)

        normals = calculate_normals2(vertices, indices)

        mesh = Mesh(shader, attributes=dict(position=vertices,
                    texcoord=tex_coords, normal=normals), index=indices, **{**uniforms, **params})
        super().__init__(mesh, texture_sampler=leave_tex)

    def generate_leaves_vertices(self, nb_trees, position):
        vertices = []
        for x, y, z in position:
            for h, w in zip(self.heights, self.widths):
                y_base = y + h - w / 2
                tip = [x, y_base + w / 2, z]

                base_vertices = []
                for i in range(self.subdivisions):
                    angle = 2 * np.pi * i / self.subdivisions
                    base_vertices.append(
                        [x + w / 2 * np.cos(angle), y_base, z + w / 2 * np.sin(angle)])

                vertices.append(tip)
                vertices.extend(base_vertices)

        return np.array(vertices, dtype=np.float32)

    def generate_leaves_tex_coords(self, nb_trees):
        tex_coords = []
        for i in range(nb_trees * len(self.heights)):
            tex_coords.append([0.5, 1])
            for j in range(self.subdivisions):
                tex_coords.append([0.5 * (1 + np.cos(2 * np.pi * j / self.subdivisions)),
                                   0.5 * (1 + np.sin(2 * np.pi * j / self.subdivisions))])

        return np.array(tex_coords, dtype=np.float32)

    def generate_leaves_indices(self, nb_trees):
        indices = []
        for i in range(nb_trees * len(self.heights)):
            base = i * (self.subdivisions + 1)
            for j in range(self.subdivisions):
                indices.append([base, base + 1 + (j + 1) %
                                self.subdivisions, base + 1 + j])

        return np.array(indices, dtype=np.uint32)
