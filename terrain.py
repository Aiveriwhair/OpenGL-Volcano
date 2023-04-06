# Python built-in modules
import math
import os                           # os function, i.e. checking file status
from itertools import cycle         # allows easy circular choice list
import atexit                       # launch a function at exit
import random                       # random number generator

# External, non built-in modules
import OpenGL.GL as GL              # standard Python OpenGL wrapper
import glfw                         # lean window system wrapper for OpenGL
import numpy as np                  # all matrix manipulations & OpenGL args
import assimpcy                     # 3D resource loader

# our transform functions
from transform import Trackball, identity
from transform import translate, rotate, scale, vec, perspective, calculate_normals
from PIL import Image
from texture import Textured, Texture
from core import Mesh, Shader, Viewer
from tree import forestGenerator


class Terrain(Mesh):
    def __init__(self, shader, height=100, width=100, step=10):
        self.h = height
        self.w = width
        self.s = step
        self.shader = shader
        (attributes, index) = self.generateTerrain()

        self.color = (1, 0, 1)

        super().__init__(shader, attributes=attributes, index=index)

    def draw(self, primitives=GL.GL_TRIANGLES, **uniforms):
        super().draw(primitives=primitives, global_color=self.color, **uniforms)

    def generateTerrain(self):
        w = self.w
        h = self.h
        step = self.s

        # Compute the number of vertices and indices needed
        num_vertices = w * h
        num_indices = (w - 1) * (h - 1) * 6

        # Create arrays to hold the vertices and indices
        position = np.zeros((num_vertices, 3), dtype=np.float32)
        indices = np.zeros(num_indices, dtype=np.uint32)
        color = np.zeros((num_vertices, 3), dtype=np.float32)
        texcoords = np.zeros((num_vertices, 2), dtype=np.float32)

        # Fill in the vertex positions
        for y in range(h):
            for x in range(w):
                i = y * w + x
                position[i, 0] = x * step
                position[i, 1] = 0
                position[i, 2] = y * step

                texcoords[i, 0] = x / (w - 1)
                texcoords[i, 1] = y / (h - 1)

        # Fill in the indices to draw triangles
        idx = 0
        for y in range(h - 1):
            for x in range(w - 1):
                i = y * w + x
                indices[idx] = i
                indices[idx + 1] = i + 1
                indices[idx + 2] = i + w
                indices[idx + 3] = i + 1
                indices[idx + 4] = i + w + 1
                indices[idx + 5] = i + w
                idx += 6

        return (dict(position=position, color=color, texcoords=texcoords), indices)


class randomTerrain(Mesh):
    def __init__(self, shader, height=100, width=100, step=1):
        self.h = height
        self.w = width
        self.s = step
        self.shader = shader
        (attributes, index) = self.generateTerrain()

        self.color = (1, 0, 1)

        super().__init__(shader, attributes=attributes, index=index)

    def draw(self, primitives=GL.GL_TRIANGLES, **uniforms):
        super().draw(primitives=primitives, global_color=self.color, **uniforms)

    def generateTerrain(self):
        w = self.w
        h = self.h
        step = self.s

        # Compute the number of vertices and indices needed
        num_vertices = w * h
        num_indices = (w - 1) * (h - 1) * 6

        # Create arrays to hold the vertices and indices
        position = np.zeros((num_vertices, 3), dtype=np.float32)
        indices = np.zeros(num_indices, dtype=np.uint32)
        color = np.zeros((num_vertices, 3), dtype=np.float32)

        # Fill in the vertex positions
        for y in range(h):
            for x in range(w):
                i = y * w + x
                z = random.randint(0, 10)
                position[i, 0] = x * step
                position[i, 1] = z
                position[i, 2] = y * step
                color[i] = (z/10, z/10, z/10)

        # Fill in the indices to draw triangles
        idx = 0
        for y in range(h - 1):
            for x in range(w - 1):
                i = y * w + x
                indices[idx] = i
                indices[idx + 1] = i + 1
                indices[idx + 2] = i + w
                indices[idx + 3] = i + 1
                indices[idx + 4] = i + w + 1
                indices[idx + 5] = i + w
                idx += 6

        return (dict(position=position, color=color), indices)


class circularTerrain(Mesh):
    def __init__(self, shader, height=100, width=100, radius=20, step=5):
        self.h = height
        self.w = width
        self.s = step
        self.r = radius
        self.shader = shader
        (attributes, index) = self.generateTerrain()

        self.color = (1, 0, 1)

        super().__init__(shader, attributes=attributes, index=index)

    def draw(self, primitives=GL.GL_TRIANGLES, **uniforms):
        super().draw(primitives=primitives, global_color=self.color, **uniforms)

    def generateTerrain(self):
        w = self.w
        h = self.h
        r = self.r
        maxheight = 100
        step = self.s

        # Compute the number of vertices and indices needed
        num_vertices = w * h
        num_indices = (w - 1) * (h - 1) * 6

        # Create arrays to hold the vertices and indices
        position = np.zeros((num_vertices, 3), dtype=np.float32)
        indices = np.zeros(num_indices, dtype=np.uint32)
        color = np.zeros((num_vertices, 3), dtype=np.float32)

        # Fill in the vertex positions
        for y in range(h):
            for x in range(w):
                i = y * w + x
                z = random.randint(0, maxheight)
                # distance from center of the plane
                dist = math.sqrt((x - w/2)**2 + (y - h/2)**2)
                z_factor = 1 - (dist/r)
                if (z_factor < 0):
                    z_factor = 0
                z = z * z_factor
                position[i, 0] = x * step
                position[i, 1] = z
                position[i, 2] = y * step
                color[i] = (z / maxheight, z/maxheight, z/maxheight)

        # Fill in the indices to draw triangles
        idx = 0
        for y in range(h - 1):
            for x in range(w - 1):
                i = y * w + x
                indices[idx] = i
                indices[idx + 1] = i + 1
                indices[idx + 2] = i + w
                indices[idx + 3] = i + 1
                indices[idx + 4] = i + w + 1
                indices[idx + 5] = i + w
                idx += 6

        return (dict(position=position, color=color), indices)


class heightMapTerrain(Textured):
    def __init__(self, shader, heightmappath, height_factor=1, numbertrees=10, **params):
        self.shader = shader
        self.height_factor = height_factor
        self.heightmappath = heightmappath
        self.numbertrees = numbertrees

        (attributes, index) = self.generateTerrain()

        self.color = (1, 1, 1)
        uniforms = dict(
            k_d=np.array((0.8, 0.8, 0.8), dtype=np.float32),
            k_s=np.array((0.5673, 0.5673, 0.5673), dtype=np.float32),
            k_a=np.array((0.5, 0.5, 0.5), dtype=np.float32),
            s=1,
            height_threshold1=40.0,
            height_threshold2=110.0,
            use_texture2=1,
            use_texture3=1,
            mix_range=20,

        )

        mesh = Mesh(shader, attributes=attributes,
                    index=index, **{**uniforms, **params})
        texture = Texture('./ress/bas.jpg')
        texture1 = Texture('./ress/milieu.jpg')
        texture2 = Texture('./ress/snow.jpg')
        super().__init__(mesh, texture_sampler=texture,
                         texture_sampler1=texture1, texture_sampler2=texture2)

    def generateTerrain(self):
        map = Image.open(self.heightmappath).convert('L')
        w, h = map.size
        iters = 0
        height_factor = self.height_factor

        # Compute the number of vertices and indices needed
        num_vertices = w * h
        num_indices = (w - 1) * (h - 1) * 6

        # Create arrays to hold the vertices and indices
        position = np.zeros((num_vertices, 3), dtype=np.float32)
        indices = np.zeros(num_indices, dtype=np.uint32)
        color = np.zeros((num_vertices, 3), dtype=np.float32)
        texcoords = np.zeros((w*h, 2), dtype=np.int32)
        i, j = np.indices((w, h), dtype=np.int32)
        texcoords[:, 0] = i.ravel() % 2
        texcoords[:, 1] = j.ravel() % 2

        # Fill in the vertex positions
        for y in range(h):
            for x in range(w):
                iters += 1
                i = y * w + x
                z = map.getpixel((x, y))
                position[i, 0] = x
                position[i, 1] = z * height_factor
                position[i, 2] = y
                color[i] = (z/(255*height_factor), z /
                            (255*height_factor), z/(255*height_factor))

        # Fill in the indices to draw triangles
        idx = 0
        for y in range(h - 1):
            for x in range(w - 1):
                iters += 1
                i = y * w + x
                indices[idx + 2] = i
                indices[idx + 1] = i + 1
                indices[idx] = i + w
                indices[idx + 5] = i + 1
                indices[idx + 4] = i + w + 1
                indices[idx + 3] = i + w
                idx += 6
        normals = calculate_normals(w, h, position)
        self.pos_trees = self.generate_trees(
            position, normals, self.numbertrees)

        return (dict(position=position, normal=normals, color=color, texcoord=texcoords), indices)

    def generate_trees(self, positions, normals, tree_number, max_angle=30, min_height=10):
        # Convertir l'angle maximal en radians
        max_angle_rad = np.radians(max_angle)

        # Calculer le cosinus de l'angle maximal
        max_cos_angle = np.cos(max_angle_rad)

        # Filtrer les positions et les normales en fonction de l'angle maximal autorisé

        valid_positions = positions[(np.dot(normals, np.array(
            [0, 1, 0])) >= max_cos_angle) & (positions[:, 1] >= min_height)]

        # Choisir un nombre aléatoire de positions valides pour les arbres
        if len(valid_positions) > tree_number:
            tree_positions = valid_positions[np.random.choice(
                len(valid_positions), tree_number, replace=False)]
        else:
            tree_positions = valid_positions

        return tree_positions
