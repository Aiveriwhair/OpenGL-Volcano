import math
import numpy as np
from core import Mesh
from texture import Texture
import sys                          # for system arguments

from OpenGL.GL import *
from core import *
from texture import *
from skybox import *
from tree import *


class Cone(Mesh):
    """ A class for cones """

    def __init__(self, shader, height=1.0, radius=1.0, divisions=32, **params):
        self.shader = shader
        self.height = height
        self.radius = radius
        self.divisions = divisions
        self.color = (1, 1, 1)

        # Vertices
        vertices = [(0, self.height, 0)]
        for angle in np.linspace(0, 2 * np.pi, self.divisions, endpoint=False):
            x = self.radius * np.cos(angle)
            z = self.radius * np.sin(angle)
            vertices.append((x, 0, z))
        vertices.append((0, 0, 0))

        position = np.array(vertices, np.float32)

        # Indices
        indices = []
        for i in range(1, self.divisions):
            indices.extend([0, i + 1, i])
        indices.extend([0, 1, self.divisions + 1])

        for i in range(1, self.divisions+1):
            indices.extend([self.divisions + 1, i, i + 1])
        indices.extend([self.divisions + 1, self.divisions, 1])
        indices.extend([0, 1, self.divisions])

        index = np.array(indices, np.uint32)

        uniforms = dict(
            k_d=np.array((0., .5, .5), dtype=np.float32),
            k_s=np.array((0.5673, 0.5673, 0.5673), dtype=np.float32),
            k_a=np.array((0., 0.4, 0.4), dtype=np.float32),
            s=60,
        )

        normals = calculate_normals(position, index)

        super().__init__(shader, attributes=dict(position=position,
                                                 normal=normals), index=index, **{**uniforms, **params})

    def draw(self, primitives=GL.GL_TRIANGLES, **uniforms):
        super().draw(primitives=primitives,
                     global_color=self.color, **uniforms)


class Dodecahedron(Mesh):
    """ A class for dodecahedrons """

    def __init__(self, shader, radius=1.0, **params):
        self.shader = shader
        self.radius = radius

        # Vertices
        phi = (1 + math.sqrt(5)) / 2
        vertices = [
            (-1, phi, 0), (1, phi, 0), (-1, -phi, 0), (1, -phi, 0),
            (0, -1, phi), (0, 1, phi), (0, -1, -phi), (0, 1, -phi),
            (phi, 0, -1), (phi, 0, 1), (-phi, 0, -1), (-phi, 0, 1)
        ]

        position = np.array(vertices, np.float32)

        # Indices
        indices = [
            0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11,
            1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6, 7, 1, 8,
            3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9,
            4, 9, 5, 2, 4, 11, 6, 2, 10, 8, 6, 7, 9, 8, 1
        ]

        index = np.array(indices, np.uint32)
        normals = calculate_normals(position, index)
        uniforms = dict(
            k_d=np.array((0., .5, .5), dtype=np.float32),
            k_s=np.array((0.5673, 0.5673, 0.5673), dtype=np.float32),
            k_a=np.array((0., 0.4, 0.4), dtype=np.float32),
            s=60,
        )

        super().__init__(shader, attributes=dict(position=position,
                                                 normal=normals), index=index, **{**uniforms, **params})


def treeGenerator(shader, pos, **params):
    """ create a tree trunk """
    mainTrunk = Cone(shader, height=10.0, radius=1.0, divisions=32, **params)
    firstBranch = Cone(shader, height=4.0, radius=0.5, divisions=32, **params)
    secondBranch = Cone(shader, height=5.0, radius=0.5, divisions=32, **params)
    thirdBranch = Cone(shader, height=2.0, radius=0.3, divisions=32, **params)

    axis = Axis(shader)

    phi = 60.0
    random_size = np.random.uniform(1.5, 3)

    mainLeaf = Dodecahedron(shader)
    leaf1 = Dodecahedron(shader)
    leaf2 = Dodecahedron(shader)
    leaf3 = Dodecahedron(shader)

    transform_mainLeaf1 = Node(
        transform=translate(0, 12, 0)@scale(random_size, random_size, random_size))
    transform_mainLeaf1.add(mainLeaf)

    transform_mainLeaf2 = Node(
        transform=translate(2, 12, 0))
    transform_mainLeaf2.add(mainLeaf)

    transform_firstBranch = Node(transform=translate(
        0, 5, 0)@rotate((0., 0., 1.), phi))
    transform_leaf1 = Node(transform=translate(0, 5, 0)@scale(1.5, 1.5, 1.5))
    transform_leaf1.add(leaf1)
    transform_firstBranch.add(firstBranch, transform_leaf1)

    transform_secondBranch = Node(transform=translate(
        0, 3, 0)@rotate((1., 0., 0.), phi))
    transform_leaf2 = Node(transform=translate(0, 5, 0)@scale(1, 1, 1))
    transform_leaf2.add(leaf2)
    transform_secondBranch.add(secondBranch, transform_leaf2)

    transform_thirdBranch = Node(transform=translate(
        0, 6, 0)@rotate((0., 1., 0.), 90.0)@rotate((1., 0., 0.), phi-25.0))
    transform_leaf3 = Node(transform=translate(0, 2.5, 0)@scale(0.5, 0.5, 0.5))
    transform_leaf3.add(leaf3)
    transform_thirdBranch.add(thirdBranch, transform_leaf3)

    transform_mainTrunk = Node(transform=translate(pos, 0, 0))
    transform_mainTrunk.add(
        mainTrunk, transform_firstBranch, transform_secondBranch, transform_thirdBranch, transform_mainLeaf1, transform_mainLeaf2)
    return transform_mainTrunk


def forestGenerator(shader, numTrees):
    """ create a forest of trees """
    # create numpy array for each tree that i will return
    forest = np.empty(numTrees, dtype=object)
    for i in range(numTrees):
        forest[i] = treeGenerator(shader, 10*i)
    return forest
