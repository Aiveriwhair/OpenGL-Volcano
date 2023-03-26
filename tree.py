import math
import numpy as np
from OpenGL.GL import *
from core import *
from texture import *
from skybox import *
from tree import *
from transform import *

# parralelisation
from multiprocessing import Pool, Manager


class Cone(Mesh):
    """ A class for cones """

    def __init__(self, shader, height=1.0, radius=1.0, divisions=32, **params):
        self.shader = shader
        self.height = height
        self.radius = radius
        self.divisions = divisions
        self.textures = {}

        # Vertices
        vertices = [(0, self.height, 0)]
        for angle in np.linspace(0, 2 * np.pi, self.divisions, endpoint=False):
            x = self.radius * np.cos(angle)
            z = self.radius * np.sin(angle)
            vertices.append((x, 0, z))
        vertices.append((0, 0, 0))

        position = np.array(vertices, np.float32)

        # Calculate texture coordinates for the apex
        texcoords = [(0.5, 1)]

        # Calculate texture coordinates for the base vertices
        angle_step = 2 * np.pi / self.divisions
        for i in range(self.divisions):
            angle = i * angle_step
            u = 0.5 * np.cos(angle) + 0.5
            v = 0.5 * np.sin(angle) + 0.5
            texcoords.append((u, v))

        # Calculate texture coordinates for the center of the base
        texcoords.append((0.5, 0.5))

        texcoord = np.array(texcoords, np.float32)

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
            k_d=np.array((0.8, 0.8, 0.8), dtype=np.float32),
            k_s=np.array((0.5673, 0.5673, 0.5673), dtype=np.float32),
            k_a=np.array((0.5, 0.5, 0.5), dtype=np.float32),
            s=60,
        )

        normals = calculate_normals2(position, index)

        super().__init__(shader, attributes=dict(position=position,
                                                 normal=normals,
                                                 texcoord=texcoord), index=index, **{**uniforms, **params})

    def draw(self, primitives=GL.GL_TRIANGLES, **uniforms):
        super().draw(primitives=primitives,
                     **uniforms)


class Cone2(Mesh):
    def __init__(self, shader, height=1.0, radius=1.0, divisions=32, subdivisions=2, **params):
        self.shader = shader
        self.height = height
        self.radius = radius
        self.divisions = divisions
        self.subdivisions = subdivisions
        self.textures = {}

        # Vertices
        self.vertices = [(0, self.height, 0)]
        for angle in np.linspace(0, 2 * np.pi, self.divisions, endpoint=False):
            x = self.radius * np.cos(angle)
            z = self.radius * np.sin(angle)
            self.vertices.append((x, 0, z))
        self.vertices.append((0, 0, 0))

        # Indices
        indices = []
        for i in range(1, self.divisions):
            indices.extend(self.subdivide_triangle(
                0, i + 1, i, self.subdivisions))
        indices.extend(self.subdivide_triangle(
            0, 1, self.divisions, self.subdivisions))

        for i in range(1, self.divisions+1):
            indices.extend(self.subdivide_triangle(
                self.divisions + 1, i, i + 1, self.subdivisions))
        indices.extend(self.subdivide_triangle(
            self.divisions + 1, self.divisions, 1, self.subdivisions))

        index = np.array(indices, np.uint32)
        position = np.array(self.vertices, np.float32)

        # Calculate texture coordinates
        texcoords = self.calculate_texcoords()

        texcoord = np.array(texcoords, np.float32)

        uniforms = dict(
            k_d=np.array((0.8, 0.8, 0.8), dtype=np.float32),
            k_s=np.array((0.5673, 0.5673, 0.5673), dtype=np.float32),
            k_a=np.array((0.5, 0.5, 0.5), dtype=np.float32),
            s=60,
        )

        normals = calculate_normals2(position, index)

        super().__init__(shader, attributes=dict(position=position,
                                                 normal=normals,
                                                 texcoord=texcoord), index=index, **{**uniforms, **params})

    def calculate_texcoords(self):
        texcoords = []
        for vertex in self.vertices:
            x, y, z = vertex

            # Calculate the height ratio (between 0 and 1)
            v = 1.0 - (y / self.height)

            # Calculate the angle in the range of [0, 2 * pi]
            angle = np.arctan2(z, x) % (2 * np.pi)

            # Convert the angle to the u coordinate (between 0 and 1)
            u = angle / (2 * np.pi)

            texcoords.append((u, v))

        return texcoords

    def subdivide_triangle(self, a, b, c, n):
        if n == 0:
            return [a, b, c]

        ab = len(self.vertices)
        self.vertices.append(self.interpolate_vertex(a, b))
        bc = len(self.vertices)
        self.vertices.append(self.interpolate_vertex(b, c))
        ca = len(self.vertices)
        self.vertices.append(self.interpolate_vertex(c, a))

        triangles = []
        triangles.extend(self.subdivide_triangle(a, ab, ca, n-1))
        triangles.extend(self.subdivide_triangle(ab, b, bc, n-1))
        triangles.extend(self.subdivide_triangle(ca, bc, c, n-1))
        triangles.extend(self.subdivide_triangle(ab, bc, ca, n-1))

        return triangles

    def interpolate_vertex(self, a, b):
        va = np.array(self.vertices[a])
        vb = np.array(self.vertices[b])
        vc = (va + vb) / 2.0
        return tuple(vc)


class Icosahedron(Mesh):
    """ A class for icosahedrons """

    def __init__(self, shader, radius=1.0, **params):
        self.shader = shader
        self.radius = radius
        self.textures = {}

        # Vertices
        phi = (1 + math.sqrt(5)) / 2
        vertices = [
            (-1, phi, 0), (1, phi, 0), (-1, -phi, 0), (1, -phi, 0),
            (0, -1, phi), (0, 1, phi), (0, -1, -phi), (0, 1, -phi),
            (phi, 0, -1), (phi, 0, 1), (-phi, 0, -1), (-phi, 0, 1)
        ]

        position = np.array(vertices, np.float32)

        # Normalize vertices
        normalized_vertices = [
            np.array(vertex) / np.linalg.norm(vertex) for vertex in vertices]

        # Calculate texture coordinates
        texcoords = [spherical_mapping(vertex)
                     for vertex in normalized_vertices]
        texcoord = np.array(texcoords, np.float32)

        # Indices
        indices = [
            0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11,
            1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6, 7, 1, 8,
            3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9,
            4, 9, 5, 2, 4, 11, 6, 2, 10, 8, 6, 7, 9, 8, 1
        ]

        index = np.array(indices, np.uint32)
        normals = calculate_normals2(position, index)

        uniforms = dict(
            k_d=np.array((0.8, 0.8, 0.8), dtype=np.float32),
            k_s=np.array((0.5673, 0.5673, 0.5673), dtype=np.float32),
            k_a=np.array((0.5, 0.5, 0.5), dtype=np.float32),
            s=60,
        )

        super().__init__(shader, attributes=dict(position=position,
                                                 normal=normals,
                                                 texcoord=texcoord), index=index, **{**uniforms, **params})


class Icosahedron2(Mesh):
    """ A class for icosahedrons """

    def __init__(self, shader, radius=1.8, subdivisions=1, **params):
        self.shader = shader
        self.radius = radius
        self.subdivisions = subdivisions
        self.textures = {}

        # Vertices
        phi = (1 + math.sqrt(5)) / 2
        vertices = [
            (-1, phi, 0), (1, phi, 0), (-1, -phi, 0), (1, -phi, 0),
            (0, -1, phi), (0, 1, phi), (0, -1, -phi), (0, 1, -phi),
            (phi, 0, -1), (phi, 0, 1), (-phi, 0, -1), (-phi, 0, 1)
        ]

        # Normalize vertices
        normalized_vertices = [
            np.array(vertex) / np.linalg.norm(vertex) for vertex in vertices]

        # Indices
        base_indices = [
            0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11,
            1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6, 7, 1, 8,
            3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9,
            4, 9, 5, 2, 4, 11, 6, 2, 10, 8, 6, 7, 9, 8, 1
        ]

        # Subdivide triangles
        vertices, indices = self.subdivide_triangles(
            normalized_vertices, base_indices, self.subdivisions)

        position = np.array(vertices, np.float32) * self.radius

        texcoords = [spherical_mapping(vertex)
                     for vertex in vertices]
        texcoord = np.array(texcoords, np.float32)

        index = np.array(indices, np.uint32)
        normals = calculate_normals2(position, index)

        uniforms = dict(
            k_d=np.array((0.8, 0.8, 0.8), dtype=np.float32),
            k_s=np.array((0.5673, 0.5673, 0.5673), dtype=np.float32),
            k_a=np.array((0.5, 0.5, 0.5), dtype=np.float32),
            s=60,
        )

        super().__init__(shader, attributes=dict(position=position,
                                                 normal=normals,
                                                 texcoord=texcoord), index=index, **{**uniforms, **params})

    def subdivide_triangles(self, vertices, indices, subdivisions):
        for _ in range(subdivisions):
            new_indices = []
            new_vertices = list(vertices)

            for i in range(0, len(indices), 3):
                a = indices[i]
                b = indices[i + 1]
                c = indices[i + 2]

                ab = self.get_midpoint(a, b, new_vertices)
                bc = self.get_midpoint(b, c, new_vertices)
                ca = self.get_midpoint(c, a, new_vertices)

                new_indices.extend([
                    a, ab, ca,
                    ab, b, bc,
                    ca, bc, c,
                    ab, bc, ca
                ])

            indices = new_indices

        return new_vertices, indices

    def get_midpoint(self, a, b, vertices):
        midpoint = (vertices[a] + vertices[b]) / 2
        normalized_midpoint = midpoint / np.linalg.norm(midpoint)
        midpoint_list = normalized_midpoint.tolist()

        for idx, vertex in enumerate(vertices):
            if np.allclose(vertex, midpoint_list):
                return idx

        vertices.append(midpoint_list)
        return len(vertices) - 1


def trunkGen(shader, **params):
    mainTrunk = Cone2(shader, height=10.0, radius=1.0, divisions=32, **params)
    firstBranch = Cone2(shader, height=4.0, radius=0.5, divisions=32, **params)
    secondBranch = Cone2(shader, height=5.0, radius=0.5,
                         divisions=32, **params)
    thirdBranch = Cone2(shader, height=2.0, radius=0.3, divisions=32, **params)

    wood_texture = Texture('./ress/wood3.jpg')

    mainTrunk_textured = Textured(mainTrunk, texture_sampler=wood_texture)
    firstBranch_textured = Textured(firstBranch, texture_sampler=wood_texture)
    secondBranch_textured = Textured(
        secondBranch, texture_sampler=wood_texture)
    thirdBranch_textured = Textured(thirdBranch, texture_sampler=wood_texture)

    return mainTrunk_textured, firstBranch_textured, secondBranch_textured, thirdBranch_textured


def leafGen(shader, random_subs, **params):

    leaf_texture = Texture('./ress/leaf.png')

    if random_subs == 0:
        mainLeaf = Icosahedron(shader, **params)
        leaf1 = Icosahedron(shader, **params)
        leaf2 = Icosahedron(shader, **params)
        leaf3 = Icosahedron(shader, **params)
    else:
        mainLeaf = Icosahedron2(shader, subdivisions=random_subs, **params)
        leaf1 = Icosahedron2(shader, subdivisions=random_subs, **params)
        leaf2 = Icosahedron2(shader, subdivisions=random_subs, **params)
        leaf3 = Icosahedron2(shader, subdivisions=random_subs, **params)

    mainLeaf_textured = Textured(mainLeaf, texture_sampler=leaf_texture)
    leaf1_textured = Textured(leaf1, texture_sampler=leaf_texture)
    leaf2_textured = Textured(leaf2, texture_sampler=leaf_texture)
    leaf3_textured = Textured(leaf3, texture_sampler=leaf_texture)

    return mainLeaf_textured, leaf1_textured, leaf2_textured, leaf3_textured


def treeGenerator(x, y, z, mainTrunk_textured, firstBranch_textured, secondBranch_textured, thirdBranch_textured, mainLeaf_textured, leaf1_textured, leaf2_textured, leaf3_textured, **params):
    """ create a tree trunk """

    phi = np.random.uniform(20., 70.)
    random_size = np.random.uniform(1.5, 3)

    transform_mainLeaf1 = Node(
        transform=translate(0, 12, 0)@scale(random_size, random_size, random_size))
    transform_mainLeaf1.add(mainLeaf_textured)

    transform_mainLeaf2 = Node(
        transform=translate(2, 12, 0))
    transform_mainLeaf2.add(mainLeaf_textured)

    transform_firstBranch = Node(transform=translate(
        0, 5, 0)@rotate((0., 0., 1.), phi))
    transform_leaf1 = Node(transform=translate(0, 5, 0)@scale(1.5, 1.5, 1.5))
    transform_leaf1.add(leaf1_textured)
    transform_firstBranch.add(firstBranch_textured, transform_leaf1)

    transform_secondBranch = Node(transform=translate(
        0, 3, 0)@rotate((1., 0., 0.), phi))
    transform_leaf2 = Node(transform=translate(0, 5, 0)@scale(1, 1, 1))
    transform_leaf2.add(leaf2_textured)
    transform_secondBranch.add(secondBranch_textured, transform_leaf2)

    transform_thirdBranch = Node(transform=translate(
        0, 6, 0)@rotate((0., 1., 0.), 90.0)@rotate((1., 0., 0.), phi))
    transform_leaf3 = Node(transform=translate(0, 2.5, 0)@scale(0.5, 0.5, 0.5))
    transform_leaf3.add(leaf3_textured)
    transform_thirdBranch.add(thirdBranch_textured, transform_leaf3)

    transform_mainTrunk = Node(transform=translate(x, y, z))
    transform_mainTrunk.add(
        mainTrunk_textured, transform_firstBranch, transform_secondBranch, transform_thirdBranch, transform_mainLeaf1, transform_mainLeaf2)

    return transform_mainTrunk


def forestGenerator(shader, numTrees, x, y, z, **params):
    """ create a forest of trees """

    mainTrunk_textured, firstBranch_textured, secondBranch_textured, thirdBranch_textured = trunkGen(
        shader, **params)

    mainLeaf_textured_0, leaf1_textured_0, leaf2_textured_0, leaf3_textured_0 = leafGen(
        shader, 0, **params)

    mainLeaf_textured_1, leaf1_textured_1, leaf2_textured_1, leaf3_textured_1 = leafGen(
        shader, 1, **params)

    # create numpy array for each tree that i will return
    forest = np.empty(numTrees, dtype=object)
    for i in range(numTrees):
        random_int = np.random.randint(low=0, high=2)

        if random_int == 0:
            forest[i] = treeGenerator(
                x, y, z,   mainTrunk_textured, firstBranch_textured, secondBranch_textured, thirdBranch_textured, mainLeaf_textured_0, leaf1_textured_0, leaf2_textured_0, leaf3_textured_0,
                red_tint_factor=round(random.uniform(0, 0.2), 1), **params)

        elif random_int == 1:
            forest[i] = treeGenerator(
                x, y, z,   mainTrunk_textured, firstBranch_textured, secondBranch_textured, thirdBranch_textured, mainLeaf_textured_1, leaf1_textured_1, leaf2_textured_1, leaf3_textured_1,
                red_tint_factor=round(random.uniform(0, 0.2), 1), **params)

    return forest


def spherical_mapping(vertex):
    x, y, z = vertex
    r = np.sqrt(x * x + y * y + z * z)
    theta = np.arccos(z / r) / np.pi  # Latitude (0 <= theta <= 1)
    phi = (np.arctan2(y, x) / (2 * np.pi) + 0.5)  # Longitude (0 <= phi <= 1)
    return phi, theta
