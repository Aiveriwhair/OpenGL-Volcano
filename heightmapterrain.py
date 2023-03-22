#!/usr/bin/env python3
import OpenGL.GL as GL              # standard Python OpenGL wrapper
import numpy as np                  # all matrix manipulations & OpenGL args
from core import  Mesh
from texture import Texture, Textured
from transform import calculate_normals, normalized
from PIL import Image, ImageOps



class heightMapTerrain(Textured):
    def __init__(self, shader, texture_path, heightmappath, height_factor=1, **params):
        self.shader = shader
        self.height_factor = height_factor
        self.heightmappath = heightmappath
        self.file = texture_path
        self.attributes = {}
        print ("texture_path", texture_path)
        
        (attributes, index) = self.generateTerrain()
    
        self.color = (1, 1, 1)
        uniforms = dict(
            k_d=np.array((0., .5, .5), dtype=np.float32),
            k_s=np.array((0.5673, 0.5673, 0.5673), dtype=np.float32),
            k_a=np.array((0. , 0.4, 0.4), dtype=np.float32),
            s=60,
            )
    
        # setup plane mesh to be textured
        mesh = Mesh(self.shader, attributes=attributes, index=index,k_a=(0.75,0.75,0.75), k_d=(0.9,0.9,0.9), k_s=(0.2,0.3,0.2), s=16)
    
        if self.file is not None:
        # Création d'une instance de la classe Texture et assignation à l'objet Mesh
            texture = Texture(texture_path, GL.GL_MIRRORED_REPEAT, GL.GL_LINEAR, GL.GL_LINEAR_MIPMAP_LINEAR)
           
        super().__init__(mesh, diffuse_map=texture)


    def draw(self, primitives=GL.GL_TRIANGLES, **uniforms):
    # Dessiner le terrain avec les couleurs calculées à partir de l'image de hauteur
        super().draw(primitives=primitives, global_color=self.color, **uniforms)


    def generateTerrain(self):
        map = Image.open(self.heightmappath).convert('L')
        w,h = map.size

        height_factor = self.height_factor

        # Compute the number of vertices and indices needed
        num_vertices = w * h
        num_indices = (w - 1) * (h - 1) * 6

        # Create arrays to hold the vertices and indices
        position = np.zeros((num_vertices, 3), dtype=np.float32)
        indices = np.zeros(num_indices, dtype=np.uint32)
        color = np.zeros((num_vertices, 3), dtype=np.float32)
        tex_coords = np.zeros((num_vertices, 2), dtype=np.float32)

        # Fill in the vertex positions
        for y in range(h):
            for x in range(w):
                i = y * w + x
                z = map.getpixel((x,y))
                position[i, 0] = x
                position[i, 1] = z * height_factor
                position[i, 2] = y
                color[i] = (z/(255*height_factor), z/(255*height_factor), z/(255*height_factor))
                tex_coords[i] = (x/w, y/h)
            # Fill in the texture coordinates
        for y in range(h):
            for x in range(w):
                i = y * w + x
                tex_coords[i] = (x/w, y/h)

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

        if self.file is not None:
            # Add the texture coordinate attribute to the existing attributes
            attributes = dict(
                position=position,
                normal=calculate_normals(position, indices),
                color=color,
                tex_coord=tex_coords,
                )
            self.attributes['tex_coord'] = attributes['tex_coord']
        else:
            attributes = dict(
                position=position,
                normal=calculate_normals(position, indices),
                color=color,
                )

        return (attributes, indices)
