import sys                          # for system arguments

# External, non built-in modules
import OpenGL.GL as GL              # standard Python OpenGL wrapper
import numpy as np                  # all matrix manipulations & OpenGL args
from core import *
from heightmapterrain import heightMapTerrain

# -------------- Example textured plane class ---------------------------------


class TexturedPlane(Textured):
    """ Simple first textured object """

    def __init__(self, shader, tex_file):
        # prepare texture modes cycling variables for interactive toggling
        self.wraps = cycle([GL.GL_REPEAT, GL.GL_MIRRORED_REPEAT,
                            GL.GL_CLAMP_TO_BORDER, GL.GL_CLAMP_TO_EDGE])
        self.filters = cycle([(GL.GL_NEAREST, GL.GL_NEAREST),
                              (GL.GL_LINEAR, GL.GL_LINEAR),
                              (GL.GL_LINEAR, GL.GL_LINEAR_MIPMAP_LINEAR)])
        self.wrap, self.filter = next(self.wraps), next(self.filters)
        self.file = tex_file

        # setup plane mesh to be textured
        base_coords = ((-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0))
        scaled = 100 * np.array(base_coords, np.float32)
        indices = np.array((0, 1, 2, 0, 2, 3), np.uint32)
        mesh = Mesh(shader, attributes=dict(position=scaled), index=indices)

        # setup & upload texture to GPU, bind it to shader name 'diffuse_map'
        texture = Texture(tex_file, self.wrap, *self.filter)
        super().__init__(mesh, diffuse_map=texture)





# -------------- main program and scene setup --------------------------------
def main():
    """ create a window, add scene objects, then run rendering loop """
    viewer = Viewer()
    
    # load shaders
    shader = Shader('./shaders/phong.vert','./shaders/phong.frag')
    shadertexture = Shader('./shaders/texture.vert','./shaders/texture.frag')

    # Add scene objects
    # viewer.add(randomTerrain(shader, 100, 100))
    # viewer.add(circularTerrain(shader))
    viewer.add(heightMapTerrain(shadertexture, './ress/grass.png', './ress/heightmap.png'))
    # viewer.add(Cube(shader, light_dir=(0, -1, 0), K_d=(.6,.7,.8), s=10))

    # start rendering loop
    viewer.run()


if __name__ == '__main__':
    main()
