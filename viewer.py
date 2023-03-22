import sys                          # for system arguments

# External, non built-in modules
import OpenGL.GL as GL              # standard Python OpenGL wrapper
import numpy as np                  # all matrix manipulations & OpenGL args
from core import *


# -------------- main program and scene setup --------------------------------
def main():
    """ create a window, add scene objects, then run rendering loop """
    viewer = Viewer()
    
    # load shaders
    shader = Shader('./shaders/phong.vert','./shaders/phong.frag')


    # Add scene objects
    # viewer.add(randomTerrain(shader, 100, 100))
    # viewer.add(circularTerrain(shader))
    viewer.add(heightMapTerrain(shader, './ress/heightmap.png'))
    # viewer.add(Cube(shader, light_dir=(0, -1, 0), K_d=(.6,.7,.8), s=10))

    # start rendering loop
    viewer.run()


if __name__ == '__main__':
    main()
