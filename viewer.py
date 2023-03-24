from tree import *
from skybox import *
from texture import *
import sys                          # for system arguments

# External, non built-in modules
import OpenGL.GL as GL              # standard Python OpenGL wrapper
import numpy as np                  # all matrix manipulations & OpenGL args
from core import *
from water import FftWater

# -------------- main program and scene setup --------------------------------


def main():
    """ create a window, add scene objects, then run rendering loop """
    viewer = Viewer()

    # load shaders
    shader = Shader('./shaders/phong.vert', './shaders/phong.frag')
    skyboxShader = Shader('./shaders/skybox.vert', './shaders/skybox.frag')


    # # Add scene objects
    
    # viewer.add(heightMapTerrain(shader, './ress/heightmap.png', light_dir=(0, -1, 4)))
    viewer.add(WaterTerrain(shader, light_dir=(0, -1, 0.7)))


    # start rendering loop
    viewer.run()


if __name__ == '__main__':
    main()
