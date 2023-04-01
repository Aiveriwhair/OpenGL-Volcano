from particle import PointAnimation
from tree import *
from skybox import *
from texture import *
import sys                          # for system arguments

# External, non built-in modules
import OpenGL.GL as GL              # standard Python OpenGL wrapper
import numpy as np                  # all matrix manipulations & OpenGL args
from core import *
from terrain import *
from water import WaterTerrain


# -------------- main program and scene setup --------------------------------


def main():
    """ create a window, add scene objects, then run rendering loop """
    viewer = Viewer()

    waterShader = Shader("shaders/water.vert", "shaders/water.frag")

    viewer.add(WaterTerrain(waterShader))

    # start rendering loop
    viewer.run()


if __name__ == '__main__':
    main()
