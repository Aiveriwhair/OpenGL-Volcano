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
from fluid import FluidTerrain


# -------------- main program and scene setup --------------------------------


def main():
    """ create a window, add scene objects, then run rendering loop """
    viewer = Viewer()

    
    # start rendering loop
    viewer.run()


if __name__ == '__main__':
    main()


