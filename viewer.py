from tree import *
from skybox import *
from texture import *
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
    shader = Shader('./shaders/phong.vert', './shaders/phong.frag')
    skyboxShader = Shader('./shaders/skybox.vert', './shaders/skybox.frag')

    viewer.add(heightMapTerrain(shader, './ress/heightmap.png', light_dir=(1, 0, 0)))
    viewer.add(SkyBoxTexture(skyboxShader, np.array(['./ress/skybox/xpos.png', './ress/skybox/xneg.png',
               './ress/skybox/ypos.png', './ress/skybox/yneg.png', './ress/skybox/zpos.png', './ress/skybox/zneg.png'])))

    # start rendering loop
    viewer.run()


if __name__ == '__main__':
    main()
