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


# -------------- main program and scene setup --------------------------------


def main():
    """ create a window, add scene objects, then run rendering loop """
    viewer = Viewer()

    # load shaders
    shader = Shader('./shaders/phong.vert', './shaders/phong.frag')
    skyboxShader = Shader('./shaders/skybox.vert', './shaders/skybox.frag')

    shaderTree = Shader('./shaders/tree.vert', './shaders/tree.frag')

    # forest = forestGenerator(shaderTree, 100, 0, 0, 0, light_dir=(1, 0, 0))
    # for i in range(len(forest)):
    #     viewer.add(forest[i])

    terrain = heightMapTerrain(
        shaderTree, './ress/hm2.png', light_dir=(1, 0, 0), height_factor=0.6, numbertrees=20, red_tint_factor=0.)
    pos = terrain.pos_trees
    for i in range(len(pos)):
        viewer.add(forestGenerator(shaderTree, 1,
                   pos[i][0], pos[i][1], pos[i][2], light_dir=(1, 0, 0))[0])
    viewer.add(terrain)
    viewer.add(SkyBoxTexture(skyboxShader, np.array(['./ress/skybox/xpos.png', './ress/skybox/xneg.png',
               './ress/skybox/ypos.png', './ress/skybox/yneg.png', './ress/skybox/zpos.png', './ress/skybox/zneg.png'])))
    viewer.add(PointAnimation(shaderTree, 0, 0, 0, './ress/grass.png', num_particles=1,
               point_size=10.0, light_dir=(1, 0, 0)))


    # start rendering loop
    viewer.run()


if __name__ == '__main__':
    main()
