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
from billboard import *


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
        shaderTree, './ress/hm2.png', light_dir=(1, 0, 0), height_factor=0.6, numbertrees=200, red_tint_factor=0.)
    pos_trees = terrain.pos_trees
    # cut the first 15 pos_trees to have two different pos_trees arrays
    pos_trees2_pine = pos_trees[0:]
    pos_trees_oak = pos_trees[:0]

    for i in range(len(pos_trees_oak)):
        viewer.add(forestGenerator(shaderTree, 1,
                                   pos_trees_oak[i][0], pos_trees_oak[i][1], pos_trees_oak[i][2], light_dir=(1, 0, 0))[0])
    viewer.add(terrain)
    viewer.add(SkyBoxTexture(skyboxShader, np.array(['./ress/skybox/xpos.png', './ress/skybox/xneg.png',
               './ress/skybox/ypos.png', './ress/skybox/yneg.png', './ress/skybox/zpos.png', './ress/skybox/zneg.png'])))
    # for i in range(len(pos)):
    # viewer.add(PointAnimation(shaderTree,  pos_trees[i][0], pos_trees[i][1]+8, pos_trees[i][2], './ress/grass.png', num_particles=15,
    #                           point_size=10.0, light_dir=(1, 0, 0)))
    # viewer.add(*load('./drag1.obj', shaderTree,
    #            light_dir=(1, 0, 0), K_d=(.6, .7, .8), s=100))

    # start rendering loop
    viewer.run()


if __name__ == '__main__':
    main()
