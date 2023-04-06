from particulesGen import particuleGenerator
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
from billboard import *
from optimize_trees import *
from smaug import *
from plate import *
from eruption import *
from generatefluid import *

# -------------- main program and scene setup --------------------------------


def main():
    """ create a window, add scene objects, then run rendering loop """
    viewer = Viewer()

    # load shaders
    skyboxShader = Shader('./shaders/skybox.vert', './shaders/skybox.frag')
    generalShader = Shader('./shaders/general.vert', './shaders/general.frag')
    objShader = Shader('./shaders/obj.vert', './shaders/obj.frag')
    fluidShader = Shader('./shaders/fluid.vert', './shaders/fluid.frag')
    partShader = Shader('./shaders/part.vert', './shaders/part.frag')

    terrain = heightMapTerrain(
        generalShader, './ress/j2.png', light_dir=(-2, -1, -2), height_factor=0.6, numbertrees=2000, red_tint_factor=0.)
    pos_trees = terrain.pos_trees

    # cut the first 15 pos_trees to have two different pos_trees arrays
    pos_trees2_pine = pos_trees[15:]
    pos_trees_oak = pos_trees[:15]

    viewer.add(PineTrees(generalShader, pos_trees2_pine,
               './ress/wood.png', './ress/pine.jpg', light_dir=(-2, -1, -2)))

    for i in range(len(pos_trees_oak)):
        viewer.add(forestGenerator(generalShader, 1,
                                   pos_trees_oak[i][0], pos_trees_oak[i][1], pos_trees_oak[i][2], light_dir=(-2, -1, -2))[0])
    viewer.add(terrain)
    viewer.add(SkyBoxTexture(skyboxShader, np.array(['./ress/skybox/xpos.png', './ress/skybox/xneg.png',
               './ress/skybox/ypos.png', './ress/skybox/yneg.png', './ress/skybox/zpos.png', './ress/skybox/zneg.png'])))
    for i in range(len(pos_trees_oak)):
        viewer.add(BillboardAnimation(generalShader,  pos_trees_oak[i][0], pos_trees_oak[i][1]+8, pos_trees_oak[i][2], './ress/grass.png', num_particles=15,
                                      point_size=0.5, light_dir=(-2, -1, -2)))
    viewer.add(Smaug(objShader))
    viewer.add(Plate(objShader))
    viewer.add(Eruption(objShader))
    viewer.add(positionFluid(fluidShader))
    viewer.add(particuleGenerator(partShader))
    # start rendering loop
    viewer.run()


if __name__ == '__main__':
    main()
