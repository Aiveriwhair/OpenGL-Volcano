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

    waterShader = Shader("shaders/fluid.vert", "shaders/fluid.frag")

    lava_uniforms = dict(
           k_ambient=(0.2, 0.2, 0.2),
            k_shadow=(1,0,0),
            k_a=(1.0, 0.2, 0.0),
            k_d=(1.0, 0.6, 0.0),
            k_s=(1.0, 0.8, 0.0),
            s=200,
            light_dir=(0, -3, -6),
            n_repeat_texture=1,
        )
    viewer.add(FluidTerrain(waterShader, dudv_path="ress/watermaps/dudv.png", normal_path="ress/lava/lavanormal.jpg", size=(8, 8),**lava_uniforms))

    # water_uniforms = dict(
    #     k_ambient=(0., 0.1, 0.2),
    #     k_shadow=(1,1,1),
    #     k_a=(0., 0.3, 0.6),
    #     k_d=(0., 0.3, 0.7),
    #     k_s=(1.0, 1.0, 1.0),
    #     s=30,
    #     light_dir=(0, -1, -1),
    #     n_repeat_texture=2,
    # )
    # viewer.add(FluidTerrain(waterShader, dudv_path="ress/watermaps/dudv.png", normal_path="ress/watermaps/normalmap.png", world_height=-2, size=(8, 8),**water_uniforms))

    # start rendering loop
    viewer.run()


if __name__ == '__main__':
    main()


