
from core import *
from transform import *
from fluid import FluidTerrain


class positionFluid(Node):
    def __init__(self, shader):
        super().__init__()
        water_uniforms = dict(
            k_ambient=(0., 0.1, 0.2),
            k_shadow=(1, 1, 1),
            k_a=(0., 0.1, 0.4),
            k_d=(0., 0.3, 0.7),
            k_s=(1.0, 1.0, 1.0),
            s=30,
            light_dir=(-2, -1, -2),
            n_repeat_texture=2,
        )
        water = FluidTerrain(shader, dudv_path="ress/watermaps/dudv.png",
                             normal_path="ress/watermaps/normalmap.png", world_height=-2, size=(300, 300), **water_uniforms)
        transform_water = Node(transform=translate(250, 50, 250))
        transform_water.add(water)
        lava_uniforms = dict(
            k_ambient=(0.2, 0.2, 0.2),
            k_shadow=(1, 0, 0),
            k_a=(1.0, 0.2, 0.0),
            k_d=(1.0, 0.6, 0.0),
            k_s=(1.0, 0.8, 0.0),
            s=200,
            light_dir=(-2, -1, -2),
            n_repeat_texture=1,
        )
        lava = FluidTerrain(shader, dudv_path="ress/watermaps/dudv.png",
                            normal_path="ress/lava/lavanormal.jpg", size=(142, 142), **lava_uniforms)
        transform_lava = Node(transform=translate(
            256, 90, 256))
        transform_lava.add(lava)

        self.add(transform_water, transform_lava)
