from core import *


class particuleGenerator(Node):
    def __init__(self, shader):
        super().__init__()
        self.particules = []
        self.particules.append(PointAnimation(shader, 256, 90, 256, num_particles=30,
                                              point_size=30.0, light_dir=(-2, -1, -2)))
        self.particules.append(PointAnimation(shader, 245, 90, 245, num_particles=30,
                                              point_size=30.0, light_dir=(-2, -1, -2)))

        transform_particules = Node(transform=translate(0, 0, 0))
        transform_particules.add(self.particules[0])
        self.add(transform_particules)

    def key_handler(self, key):
        if key == glfw.KEY_T:
            for part in self.particules:
                part.accelerate()
        if key == glfw.KEY_Y:
            for part in self.particules:
                part.stop()
