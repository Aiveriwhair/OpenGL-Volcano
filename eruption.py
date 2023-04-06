import numpy as np
import random
from core import load, Node, KeyFrameControlNode
from transform import *
import OpenGL.GL as GL


class Eruption(Node):
    def __init__(self, shader):
        super().__init__()

        fireball = load("./fireball/rock.obj", shader, light_dir=(1, 0, 0))

        translate_keyFrames, rotate_keyFrames, scale_keyFrames = self.generate_keyframes()

        rock = KeyFrameControlNode(
            translate_keyFrames, rotate_keyFrames, scale_keyFrames, duration=10)

        rock.add(fireball[0])
        self.add(rock)

    def generate_keyframes(self):
        trans_keys = {
            0: vec(250, 15, 300),
            3: vec(300, 300, 300),
            5: vec(320, 320, 300),
            7: vec(350, 200, 300),
            10: vec(400, 110, 300),
        }

        rot_keys = {
            0: quaternion(),
            5: quaternion_from_euler(-200, 300, 150),
            7: quaternion_from_euler(-300, 400, 200),
            10: quaternion(),
        }
        scale_keys = {
            0: vec(0.5, 0.5, 0.5),
            5: vec(0.5, 0.5, 0.5),
            7: vec(0.5, 0.5, 0.5),
            10: vec(0.5, 0.5, 0.5),
        }

        return trans_keys, rot_keys, scale_keys
