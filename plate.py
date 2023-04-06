from core import load, Node
from transform import *


class Plate(Node):
    def __init__(self, shader):
        super().__init__()

        # Charger le modèle
        plate = load("./island/island.obj", shader, light_dir=(1, 0, 0))

        transform_size = Node(transform=scale(
            80.0, 80.0, 80.0)@translate(3, -2.35, 3.2))
        transform_size.add(plate[0])

        # Ajouter le modèle chargé à la liste des enfants
        self.add(transform_size)
