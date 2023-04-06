from core import load, Node
from transform import *


class Smaug(Node):
    def __init__(self, shader):
        super().__init__()

        # Charger le modèle
        Smaug = load("./dragon/Smaug/smaug.obj", shader, light_dir=(1, 0, 0))

        transform_size = Node(transform=translate(
            205, 135, 205)@rotate((0., 1., 0.), 30.0))
        transform_size.add(Smaug[0])

        # Ajouter le modèle chargé à la liste des enfants
        self.add(transform_size)
