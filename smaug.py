from core import load, Node


class Smaug(Node):
    def __init__(self, shader):
        super().__init__()

        # Charger le modèle
        Smaug = load("./dragon/Smaug/smaug.obj", shader, light_dir=(1, 0, 0))

        # Ajouter le modèle chargé à la liste des enfants
        self.add(Smaug[0])
