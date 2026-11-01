"""Planificador de la exploración: qué objetivos visitar y en qué orden.

La misión (nodo_equipo) lo usa así:

    exploracion = Exploracion(mapa, logger)
    exploracion.planificar(pose_inicial)       # una vez, al recibir /map
    exploracion.siguiente(pose)                # (x, y, yaw) en `map`, o None
    exploracion.objetivo_terminado(exito)      # cuando Nav2 termina ese objetivo
    exploracion.distancia_camino(a, b)         # metros por el laberinto, o None

TODO: calcular la ruta de cobertura. Por ahora no hay objetivos.
"""


class Exploracion:

    def __init__(self, mapa, logger):
        self.mapa = mapa
        self.logger = logger
        self.pendientes = []

    def planificar(self, pose):
        self.pendientes = []

    def siguiente(self, pose):
        return self.pendientes[0] if self.pendientes else None

    def objetivo_terminado(self, exito):
        if self.pendientes:
            self.pendientes.pop(0)

    def distancia_camino(self, a, b):
        return None
