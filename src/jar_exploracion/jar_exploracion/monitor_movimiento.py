"""Compara desplazamientos relativos sin mezclar los marcos de sus posiciones."""

import math


class DetectorDeslizamiento:
    """Detecta avance estimado con chasis inmóvil durante una orden sostenida."""

    def __init__(self, ventana_s=3.0):
        self.ventana_s = ventana_s
        self.reiniciar()

    def reiniciar(self):
        """Descarta la ventana después de una pausa o una nueva localización."""
        self.inicio = None
        self.ultimo_tiempo = None

    def actualizar(self, tiempo, real, estimada, velocidad):
        """Devuelve True si en la ventana casi no hay movimiento real."""
        if velocidad < 0.05:
            self.reiniciar()
            return False
        if (self.inicio is None or self.ultimo_tiempo is None
                or tiempo < self.ultimo_tiempo or tiempo - self.ultimo_tiempo > 0.5):
            self.inicio = (tiempo, real, estimada)
        self.ultimo_tiempo = tiempo
        t0, real0, estimada0 = self.inicio
        if tiempo - t0 < self.ventana_s:
            return False
        avance_real = math.dist(real, real0)
        avance_estimado = math.dist(estimada, estimada0)
        self.inicio = (tiempo, real, estimada)
        return avance_real < 0.02 and avance_estimado > 0.08
