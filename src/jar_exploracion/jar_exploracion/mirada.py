"""Giro de 360° por pasos para que la cámara frontal mire alrededor, sin ROS."""

import math

import numpy as np


def normalizar(angulo):
    return math.atan2(math.sin(angulo), math.cos(angulo))


def eje_mas_cercano(yaw, eje=math.pi / 2):
    """Orientación múltiplo de `eje` más cercana: alineada con los pasillos del mapa."""
    return normalizar(round(yaw / eje) * eje)


def radio_de_giro(largo, ancho, margen):
    """Radio del círculo que barre la huella rectangular al girar en el lugar."""
    return math.hypot(largo / 2.0, ancho / 2.0) + margen


def giro_seguro(impactos, distancia_a_pared, radio):
    """
    Indica si el robot puede girar en el lugar.

    `impactos` son puntos del scan en base_footprint (Nx2); `distancia_a_pared`
    es la distancia del centro del robot al obstáculo más cercano del mapa.
    """
    if distancia_a_pared is None or distancia_a_pared < radio:
        return False
    puntos = np.asarray(impactos, dtype=float).reshape(-1, 2)
    return not len(puntos) or float(np.min(np.hypot(puntos[:, 0], puntos[:, 1]))) >= radio


class MiradaPorPasos:
    """
    Gira 360° en `pasos` tramos iguales y se detiene `pausa_s` en cada uno.

    En las pausas la imagen sale nítida y detector_scan no la descarta por
    girar rápido. El último tramo termina en la orientación alineada con los
    ejes del mapa más cercana a la inicial: con una orientación intermedia la
    huella rectangular no entra en pasos que alineada sí entra. Si se agota
    `timeout_s`, salta directo a alinearse en vez de quedar en cualquier ángulo.
    """

    def __init__(self, pasos=8, pausa_s=0.6, velocidad=0.6, tolerancia=0.02,
                 timeout_s=60.0, timeout_alineacion_s=30.0):
        self.pasos = pasos
        self.pausa_s = pausa_s
        self.velocidad = velocidad
        self.tolerancia = tolerancia
        self.timeout_s = timeout_s
        self.timeout_alineacion_s = timeout_alineacion_s
        self.activa = False
        self.resultado = None

    def iniciar(self, yaw, ahora):
        self.activa = True
        self.resultado = None
        self.yaw_inicial = yaw
        self.yaw_final = eje_mas_cercano(yaw)
        self.paso = 1
        self.inicio = ahora
        self.fin_pausa = None
        self.por_tiempo = False

    def objetivo(self):
        if self.paso >= self.pasos:
            return self.yaw_final
        return normalizar(self.yaw_inicial + 2.0 * math.pi * self.paso / self.pasos)

    def terminar(self, resultado):
        self.activa = False
        self.resultado = resultado
        return 0.0

    def actualizar(self, yaw, ahora):
        """Devuelve la velocidad angular a ordenar; desactiva la mirada al terminar."""
        if not self.activa:
            return 0.0
        transcurrido = ahora - self.inicio
        if transcurrido > self.timeout_s + self.timeout_alineacion_s:
            return self.terminar('abandonada')
        if transcurrido > self.timeout_s and not self.por_tiempo:
            # Sin tiempo para la vuelta completa: alinearse antes de seguir explorando.
            self.por_tiempo = True
            self.paso = self.pasos
            self.fin_pausa = None
        if self.fin_pausa is not None:
            if ahora < self.fin_pausa:
                return 0.0
            self.fin_pausa = None
            self.paso += 1
            if self.paso > self.pasos:
                return self.terminar('alineada_por_tiempo' if self.por_tiempo else 'completa')
        error = normalizar(self.objetivo() - yaw)
        if abs(error) <= self.tolerancia:
            if self.paso >= self.pasos:
                return self.terminar('alineada_por_tiempo' if self.por_tiempo else 'completa')
            self.fin_pausa = ahora + self.pausa_s
            return 0.0
        # Siempre antihorario mientras falte más de media vuelta del paso.
        if self.paso < self.pasos and error < -math.pi / 2:
            error += 2.0 * math.pi
        rapidez = min(self.velocidad, max(0.15, 1.5 * abs(error)))
        return math.copysign(rapidez, error)
