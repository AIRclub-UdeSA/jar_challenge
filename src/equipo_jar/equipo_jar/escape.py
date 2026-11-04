"""Salir de una traba: alejarse despacio de lo que tiene al lado, usando solo el lidar.

Pegado a una pared, Nav2 no se destraba solo: RPP ve la huella del robot sobre un obstáculo y no
avanza, y Spin y BackUp abortan por el mismo motivo. Acá el robot usa que es mecanum: prueba varias
direcciones, también de costado, y va hacia la que más lo aleja de los obstáculos.

Es lógica pura (sin ROS), para poder probarla con pytest. Todo está en el marco del robot.
"""
import math
from typing import NamedTuple

import numpy as np


class Cuerpo(NamedTuple):
    """Rectángulo del robot respecto de base_footprint (m): el centro no es el del rectángulo."""

    frente: float
    atras: float
    semiancho: float


class Direccion(NamedTuple):
    ux: float           # dirección unitaria en el marco del robot
    uy: float
    libre: float        # cuánto se puede recorrer sin acercarse de más a nada (m)
    avance: float       # en qué punto de ese recorrido el robot queda más despejado (m)
    despeje: float      # distancia a lo más cercano en ese punto (m)


def puntos_en_robot(rangos, angulo_min, incremento, rango_min, rango_max, lidar):
    """Pasa un scan a puntos (x, y) en el marco del robot, descartando las lecturas inválidas.

    `lidar` es (tx, ty, r00, r01, r10, r11): la posición del lidar en el robot y la parte xy de su
    rotación. Se usa la rotación entera porque el lidar simulado está montado al revés.
    """
    rangos = np.asarray(rangos, dtype=float)
    validos = np.isfinite(rangos) & (rangos > rango_min) & (rangos < rango_max)
    angulos = angulo_min + np.nonzero(validos)[0] * incremento
    lx, ly = rangos[validos] * np.cos(angulos), rangos[validos] * np.sin(angulos)
    tx, ty, r00, r01, r10, r11 = lidar
    return tx + r00 * lx + r01 * ly, ty + r10 * lx + r11 * ly


def distancia_al_cuerpo(px, py, cuerpo):
    """Distancia de cada punto al rectángulo del robot (0 si cae adentro)."""
    dx = np.maximum(np.maximum(-cuerpo.atras - px, px - cuerpo.frente), 0.0)
    dy = np.maximum(np.abs(py) - cuerpo.semiancho, 0.0)
    return np.hypot(dx, dy)


def despeje_actual(px, py, cuerpo):
    """Distancia a lo más cercano. Un punto dentro del cuerpo es el propio robot: no cuenta."""
    d = distancia_al_cuerpo(px, py, cuerpo)
    d = d[d > 0.0]
    return float(d.min()) if d.size else math.inf


def _barrido(px, py, ux, uy, cuerpo, margen, alcance, paso):
    """Simula el avance en (ux, uy): da las distancias, el despeje en cada una y lo libre."""
    t = np.arange(paso, alcance + paso / 2, paso)
    inicial = distancia_al_cuerpo(px, py, cuerpo)[:, None]
    # Avanzar el robot equivale a mover los puntos para el otro lado.
    d = distancia_al_cuerpo(px[:, None] - t * ux, py[:, None] - t * uy, cuerpo)
    # Lo que ya está más cerca que el margen no bloquea, salvo que el movimiento lo acerque más.
    bloqueado = (d < np.minimum(margen, inicial - 0.005)).any(axis=0)
    n = int(bloqueado.argmax()) if bloqueado.any() else len(t)
    return t, d.min(axis=0), float(t[n - 1]) if n > 0 else 0.0


def recorrido_libre(px, py, ux, uy, cuerpo, margen, alcance, paso=0.01):
    """Cuánto puede avanzar el robot en (ux, uy) sin acercarse de más a nada."""
    if px.size == 0:
        return alcance
    return _barrido(px, py, ux, uy, cuerpo, margen, alcance, paso)[2]


def elegir_direccion(px, py, cuerpo, margen, alcance, direcciones=16, paso=0.01):
    """La dirección que más despeja al robot; a igual despeje, la que deja recorrer más."""
    mejor, puntaje_mejor = None, None
    for k in range(direcciones):
        angulo = 2.0 * math.pi * k / direcciones
        ux, uy = math.cos(angulo), math.sin(angulo)
        if px.size == 0:
            return Direccion(ux, uy, alcance, alcance, math.inf)
        t, despejes, libre = _barrido(px, py, ux, uy, cuerpo, margen, alcance, paso)
        alcanzables = t <= libre + 1e-9
        if not alcanzables.any():
            continue
        i = int(despejes[alcanzables].argmax())
        puntaje = (round(float(despejes[i]), 3), libre)
        if puntaje_mejor is None or puntaje > puntaje_mejor:
            puntaje_mejor = puntaje
            mejor = Direccion(ux, uy, libre, float(t[i]), float(despejes[i]))
    return mejor


class DetectorTraba:
    """Avisa si el robot está quieto o si gira sin ir a ningún lado.

    Quieto: `tiempo_s` sin moverse `radio_m` ni girar `angulo_rad`. Sin avanzar: `tiempo_avance_s`
    sin alejarse `radio_avance_m`, aunque gire. Hace falta el segundo caso porque los giros de
    recuperación de Nav2 cuentan como movimiento y pueden tapar una traba.
    """

    def __init__(self, radio_m, angulo_rad, tiempo_s, radio_avance_m, tiempo_avance_s):
        self.radio_m = radio_m
        self.angulo_rad = angulo_rad
        self.tiempo_s = tiempo_s
        self.radio_avance_m = radio_avance_m
        self.tiempo_avance_s = tiempo_avance_s
        self.ancla = None
        self.t_ancla = 0.0
        self.ancla_avance = None
        self.t_ancla_avance = 0.0

    def reiniciar(self):
        self.ancla = None
        self.ancla_avance = None

    def actualizar(self, x, y, yaw, ahora):
        if self.ancla_avance is None or math.hypot(
                x - self.ancla_avance[0], y - self.ancla_avance[1]) > self.radio_avance_m:
            self.ancla_avance, self.t_ancla_avance = (x, y), ahora
        sin_avanzar = ahora - self.t_ancla_avance > self.tiempo_avance_s
        return self._quieto(x, y, yaw, ahora) or sin_avanzar

    def _quieto(self, x, y, yaw, ahora):
        if self.ancla is None:
            self.ancla, self.t_ancla = (x, y, yaw), ahora
            return False
        giro = math.atan2(math.sin(yaw - self.ancla[2]), math.cos(yaw - self.ancla[2]))
        if math.hypot(x - self.ancla[0], y - self.ancla[1]) > self.radio_m \
                or abs(giro) > self.angulo_rad:
            self.ancla, self.t_ancla = (x, y, yaw), ahora
            return False
        return ahora - self.t_ancla > self.tiempo_s


class Escape:
    """Una maniobra de escape: da la velocidad de cada ciclo hasta quedar despejado.

    Mira el lidar en cada ciclo, no solo al arrancar: en la zona de la traba el mapa y la pose
    no son confiables a escala de centímetros.
    """

    def __init__(self, cuerpo, velocidad, despeje_objetivo, margen, alcance, duracion_max, ahora):
        self.cuerpo = cuerpo
        self.velocidad = velocidad
        self.despeje_objetivo = despeje_objetivo
        self.margen = margen
        self.alcance = alcance
        self.t_limite = ahora + duracion_max
        self.direccion = None
        self.t_tramo = 0.0

    def comando(self, px, py, ahora):
        """Devuelve (vx, vy) en el marco del robot, o None cuando la maniobra terminó."""
        despeje = despeje_actual(px, py, self.cuerpo)
        if despeje >= self.despeje_objetivo or ahora >= self.t_limite:
            return None
        bloqueada = self.direccion is not None and recorrido_libre(
            px, py, self.direccion.ux, self.direccion.uy, self.cuerpo, self.margen,
            self.alcance) < self.velocidad * 0.2
        if self.direccion is None or bloqueada or ahora >= self.t_tramo:
            self.direccion = elegir_direccion(px, py, self.cuerpo, self.margen, self.alcance)
            if self.direccion is None or self.direccion.despeje <= despeje + 0.005:
                return None
            self.t_tramo = ahora + self.direccion.avance / self.velocidad
        return self.velocidad * self.direccion.ux, self.velocidad * self.direccion.uy
