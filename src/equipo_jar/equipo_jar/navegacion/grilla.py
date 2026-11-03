"""Grilla del mapa recibido, con origen rotado y celdas desconocidas ocupadas."""
from dataclasses import dataclass, replace
import math

import numpy as np


@dataclass(frozen=True)
class Grilla:
    resolucion: float
    origen_x: float
    origen_y: float
    origen_yaw: float
    ancho: int
    alto: int
    ocupada: np.ndarray  # bool (alto, ancho); fila = y local, columna = x local

    def a_local(self, x, y):
        dx, dy = x - self.origen_x, y - self.origen_y
        c, s = math.cos(self.origen_yaw), math.sin(self.origen_yaw)
        return c * dx + s * dy, -s * dx + c * dy

    def a_celda(self, x, y):
        u, v = self.a_local(x, y)
        col, fila = math.floor(u / self.resolucion), math.floor(v / self.resolucion)
        if 0 <= fila < self.alto and 0 <= col < self.ancho:
            return fila, col
        return None

    def a_mundo(self, fila, col):
        u, v = (col + 0.5) * self.resolucion, (fila + 0.5) * self.resolucion
        c, s = math.cos(self.origen_yaw), math.sin(self.origen_yaw)
        return self.origen_x + c * u - s * v, self.origen_y + s * u + c * v


def grilla_desde_occupancy(datos, ancho, alto, resolucion, origen_x, origen_y, origen_yaw,
                           umbral_ocupado=50):
    """Construye la grilla con los campos de un OccupancyGrid (no el mensaje)."""
    if ancho <= 0 or alto <= 0 or len(datos) != ancho * alto:
        raise ValueError('Dimensiones del mapa inválidas')
    if not (math.isfinite(resolucion) and resolucion > 0):
        raise ValueError('Resolución del mapa inválida')
    if not all(math.isfinite(v) for v in (origen_x, origen_y, origen_yaw)):
        raise ValueError('Origen del mapa inválido')
    valores = np.asarray(datos, dtype=np.int16).reshape(alto, ancho)
    ocupada = (valores < 0) | (valores >= umbral_ocupado)
    return Grilla(float(resolucion), float(origen_x), float(origen_y), float(origen_yaw),
                  int(ancho), int(alto), ocupada)


def refinar_para_obstaculos(grilla, resolucion_max_m=0.025):
    """Conserva paredes completas y permite registrar retornos LiDAR más finos.

    Refinar después de guardar un retorno como celda gruesa sólo replica el
    obstáculo ensanchado. La capa viva debe nacer en esta misma grilla.
    """
    if not math.isfinite(resolucion_max_m) or resolucion_max_m <= 0:
        raise ValueError('Resolución de navegación inválida')
    factor = max(1, math.ceil(grilla.resolucion / resolucion_max_m - 1e-6))
    if factor == 1:
        return grilla
    return replace(grilla, resolucion=grilla.resolucion / factor,
                   ancho=grilla.ancho * factor, alto=grilla.alto * factor,
                   ocupada=np.repeat(np.repeat(grilla.ocupada, factor, axis=0),
                                     factor, axis=1))
