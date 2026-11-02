"""Geometría del regreso a la largada, sin ROS (se testea en test/test_regreso.py)."""
import math

import numpy as np
from scipy import ndimage

UMBRAL_OCUPADO = 65


def distancia_a_paredes(grilla: np.ndarray, resolucion: float) -> np.ndarray:
    """Para cada celda, distancia en metros a la celda ocupada más cercana. Las celdas
    desconocidas (-1) cuentan como ocupadas: por ahí no se puede asegurar que haya paso."""
    bloqueada = (grilla >= UMBRAL_OCUPADO) | (grilla < 0)
    return ndimage.distance_transform_edt(~bloqueada) * resolucion


def linea_libre(distancias: np.ndarray, resolucion: float, origen_x: float, origen_y: float,
                desde, hasta, margen: float) -> bool:
    """True si el segmento desde -> hasta (en metros, marco `map`) pasa a más de `margen` de
    toda pared, es decir, si el robot puede ir en línea recta sin rozar nada."""
    largo = math.hypot(hasta[0] - desde[0], hasta[1] - desde[1])
    pasos = max(2, int(largo / (resolucion / 2)) + 1)
    xs = np.linspace(desde[0], hasta[0], pasos)
    ys = np.linspace(desde[1], hasta[1], pasos)
    col = ((xs - origen_x) / resolucion).astype(int)
    fila = ((ys - origen_y) / resolucion).astype(int)
    alto, ancho = distancias.shape
    if (col < 0).any() or (col >= ancho).any() or (fila < 0).any() or (fila >= alto).any():
        return False
    return bool((distancias[fila, col] >= margen).all())


def velocidad_hacia(x: float, y: float, yaw: float, objetivo_x: float, objetivo_y: float,
                    velocidad_max: float, ganancia: float):
    """(vx, vy) en el frame del robot para ir derecho al objetivo sin girar (base mecanum):
    proporcional a la distancia y recortada a velocidad_max."""
    dx, dy = objetivo_x - x, objetivo_y - y
    distancia = math.hypot(dx, dy)
    if distancia < 1e-9:
        return 0.0, 0.0
    rapidez = min(velocidad_max, ganancia * distancia)
    c, s = math.cos(yaw), math.sin(yaw)
    # Del frame map al del robot: rotar por -yaw.
    return (c * dx + s * dy) / distancia * rapidez, (-s * dx + c * dy) / distancia * rapidez
