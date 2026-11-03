"""Geometría del regreso a la largada, sin ROS (se testea en test/test_regreso.py)."""
import heapq
import math

import numpy as np
from scipy import ndimage

UMBRAL_OCUPADO = 65


def distancia_a_paredes(grilla: np.ndarray, resolucion: float) -> np.ndarray:
    """Para cada celda, distancia en metros a la celda ocupada más cercana. Las celdas
    desconocidas (-1) cuentan como ocupadas: por ahí no se puede asegurar que haya paso."""
    bloqueada = (grilla >= UMBRAL_OCUPADO) | (grilla < 0)
    return ndimage.distance_transform_edt(~bloqueada) * resolucion


def campo_hacia(distancias: np.ndarray, resolucion: float, origen_x: float, origen_y: float,
                objetivo, margen: float) -> np.ndarray:
    """Para cada celda, cuántos metros hay que RECORRER por el laberinto hasta `objetivo`
    (Dijkstra desde el objetivo, vecindad de 8), pasando solo por celdas a más de `margen` de
    las paredes. inf = no se llega. A diferencia de la distancia en línea recta, no tiene
    mínimos locales: bajando por este campo nunca se queda atrapado contra una pared."""
    alto, ancho = distancias.shape
    campo = np.full((alto, ancho), np.inf)
    f0 = int((objetivo[1] - origen_y) / resolucion)
    c0 = int((objetivo[0] - origen_x) / resolucion)
    if not (0 <= f0 < alto and 0 <= c0 < ancho):
        return campo
    transitable = distancias >= margen
    diagonal = math.sqrt(2.0) * resolucion
    vecinos = [(-1, 0, resolucion), (1, 0, resolucion), (0, -1, resolucion), (0, 1, resolucion),
               (-1, -1, diagonal), (-1, 1, diagonal), (1, -1, diagonal), (1, 1, diagonal)]
    campo[f0, c0] = 0.0
    pendientes = [(0.0, f0, c0)]
    while pendientes:
        d, f, c = heapq.heappop(pendientes)
        if d > campo[f, c]:
            continue
        for df, dc, costo in vecinos:
            nf, nc = f + df, c + dc
            if 0 <= nf < alto and 0 <= nc < ancho and transitable[nf, nc]:
                nd = d + costo
                if nd < campo[nf, nc]:
                    campo[nf, nc] = nd
                    heapq.heappush(pendientes, (nd, nf, nc))
    return campo


def valor_cercano(campo: np.ndarray, resolucion: float, origen_x: float, origen_y: float,
                  x: float, y: float, radio: float) -> float:
    """Menor valor del campo a menos de `radio` de (x, y): el robot puede estar en una celda
    pegada a una pared que el campo no cubre (por el margen)."""
    alto, ancho = campo.shape
    f = int((y - origen_y) / resolucion)
    c = int((x - origen_x) / resolucion)
    r = max(0, int(radio / resolucion))
    ventana = campo[max(0, f - r):min(alto, f + r + 1), max(0, c - r):min(ancho, c + r + 1)]
    return float(ventana.min()) if ventana.size else math.inf


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
