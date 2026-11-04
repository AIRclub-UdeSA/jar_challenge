"""Segmentación del rojo en la imagen (ADR-0006), sin ROS.

El rojo cae en los dos extremos del tono de OpenCV (0-179), así que se usan dos
rangos. Ningún distractor es rojo (§6.3 del reglamento).
"""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ConfigColor:
    h_bajo_1: int = 0
    h_alto_1: int = 10
    h_bajo_2: int = 170
    h_alto_2: int = 179
    s_min: int = 120
    v_min: int = 70
    area_min_px: int = 150
    margen_borde_px: int = 2       # blobs cortados por el borde lateral se descartan
    kernel_px: int = 3
    fila_min_px: int = 0           # se ignora el rojo de las filas 0..fila_min_px-1 (arriba)


@dataclass
class Blob:
    u_min: int
    u_max: int
    v_min: int
    v_max: int
    area: int


def mascara_roja(rgb, cfg):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    m1 = cv2.inRange(hsv, (cfg.h_bajo_1, cfg.s_min, cfg.v_min), (cfg.h_alto_1, 255, 255))
    m2 = cv2.inRange(hsv, (cfg.h_bajo_2, cfg.s_min, cfg.v_min), (cfg.h_alto_2, 255, 255))
    mascara = cv2.bitwise_or(m1, m2)
    if cfg.kernel_px > 1:
        kernel = np.ones((cfg.kernel_px, cfg.kernel_px), np.uint8)
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)
    mascara[:cfg.fila_min_px, :] = 0
    return mascara


def toca_borde(u_min, u_max, ancho, cfg):
    """¿El blob (columnas u_min..u_max) está cortado por el borde lateral de la imagen?"""
    return u_min <= cfg.margen_borde_px or u_max + 1 >= ancho - cfg.margen_borde_px


def segmentar_rojo(rgb, cfg, descartar_borde=True):
    """Blobs rojos de una imagen RGB (H×W×3, uint8), del más grande al más chico.

    Con `descartar_borde=False` también los cortados por el borde lateral: no sirven para
    ubicar la caja, pero sí para saber que en esa dirección hay rojo (una caja roja de cerca
    ocupa toda la imagen y toca los bordes).
    """
    mascara = mascara_roja(rgb, cfg)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mascara, connectivity=8)
    ancho = rgb.shape[1]
    blobs = []
    for k in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[k])
        if area < cfg.area_min_px:
            continue
        if descartar_borde and toca_borde(x, x + w - 1, ancho, cfg):
            continue
        blobs.append(Blob(x, x + w - 1, y, y + h - 1, area))
    return sorted(blobs, key=lambda b: -b.area)
