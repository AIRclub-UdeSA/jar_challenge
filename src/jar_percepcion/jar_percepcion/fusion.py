"""Fusión cámara + LiDAR (ADR-0006), sin ROS.

El blob rojo da una ventana de rumbos; el LiDAR, dentro de esa ventana, da los
puntos de la cara visible de la caja. El centro de la caja (lo que se puntúa)
queda media caja más allá de la cara, en la dirección en que se la mira.
"""
import math

import numpy as np


def envolver(a):
    return np.arctan2(np.sin(a), np.cos(a))


def rotacion_de_cuaternion(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def rumbo_en_laser(u, fx, cx, r_laser_optico):
    """Rumbo (rad) en el plano del láser del rayo de la columna `u` de la imagen."""
    d = r_laser_optico @ np.array([(u - cx) / fx, 0.0, 1.0])
    return math.atan2(d[1], d[0])


def ventana_angular(u_min, u_max, fx, cx, r_laser_optico, recorte=0.15):
    """(centro, semiancho) de la ventana de rumbos del blob, achicada un `recorte` por lado."""
    a1 = rumbo_en_laser(u_min, fx, cx, r_laser_optico)
    a2 = rumbo_en_laser(u_max + 1, fx, cx, r_laser_optico)
    delta = float(envolver(a2 - a1))
    centro = float(envolver(a1 + delta / 2.0))
    return centro, abs(delta) / 2.0 * (1.0 - recorte)


def puntos_en_ventana(ranges, angle_min, angle_increment, range_min, range_max,
                      centro, semiancho, alcance_max=2.5, salto=0.15, min_puntos=3,
                      fraccion_cerca_min=0.0):
    """Puntos (marco del láser) del objeto más cercano dentro de la ventana, o None.

    Con `fraccion_cerca_min`, también None si ese objeto ocupa menos de esa fracción de los
    rayos de la ventana: el rojo está detrás, asomando al lado de algo más cercano, y no se
    sabe a qué distancia (laberinto_puertas: V1 asomando junto a la punta de una pared quedó
    ubicada en la punta, a 1 m, y se reportó dos veces)."""
    r = np.asarray(ranges, dtype=float)
    ang = angle_min + angle_increment * np.arange(len(r))
    sel = ((np.abs(envolver(ang - centro)) <= semiancho) & np.isfinite(r)
           & (r >= range_min) & (r <= min(range_max, alcance_max)))
    if np.count_nonzero(sel) < min_puntos:
        return None
    r, ang = r[sel], ang[sel]
    cerca = r <= r.min() + salto
    if np.count_nonzero(cerca) < min_puntos:
        return None
    if np.count_nonzero(cerca) < fraccion_cerca_min * len(r):
        return None
    r, ang = r[cerca], ang[cerca]
    return np.column_stack((r * np.cos(ang), r * np.sin(ang)))


def alto_coherente(v_min, v_max, alto_img, fy, distancia, alto_caja, fraccion, margen_px=2):
    """¿El blob (filas v_min..v_max) es tan alto como una caja de `alto_caja` m a `distancia`?

    Uno de menos de `fraccion` del alto esperado está más lejos que el objeto del LiDAR donde se
    lo ubicó: una víctima vista detrás, por un hueco al costado de una caja más cercana
    (laberinto_puertas: V1 a 2.1 m, por el hueco junto a la caja azul de la puerta de A a 0.6 m,
    hizo víctima a la caja azul). El ancho no sirve (lo corta lo que tapa al costado); el alto
    no cambia. Cortado por arriba o por abajo no se puede medir: vale. `fraccion` 0: apagado.
    """
    if fraccion <= 0 or distancia <= 0:
        return True
    if v_min <= margen_px or v_max + 1 >= alto_img - margen_px:
        return True
    return v_max - v_min + 1 >= fraccion * fy * alto_caja / distancia


def transformar(puntos, tx, ty, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    p = np.asarray(puntos, dtype=float)
    return np.column_stack((tx + c * p[:, 0] - s * p[:, 1], ty + s * p[:, 0] + c * p[:, 1]))


def estimar_centro(puntos, origen, profundidad):
    """Centro de la caja: la media de la cara vista corrida `profundidad` alejándose del origen."""
    media = np.mean(np.asarray(puntos, dtype=float), axis=0)
    v = media - np.asarray(origen, dtype=float)
    norma = float(np.hypot(*v))
    if norma < 1e-6:
        return float(media[0]), float(media[1])
    c = media + profundidad * v / norma
    return float(c[0]), float(c[1])
