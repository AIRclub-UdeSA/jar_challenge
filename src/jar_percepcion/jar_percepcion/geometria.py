"""Geometría de lidar, cámara y mapa sin dependencias de ROS."""

import math

import numpy as np


def yaw_de_quaternion(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def cuaternion_a_rotacion(x, y, z, w):
    """Matriz de rotación 3x3 equivalente a un quaternion (x, y, z, w)."""
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def puntos_laser_a_pixeles(rangos, angulos, rotacion, traslacion, fx, fy, cx, cy):
    """
    Proyecta los puntos del lidar en la imagen con el modelo pinhole.

    Devuelve (u, v, adelante); u y v no tienen sentido donde adelante es False.
    """
    puntos_laser = np.stack(
        [rangos * np.cos(angulos), rangos * np.sin(angulos), np.zeros_like(rangos)], axis=1)
    puntos_camara = puntos_laser @ rotacion.T + traslacion
    x_cam, y_cam, z_cam = puntos_camara[:, 0], puntos_camara[:, 1], puntos_camara[:, 2]
    adelante = z_cam > 0.01
    with np.errstate(divide='ignore', invalid='ignore'):
        u = fx * x_cam / z_cam + cx
        v = fy * y_cam / z_cam + cy
    return u, v, adelante


def pixeles_en_mascara(u, v, adelante, mascara):
    """Marca los puntos que caen dentro de la imagen sobre un pixel activo."""
    alto, ancho = mascara.shape[:2]
    with np.errstate(invalid='ignore'):
        dentro = adelante & (u >= 0) & (u < ancho) & (v >= 0) & (v < alto)
    activos = np.zeros(len(u), dtype=bool)
    indices = np.flatnonzero(dentro)
    # Las imágenes se indexan [fila, columna] = [v, u].
    activos[indices] = mascara[v[indices].astype(int), u[indices].astype(int)] > 0
    return activos


def agrupar_en_rachas(booleanos):
    """
    Agrupa los índices contiguos en True, tratando el scan como circular.

    [F, F, T, T, T, F, T, T, F] -> [[2, 3, 4], [6, 7]].
    """
    booleanos = np.asarray(booleanos, dtype=bool)
    indices = np.flatnonzero(booleanos)
    if len(indices) == 0:
        return []
    rachas = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
    if (len(rachas) > 1 and rachas[0][0] == 0
            and rachas[-1][-1] == len(booleanos) - 1):
        rachas[0] = np.concatenate([rachas.pop(), rachas[0]])
    return rachas


def scan_a_map(rangos, angulos, tx, ty, yaw):
    """Pasa los puntos polares del lidar al frame map."""
    xs_laser = rangos * np.cos(angulos)
    ys_laser = rangos * np.sin(angulos)
    c, s = math.cos(yaw), math.sin(yaw)
    return tx + c * xs_laser - s * ys_laser, ty + s * xs_laser + c * ys_laser


def probabilidad_en_campo(xs, ys, campo, resolucion, origen_x, origen_y):
    """Lee el campo en cada punto; devuelve (probabilidad, dentro_de_la_grilla)."""
    col = np.floor((xs - origen_x) / resolucion).astype(np.int64)
    fila = np.floor((ys - origen_y) / resolucion).astype(np.int64)
    alto, ancho = campo.shape
    dentro = (fila >= 0) & (fila < alto) & (col >= 0) & (col < ancho)
    probabilidad = np.zeros(len(xs))
    probabilidad[dentro] = campo[fila[dentro], col[dentro]]
    return probabilidad, dentro
