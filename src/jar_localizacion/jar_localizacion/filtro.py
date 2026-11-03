"""Filtro de partículas (Monte Carlo Localization) sin dependencias de ROS."""

import math

import numpy as np
from scipy import ndimage

UMBRAL_OCUPADO = 65


def normalizar_angulo(angulo):
    """Lleva un ángulo cualquiera al rango (-pi, pi]."""
    return math.atan2(math.sin(angulo), math.cos(angulo))


def invertir_pose(pose):
    """Inversa de una pose 2D (x, y, theta) como transformada rígida."""
    x, y, theta = pose
    c, s = math.cos(-theta), math.sin(-theta)
    return (-(c * x - s * y), -(s * x + c * y), -theta)


def componer_poses(pose_a, pose_b):
    """Aplica primero pose_b y después pose_a."""
    xa, ya, ta = pose_a
    xb, yb, tb = pose_b
    c, s = math.cos(ta), math.sin(ta)
    return (xa + c * xb - s * yb, ya + s * xb + c * yb, normalizar_angulo(ta + tb))


def campo_de_probabilidad(grid, resolucion, sigma):
    """Probabilidad 0-100 de que un rayo termine en cada celda de la grilla."""
    obstaculos = np.asarray(grid) >= UMBRAL_OCUPADO
    if not obstaculos.any():
        return np.zeros(obstaculos.shape)
    distancia = ndimage.distance_transform_edt(~obstaculos) * resolucion
    return 100.0 * np.exp(-distancia ** 2 / (2.0 * sigma ** 2))


def sembrar_particulas(x, y, theta, dispersion_xy, dispersion_theta, n, rng=np.random):
    """Nube gaussiana de n partículas (x, y, theta, peso) alrededor de una pose."""
    return np.stack([
        rng.normal(x, dispersion_xy, n),
        rng.normal(y, dispersion_xy, n),
        rng.normal(theta, dispersion_theta, n),
        np.full(n, 1.0 / n),
    ], axis=1)


def mover_particulas(particulas, dx, dy, dtheta, theta0, gammas,
                     escala_avance=1.0, escala_giro=1.0, rng=np.random):
    """
    Modelo de movimiento odométrico mecanum con ruido por partícula.

    El desplazamiento (dx, dy) llega en el frame odom y se expresa en el
    frame del robot al inicio del paso. Las gammas son varianzas.
    """
    g1, g2, g3, g4, g5, g6 = gammas
    c, s = math.cos(theta0), math.sin(theta0)
    avance = (dx * c + dy * s) * escala_avance
    costado = (-dx * s + dy * c) * escala_avance
    giro = dtheta * escala_giro

    std_x = math.sqrt(g1 * avance ** 2 + g2 * costado ** 2 + g3 * giro ** 2)
    std_y = math.sqrt(g2 * avance ** 2 + g1 * costado ** 2 + g3 * giro ** 2)
    std_rot = math.sqrt(g4 * avance ** 2 + g5 * costado ** 2 + g6 * giro ** 2)

    n = len(particulas)
    avance_r = avance - rng.normal(0.0, std_x, n)
    costado_r = costado - rng.normal(0.0, std_y, n)
    giro_r = giro - rng.normal(0.0, std_rot, n)

    theta = particulas[:, 2]
    nuevas = particulas.copy()
    nuevas[:, 0] += avance_r * np.cos(theta) - costado_r * np.sin(theta)
    nuevas[:, 1] += avance_r * np.sin(theta) + costado_r * np.cos(theta)
    nuevas[:, 2] = np.arctan2(np.sin(theta + giro_r), np.cos(theta + giro_r))
    return nuevas


def probabilidades_scan(particulas, puntos, campo, resolucion, origen_x, origen_y):
    """
    Probabilidad del campo (1-100) donde cae cada punto, para cada partícula.

    `puntos` es un array Mx2 en el frame del robot; devuelve una matriz NxM.
    Los puntos fuera del mapa o en celdas de probabilidad nula valen 1.
    """
    x = particulas[:, 0][:, None]
    y = particulas[:, 1][:, None]
    c = np.cos(particulas[:, 2])[:, None]
    s = np.sin(particulas[:, 2])[:, None]
    px = puntos[:, 0][None, :]
    py = puntos[:, 1][None, :]
    xs = x + c * px - s * py
    ys = y + s * px + c * py

    col = np.floor((xs - origen_x) / resolucion).astype(np.int64)
    fila = np.floor((ys - origen_y) / resolucion).astype(np.int64)
    alto, ancho = campo.shape
    dentro = (fila >= 0) & (fila < alto) & (col >= 0) & (col < ancho)

    probabilidad = np.ones(xs.shape)
    probabilidad[dentro] = np.maximum(campo[fila[dentro], col[dentro]], 1.0)
    return probabilidad


def pesar_particulas(particulas, puntos, campo, resolucion, origen_x, origen_y):
    """Pesos normalizados según cuánto calza el scan contra el campo."""
    probabilidad = probabilidades_scan(particulas, puntos, campo, resolucion,
                                       origen_x, origen_y)
    log_pesos = np.log(probabilidad).sum(axis=1)
    log_pesos -= log_pesos.max()
    pesos = np.exp(log_pesos)
    total = pesos.sum()
    if not np.isfinite(total) or total <= 0.0:
        return np.full(len(particulas), 1.0 / len(particulas))
    return pesos / total


def calidad_scan(pose, puntos, campo, resolucion, origen_x, origen_y):
    """Probabilidad media (0-100) del campo donde caen los puntos vistos desde `pose`."""
    particula = np.array([[pose[0], pose[1], pose[2], 1.0]])
    return float(np.mean(probabilidades_scan(particula, puntos, campo, resolucion,
                                             origen_x, origen_y)))


def remuestrear(particulas, pesos, rng=np.random):
    """Resampling sistemático: copia partículas en proporción a su peso."""
    n = len(pesos)
    pasos = (np.arange(n) + rng.uniform()) / n
    acumulados = np.cumsum(pesos)
    acumulados[-1] = 1.0
    nuevas = particulas[np.searchsorted(acumulados, pasos)].copy()
    nuevas[:, 3] = 1.0 / n
    return nuevas


def dispersar(particulas, desvio_xy, desvio_theta, rng=np.random):
    """
    Suma ruido chico a cada partícula después de remuestrear.

    Con el robot quieto no hay ruido de movimiento y el remuestreo deja todas
    las partículas copiadas de una sola; esto conserva la diversidad.
    """
    n = len(particulas)
    nuevas = particulas.copy()
    nuevas[:, 0] += rng.normal(0.0, desvio_xy, n)
    nuevas[:, 1] += rng.normal(0.0, desvio_xy, n)
    theta = nuevas[:, 2] + rng.normal(0.0, desvio_theta, n)
    nuevas[:, 2] = np.arctan2(np.sin(theta), np.cos(theta))
    return nuevas


def estimar_pose(particulas):
    """Media de x/y y media circular de theta."""
    return (
        float(np.mean(particulas[:, 0])),
        float(np.mean(particulas[:, 1])),
        float(math.atan2(np.mean(np.sin(particulas[:, 2])),
                         np.mean(np.cos(particulas[:, 2])))),
    )


def dispersion_xy(particulas):
    """Desvío de la nube en metros: raíz de la suma de las varianzas en x e y."""
    return float(math.sqrt(np.var(particulas[:, 0]) + np.var(particulas[:, 1])))


def usar_filtro(calidad_filtro, calidad_odometria, dispersion, dispersion_maxima, tolerancia):
    """
    Decide si la pose del filtro es confiable o conviene seguir con odometría.

    El filtro se descarta si la nube está muy abierta o si el scan calza
    peor desde su pose que desde la que da la odometría desde la última
    corrección aceptada (con `tolerancia` en puntos de probabilidad).
    """
    if dispersion > dispersion_maxima:
        return False
    return calidad_filtro + tolerancia >= calidad_odometria
