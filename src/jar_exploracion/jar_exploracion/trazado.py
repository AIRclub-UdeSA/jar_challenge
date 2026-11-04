"""Trazado de segmentos entre celdas y visibilidad en la grilla."""

import numpy as np


def trazar_rayo(fila_desde: int, col_desde: int, fila_hasta: int, col_hasta: int):
    """Devuelve las celdas del segmento, incluyendo ambos extremos."""
    f0 = int(fila_desde)
    c0 = int(col_desde)
    f1 = int(fila_hasta)
    c1 = int(col_hasta)

    dc = abs(c1 - c0)
    df = abs(f1 - f0)
    sc = 1 if c0 < c1 else -1
    sf = 1 if f0 < f1 else -1
    celdas = []
    error = dc - df
    f, c = f0, c0
    while True:
        celdas.append((f, c))
        if f == f1 and c == c1:
            break
        e2 = 2 * error
        if e2 > -df:
            error -= df
            c += sc
        if e2 < dc:
            error += dc
            f += sf
    return celdas


def celdas_visibles(libre, origenes, alcance_celdas, rayos=720):
    """
    Celdas libres que se ven en línea recta desde alguno de los orígenes.

    Lanza `rayos` direcciones desde el centro de cada origen a pasos de media
    celda. Cada rayo se corta en la primera celda no libre, al pasar en
    diagonal entre dos celdas si alguna está ocupada, o en `alcance_celdas`.
    """
    libre = np.asarray(libre, dtype=bool)
    alto, ancho = libre.shape
    visibles = np.zeros_like(libre)
    angulos = np.linspace(0.0, 2.0 * np.pi, rayos, endpoint=False)
    pasos = np.arange(0.0, alcance_celdas, 0.5)
    for f0, c0 in origenes:
        filas = np.floor(f0 + 0.5 + np.outer(np.sin(angulos), pasos)).astype(np.int64)
        cols = np.floor(c0 + 0.5 + np.outer(np.cos(angulos), pasos)).astype(np.int64)
        dentro = (filas >= 0) & (filas < alto) & (cols >= 0) & (cols < ancho)
        filas = np.clip(filas, 0, alto - 1)
        cols = np.clip(cols, 0, ancho - 1)
        ok = dentro & libre[filas, cols]
        diagonal = (np.diff(filas, axis=1) != 0) & (np.diff(cols, axis=1) != 0)
        esquinas = (libre[filas[:, :-1], cols[:, 1:]] & libre[filas[:, 1:], cols[:, :-1]])
        ok[:, 1:] &= ~(diagonal & ~esquinas)
        hasta = np.cumprod(ok, axis=1).astype(bool)
        visibles[filas[hasta], cols[hasta]] = True
    return visibles
