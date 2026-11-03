"""Estimaciones de último recurso a partir del mapa y de la cobertura real."""
import math

import cv2
import numpy as np


def lugares_probables(mapa, visto, recorrido, excluidos, cantidad, margen, separacion):
    """Reparte estimaciones sobre piso poco observado, sin coordenadas prefijadas.

    Son hipótesis sin confirmación visual. Prioriza el área sin observar dentro
    de 0,5 m y separa las hipótesis para aprovechar los reportes disponibles.
    """
    libre, res = mapa['libre'], mapa['res']
    valido = libre & (mapa['dist'] >= margen)
    if recorrido is not None:
        valido &= ~recorrido
    pendiente = libre if visto is None else libre & ~visto
    radio = max(1, int(math.ceil(0.5 / res)))
    ff, cc = np.ogrid[-radio:radio + 1, -radio:radio + 1]
    disco = ((ff * res) ** 2 + (cc * res) ** 2 <= 0.5 ** 2).astype(np.float32)
    ganancia = cv2.filter2D(pendiente.astype(np.float32), -1, disco,
                            borderType=cv2.BORDER_CONSTANT)
    # Entre áreas de igual ganancia, prefiere el centro del piso sin observar.
    profundidad = cv2.distanceTransform(np.pad(pendiente.astype(np.uint8), 1),
                                        cv2.DIST_L2, 5)[1:-1, 1:-1]
    utilidad = ganancia + pendiente.astype(float) * (libre.size + 1) + 0.001 * profundidad
    ys = mapa['oy'] + (np.arange(libre.shape[0]) + 0.5) * res
    xs = mapa['ox'] + (np.arange(libre.shape[1]) + 0.5) * res

    def excluir(x, y):
        if math.isfinite(x) and math.isfinite(y):
            valido[:] &= (xs[None, :] - x) ** 2 + (ys[:, None] - y) ** 2 >= separacion ** 2

    for x, y in excluidos:
        excluir(x, y)
    elegidos = []
    for _ in range(max(0, cantidad)):
        if not valido.any():
            break
        f, c = np.unravel_index(np.argmax(np.where(valido, utilidad, -np.inf)), libre.shape)
        punto = (float(xs[c]), float(ys[f]))
        elegidos.append(punto)
        excluir(*punto)
    return elegidos
