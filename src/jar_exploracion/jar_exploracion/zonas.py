"""Orden de las zonas sin ver (planificación por zonas), sin ROS.

El goal de cobertura greedy compara un goal contra otro mirando sólo el paso siguiente: una zona
chica que queda atrás se vuelve cada vez más cara y se posterga hasta el final (en una práctica: un
callejón de ~2 m² a 1.6 m del robot, dejado por una sala grande, quedó a 4 m).

Acá se ordenan las zonas sin ver para minimizar el tiempo esperado hasta encontrar las víctimas
que faltan. Sin saber dónde están, cualquier lugar sin ver es igual de probable: cada zona pesa
por su área (más si hay objetos "posibles" adentro). Si F_k es la fracción del peso ya cubierta
al terminar la k-ésima zona del orden, la probabilidad de haber encontrado las m víctimas que
faltan es F_k^m, y

    E[T] = Σ_k T_k × (F_k^m − F_(k−1)^m),   T_k = momento en que se termina la zona k

con T sumando los viajes por camino entre zonas y el tiempo de cubrir cada una. Con m = 1 es
el tiempo hasta la primera víctima (prioriza lo grande); con m grande se acerca al recorrido
más corto (no deja rincones atrás). El viaje de una zona a otra es la distancia promedio desde
sus celdas: después de cubrir una sala el robot puede estar en cualquier parte de ella. Ninguna
zona se olvida: el orden las incluye a todas y se rehace en cada decisión.
"""
import itertools
import math

import numpy as np
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


def grafo(mascara, res):
    """Grafo de 8 vecinos sobre las celdas de `mascara`. Devuelve (matriz, índice por celda)."""
    m = np.asarray(mascara, dtype=bool)
    alto, ancho = m.shape
    indice = np.full(m.shape, -1, dtype=np.int64)
    indice[m] = np.arange(int(m.sum()))
    filas, columnas, pesos = [], [], []
    for di, dj in ((0, 1), (1, 0), (1, 1), (1, -1)):
        j0, j1 = max(0, -dj), ancho - max(0, dj)
        a = m[0:alto - di, j0:j1] & m[di:alto, j0 + dj:j1 + dj]
        ii, jj = np.nonzero(a)
        jj = jj + j0
        filas.append(indice[ii, jj])
        columnas.append(indice[ii + di, jj + dj])
        pesos.append(np.full(len(ii), math.hypot(di, dj) * res))
    n = int(m.sum())
    matriz = csr_matrix((np.concatenate(pesos), (np.concatenate(filas),
                                                 np.concatenate(columnas))), shape=(n, n))
    return matriz, indice


def distancias(matriz, indice, fuentes):
    """Metros por camino desde el conjunto de celdas `fuentes` (N×2) a cada celda (inf: no)."""
    nodos = indice[fuentes[:, 0], fuentes[:, 1]]
    nodos = nodos[nodos >= 0]
    salida = np.full(indice.shape, np.inf)
    if len(nodos) == 0:
        return salida
    d = dijkstra(matriz, directed=False, indices=nodos, min_only=True)
    salida[indice >= 0] = d
    return salida


def ordenar(pesos, tiempos, viaje_robot, viaje_entre, victimas=1, actual=None,
            histeresis=0.0):
    """Orden de las zonas que minimiza el tiempo esperado hasta encontrar `victimas` víctimas.

    `tiempos[k]`: cubrir la zona k (s); `viaje_robot[k]`: del robot a k (s); `viaje_entre[a][b]`:
    de a a b (s). Prueba todos los órdenes (pocas zonas). Con `actual` (la zona que se estaba
    cubriendo), se queda con el mejor orden que empieza por ella si no es más de `histeresis`
    (fracción) peor que el mejor: así no cambia de zona por diferencias chicas.
    Devuelve (orden, costo).
    """
    n = len(pesos)
    if n == 0:
        return [], 0.0
    total = float(sum(pesos))
    m = max(1, int(victimas))
    mejor, mejor_actual = (None, math.inf), (None, math.inf)
    for orden in itertools.permutations(range(n)):
        t, costo, previa, f_previa = 0.0, 0.0, None, 0.0
        for k in orden:
            t += (viaje_robot[k] if previa is None else viaje_entre[previa][k]) + tiempos[k]
            f = f_previa + pesos[k] / total
            costo += t * (f ** m - f_previa ** m)
            previa, f_previa = k, f
            if costo >= mejor[1] and (actual is None or orden[0] != actual):
                break                             # ya no puede ganar
        if costo < mejor[1]:
            mejor = (list(orden), costo)
        if actual is not None and orden[0] == actual and costo < mejor_actual[1]:
            mejor_actual = (list(orden), costo)
    if mejor_actual[0] is not None and mejor_actual[1] <= mejor[1] * (1.0 + histeresis):
        return mejor_actual
    return mejor


def zonas_sin_ver(navegable, visto, celdas_min):
    """Etiquetas de las zonas conexas sin ver de al menos `celdas_min` celdas (8 vecinos)."""
    etiquetas, n = ndimage.label(navegable & ~visto, structure=np.ones((3, 3)))
    tamanios = np.bincount(etiquetas.ravel(), minlength=n + 1)
    grandes = [k for k in range(1, n + 1) if tamanios[k] >= celdas_min]
    return etiquetas, grandes
