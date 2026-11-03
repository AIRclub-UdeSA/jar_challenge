"""Planificación A* sobre una grilla de celdas transitables."""

import heapq
import math

import numpy as np

from jar_exploracion.trazado import trazar_rayo


def segmento_transitable(libres, inicio, fin):
    """Verifica un segmento sin salir del mapa ni cortar esquinas bloqueadas."""
    if inicio is None or fin is None:
        return False
    alto, ancho = libres.shape
    celdas = trazar_rayo(*inicio, *fin)
    for f, c in celdas:
        if not (0 <= f < alto and 0 <= c < ancho) or not libres[f, c]:
            return False
    for a, b in zip(celdas, celdas[1:]):
        if a[0] != b[0] and a[1] != b[1]:
            if not libres[a[0], b[1]] or not libres[b[0], a[1]]:
                return False
    return True


def a_estrella_por_etapas(libres, inicio, objetivo, costos=None, lote=200):
    """Cede con None cada lote; al finalizar retorna el camino o None."""
    if lote <= 0:
        raise ValueError('El lote de planificación debe ser positivo')
    libres = np.asarray(libres, dtype=bool)
    if libres.ndim != 2:
        raise ValueError('La grilla debe tener dos dimensiones')
    if costos is None:
        costos = np.zeros(libres.shape, dtype=float)
    else:
        costos = np.asarray(costos, dtype=float)
    if costos.shape != libres.shape or not np.all(np.isfinite(costos)):
        raise ValueError('Los costos deben ser finitos y tener la forma de la grilla')
    if np.any(costos < 0):
        raise ValueError('Los costos no pueden ser negativos')

    alto, ancho = libres.shape

    def valida(celda):
        return (0 <= celda[0] < alto and 0 <= celda[1] < ancho
                and libres[celda])

    if inicio is None or objetivo is None:
        return None
    inicio, objetivo = tuple(inicio), tuple(objetivo)
    if not valida(inicio) or not valida(objetivo):
        return None

    def heuristica(celda):
        return math.hypot(celda[0] - objetivo[0], celda[1] - objetivo[1])

    pendientes = [(heuristica(inicio), 0.0, inicio)]
    distancias = {inicio: 0.0}
    anteriores = {}
    vecinos = [(df, dc) for df in (-1, 0, 1) for dc in (-1, 0, 1)
               if df != 0 or dc != 0]
    expansiones = 0
    while pendientes:
        if expansiones >= lote:
            expansiones = 0
            yield None
        _, distancia, actual = heapq.heappop(pendientes)
        expansiones += 1
        if distancia > distancias[actual]:
            continue
        if actual == objetivo:
            camino = [actual]
            while actual in anteriores:
                actual = anteriores[actual]
                camino.append(actual)
            return camino[::-1]
        fila, col = actual
        for df, dc in vecinos:
            vecino = (fila + df, col + dc)
            if not valida(vecino):
                continue
            # Las diagonales no pueden atravesar las esquinas de obstáculos.
            if df and dc and (not libres[fila + df, col] or not libres[fila, col + dc]):
                continue
            nueva = distancia + math.hypot(df, dc) * (1.0 + costos[vecino])
            if nueva < distancias.get(vecino, math.inf):
                distancias[vecino] = nueva
                anteriores[vecino] = actual
                heapq.heappush(pendientes, (nueva + heuristica(vecino), nueva, vecino))
    return None


def a_estrella(libres, inicio, objetivo, costos=None):
    """Devuelve el camino incluyendo sus extremos, o None si no existe."""
    busqueda = a_estrella_por_etapas(libres, inicio, objetivo, costos)
    while True:
        try:
            next(busqueda)
        except StopIteration as fin:
            return fin.value


class PlanificadorCompilado:
    """Cachea el grafo de ocho vecinos y calcula rutas con SciPy en código compilado."""

    def __init__(self, libres, costos=None):
        from scipy.sparse import csr_matrix

        self.libres = np.asarray(libres, dtype=bool)
        if self.libres.ndim != 2 or not self.libres.size:
            raise ValueError('La grilla debe tener dos dimensiones y no estar vacía')
        self.costos = np.zeros(self.libres.shape) if costos is None else np.asarray(costos)
        if (self.costos.shape != self.libres.shape or not np.isfinite(self.costos).all()
                or np.any(self.costos < 0)):
            raise ValueError('Costos inválidos')
        alto, ancho = self.libres.shape
        ids = np.arange(alto * ancho, dtype=np.int32).reshape(alto, ancho)
        filas, columnas, pesos = [], [], []
        for df, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            f0, f1 = max(0, -df), min(alto, alto - df)
            c0, c1 = max(0, -dc), min(ancho, ancho - dc)
            origen = (slice(f0, f1), slice(c0, c1))
            destino = (slice(f0 + df, f1 + df), slice(c0 + dc, c1 + dc))
            validos = self.libres[origen] & self.libres[destino]
            if df and dc:
                validos &= (self.libres[f0:f1, c0 + dc:c1 + dc]
                            & self.libres[f0 + df:f1 + df, c0:c1])
            a, b = ids[origen][validos], ids[destino][validos]
            filas.extend((a, b))
            columnas.extend((b, a))
            paso = math.hypot(df, dc)
            pesos.extend((paso * (1 + self.costos[destino][validos]),
                          paso * (1 + self.costos[origen][validos])))
        self.grafo = csr_matrix(
            (np.concatenate(pesos), (np.concatenate(filas), np.concatenate(columnas))),
            shape=(alto * ancho, alto * ancho))

    def buscar(self, inicio, objetivo):
        """Devuelve un camino óptimo con los mismos costos y esquinas que A*."""
        from scipy.sparse.csgraph import dijkstra

        alto, ancho = self.libres.shape
        for celda in (inicio, objetivo):
            if (celda is None or not (0 <= celda[0] < alto and 0 <= celda[1] < ancho)
                    or not self.libres[celda]):
                return None
        desde, hasta = inicio[0] * ancho + inicio[1], objetivo[0] * ancho + objetivo[1]
        if desde == hasta:
            return [inicio]
        distancias, anteriores = dijkstra(
            self.grafo, indices=desde, directed=True, return_predecessors=True)
        if not np.isfinite(distancias[hasta]):
            return None
        camino = []
        actual = hasta
        while actual != desde:
            camino.append((int(actual // ancho), int(actual % ancho)))
            actual = anteriores[actual]
        camino.append(inicio)
        return camino[::-1]
