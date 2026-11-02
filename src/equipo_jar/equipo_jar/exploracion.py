"""Planificador de la exploración: qué puntos visitar y en qué orden.

Al recibir /map se calcula, una sola vez, un conjunto chico de puntos de vista
desde donde el lidar (360°) ve todas las celdas donde puede haber una caja, y
se los ordena como un viajante por distancias reales por el laberinto. El lidar
ve las cajas (objetos que no están en el mapa) desde lejos y por los pasillos;
cubrir con la cámara (61°) pediría 2 a 3 veces más paradas, cada una con un
giro.

La misión (nodo_equipo) lo usa así:

    exploracion = Exploracion(mapa, logger, parametros)
    exploracion.planificar(pose_inicial)       # una vez, al recibir /map
    exploracion.siguiente(pose)                # (x, y, yaw) en `map`, o None
    exploracion.objetivo_terminado(exito)      # cuando Nav2 termina ese objetivo
    exploracion.distancia_camino(a, b)         # metros por el laberinto, o None

Este módulo no depende de ROS más allá de leer el OccupancyGrid, así que se
puede probar sin simulador (test/test_exploracion.py).
"""
import math
import time

import numpy as np
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

PARAMETROS_POR_DEFECTO = {
    # Alcance (m) del lidar que se cuenta como "visto" para la cobertura. Una
    # caja de 0.18 m a 3.5 m son ~9 rayos, de sobra para detectarla.
    'alcance_cobertura': 3.5,
    # Distancia mínima (m) de un punto de vista a la pared más cercana.
    'despeje_punto_vista': 0.25,
    # Distancia mínima (m) a la pared por la que el robot puede pasar
    # (medio ancho del footprint más el padding).
    'despeje_transito': 0.12,
    # Celdas donde puede estar el centro de una caja: a esta distancia (m) o
    # más de una pared.
    'despeje_caja': 0.10,
    # Fracción de esas celdas que tiene que ver el conjunto de puntos de vista.
    'fraccion_cobertura': 0.98,
    # Cuántas veces se reintenta un punto de vista al que Nav2 no pudo llegar.
    'reintentos_punto_vista': 1,
    'max_candidatos': 400,
    'rayos': 180,
}


def raycast(ocupado, fila, col, alcance_celdas, rayos):
    """Celdas visibles desde (fila, col) hasta alcance_celdas, cortando en paredes."""
    alto, ancho = ocupado.shape
    angulos = np.linspace(0.0, 2.0 * np.pi, rayos, endpoint=False)
    pasos = np.arange(0.0, alcance_celdas, 0.5)
    filas = fila + 0.5 + np.outer(np.sin(angulos), pasos)
    cols = col + 0.5 + np.outer(np.cos(angulos), pasos)
    fuera = (filas < 0) | (filas >= alto) | (cols < 0) | (cols >= ancho)
    fi = np.clip(filas.astype(int), 0, alto - 1)
    ci = np.clip(cols.astype(int), 0, ancho - 1)
    bloqueado = ocupado[fi, ci] | fuera
    corte = np.where(bloqueado.any(axis=1), bloqueado.argmax(axis=1), len(pasos))
    validos = np.arange(len(pasos))[None, :] < corte[:, None]
    visibles = np.zeros_like(ocupado)
    visibles[fi[validos], ci[validos]] = True
    return visibles


def grafo_de_grilla(transitable, resolucion):
    """Grafo de 8 vecinos sobre las celdas transitables (pesos en metros)."""
    alto, ancho = transitable.shape
    indices = np.arange(alto * ancho).reshape(alto, ancho)
    origen, destino, peso = [], [], []
    for df, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        # Celdas de origen cuyo vecino (fila + df, col + dc) está dentro de la grilla.
        f0, f1 = max(0, -df), alto - max(0, df)
        c0, c1 = max(0, -dc), ancho - max(0, dc)
        ok = transitable[f0:f1, c0:c1] & transitable[f0 + df:f1 + df, c0 + dc:c1 + dc]
        ia = indices[f0:f1, c0:c1]
        ib = ia + df * ancho + dc
        origen.append(ia[ok])
        destino.append(ib[ok])
        peso.append(np.full(ok.sum(), math.hypot(df, dc) * resolucion))
    origen = np.concatenate(origen)
    destino = np.concatenate(destino)
    peso = np.concatenate(peso)
    n = alto * ancho
    return csr_matrix((np.concatenate([peso, peso]),
                       (np.concatenate([origen, destino]), np.concatenate([destino, origen]))),
                      shape=(n, n))


def ordenar_recorrido(distancias):
    """Orden de visita (empezando en 0 y volviendo a 0): vecino más cercano + 2-opt."""
    n = len(distancias)
    if n <= 2:
        return list(range(n))
    orden = [0]
    restantes = set(range(1, n))
    while restantes:
        j = min(restantes, key=lambda k: distancias[orden[-1], k])
        orden.append(j)
        restantes.remove(j)
    ciclo = orden + [0]

    def largo(o):
        return sum(distancias[o[i], o[i + 1]] for i in range(len(o) - 1))

    mejoro = True
    while mejoro:
        mejoro = False
        for i in range(1, len(ciclo) - 2):
            for j in range(i + 1, len(ciclo) - 1):
                nuevo = ciclo[:i] + ciclo[i:j + 1][::-1] + ciclo[j + 1:]
                if largo(nuevo) < largo(ciclo) - 1e-6:
                    ciclo = nuevo
                    mejoro = True
    return ciclo[:-1]


class Exploracion:

    def __init__(self, mapa, logger, parametros=None):
        self.logger = logger
        self.p = dict(PARAMETROS_POR_DEFECTO)
        self.p.update(parametros or {})
        info = mapa.info
        self.res = info.resolution
        self.ox = info.origin.position.x
        self.oy = info.origin.position.y
        datos = np.array(mapa.data, dtype=np.int16).reshape(info.height, info.width)
        # Lo desconocido se trata como pared.
        self.ocupado = (datos >= 50) | (datos < 0)
        self.libre = datos == 0
        self.dist_pared = ndimage.distance_transform_edt(~self.ocupado) * self.res
        self.pendientes = []      # [(x, y, intentos)]
        self.distancias_largada = None
        self.celda_largada = None
        self.grafo = None

    # ---------- conversiones ----------

    def celda(self, x, y):
        return int((y - self.oy) / self.res), int((x - self.ox) / self.res)

    def punto(self, fila, col):
        return self.ox + (col + 0.5) * self.res, self.oy + (fila + 0.5) * self.res

    def celda_mas_cercana(self, mascara, fila, col):
        filas, cols = np.nonzero(mascara)
        if len(filas) == 0:
            return None
        k = np.argmin((filas - fila) ** 2 + (cols - col) ** 2)
        return int(filas[k]), int(cols[k])

    # ---------- plan ----------

    def planificar(self, pose):
        t0 = time.monotonic()
        alto, ancho = self.ocupado.shape
        transitable = self.libre & (self.dist_pared >= self.p['despeje_transito'])
        inicio = self.celda(pose[0], pose[1])
        if not (0 <= inicio[0] < alto and 0 <= inicio[1] < ancho) or not transitable[inicio]:
            inicio = self.celda_mas_cercana(transitable, *inicio)
        if inicio is None:
            self.logger.error('Exploración: no hay celdas transitables en el mapa')
            return
        # Solo la parte del laberinto a la que el robot puede llegar.
        etiquetas, _ = ndimage.label(transitable, structure=np.ones((3, 3)))
        transitable = etiquetas == etiquetas[inicio]
        seguro = transitable & (self.dist_pared >= self.p['despeje_punto_vista'])
        cajas = self.libre & (self.dist_pared >= self.p['despeje_caja'])

        # Candidatos a punto de vista: centros de pasillo (máximos locales de la
        # distancia a la pared).
        maximos = self.dist_pared == ndimage.maximum_filter(self.dist_pared, size=5)
        candidatos = np.argwhere(maximos & seguro)
        if len(candidatos) == 0:
            candidatos = np.argwhere(seguro)
        paso = max(1, len(candidatos) // self.p['max_candidatos'])
        candidatos = candidatos[::paso]
        alcance = self.p['alcance_cobertura'] / self.res
        vistas = [raycast(self.ocupado, f, c, alcance, self.p['rayos']) & cajas
                  for f, c in candidatos]

        # Cobertura greedy (set cover) de lo que se puede ver desde algún candidato.
        visible = np.zeros_like(cajas)
        for v in vistas:
            visible |= v
        meta = self.p['fraccion_cobertura'] * visible.sum()
        # Lo que ya se ve desde la largada no hace falta ir a buscarlo.
        cubierto = raycast(self.ocupado, inicio[0], inicio[1], alcance, self.p['rayos']) & cajas
        elegidos = []
        while cubierto.sum() < meta:
            ganancia = [int((v & ~cubierto).sum()) for v in vistas]
            i = int(np.argmax(ganancia))
            if ganancia[i] == 0:
                break
            elegidos.append(i)
            cubierto |= vistas[i]

        # Distancias por el laberinto entre la largada y los puntos elegidos.
        self.grafo = grafo_de_grilla(transitable, self.res)
        celdas = [inicio] + [tuple(candidatos[i]) for i in elegidos]
        nodos = [f * ancho + c for f, c in celdas]
        distancias_completas = dijkstra(self.grafo, indices=nodos)
        distancias = distancias_completas[:, nodos]
        self.distancias_largada = distancias_completas[0].reshape(alto, ancho)
        self.celda_largada = inicio
        orden = ordenar_recorrido(distancias)[1:]   # el 0 es la largada
        self.pendientes = [self.punto(*celdas[k]) + (0,) for k in orden]
        largo = sum(distancias[a, b] for a, b in zip([0] + orden, orden + [0]))
        self.logger.info(
            f'Exploración: {len(self.pendientes)} puntos de vista, '
            f'{100.0 * cubierto.sum() / max(1, visible.sum()):.0f}% cubierto, '
            f'recorrido de {largo:.1f} m ida y vuelta, calculado en '
            f'{time.monotonic() - t0:.1f} s')

    # ---------- uso durante la corrida ----------

    def siguiente(self, pose):
        if not self.pendientes:
            return None
        x, y, _ = self.pendientes[0]
        # Mirando hacia el punto siguiente (el yaw casi no importa: Nav2 lo
        # tolera entero), o hacia donde se viene si es el último.
        if len(self.pendientes) > 1:
            xs, ys, _ = self.pendientes[1]
            yaw = math.atan2(ys - y, xs - x)
        else:
            yaw = math.atan2(y - pose[1], x - pose[0])
        return x, y, yaw

    def objetivo_terminado(self, exito):
        if not self.pendientes:
            return
        x, y, intentos = self.pendientes.pop(0)
        if not exito and intentos < self.p['reintentos_punto_vista']:
            # Se reintenta al final, cuando el robot venga desde otro lado.
            self.pendientes.append((x, y, intentos + 1))

    def dentro(self, fila, col):
        alto, ancho = self.ocupado.shape
        return 0 <= fila < alto and 0 <= col < ancho

    def despeje(self, x, y):
        """Distancia (m) de (x, y) a la pared más cercana del mapa (0 fuera)."""
        fila, col = self.celda(x, y)
        return float(self.dist_pared[fila, col]) if self.dentro(fila, col) else 0.0

    def linea_de_vista(self, a, b, margen_final=0.2):
        """True si entre a y b no hay paredes del mapa (sin contar los últimos cm)."""
        d = math.hypot(b[0] - a[0], b[1] - a[1])
        pasos = max(2, int(d / (self.res / 2)))
        for i in range(pasos):
            t = i / pasos
            if t * d > d - margen_final:
                break
            fila, col = self.celda(a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
            if not self.dentro(fila, col) or self.ocupado[fila, col]:
                return False
        return True

    def punto_de_inspeccion(self, objetivo, desde, d_min=0.6, d_max=1.5):
        """Celda libre a d_min-d_max de `objetivo`, con vista directa, cerca de `desde`.

        Se busca en un anillo alrededor del objetivo, sobre celdas con
        despeje_punto_vista (el robot tiene que poder girar en el lugar). Entre
        las candidatas se prefieren las más despejadas: en un rincón el robot
        después tarda en salir (cada 0.1 m de despeje vale 0.2 m de camino).
        """
        r = self.res
        fila0, col0 = self.celda(*objetivo)
        radio = int(d_max / r) + 1
        mejor, mejor_d = None, math.inf
        for df in range(-radio, radio + 1, 2):
            for dc in range(-radio, radio + 1, 2):
                fila, col = fila0 + df, col0 + dc
                if not self.dentro(fila, col):
                    continue
                if self.dist_pared[fila, col] < self.p['despeje_punto_vista']:
                    continue
                x, y = self.punto(fila, col)
                d = math.hypot(x - objetivo[0], y - objetivo[1])
                if not (d_min <= d <= d_max) or not self.linea_de_vista((x, y), objetivo):
                    continue
                dd = (math.hypot(x - desde[0], y - desde[1])
                      - 2.0 * min(self.dist_pared[fila, col], 0.6))
                if dd < mejor_d:
                    mejor, mejor_d = (x, y), dd
        return mejor

    def distancia_camino(self, a, b):
        """Distancia (m) por el laberinto de `a` a la largada, si `b` es la largada."""
        if self.distancias_largada is None:
            return None
        fila, col = self.celda(a[0], a[1])
        alto, ancho = self.distancias_largada.shape
        if not (0 <= fila < alto and 0 <= col < ancho):
            return None
        d = self.distancias_largada[fila, col]
        return float(d) if np.isfinite(d) else None
