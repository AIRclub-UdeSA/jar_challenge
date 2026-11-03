import math
import numpy as np
from collections import deque


def yaw_a_quat(yaw):
    return math.sin(yaw / 2.0), math.cos(yaw / 2.0)


def quat_a_yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def ang_norm(a):
    return math.atan2(math.sin(a), math.cos(a))


def contar_huecos_pequenos(visto, piso, resolucion, area_caja_m2):
    """Cuenta como visto cada componente sin revisar menor que una caja."""
    if area_caja_m2 <= resolucion * resolucion:
        return 0
    pendiente = np.asarray(piso, dtype=bool) & ~visto
    agregado = 0
    alto, ancho = pendiente.shape
    for fila, col in zip(*np.nonzero(pendiente)):
        if not pendiente[fila, col]:
            continue
        cola = [(int(fila), int(col))]
        pendiente[fila, col] = False
        componente = []
        while cola:
            f, c = cola.pop()
            componente.append((f, c))
            for df in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nf, nc = f + df, c + dc
                    if ((df or dc) and 0 <= nf < alto and 0 <= nc < ancho
                            and pendiente[nf, nc]):
                        pendiente[nf, nc] = False
                        cola.append((nf, nc))
        if len(componente) * resolucion ** 2 < area_caja_m2:
            for f, c in componente:
                visto[f, c] = True
            agregado += len(componente)
    return agregado


def punto_para_mirar(candidatable, libre, dist_bfs, res, origen, caja, d_min=0.6, d_max=1.2,
                     d_ideal=0.9, fin_vista=0.12):
    """Celda desde donde mirar una caja de cerca: a d_min..d_max de ella, en piso
    candidato, alcanzable (dist_bfs >= 0) y con línea de vista libre (sin paredes
    del mapa entre medio) hasta `fin_vista` m antes del centro de la caja (su cara
    está a ~0.09 m). Minimiza camino + desvío de d_ideal. Devuelve
    (x, y, yaw_hacia_la_caja, metros_de_camino) o None."""
    ox, oy = origen
    bx, by = caja
    alto, ancho = candidatable.shape
    r = int(math.ceil(d_max / res)) + 1
    fb, cb = int(math.floor((by - oy) / res)), int(math.floor((bx - ox) / res))
    mejor = None
    for f in range(max(0, fb - r), min(alto, fb + r + 1)):
        for c in range(max(0, cb - r), min(ancho, cb + r + 1)):
            if not candidatable[f, c] or dist_bfs[f, c] < 0:
                continue
            x, y = ox + (c + 0.5) * res, oy + (f + 0.5) * res
            d = math.hypot(bx - x, by - y)
            if not d_min <= d <= d_max:
                continue
            # Línea de vista hasta casi la cara de la caja: una pared entre medio
            # (p. ej. la caja apoyada del otro lado) descarta el punto.
            pasos = max(1, int((d - fin_vista) / (0.5 * res)))
            visible = True
            for k in range(1, pasos + 1):
                t = k * (d - fin_vista) / pasos / d
                ff = int(math.floor((y + t * (by - y) - oy) / res))
                cc = int(math.floor((x + t * (bx - x) - ox) / res))
                if not (0 <= ff < alto and 0 <= cc < ancho) or not libre[ff, cc]:
                    visible = False
                    break
            if not visible:
                continue
            costo = dist_bfs[f, c] * res + 0.5 * abs(d - d_ideal)
            if mejor is None or costo < mejor[0]:
                mejor = (costo, x, y, math.atan2(by - y, bx - x), dist_bfs[f, c] * res)
    return None if mejor is None else mejor[1:]


def dilatar(mascara, radio_celdas):
    """Dilata una máscara booleana con un disco (sin scipy)."""
    if radio_celdas <= 0:
        return mascara.copy()
    r = int(math.ceil(radio_celdas))
    alto, ancho = mascara.shape
    pad = np.pad(mascara, r, mode='constant', constant_values=True)
    salida = np.zeros_like(mascara)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx * dx + dy * dy <= radio_celdas * radio_celdas:
                salida |= pad[r + dy:r + dy + alto, r + dx:r + dx + ancho]
    return salida


def bfs(transitable, fila, col):
    """Distancias en celdas, sin atravesar esquinas ocupadas. -1 = inalcanzable."""
    alto, ancho = transitable.shape
    if not (0 <= fila < alto and 0 <= col < ancho):
        return None
    if not transitable[fila, col]:
        filas, cols = np.nonzero(transitable)
        if filas.size == 0:
            return None
        k = int(np.argmin((filas - fila) ** 2 + (cols - col) ** 2))
        fila, col = int(filas[k]), int(cols[k])
    dist = np.full(transitable.shape, -1.0)
    dist[fila, col] = 0.0
    cola = deque([(fila, col)])
    vecinos = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
               (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414))
    while cola:   # BFS por capas con costo diagonal: buena aproximación de la distancia real
        f, c = cola.popleft()
        d0 = dist[f, c]
        for df, dc, w in vecinos:
            nf, nc = f + df, c + dc
            if 0 <= nf < alto and 0 <= nc < ancho and transitable[nf, nc]:
                if df and dc and not (transitable[f, nc] and transitable[nf, c]):
                    continue
                nd = d0 + w
                if dist[nf, nc] < 0 or nd < dist[nf, nc] - 1e-6:
                    dist[nf, nc] = nd
                    cola.append((nf, nc))
    return dist


class Grilla:
    """Mapa + máscaras precalculadas."""

    def __init__(self, msg, margen_transito, margen_candidato):
        info = msg.info
        self.res = info.resolution
        self.ancho = info.width
        self.alto = info.height
        self.ox = info.origin.position.x
        self.oy = info.origin.position.y
        self.frame = msg.header.frame_id or 'map'
        datos = np.array(msg.data, dtype=np.int16).reshape(self.alto, self.ancho)
        self.libre = (datos >= 0) & (datos < 50)
        self.bloqueo = ~self.libre                    # ocupado o desconocido: no se ve ni se pasa
        self.m_trans = margen_transito / self.res
        self.m_cand = margen_candidato / self.res
        self.transitable_base = ~dilatar(self.bloqueo, self.m_trans)
        self.candidatable_base = ~dilatar(self.bloqueo, self.m_cand)
        self.transitable = self.transitable_base
        self.candidatable = self.candidatable_base
        self.bloques_conocidos = np.zeros_like(self.libre)
        self.obstaculos_costmap = np.zeros_like(self.libre)
        self.obstaculos_extra = np.zeros_like(self.libre)
        self.total_libre = max(1, int(self.libre.sum()))

    def actualizar_obstaculos(self, ocupado_extra):
        """Suma los obstáculos no mapeados (del costmap) a las máscaras."""
        self.obstaculos_costmap = np.asarray(ocupado_extra, dtype=bool).copy()
        self.obstaculos_extra = self.obstaculos_costmap | self.bloques_conocidos
        self.transitable = self.transitable_base & ~dilatar(self.obstaculos_extra, self.m_trans)
        self.candidatable = self.candidatable_base & ~dilatar(self.obstaculos_extra, self.m_cand)

    def registrar_bloque_objeto(self, x, y, lado_m):
        """Ocupa un cuadrado conocido alrededor del objeto confirmado."""
        if lado_m <= 0:
            raise ValueError('lado_m debe ser positivo')
        radio = lado_m / 2.0
        f0, c0 = self.a_celda(x - radio, y - radio)
        f1, c1 = self.a_celda(x + radio, y + radio)
        f0, c0 = max(0, f0), max(0, c0)
        f1, c1 = min(self.alto - 1, f1), min(self.ancho - 1, c1)
        if f0 > f1 or c0 > c1:
            return
        self.bloques_conocidos[f0:f1 + 1, c0:c1 + 1] |= self.libre[f0:f1 + 1, c0:c1 + 1]
        self.actualizar_obstaculos(self.obstaculos_costmap)

    def a_celda(self, x, y):
        return int(math.floor((y - self.oy) / self.res)), int(math.floor((x - self.ox) / self.res))

    def a_mundo(self, fila, col):
        return self.ox + (col + 0.5) * self.res, self.oy + (fila + 0.5) * self.res

    def dentro(self, fila, col):
        return 0 <= fila < self.alto and 0 <= col < self.ancho
