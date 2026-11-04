"""Huecos medidos con el LiDAR (r9), sin ROS.

Nav2 decide si un hueco se puede pasar con el costmap global: celdas de 5 cm, `robot_radius`
0.15 y las cajas marcadas con la pose de AMCL, que al principio de la corrida anda 0.10-0.15 m
corrida. En la puerta de E de `laberinto_puertas` (0.36 m de hueco, robot de 0.22 m) eso deja
0-6 cm de franja y Nav2 no planifica, aunque el robot pasa con 7 cm por lado. El LiDAR mide
el hueco relativo al robot: ni AMCL ni el costmap lo afectan.

Un hueco en el scan es un tramo de rayos que pegan lejos (o no pegan) entre dos objetos
cercanos: el rayo anterior al tramo y el posterior son bordes de oclusión (`salto_m` o más de
diferencia con su vecino), los dos a menos de `alcance_m`, y todo el tramo pega al menos
`profundidad_min_m` más lejos que los dos bordes (hay lugar detrás). El ancho es la menor
distancia entre los puntos de un lado y los del otro (hasta `vecindad_borde_m` de cada borde):
visto de costado, los extremos que se ven no son los más cercanos entre sí.

Un hueco es pasable si mide al menos el ancho del robot más `margen_m` por lado. Visto de
costado mide de más: lo más angosto puede estar en una cara que no se ve (la de una caja dentro
de una puerta, vista de frente a la caja: 0.41 m para un hueco de 0.36). Medido de frente al
hueco esa cara se ve. Por eso el registro junta las mediciones del mismo hueco (centro a menos
de `radio_union_m`, en `map`) y se queda con las más chicas (`cuantil`): medir de menos sólo
hace que no se intente; de más, que se intente y se trabe.

Todo en el marco del robot (x adelante, y a la izquierda) salvo el registro, en `map`.
"""
import heapq
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class ConfigHuecos:
    salto_m: float = 0.15            # diferencia entre rayos vecinos que hace un borde
    profundidad_min_m: float = 0.30  # el tramo del hueco pega al menos esto más lejos
    alcance_m: float = 2.0           # bordes más lejos: no se miden (la medida empeora)
    ancho_min_m: float = 0.10        # más angosto: ruido entre dos rayos, no un hueco
    ancho_max_m: float = 0.80        # más ancho: no es un hueco que haya que medir
    abertura_max_rad: float = 2.0    # tramo de rayos lejanos más largo: no es un hueco
    vecindad_borde_m: float = 0.30   # puntos de cada lado que cuentan para el ancho
    ancho_robot_m: float = 0.22
    margen_m: float = 0.04           # por lado: pasable con ancho_robot + 2 margen
    radio_union_m: float = 0.25      # dos medidas con el centro más cerca: el mismo hueco
    mediciones_max: int = 9          # se guardan las últimas
    mediciones_min: int = 3          # antes, el hueco no se da por medido
    cuantil: float = 0.25            # de las mediciones guardadas: el ancho del hueco


@dataclass
class Hueco:
    """Un hueco medido en un scan, en el marco del robot."""
    borde_a: Tuple[float, float]     # los dos puntos más cercanos entre los lados
    borde_b: Tuple[float, float]
    ancho: float

    @property
    def centro(self):
        return ((self.borde_a[0] + self.borde_b[0]) / 2.0,
                (self.borde_a[1] + self.borde_b[1]) / 2.0)

    def normal(self):
        """Dirección unitaria a través del hueco, alejándose del robot."""
        dx, dy = self.borde_b[0] - self.borde_a[0], self.borde_b[1] - self.borde_a[1]
        n = np.array([-dy, dx]) / max(math.hypot(dx, dy), 1e-9)
        return tuple(n if np.dot(n, self.centro) >= 0.0 else -n)


def pasable(ancho, c=ConfigHuecos()):
    return ancho >= c.ancho_robot_m + 2.0 * c.margen_m


def _puntos(rangos, angle_min, angle_increment, tf_laser):
    """Rayo k → punto en el marco del robot (NaN si no pegó)."""
    tx, ty, yaw = tf_laser
    ang = angle_min + angle_increment * np.arange(len(rangos)) + yaw
    return np.column_stack((tx + rangos * np.cos(ang), ty + rangos * np.sin(ang)))


def _lado(r, p, k, paso, c):
    """Índices del objeto que termina en el rayo k, yendo en dirección `paso` (-1 o +1) hacia
    afuera del hueco, hasta `vecindad_borde_m` del borde o hasta un salto."""
    n = len(r)
    lado = [k]
    j = k
    while len(lado) < n:
        sig = (j + paso) % n
        if (not np.isfinite(r[sig]) or abs(r[sig] - r[j]) >= c.salto_m
                or math.hypot(*(p[sig] - p[k])) > c.vecindad_borde_m):
            break
        lado.append(sig)
        j = sig
    return lado


def huecos_del_scan(ranges, angle_min, angle_increment, range_min, range_max,
                    tf_laser=(0.0, 0.0, 0.0), c=ConfigHuecos()):
    """Huecos que se ven en el scan, en el marco del robot (`tf_laser`: láser → robot)."""
    r = np.asarray(ranges, dtype=float).copy()
    n = len(r)
    if n < 3:
        return []
    sin_eco = ~np.isfinite(r) | (r > range_max)
    r[sin_eco] = np.inf                       # no pegó: lejos
    r[r < range_min] = np.nan                 # inválido: corta todo
    p = _puntos(np.where(np.isfinite(r), r, 0.0), angle_min, angle_increment, tf_laser)
    max_rayos = int(c.abertura_max_rad / abs(angle_increment))
    huecos = []
    for s in range(n):
        izq = (s - 1) % n
        if not (np.isfinite(r[izq]) and r[izq] <= c.alcance_m):
            continue
        if np.isnan(r[s]) or r[s] - r[izq] < c.salto_m:
            continue
        # Tramo lejano desde s hasta el primer rayo que vuelve a pegar cerca.
        k, tramo = s, []
        while len(tramo) <= max_rayos:
            if np.isnan(r[k]):
                tramo = None
                break
            sig = (k + 1) % n
            tramo.append(k)
            if np.isfinite(r[sig]) and r[k] - r[sig] >= c.salto_m:
                break
            k = sig
        if not tramo or len(tramo) > max_rayos:
            continue
        der = (tramo[-1] + 1) % n
        if not (np.isfinite(r[der]) and r[der] <= c.alcance_m):
            continue
        if np.min(r[tramo]) < max(r[izq], r[der]) + c.profundidad_min_m:
            continue
        a, b = p[_lado(r, p, izq, -1, c)], p[_lado(r, p, der, +1, c)]
        dist = np.hypot(a[:, None, 0] - b[None, :, 0], a[:, None, 1] - b[None, :, 1])
        i, j = np.unravel_index(int(np.argmin(dist)), dist.shape)
        ancho = float(dist[i, j])
        if c.ancho_min_m <= ancho <= c.ancho_max_m:
            huecos.append(Hueco((float(a[i, 0]), float(a[i, 1])),
                                (float(b[j, 0]), float(b[j, 1])), ancho))
    return huecos


def a_mapa(punto, pose):
    """Punto del marco del robot a `map`, con la pose (x, y, yaw) del robot."""
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    return (x + c * punto[0] - s * punto[1], y + s * punto[0] + c * punto[1])


@dataclass
class HuecoMapa:
    """Un hueco en `map`, con sus últimas mediciones."""
    id: int
    borde_a: Tuple[float, float]
    borde_b: Tuple[float, float]
    anchos: List[float] = field(default_factory=list)
    cuantil: float = 0.25

    @property
    def centro(self):
        return ((self.borde_a[0] + self.borde_b[0]) / 2.0,
                (self.borde_a[1] + self.borde_b[1]) / 2.0)

    @property
    def ancho(self):
        return float(np.quantile(self.anchos, self.cuantil))


class RegistroHuecos:
    """Junta las mediciones de cada hueco en `map` (ver el docstring del módulo)."""

    def __init__(self, config=ConfigHuecos()):
        self.config = config
        self.lista: List[HuecoMapa] = []
        self._proximo_id = 1

    def medidos(self):
        return [h for h in self.lista if len(h.anchos) >= self.config.mediciones_min]

    def cercano(self, centro) -> Optional[HuecoMapa]:
        mejor, d_min = None, self.config.radio_union_m
        for h in self.lista:
            d = math.hypot(h.centro[0] - centro[0], h.centro[1] - centro[1])
            if d < d_min:
                mejor, d_min = h, d
        return mejor

    def agregar(self, huecos, pose):
        """Suma los huecos de un scan (marco del robot) con la pose del robot en `map`.
        Devuelve [(hueco, cambio)] con cambio 'medido' (llegó a mediciones_min) o 'pasable' /
        'no_pasable' (la mediana cruzó el umbral), para avisar."""
        c = self.config
        avisos = []
        for hueco in huecos:
            a, b = a_mapa(hueco.borde_a, pose), a_mapa(hueco.borde_b, pose)
            centro = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
            h = self.cercano(centro)
            if h is None:
                h = HuecoMapa(self._proximo_id, a, b, cuantil=c.cuantil)
                self._proximo_id += 1
                self.lista.append(h)
            antes = pasable(h.ancho, c) if len(h.anchos) >= c.mediciones_min else None
            h.borde_a, h.borde_b = a, b
            h.anchos.append(hueco.ancho)
            del h.anchos[:-c.mediciones_max]
            if len(h.anchos) < c.mediciones_min:
                continue
            ahora = pasable(h.ancho, c)
            if antes is None:
                avisos.append((h, 'medido'))
            elif ahora != antes:
                avisos.append((h, 'pasable' if ahora else 'no_pasable'))
        return avisos


def cuello_en_camino(libre, resolucion, origen, desde, hasta, despeje_min=0.11,
                     despeje_cuello_max=0.47, peso_centro=0.3, bloqueado=None,
                     radio_puntas=0.30):
    """El lugar más angosto del camino de `desde` a `hasta` por el mapa estático.

    `libre`: celdas libres del mapa (fila 0 = y del origen). El mapa estático no tiene las
    cajas de las puertas: el camino pasa por el medio de la abertura y el punto de menor despeje
    es la puerta, donde puede haber una caja que deje un hueco. Se toma la primera del camino
    (la primera que el robot tiene que cruzar): una abertura con a lo sumo `despeje_cuello_max`
    de despeje (una puerta de 0.9 m tiene 0.45; un pasillo de 1 m, 0.5). Devuelve ((x, y) del
    centro de la abertura, dirección unitaria para cruzarla) o None si no hay camino o puerta.

    Con el mapa estático solo, una puerta y un pasillo del mismo ancho son iguales. Con
    `bloqueado(x, y)` (el costmap global de Nav2: ahí están las cajas) el cuello es la primera
    celda del camino que Nav2 tiene cerrada, que es donde se traba.
    """
    from scipy import ndimage
    libre = np.asarray(libre, dtype=bool)
    despeje = ndimage.distance_transform_edt(libre) * resolucion
    alto, ancho = libre.shape

    def celda(p):
        return (int((p[1] - origen[1]) / resolucion), int((p[0] - origen[0]) / resolucion))

    paso_ok = despeje >= despeje_min
    inicio, fin = celda(desde), celda(hasta)
    for i, j in (inicio, fin):
        if not (0 <= i < alto and 0 <= j < ancho):
            return None
    # El robot o el goal pueden caer pegados a una pared: se arranca de la celda pasable más
    # cercana.
    pasables = np.argwhere(paso_ok)
    if len(pasables) == 0:
        return None

    def mas_cercana(c):
        if paso_ok[c]:
            return c
        k = int(np.argmin((pasables[:, 0] - c[0]) ** 2 + (pasables[:, 1] - c[1]) ** 2))
        return tuple(int(v) for v in pasables[k])
    inicio, fin = mas_cercana(inicio), mas_cercana(fin)
    dist = np.full(libre.shape, np.inf)
    previo = {}
    dist[inicio] = 0.0
    cola = [(0.0, inicio)]
    vecinos = [(di, dj, math.hypot(di, dj)) for di in (-1, 0, 1) for dj in (-1, 0, 1) if di or dj]
    # Por el medio de los pasos: pegado a una pared cuesta más. Así el despeje a lo largo del
    # camino es el medio ancho del paso y la primera puerta es su primer mínimo.
    costo = 1.0 + peso_centro / np.maximum(despeje, resolucion)
    while cola:
        d, (i, j) = heapq.heappop(cola)
        if (i, j) == fin:
            break
        if d > dist[i, j]:
            continue
        for di, dj, paso in vecinos:
            a, b = i + di, j + dj
            if not (0 <= a < alto and 0 <= b < ancho and paso_ok[a, b]):
                continue
            nuevo = d + paso * costo[a, b]
            if nuevo < dist[a, b]:
                dist[a, b] = nuevo
                previo[(a, b)] = (i, j)
                heapq.heappush(cola, (nuevo, (a, b)))
    if not np.isfinite(dist[fin]):
        return None
    camino = [fin]
    while camino[-1] != inicio:
        camino.append(previo[camino[-1]])
    camino.reverse()

    def punto(c):
        return (origen[0] + (c[1] + 0.5) * resolucion, origen[1] + (c[0] + 0.5) * resolucion)

    # El robot o el goal pegados a una pared no son una puerta: no cuenta lo que está a menos
    # de radio_puntas de las puntas del camino.
    p_ini, p_fin = np.asarray(punto(camino[0])), np.asarray(punto(camino[-1]))
    medio = [k for k, ck in enumerate(camino)
             if np.hypot(*(np.asarray(punto(ck)) - p_ini)) > radio_puntas
             and np.hypot(*(np.asarray(punto(ck)) - p_fin)) > radio_puntas]
    if bloqueado is not None:
        cerrados = [k for k in medio if bloqueado(*punto(camino[k]))]
        if not cerrados:
            return None
        k = cerrados[0]
    else:
        tramos, actual = [], []
        for k in medio:
            if despeje[camino[k]] <= despeje_cuello_max and (not actual or k == actual[-1] + 1):
                actual.append(k)
            elif actual:
                tramos.append(actual)
                actual = [k] if despeje[camino[k]] <= despeje_cuello_max else []
        if actual:
            tramos.append(actual)
        if not tramos:
            return None
        k = min(tramos[0], key=lambda k: despeje[camino[k]])
    p0 = np.asarray(punto(camino[max(0, k - 3)]))
    p1 = np.asarray(punto(camino[min(len(camino) - 1, k + 3)]))
    d = p1 - p0
    norma = float(np.hypot(*d))
    if norma < 1e-9:
        return None
    d = d / norma
    # El camino más corto pasa pegado a un marco y cruza en diagonal. Los dos marcos son la
    # pared más cercana al cuello y la más cercana del lado opuesto: el centro es el medio entre
    # ellos y la dirección para cruzar, su perpendicular hacia el goal. De frente a ese centro
    # el hueco se mide bien.
    c = np.asarray(punto(camino[k]))
    ocupadas = np.argwhere(~libre)
    paredes = np.column_stack((origen[0] + (ocupadas[:, 1] + 0.5) * resolucion,
                               origen[1] + (ocupadas[:, 0] + 0.5) * resolucion))
    rel = paredes - c
    dist = np.hypot(rel[:, 0], rel[:, 1])
    q1 = paredes[int(np.argmin(dist))]
    opuestas = rel @ (q1 - c) < 0.0
    if not np.any(opuestas):
        return None
    q2 = paredes[np.flatnonzero(opuestas)[int(np.argmin(dist[opuestas]))]]
    medio = (q1 + q2) / 2.0
    u = (q2 - q1) / max(float(np.hypot(*(q2 - q1))), 1e-9)
    normal = np.array([-u[1], u[0]])
    if np.dot(normal, d) < 0.0:
        normal = -normal
    return (float(medio[0]), float(medio[1])), (float(normal[0]), float(normal[1]))
