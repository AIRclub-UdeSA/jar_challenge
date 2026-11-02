"""Qué falta ver del laberinto y a dónde conviene ir para verlo.

Trabaja sobre la grilla de /map: lleva la cuenta de lo que el lidar ya barrió, detecta objetos
que no están en el mapa (las cajas son candidatas a víctima) y elige la próxima meta. Es lógica
pura (sin ROS), para poder probarla con pytest. Las poses son (x, y, yaw) en el marco del mapa.
"""
import heapq
import math

import cv2
from equipo_jar.memoria_camara import veredicto
import numpy as np

# Vecinos de una celda: (fila, columna, largo del paso en celdas).
VECINOS = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
           (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)]
PASO_GRILLA = 2             # el Dijkstra usa una celda de cada dos: alcanza y es mucho más rápido
SECTORES_VISTA = 8          # un objeto se confirma visto desde dos de estos sectores


class Cobertura:
    def __init__(self, grilla, resolucion, origen_x, origen_y, largada, radio_paso_m,
                 despeje_meta_m, alcance_vista_m=3.0, alcance_vista_fino_m=1.8,
                 despeje_cobertura_m=0.36, separacion_candidatos_m=0.3, ganancia_minima=40,
                 margen_pared_m=0.20):
        """`grilla` es el OccupancyGrid como matriz filas×columnas (-1 desconocido, 0 libre).

        `margen_pared_m`: lo que el lidar ve a menos de esto de una pared del mapa no cuenta como
        objeto (ver acumular_lidar).
        """
        self.res = resolucion
        self.margen_pared_m = margen_pared_m
        self.ox = origen_x
        self.oy = origen_y
        self.alto, self.ancho = grilla.shape
        self.libre = grilla == 0
        self.ocupado = ~self.libre
        self.radio_paso_m = radio_paso_m
        self.despeje_meta_m = despeje_meta_m
        self.ganancia_minima = ganancia_minima

        self.d_mapa = cv2.distanceTransform(
            self.libre.astype(np.uint8), cv2.DIST_L2, 5) * self.res
        _, etiquetas = cv2.connectedComponents(self.libre.astype(np.uint8))
        i, j = self.celda(*largada)
        self.alcanzable = etiquetas == etiquetas[i, j]
        self.objetivo = self.alcanzable & (self.d_mapa >= 0.10)
        self.total = max(int(self.objetivo.sum()), 1)

        self.visto = np.zeros_like(self.libre)
        self.nuevos = np.zeros_like(self.libre)     # celdas con obstáculos que no están en /map
        self.cuenta = np.zeros(grilla.shape, np.int16)
        self.vistas = np.zeros(grilla.shape, np.uint8)
        self.lista_negra = set()
        self.fase = 1
        self.alcance_vista_m = alcance_vista_m
        self._actualizar_despeje()

        # Puntos de observación: centros de pasillo, con lo que se ve desde cada uno en cada fase.
        self.candidatos, self.visibles, self.visibles_fino = [], [], []
        paso = max(int(separacion_candidatos_m / self.res), 1)
        centro = (self.d_mapa >= despeje_cobertura_m) & self.alcanzable
        for i in range(0, self.alto, paso):
            for j in range(0, self.ancho, paso):
                if centro[i, j]:
                    x, y = self.xy(i, j)
                    self.candidatos.append((i, j))
                    self.visibles.append(self.rayos(x, y, alcance_vista_m) & self.objetivo)
                    self.visibles_fino.append(
                        self.rayos(x, y, alcance_vista_fino_m) & self.objetivo)
        self.alcance_fino_m = alcance_vista_fino_m

    def celda(self, x, y):
        return int(round((y - self.oy) / self.res)), int(round((x - self.ox) / self.res))

    def xy(self, i, j):
        return self.ox + j * self.res, self.oy + i * self.res

    def porcentaje_visto(self):
        return 100.0 * (self.visto & self.objetivo).sum() / self.total

    def despeje_en(self, x, y):
        i, j = self.celda(x, y)
        return float(self.despeje[min(max(i, 0), self.alto - 1), min(max(j, 0), self.ancho - 1)])

    def rayos(self, x, y, alcance):
        """Celdas que se ven desde (x, y): cada rayo corta en la primera pared."""
        angulos = np.arange(0, 2 * math.pi, math.radians(1.5))
        n = int(alcance / self.res)
        t = np.arange(1, n + 1) * self.res
        jj = np.round((x + np.outer(np.cos(angulos), t) - self.ox) / self.res).astype(int)
        ii = np.round((y + np.outer(np.sin(angulos), t) - self.oy) / self.res).astype(int)
        fuera = (ii < 0) | (ii >= self.alto) | (jj < 0) | (jj >= self.ancho)
        ii, jj = np.clip(ii, 0, self.alto - 1), np.clip(jj, 0, self.ancho - 1)
        pared = self.ocupado[ii, jj] | fuera
        corte = np.where(pared.any(axis=1), pared.argmax(axis=1), n)
        antes = np.arange(n)[None, :] < corte[:, None]
        visibles = np.zeros(self.libre.shape, bool)
        visibles[ii[antes], jj[antes]] = True
        return visibles

    def marcar_visto(self, x, y):
        self.visto |= self.rayos(x, y, self.alcance_vista_m) & self.objetivo

    def _actualizar_despeje(self):
        d = self.d_mapa
        if self.nuevos.any():
            d = np.minimum(d, cv2.distanceTransform(
                (~self.nuevos).astype(np.uint8), cv2.DIST_L2, 5) * self.res)
        self.despeje = d
        self.transitable = (d >= self.radio_paso_m) & self.alcanzable

    def acumular_lidar(self, pose, wx, wy):
        """Cuenta los puntos del lidar (ya en el mapa) que caen donde /map dice que está libre."""
        x, y, _ = pose
        ii = np.round((wy - self.oy) / self.res).astype(int)
        jj = np.round((wx - self.ox) / self.res).astype(int)
        dentro = (ii >= 0) & (ii < self.alto) & (jj >= 0) & (jj < self.ancho)
        ii, jj = ii[dentro], jj[dentro]
        # Pegado a una pared del mapa no cuenta: ahí un error chico de pose ya parece un objeto.
        # Más margen saca más pedazos de pared que faltan en el mapa, pero también dejaría sin
        # inspeccionar una caja contra una pared (el reglamento no dice que no pueda haberlas).
        anomalo = self.objetivo[ii, jj] & (self.d_mapa[ii, jj] > self.margen_pared_m)
        celdas = np.unique(np.stack([ii[anomalo], jj[anomalo]], axis=1), axis=0)
        if not len(celdas):
            return
        self.cuenta[celdas[:, 0], celdas[:, 1]] += 1
        cy, cx = self.oy + celdas[:, 0] * self.res, self.ox + celdas[:, 1] * self.res
        sector = ((np.arctan2(cy - y, cx - x) + math.pi) / (2 * math.pi) * SECTORES_VISTA)
        sector = sector.astype(int) % SECTORES_VISTA
        self.vistas[celdas[:, 0], celdas[:, 1]] |= (1 << sector).astype(np.uint8)

    def objetos_nuevos(self):
        """Devuelve el centro (x, y) de cada objeto compacto confirmado y lo agrega a los obstáculos.

        Un fantasma de localización no aguanta dos puntos de vista, y algo largo es un error de pose
        contra una pared, no un objeto.
        """
        sectores = np.vectorize(lambda b: bin(b).count('1'), otypes=[np.uint8])(self.vistas)
        firmes = (self.cuenta >= 8) & (sectores >= 2)
        if self.fase == 2:
            firmes |= self.cuenta >= 14
        firmes = firmes.astype(np.uint8)
        n, etiquetas = cv2.connectedComponents(cv2.dilate(firmes, np.ones((7, 7), np.uint8)))
        objetos = []
        for etiqueta in range(1, n):
            mascara = (etiquetas == etiqueta) & (firmes > 0)
            if mascara.sum() < 4:
                continue
            ii, jj = np.nonzero(mascara)
            if max(ii.max() - ii.min(), jj.max() - jj.min()) * self.res > 0.45:
                continue
            if not self.nuevos[mascara].all():
                self.nuevos |= mascara
                self._actualizar_despeje()
            objetos.append(self.xy(ii.mean(), jj.mean()))
        return objetos

    def veredicto_camara(self, cx, cy, camara, veces_min, celdas_min, radio_m=0.30):
        """Qué vio la cámara del objeto en (cx, cy): memoria_camara.ROJO_VISTO, SIN_ROJO o SIN_VER.

        `camara`: la grilla de /equipo/vistas_camara, del mismo tamaño que el mapa. Las celdas del
        objeto son las que el lidar vio fuera del mapa (al menos 2 veces) a menos de `radio_m` de
        su centro: así no cuentan las paredes de alrededor, que la cámara ve sin rojo aunque el
        objeto no lo haya visto nunca.
        """
        i, j = self.celda(cx, cy)
        n = int(math.ceil(radio_m / self.res)) + 1
        f0, f1 = max(i - n, 0), min(i + n + 1, self.alto)
        c0, c1 = max(j - n, 0), min(j + n + 1, self.ancho)
        if f0 >= f1 or c0 >= c1:
            return veredicto(np.zeros((1, 1)), np.zeros((1, 1), bool), veces_min, celdas_min)
        ii, jj = np.mgrid[f0:f1, c0:c1]
        disco = np.hypot(ii - i, jj - j) * self.res <= radio_m
        objeto = disco & (self.cuenta[f0:f1, c0:c1] >= 2)
        return veredicto(camara[f0:f1, c0:c1], objeto, veces_min, celdas_min)

    def olvidar_objetos(self):
        self.nuevos[:] = False
        self.cuenta[:] = 0
        self._actualizar_despeje()

    def pasar_a_fase_fina(self):
        """Segunda pasada: se vuelve a recorrer todo mirando más de cerca."""
        self.fase = 2
        self.alcance_vista_m = self.alcance_fino_m
        self.visibles = self.visibles_fino
        self.visto[:] = False
        self.lista_negra.clear()

    def distancias_desde(self, x, y):
        """Dijkstra por las celdas transitables: distancia (m) a cada celda alcanzable."""
        g = PASO_GRILLA
        i, j = self.celda(x, y)
        inicio = (i - i % g, j - j % g)
        if not self.transitable[inicio]:
            # El robot quedó cerca de algo: se arranca desde la celda más despejada de alrededor.
            f0, c0 = max(inicio[0] - 8, 0), max(inicio[1] - 8, 0)
            zona = self.despeje[f0:inicio[0] + 9:g, c0:inicio[1] + 9:g]
            k = np.unravel_index(zona.argmax(), zona.shape)
            inicio = (f0 + k[0] * g, c0 + k[1] * g)
        distancias = {inicio: 0.0}
        cola = [(0.0, inicio)]
        while cola:
            d, (i, j) = heapq.heappop(cola)
            if d > distancias[(i, j)]:
                continue
            for a, b, largo in VECINOS:
                ni, nj = i + a * g, j + b * g
                if 0 <= ni < self.alto and 0 <= nj < self.ancho and self.transitable[ni, nj]:
                    nueva = d + largo * g * self.res
                    if nueva < distancias.get((ni, nj), math.inf):
                        distancias[(ni, nj)] = nueva
                        heapq.heappush(cola, (nueva, (ni, nj)))
        return distancias

    def meta_inspeccion(self, cx, cy, distancias):
        """Punto desde donde mirar el objeto (cx, cy): (x, y, yaw) o None si no hay ninguno."""
        g = PASO_GRILLA
        mejor = None
        for radio in (0.9, 1.1, 1.3):
            for k in range(16):
                angulo = k * math.pi / 8
                x, y = cx + radio * math.cos(angulo), cy + radio * math.sin(angulo)
                i, j = self.celda(x, y)
                c = (i - i % g, j - j % g)
                # Con poco despeje el robot llega, pero después no puede girar para mirar.
                if c not in distancias or self.despeje[c] < self.despeje_meta_m:
                    continue
                if not self._hay_linea_de_vista(x, y, cx, cy, radio):
                    continue
                if mejor is None or distancias[c] < mejor[0]:
                    px, py = self.xy(*c)
                    mejor = (distancias[c], px, py, math.atan2(cy - y, cx - x))
            if mejor:
                return mejor[1:]
        return None

    def _hay_linea_de_vista(self, x, y, cx, cy, radio):
        for t in np.linspace(0.15, 0.85, int(radio / 0.05)):
            if self.ocupado[self.celda(x + (cx - x) * t, y + (cy - y) * t)]:
                return False
        return True

    def meta_cobertura(self, distancias):
        """El candidato que más celdas nuevas muestra por metro recorrido: (índice, x, y) o None."""
        g = PASO_GRILLA
        mejor, puntaje_mejor = None, 0.0
        for k, (i, j) in enumerate(self.candidatos):
            c = (i - i % g, j - j % g)
            if c not in distancias or k in self.lista_negra \
                    or self.despeje[c] < self.despeje_meta_m:
                continue
            ganancia = self.ganancia(k)
            if ganancia < self.ganancia_minima:
                continue
            puntaje = ganancia / (distancias[c] + 0.3)
            if puntaje > puntaje_mejor:
                puntaje_mejor, mejor = puntaje, (k, c)
        if mejor is None:
            return None
        return (mejor[0], *self.xy(*mejor[1]))

    def ganancia(self, k):
        return int((self.visibles[k] & ~self.visto).sum())
