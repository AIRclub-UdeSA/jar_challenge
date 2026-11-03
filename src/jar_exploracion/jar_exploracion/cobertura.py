"""Cobertura visual sobre el mapa conocido (ADR-0005), sin ROS.

Una grilla gruesa del mapa guarda qué celdas libres ya entraron en el campo de
la cámara. El próximo goal es la pose libre y alcanzable desde la que más
superficie sin ver entra en el cono de la cámara, menos un costo por distancia.
La distancia es por los pasillos (Dijkstra sobre la grilla gruesa), no en línea
recta: en un laberinto, un punto "cerca" del otro lado de una pared puede quedar
a una vuelta entera.

Convenciones: fila i = eje y, columna j = eje x, como en nav_msgs/OccupancyGrid.
"""
import heapq
import math

import numpy as np
from scipy import ndimage

from jar_exploracion import zonas as zonas_mod


class Grilla:
    def __init__(self, datos, ancho, alto, resolucion, origen_x, origen_y,
                 resolucion_gruesa=0.10, umbral_ocupado=50):
        fino = np.asarray(datos, dtype=np.int16).reshape(alto, ancho)
        f = max(1, int(round(resolucion_gruesa / resolucion)))
        alto_g, ancho_g = -(-alto // f), -(-ancho // f)
        relleno = np.full((alto_g * f, ancho_g * f), -1, dtype=np.int16)
        relleno[:alto, :ancho] = fino
        bloqueado = (relleno < 0) | (relleno >= umbral_ocupado)
        # Una celda gruesa es obstáculo si alguna de sus celdas finas lo es (o es desconocida).
        self.ocupado_mapa = bloqueado.reshape(alto_g, f, ancho_g, f).any(axis=(1, 3))
        # Distancia (m) a la pared del mapa más cercana, en la grilla fina: para reconocer en
        # el costmap las paredes que el LiDAR vuelve a ver (corridas por el error de AMCL).
        self.fino = (resolucion, float(origen_x), float(origen_y))
        self.distancia_a_pared = ndimage.distance_transform_edt(
            ~((fino >= umbral_ocupado) | (fino < 0))) * resolucion
        self.res = resolucion * f
        self.origen = (float(origen_x), float(origen_y))
        self.visto = np.zeros_like(self.ocupado_mapa)
        self.visitados = []                      # goals a los que ya se llegó: no se repiten
        self.bloqueo_nav = np.zeros_like(self.ocupado_mapa)
        self.zona_previa = None                  # celdas de la zona que se estaba cubriendo
        self._recalcular(self.ocupado_mapa)

    def _recalcular(self, ocupado):
        self.ocupado = ocupado
        self.libre = ~ocupado
        self.despeje = ndimage.distance_transform_edt(self.libre) * self.res

    def actualizar_costmap(self, datos, ancho, alto, resolucion, origen_x, origen_y,
                           letal=100, despeje_paso=0.14, cerca_de_pared=0.15):
        """Suma lo que Nav2 ve y el mapa no (cubos de color): tapa la cámara y corta caminos.

        `datos` es el costmap global como OccupancyGrid (0-100). Una celda gruesa queda
        ocupada si tiene alguna celda letal, y bloqueada para navegar si en ninguna parte de
        ella el centro del robot queda a `despeje_paso` de lo letal (medido en la grilla fina:
        un paso de 2 × `despeje_paso`). Es permisivo a propósito: el error de AMCL corre las
        cajas respecto de las paredes; los huecos que de verdad no entran los mide la misión
        con el LiDAR y los tapa (/jar/bloqueos, que acá llegan como letales).

        Lo letal a menos de `cerca_de_pared` de una pared del mapa no se usa: es la misma pared
        vista por el LiDAR, corrida por el error de AMCL, y engordaba las paredes hasta cerrar
        pasos por donde el robot entra.
        """
        costo = np.asarray(datos, dtype=np.int16).reshape(alto, ancho)
        res_m, ox_m, oy_m = self.fino
        cy = origen_y + (np.arange(alto) + 0.5) * resolucion
        cx = origen_x + (np.arange(ancho) + 0.5) * resolucion
        mi = np.clip(np.floor((cy - oy_m) / res_m).astype(int), 0,
                     self.distancia_a_pared.shape[0] - 1)
        mj = np.clip(np.floor((cx - ox_m) / res_m).astype(int), 0,
                     self.distancia_a_pared.shape[1] - 1)
        a_pared = self.distancia_a_pared[mi[:, None], mj[None, :]]
        costo = np.where((costo >= letal) & (a_pared < cerca_de_pared), 0, costo)
        libre = (costo < letal) & (a_pared > 0)       # ni letal propio ni pared del mapa
        holgura_fina = ndimage.distance_transform_edt(libre) * resolucion
        alto_g, ancho_g = self.forma
        sub = 4                                  # muestras por lado en cada celda gruesa
        k = (np.arange(sub) + 0.5) / sub
        xs = self.origen[0] + (np.arange(ancho_g)[:, None] + k[None, :]).ravel() * self.res
        ys = self.origen[1] + (np.arange(alto_g)[:, None] + k[None, :]).ravel() * self.res
        jj = np.floor((xs - origen_x) / resolucion).astype(int)
        ii = np.floor((ys - origen_y) / resolucion).astype(int)
        en_j = (jj >= 0) & (jj < ancho)
        en_i = (ii >= 0) & (ii < alto)
        fi, fj = np.clip(ii, 0, alto - 1)[:, None], np.clip(jj, 0, ancho - 1)[None, :]
        adentro = en_i[:, None] & en_j[None, :]
        muestra = np.where(adentro, costo[fi, fj], 0).reshape(alto_g, sub, ancho_g, sub)
        holgura = np.where(adentro, holgura_fina[fi, fj], np.inf)
        holgura = holgura.reshape(alto_g, sub, ancho_g, sub)
        letales = (muestra >= letal).any(axis=(1, 3))
        self.bloqueo_nav = ~(holgura >= despeje_paso).any(axis=(1, 3))
        self._recalcular(self.ocupado_mapa | letales)

    @property
    def forma(self):
        return self.ocupado.shape

    def celda(self, x, y):
        return (int(math.floor((y - self.origen[1]) / self.res)),
                int(math.floor((x - self.origen[0]) / self.res)))

    def centro(self, i, j):
        return (self.origen[0] + (j + 0.5) * self.res, self.origen[1] + (i + 0.5) * self.res)

    def dentro(self, i, j):
        return 0 <= i < self.forma[0] and 0 <= j < self.forma[1]

    # Rayos ----------------------------------------------------------------------
    def _rayos(self, xs, ys, angulos, alcance):
        """Celdas libres alcanzadas por cada rayo antes de chocar.

        Devuelve (ii, jj, validos, distancias) de forma C×K×S para C orígenes y K ángulos.
        """
        paso = self.res * 0.5
        dist = np.arange(1, int(alcance / paso) + 1) * paso
        px = xs[:, None, None] + dist[None, None, :] * np.cos(angulos)[None, :, None]
        py = ys[:, None, None] + dist[None, None, :] * np.sin(angulos)[None, :, None]
        jj = np.floor((px - self.origen[0]) / self.res).astype(int)
        ii = np.floor((py - self.origen[1]) / self.res).astype(int)
        en_mapa = (ii >= 0) & (ii < self.forma[0]) & (jj >= 0) & (jj < self.forma[1])
        ii_c, jj_c = np.clip(ii, 0, self.forma[0] - 1), np.clip(jj, 0, self.forma[1] - 1)
        libre = en_mapa & self.libre[ii_c, jj_c]
        validos = np.cumprod(libre, axis=2).astype(bool)   # se corta en el primer obstáculo
        return ii_c, jj_c, validos, dist

    def marcar_visto(self, x, y, yaw, fov, alcance, radio_propio=0.2, rayos=31):
        """Marca lo que entra en el cono de la cámara desde (x, y, yaw)."""
        angulos = yaw + np.linspace(-fov / 2.0, fov / 2.0, rayos)
        ii, jj, validos, _ = self._rayos(np.array([x]), np.array([y]), angulos, alcance)
        self.visto[ii[validos], jj[validos]] = True
        # Donde está el robot no puede haber una caja.
        i0, j0 = self.celda(x, y)
        r = int(math.ceil(radio_propio / self.res))
        ia, ib = max(0, i0 - r), min(self.forma[0], i0 + r + 1)
        ja, jb = max(0, j0 - r), min(self.forma[1], j0 + r + 1)
        yy, xx = np.mgrid[ia:ib, ja:jb]
        cerca = (yy - i0) ** 2 + (xx - j0) ** 2 <= r * r
        self.visto[yy[cerca], xx[cerca]] = True

    def llegada(self, gx, gy, gyaw, fov, alcance):
        """El robot llegó a un goal y miró: se da por visto lo que se ve desde la pose del
        goal y el lugar no se vuelve a proponer.

        Sin esto, el robot llega unos centímetros corrido (tolerancia de Nav2, error de AMCL),
        una esquina le tapa las celdas que el goal prometía, siguen sin ver y el explorador lo
        manda otra vez al mismo lugar: vueltas sobre zonas ya exploradas.
        """
        self.marcar_visto(gx, gy, gyaw, fov, alcance)
        self.visitados.append((float(gx), float(gy)))

    # Inspección de candidatos ---------------------------------------------------
    # La percepción lleva los candidatos (todo objeto del LiDAR que el mapa no tiene) y decide
    # si son víctimas cuando la cámara los tiene de frente. Que el piso de alrededor haya
    # entrado en el cono de la cámara no alcanza: si el robot pasó pegado o de costado, la
    # caja nunca estuvo entera en la imagen. Acá se elige desde dónde mirarlos.
    def _vista_libre(self, x0, y0, x1, y1, antes=0.2):
        """¿Nada tapa la línea de (x0, y0) a (x1, y1), salvo los últimos `antes` m (el objeto)?"""
        d = math.hypot(x1 - x0, y1 - y0)
        for s in np.arange(self.res * 0.5, max(d - antes, 0.0), self.res * 0.5):
            i, j = self.celda(x0 + (x1 - x0) * s / d, y0 + (y1 - y0) * s / d)
            if not self.dentro(i, j) or self.ocupado[i, j]:
                return False
        return True

    def mejor_inspeccion(self, x, y, objetos, despeje_min=0.10, d_min=0.7, d_max=1.3,
                         despeje_goal=0.25, descartados=(), radio_descarte=0.30, rangos=None,
                         fija=None, despeje_amplio=0.22, penalidad_paso=0.0):
        """Pose para mirar de frente el candidato más cercano por camino.

        `objetos`: {id: (x, y)} de los candidatos sin inspeccionar. `rangos`: {id: (d_min,
        d_max)} para los que hay que mirar desde otra distancia (rojo cortado por el borde:
        más lejos); prefiere el medio del rango. `fija`: (id, (gx, gy, yaw)) de la inspección
        en curso; si ese candidato sigue pendiente y la pose sigue sirviendo, se devuelve la
        misma (con el yaw hacia el objeto) en vez de buscar otra: buscarla de nuevo desde cada
        pose del robot la hacía saltar de un lado al otro de un hueco angosto. Con
        `penalidad_paso` (m), una pose a la que sólo se llega pasando a menos de `despeje_amplio`
        de las paredes (un hueco angosto: maniobra lenta, Nav2 casi no planifica) cuesta eso de
        más: en laberinto_puertas cruzaba la puerta de A para mirar la caja de esa puerta teniendo
        lugar de sobra de su lado, retrocediendo. Devuelve ((gx,
        gy, yaw hacia el objeto), id) o (None, None), y la lista de ids que no se pueden mirar
        desde ningún lado (para no volver a intentarlo).
        """
        rangos = rangos or {}
        sin_pose = []
        if not objetos:
            return None, None, sin_pose
        navegable = self._componente((self.despeje >= despeje_min) & ~self.bloqueo_nav, x, y)
        camino = self.distancia_por_camino(navegable, x, y)
        costo_camino = camino
        if penalidad_paso > 0:
            amplio = self._componente((self.despeje >= despeje_amplio) & ~self.bloqueo_nav, x, y)
            camino_amplio = self.distancia_por_camino(amplio, x, y)
            costo_camino = np.where(np.isfinite(camino_amplio), camino_amplio,
                                    camino + penalidad_paso)
        if fija is not None and fija[0] in objetos:
            ident, (gx, gy, _) = fija
            ox, oy = objetos[ident]
            lo, hi = rangos.get(ident, (d_min, d_max))
            i, j = self.celda(gx, gy)
            if (self.dentro(i, j) and navegable[i, j] and self.despeje[i, j] >= despeje_goal
                    and np.isfinite(camino[i, j])
                    and all(math.hypot(gx - dx, gy - dy) >= radio_descarte
                            for dx, dy in descartados)
                    and lo <= math.hypot(ox - gx, oy - gy) <= hi
                    and self._vista_libre(gx, gy, ox, oy)):
                return (gx, gy, math.atan2(oy - gy, ox - gx)), ident, sin_pose
        cand = np.argwhere(navegable & (self.despeje >= despeje_goal))
        if len(cand) == 0:
            return None, None, sin_pose
        cx = self.origen[0] + (cand[:, 1] + 0.5) * self.res
        cy = self.origen[1] + (cand[:, 0] + 0.5) * self.res
        lejos = np.ones(len(cand), dtype=bool)
        for dx, dy in descartados:
            lejos &= np.hypot(cx - dx, cy - dy) >= radio_descarte
        mejor = None
        for ident, (ox, oy) in objetos.items():
            d = np.hypot(cx - ox, cy - oy)
            lo, hi = rangos.get(ident, (d_min, d_max))
            preferida = 1.0 if ident not in rangos else (lo + hi) / 2.0
            ok = np.flatnonzero(lejos & (d >= lo) & (d <= hi)
                                & np.isfinite(camino[cand[:, 0], cand[:, 1]]))
            ok = [k for k in ok if self._vista_libre(cx[k], cy[k], ox, oy)]
            if not ok:
                sin_pose.append(ident)                  # no hay desde dónde mirarlo
                continue
            # El más cercano por camino; a igual camino, el que queda a ~1 m del objeto.
            costo = [costo_camino[cand[k, 0], cand[k, 1]] + abs(d[k] - preferida) for k in ok]
            k = ok[int(np.argmin(costo))]
            if mejor is None or min(costo) < mejor[0]:
                mejor = (min(costo), (float(cx[k]), float(cy[k]),
                                      float(math.atan2(oy - cy[k], ox - cx[k]))), ident)
        if mejor is None:
            return None, None, sin_pose
        return mejor[1], mejor[2], sin_pose

    # Alcanzabilidad -------------------------------------------------------------
    def _componente(self, mascara, x, y):
        etiquetas, _ = ndimage.label(mascara)
        i, j = self.celda(x, y)
        if not self.dentro(i, j):
            return np.zeros_like(mascara)
        etiqueta = etiquetas[i, j]
        if etiqueta == 0:
            # El robot cae fuera de la máscara (pegado a una pared): la componente más cercana.
            libres = np.argwhere(etiquetas > 0)
            if len(libres) == 0:
                return np.zeros_like(mascara)
            k = np.argmin((libres[:, 0] - i) ** 2 + (libres[:, 1] - j) ** 2)
            etiqueta = etiquetas[tuple(libres[k])]
        return etiquetas == etiqueta

    def zona_libre(self, x, y):
        """Celdas libres conectadas con el robot: el universo de la cobertura."""
        return self._componente(self.libre, x, y)

    def cobertura(self, x, y):
        zona = self.zona_libre(x, y)
        total = int(zona.sum())
        return 1.0 if total == 0 else float((self.visto & zona).sum()) / total

    def distancia_por_camino(self, mascara, x, y):
        """Metros desde (x, y) a cada celda de `mascara` moviéndose por ella (8 vecinos).

        Arranca en la celda de la máscara más cercana al robot. Las celdas que no se
        alcanzan quedan en inf.
        """
        dist = np.full(self.forma, np.inf)
        libres = np.argwhere(mascara)
        if len(libres) == 0:
            return dist
        i, j = self.celda(x, y)
        k = np.argmin((libres[:, 0] - i) ** 2 + (libres[:, 1] - j) ** 2)
        inicio = tuple(int(v) for v in libres[k])
        dist[inicio] = 0.0
        cola = [(0.0, inicio)]
        vecinos = [(di, dj, math.hypot(di, dj) * self.res)
                   for di in (-1, 0, 1) for dj in (-1, 0, 1) if di or dj]
        alto, ancho = self.forma
        while cola:
            d, (i, j) = heapq.heappop(cola)
            if d > dist[i, j]:
                continue
            for di, dj, paso in vecinos:
                a, b = i + di, j + dj
                if 0 <= a < alto and 0 <= b < ancho and mascara[a, b] and d + paso < dist[a, b]:
                    dist[a, b] = d + paso
                    heapq.heappush(cola, (d + paso, (a, b)))
        return dist

    def zonas_sin_ver(self, navegable, x, y, cuantas=4):
        """Las zonas alcanzables más grandes sin ver: centro, celdas y camino hasta ellas."""
        etiquetas, n = ndimage.label(navegable & ~self.visto, structure=np.ones((3, 3)))
        if n == 0:
            return []
        camino = self.distancia_por_camino(navegable, x, y)
        zonas = []
        for k in range(1, n + 1):
            celdas = np.argwhere(etiquetas == k)
            i, j = celdas.mean(axis=0)
            d = camino[celdas[:, 0], celdas[:, 1]]
            zonas.append({'x': round(self.origen[0] + (j + 0.5) * self.res, 2),
                          'y': round(self.origen[1] + (i + 0.5) * self.res, 2),
                          'celdas': int(len(celdas)),
                          'camino_m': round(float(d[np.isfinite(d)].min()), 2)
                          if np.isfinite(d).any() else None})
        return sorted(zonas, key=lambda z: -z['celdas'])[:cuantas]

    # Orden de las zonas -----------------------------------------------------------
    def navegable(self, x, y, despeje_min):
        return self._componente((self.despeje >= despeje_min) & ~self.bloqueo_nav, x, y)

    def zona_objetivo(self, x, y, despeje_min=0.20, vel_m_s=0.15, cobertura_m2_s=0.15,
                      celdas_min=20, max_zonas=7, histeresis=0.1, posibles=(),
                      peso_posible=100.0, victimas=1):
        """La zona sin ver por la que conviene seguir (zonas.py): máscara de sus celdas, o
        None si no hay zonas que planificar. Deja el plan en `self.plan` para el registro.

        `posibles`: (x, y) de objetos que el LiDAR vio de lejos y no se confirmaron; cada uno
        suma `peso_posible` celdas a su zona (es más probable que haya una caja ahí).
        """
        self.plan = []
        navegable = self.navegable(x, y, despeje_min)
        etiquetas, ids = zonas_mod.zonas_sin_ver(navegable, self.visto, celdas_min)
        if not ids:
            self.zona_previa = None
            return None
        pesos = {k: float((etiquetas == k).sum()) for k in ids}
        for px, py in posibles:
            i, j = self.celda(px, py)
            # El objeto está en una celda ocupada o al borde: la zona más cercana a él.
            for di in range(-3, 4):
                for dj in range(-3, 4):
                    a, b = i + di, j + dj
                    if self.dentro(a, b) and etiquetas[a, b] in pesos:
                        pesos[etiquetas[a, b]] += peso_posible
                        break
                else:
                    continue
                break
        ids = sorted(ids, key=lambda k: -pesos[k])[:max_zonas]
        matriz, indice = zonas_mod.grafo(navegable, self.res)
        i0, j0 = self.celda(x, y)
        libres = np.argwhere(navegable)
        cerca = libres[np.argmin((libres[:, 0] - i0) ** 2 + (libres[:, 1] - j0) ** 2)]
        desde_robot = zonas_mod.distancias(matriz, indice, cerca.reshape(1, 2))
        celdas = {k: np.argwhere(etiquetas == k) for k in ids}
        viaje_robot = [float(desde_robot[celdas[k][:, 0], celdas[k][:, 1]].min()) / vel_m_s
                       for k in ids]
        # De a a b: promedio, desde las celdas de a, del camino hasta la celda más cercana de b.
        hasta = {b: zonas_mod.distancias(matriz, indice, celdas[b]) for b in ids}
        viaje_entre = [[float(np.mean(hasta[b][celdas[a][:, 0], celdas[a][:, 1]])) / vel_m_s
                        for b in ids] for a in ids]
        area = self.res ** 2
        tiempos = [len(celdas[k]) * area / cobertura_m2_s for k in ids]
        actual = None
        if self.zona_previa is not None:
            solapes = [int((self.zona_previa & (etiquetas == k)).sum()) for k in ids]
            if max(solapes) > 0:
                actual = int(np.argmax(solapes))
        orden, _ = zonas_mod.ordenar([pesos[k] for k in ids], tiempos, viaje_robot,
                                     viaje_entre, victimas=victimas, actual=actual,
                                     histeresis=histeresis)
        elegida = etiquetas == ids[orden[0]]
        self.zona_previa = elegida
        for n, o in enumerate(orden):
            ci, cj = celdas[ids[o]].mean(axis=0)
            cx, cy = self.centro(ci, cj)
            self.plan.append({'x': round(cx, 2), 'y': round(cy, 2),
                              'celdas': int(len(celdas[ids[o]])),
                              'peso': round(pesos[ids[o]], 0),
                              'viaje_s': round(viaje_robot[o] if n == 0
                                               else viaje_entre[orden[n - 1]][o], 1)})
        return elegida

    # Próximo goal ---------------------------------------------------------------
    def mejor_goal(self, x, y, fov, alcance, despeje_min=0.20, separacion=0.30,
                   costo_distancia=4.0, descartados=(), radio_descarte=0.30,
                   rayos=72, ganancia_minima=3.0, radio_local=None, ganancia_local=None,
                   despeje_goal=None, fraccion_local=0.0, zona=None):
        """Devuelve (gx, gy, gyaw, ganancia) o None si no queda nada que valga la pena ver.

        La ganancia está en celdas gruesas sin ver; el costo es por metro de camino.
        `despeje_min` decide por dónde se puede pasar; `despeje_goal` (si se da) es lo que
        se le exige a la pose del goal: en el medio del pasillo, no pegada a una pared,
        donde Nav2 ve choque al girar y se queda quieto. Con `zona` (máscara, de
        `zona_objetivo`), sólo cuenta lo que se ve de esa zona y no hay regla local.
        """
        navegable = self.navegable(x, y, despeje_min)
        self.diagnostico = {'navegables': int(navegable.sum()),
                            'sin_ver_alcanzables': int((navegable & ~self.visto).sum())}
        paso = max(1, int(round(separacion / self.res)))
        donde = navegable
        if despeje_goal is not None and (navegable & (self.despeje >= despeje_goal)).any():
            donde = navegable & (self.despeje >= despeje_goal)
        cand = np.argwhere(donde)
        cand = cand[(cand[:, 0] % paso == 0) & (cand[:, 1] % paso == 0)]
        if len(cand) == 0:
            return None
        xs = self.origen[0] + (cand[:, 1] + 0.5) * self.res
        ys = self.origen[1] + (cand[:, 0] + 0.5) * self.res
        descartados = list(descartados) + self.visitados
        if descartados:
            d = np.asarray(descartados, dtype=float).reshape(-1, 2)
            lejos = np.min(np.hypot(xs[:, None] - d[None, :, 0], ys[:, None] - d[None, :, 1]),
                           axis=1) >= radio_descarte
            xs, ys, cand = xs[lejos], ys[lejos], cand[lejos]
            if len(xs) == 0:
                return None
        camino = self.distancia_por_camino(navegable, x, y)[cand[:, 0], cand[:, 1]]

        angulos = np.linspace(0.0, 2.0 * math.pi, rayos, endpoint=False)
        ii, jj, validos, dist = self._rayos(xs, ys, angulos, alcance)
        sin_ver = validos & ~self.visto[ii, jj]
        if zona is not None:
            sin_ver &= zona[ii, jj]
            radio_local = None
        # Elemento de área en polares (r·dr·dθ), pasado a celdas: aproxima la superficie nueva.
        d_theta = 2.0 * math.pi / rayos
        area = dist * (self.res * 0.5) * d_theta / (self.res ** 2)
        por_rayo = (sin_ver * area[None, None, :]).sum(axis=2)          # C×K
        ancho = max(1, int(round(fov / d_theta)))
        circular = np.concatenate([por_rayo, por_rayo[:, :ancho - 1]], axis=1)
        acumulado = np.cumsum(np.pad(circular, ((0, 0), (1, 0))), axis=1)
        ventanas = acumulado[:, ancho:ancho + rayos] - acumulado[:, :rayos]   # C×K
        k_mejor = np.argmax(ventanas, axis=1)
        ganancia = ventanas[np.arange(len(xs)), k_mejor]
        self.diagnostico.update(candidatos=int(len(xs)), ganancia_max=float(ganancia.max()))
        puntaje = ganancia - costo_distancia * camino
        puntaje_global = puntaje.copy()
        local = False
        if radio_local is not None:
            # Terminar la zona antes de irse: si cerca queda algo que valga la pena, sólo cerca.
            umbral = ganancia_minima if ganancia_local is None else ganancia_local
            # Sólo si lo cercano vale la pena frente a lo mejor disponible: si cerca quedan
            # migajas y a un paso hay una sala entera, compiten todos con el costo por metro.
            umbral = max(umbral, fraccion_local * float(ganancia.max()))
            cerca = (camino <= radio_local) & (ganancia >= umbral)
            if cerca.any():
                puntaje = np.where(cerca, puntaje, -np.inf)
                local = True
        c = int(np.argmax(puntaje))
        if ganancia[c] < ganancia_minima:
            # El mejor puntaje puede no tener ganancia; ¿queda algún candidato que sí?
            utiles = np.flatnonzero(ganancia >= ganancia_minima)
            if len(utiles) == 0:
                return None
            c = int(utiles[np.argmax(puntaje[utiles])])
            local = False

        # Para el registro de decisiones: lo elegido y las mejores alternativas.
        def resumen(k):
            return {'x': round(float(xs[k]), 2), 'y': round(float(ys[k]), 2),
                    'ganancia': round(float(ganancia[k]), 1),
                    'camino_m': round(float(camino[k]), 2)}
        mejores = [int(k) for k in np.argsort(-puntaje_global)[:4] if int(k) != c][:3]
        if zona is not None:
            self.diagnostico['plan'] = getattr(self, 'plan', [])
        self.diagnostico.update(elegido=resumen(c), local=local,
                                alternativas=[resumen(k) for k in mejores],
                                zonas_sin_ver=self.zonas_sin_ver(navegable, x, y))
        yaw = angulos[k_mejor[c]] + (ancho - 1) * d_theta / 2.0
        return (float(xs[c]), float(ys[c]), float(math.atan2(math.sin(yaw), math.cos(yaw))),
                float(ganancia[c]))
