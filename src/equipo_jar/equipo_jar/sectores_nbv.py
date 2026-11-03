"""Orden y progreso de sectores BCD para el explorador NBV de competencia.

"""

import numpy as np
from collections import deque

from .planificador import (bcd, depurar_regiones, enriquecer_grafo,
                           ordenar_regiones, ruta_por_portales, _distancia_al_borde)


class SectoresNBV:
    def __init__(self, libre, seguro, resolucion, inicio, cobertura_objetivo,
                 area_minima_sector_m2=0.35, ancho_minimo_sector_m=0.45,
                 area_residual_ignorable_m2=0.09):
        self.libre = np.asarray(libre, dtype=bool)
        seguro = np.asarray(seguro, dtype=bool)
        if self.libre.shape != seguro.shape or self.libre.ndim != 2:
            raise ValueError('Mapa y máscara transitable incompatibles')
        if not 0 < cobertura_objetivo <= 1:
            raise ValueError('cobertura_objetivo debe estar entre 0 y 1')
        if area_residual_ignorable_m2 < 0:
            raise ValueError('area_residual_ignorable_m2 debe ser no negativa')
        c, f = inicio
        if not (0 <= f < seguro.shape[0] and 0 <= c < seguro.shape[1] and seguro[f, c]):
            filas, columnas = np.nonzero(seguro)
            if not filas.size:
                raise ValueError('Mapa sin celdas transitables')
            i = int(np.argmin((filas - f) ** 2 + (columnas - c) ** 2))
            c, f = int(columnas[i]), int(filas[i])
        regiones, _ = bcd(np.where(self.libre, 0, 100))
        depuradas, self.depuracion = depurar_regiones(
            regiones, seguro, (c, f), resolucion,
            area_minima_sector_m2, ancho_minimo_sector_m)
        self.regiones = np.asarray(depuradas, dtype=np.int32)
        self.orden, self.inaccesibles = ordenar_regiones(self.regiones, seguro, (c, f))
        _, self.conexiones = enriquecer_grafo(depuradas, resolucion)
        self.resolucion = resolucion
        self.cobertura_objetivo = cobertura_objetivo
        self.area_residual_ignorable_m2 = area_residual_ignorable_m2
        self.bloques_conocidos = np.zeros_like(self.libre)
        self.indice = 0
        self.indice_ordenado = -1
        self.largada = (c, f)          # el recorrido termina acá: hay que volver antes de los 10 min
        self.recorrido_planificado = []
        self.metros_planificados = 0.0
        self._cache_distancias = None
        self.resultados = {int(r): 'inaccesible' for r in self.inaccesibles}

    @property
    def actual(self):
        return self.orden[self.indice] if self.indice < len(self.orden) else None

    def mascara_actual(self):
        region = self.actual
        return None if region is None else (self.regiones == region) & ~self.bloques_conocidos

    def mascara_observacion(self):
        """Piso a mirar, aunque el robot no pueda entrar a esa zona.

        Tras recorrer los sectores, quedan las celdas excluidas por la
        depuración y los sectores sin entrada o sin vistas interiores útiles.
        El raycasting decide qué parte se puede ver desde posiciones seguras;
        el resto permanece sin observar, sin exigir cobertura completa.
        """
        if self.actual is not None:
            return self.mascara_actual()
        pendientes = [r for r, resultado in self.resultados.items()
                      if resultado in ('inaccesible', 'sin_vistas_alcanzables')]
        return (self.libre & ~self.bloques_conocidos
                & ((self.regiones < 0) | np.isin(self.regiones, pendientes)))

    def dentro_actual(self, celda):
        c, f = celda
        return (self.actual is not None and 0 <= f < self.regiones.shape[0]
                and 0 <= c < self.regiones.shape[1]
                and self.regiones[f, c] == self.actual)

    def elegir_proximo(self, seguro, inicio):
        """Elige el próximo sector con un recorrido GLOBAL desde donde está el robot.

        Planifica el orden completo robot -> pendientes -> largada minimizando metros
        por camino transitable y toma el primero. Se recalcula cada vez que se termina
        un sector, desde la pose de ese momento.
        """
        if self.actual is None or self.indice_ordenado == self.indice:
            return None
        seguro = np.asarray(seguro, dtype=bool)
        if seguro.shape != self.regiones.shape:
            raise ValueError('Máscara transitable incompatible')
        inicio = _celda_segura(seguro, inicio)
        if inicio is None:
            return None
        pendientes = list(dict.fromkeys(self.orden[self.indice:]))
        orden, metros = self.recorrido_global(seguro, inicio, pendientes)
        if not orden:
            return None
        elegido = orden[0]
        restantes = [r for r in pendientes if r not in orden]
        self.orden[self.indice:] = orden + restantes
        self.indice_ordenado = self.indice
        self.recorrido_planificado = orden
        self.metros_planificados = metros
        return elegido

    # ------------------------------------------------------------ recorrido global
    def _representantes(self, seguro):
        """Una celda por sector: la transitable más alejada de los bordes del sector."""
        reps = {}
        for region in set(int(r) for r in np.unique(self.regiones) if r >= 0):
            mascara = (self.regiones == region) & seguro & ~self.bloques_conocidos
            if not mascara.any():
                continue
            prof = _distancia_al_borde(mascara)
            fs, cs = np.nonzero(mascara)
            k = int(np.argmax(prof[fs, cs]))
            reps[region] = (int(cs[k]), int(fs[k]))
        return reps

    def _distancias_sectores(self, seguro):
        """Metros (BFS en la grilla transitable) entre los representantes y a la largada."""
        clave = hash(np.packbits(seguro).tobytes()) ^ hash(np.packbits(self.bloques_conocidos).tobytes())
        if self._cache_distancias is not None and self._cache_distancias[0] == clave:
            return self._cache_distancias[1:]
        reps = self._representantes(seguro)
        campos = {r: _bfs_celdas(seguro, celda) for r, celda in reps.items()}
        largada = _celda_segura(seguro, self.largada)
        campos['largada'] = _bfs_celdas(seguro, largada) if largada is not None else None
        self._cache_distancias = (clave, reps, campos)
        return reps, campos

    def recorrido_global(self, seguro, inicio, pendientes):
        """Orden de los pendientes alcanzables que minimiza robot -> ... -> largada."""
        reps, campos = self._distancias_sectores(seguro)
        desde_robot = _bfs_celdas(seguro, inicio)
        nodos = [r for r in pendientes if r in reps and desde_robot[reps[r][1], reps[r][0]] >= 0]
        if not nodos:
            return [], 0.0
        res = self.resolucion
        n = len(nodos)
        inf = float('inf')

        def d(campo, destino):
            if campo is None:
                return inf
            v = campo[destino[1], destino[0]]
            return float(v) * res if v >= 0 else inf

        inicio_a = [d(desde_robot, reps[r]) for r in nodos]
        entre = [[0.0 if i == j else d(campos[nodos[i]], reps[nodos[j]]) for j in range(n)]
                 for i in range(n)]
        a_casa = [d(campos.get('largada'), reps[r]) for r in nodos]   # simétrico (grilla 4-vecinos)
        a_casa = [x if x < inf else 0.0 for x in a_casa]

        def costo(orden):
            if not orden:
                return 0.0
            total = inicio_a[orden[0]] + a_casa[orden[-1]]
            for a, b in zip(orden, orden[1:]):
                total += entre[a][b]
            return total

        # Inserción más barata + mejoras 2-opt y reubicación (extremos fijos: robot y largada).
        orden = [min(range(n), key=lambda i: inicio_a[i])]
        libres = [i for i in range(n) if i not in orden]
        while libres:
            mejor = None
            for i in libres:
                for pos in range(len(orden) + 1):
                    candidato = orden[:pos] + [i] + orden[pos:]
                    c = costo(candidato)
                    if mejor is None or c < mejor[0] - 1e-9:
                        mejor = (c, candidato, i)
            orden = mejor[1]
            libres.remove(mejor[2])
        mejorado = True
        while mejorado:
            mejorado = False
            actual = costo(orden)
            for i in range(n - 1):
                for j in range(i + 1, n):
                    candidato = orden[:i] + orden[i:j + 1][::-1] + orden[j + 1:]
                    if costo(candidato) < actual - 1e-6:
                        orden, actual, mejorado = candidato, costo(candidato), True
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    candidato = orden[:i] + orden[i + 1:]
                    candidato = candidato[:j] + [orden[i]] + candidato[j:]
                    if costo(candidato) < actual - 1e-6:
                        orden, actual, mejorado = candidato, costo(candidato), True
        return [nodos[i] for i in orden], costo(orden)

    def ruta_entrada(self, seguro, inicio, paso_maximo_m=0.50):
        """Guías hasta el primer portal de entrada al sector activo."""
        if self.actual is None:
            return []
        seguro = np.asarray(seguro, dtype=bool)
        if seguro.shape != self.regiones.shape:
            raise ValueError('Máscara transitable incompatible')
        inicio_original = inicio
        c, f = inicio
        alto, ancho = seguro.shape
        if not (0 <= f < alto and 0 <= c < ancho and seguro[f, c]):
            filas, columnas = np.nonzero(seguro)
            if not filas.size:
                return []
            i = int(np.argmin((filas - f) ** 2 + (columnas - c) ** 2))
            c, f = int(columnas[i]), int(filas[i])
            if np.hypot(c - inicio[0], f - inicio[1]) * self.resolucion > paso_maximo_m:
                return []
        inicio = (c, f)
        profundidad = (_distancia_al_borde(self.regiones == self.actual) - 0.5) * self.resolucion
        # El goal checker de portales exige 8 cm. Una meta 15 cm dentro evita
        # declarar éxito antes del cruce, incluso en sectores angostos.
        cola = deque([inicio])
        visitados = {inicio}
        destino = None
        while cola:
            x, y = cola.popleft()
            if self.regiones[y, x] == self.actual and profundidad[y, x] >= 0.15:
                destino = (x, y)
                break
            for nx, ny in ((x+1, y), (x-1, y), (x, y+1), (x, y-1)):
                if (0 <= nx < ancho and 0 <= ny < alto and seguro[ny, nx]
                        and (nx, ny) not in visitados):
                    visitados.add((nx, ny))
                    cola.append((nx, ny))
        if destino is None:
            return []
        ruta = ruta_por_portales(seguro, self.regiones, self.conexiones,
                                inicio, destino, self.resolucion, paso_maximo_m)
        return ([(inicio, False)] + ruta) if ruta and inicio != inicio_original else ruta

    def ruta_hacia(self, seguro, inicio, destino, paso_maximo_m=0.50):
        """Ruta de regreso entre sectores, con guías en portales y tramos cortos."""
        seguro = np.asarray(seguro, dtype=bool)
        if seguro.shape != self.regiones.shape:
            raise ValueError('Máscara transitable incompatible')
        filas, columnas = np.nonzero(seguro)
        if not filas.size:
            return []

        def ajustar(celda):
            c, f = celda
            if 0 <= f < seguro.shape[0] and 0 <= c < seguro.shape[1] and seguro[f, c]:
                return celda
            i = int(np.argmin((filas - f) ** 2 + (columnas - c) ** 2))
            vecino = (int(columnas[i]), int(filas[i]))
            if np.hypot(vecino[0] - c, vecino[1] - f) * self.resolucion > paso_maximo_m:
                return None
            return vecino

        origen, meta = ajustar(inicio), ajustar(destino)
        if origen is None or meta is None:
            return []
        ruta = ruta_por_portales(seguro, self.regiones, self.conexiones,
                                origen, meta, self.resolucion, paso_maximo_m)
        return ([(origen, False)] + ruta) if ruta and origen != inicio else ruta

    def cobertura(self, visto, region=None):
        region = self.actual if region is None else region
        if region is None:
            return 1.0
        piso = (self.regiones == region) & self.libre & ~self.bloques_conocidos
        total = int(np.count_nonzero(piso))
        return float(np.count_nonzero(np.asarray(visto, bool) & piso)) / total if total else 1.0

    def actualizar_bloques(self, bloques):
        bloques = np.asarray(bloques, dtype=bool)
        if bloques.shape != self.libre.shape:
            raise ValueError('Máscara de bloques incompatible')
        self.bloques_conocidos = bloques.copy()

    def avanzar_cubiertos(self, visto):
        while self.actual is not None:
            piso = (self.regiones == self.actual) & self.libre & ~self.bloques_conocidos
            total = int(np.count_nonzero(piso))
            pendiente = int(np.count_nonzero(piso & ~np.asarray(visto, dtype=bool)))
            fraccion = 1.0 - pendiente / total if total else 1.0
            if fraccion >= self.cobertura_objetivo:
                resultado = 'cubierto'
            elif (fraccion >= 0.50
                  and pendiente * self.resolucion ** 2 <= self.area_residual_ignorable_m2):
                resultado = 'residuo_pequeno'
            else:
                break
            self.resultados[self.actual] = resultado
            self.indice += 1
        return self.actual

    def avanzar_sin_vistas(self):
        if self.actual is not None:
            self.resultados[self.actual] = 'sin_vistas_alcanzables'
            self.indice += 1
        return self.actual

    def avanzar_inaccesible(self):
        if self.actual is not None:
            self.resultados[self.actual] = 'inaccesible'
            self.indice += 1
        return self.actual


def _celda_segura(seguro, celda):
    c, f = celda
    if 0 <= f < seguro.shape[0] and 0 <= c < seguro.shape[1] and seguro[f, c]:
        return (c, f)
    filas, columnas = np.nonzero(seguro)
    if not filas.size:
        return None
    i = int(np.argmin((filas - f) ** 2 + (columnas - c) ** 2))
    return (int(columnas[i]), int(filas[i]))


def _bfs_celdas(seguro, celda):
    """Distancia en celdas (4 vecinos) desde `celda` (x, y); -1 inalcanzable."""
    alto, ancho = seguro.shape
    dist = np.full(alto * ancho, -1, dtype=np.int32)
    libre = seguro.ravel()
    inicio = celda[1] * ancho + celda[0]
    dist[inicio] = 0
    cola = deque([inicio])
    while cola:
        i = cola.popleft()
        siguiente = dist[i] + 1
        y, x = divmod(i, ancho)
        if x > 0 and libre[i - 1] and dist[i - 1] < 0:
            dist[i - 1] = siguiente; cola.append(i - 1)
        if x < ancho - 1 and libre[i + 1] and dist[i + 1] < 0:
            dist[i + 1] = siguiente; cola.append(i + 1)
        if y > 0 and libre[i - ancho] and dist[i - ancho] < 0:
            dist[i - ancho] = siguiente; cola.append(i - ancho)
        if y < alto - 1 and libre[i + ancho] and dist[i + ancho] < 0:
            dist[i + ancho] = siguiente; cola.append(i + ancho)
    return dist.reshape(alto, ancho)
