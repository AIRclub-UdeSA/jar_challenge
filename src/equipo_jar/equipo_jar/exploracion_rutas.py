import math
import numpy as np
from collections import deque
from equipo_jar.sectores_nbv import SectoresNBV
from equipo_jar.visualizacion_cobertura import marcadores_sectores
from equipo_jar.planificador import (
    ruta_por_portales,
    guias_por_zona,
    segmento_transitable,
    _distancia_al_borde,
)
from equipo_jar.exploracion_estados import ELIGIENDO, NAVEGANDO
from equipo_jar.exploracion_geometria import ang_norm, dilatar, bfs


class RutasExploracion:

    def preparar_sectores(self, pose):
        if self.cfg['modo_exploracion'] != 'sectores_nbv' or self.sectorizador is not None:
            return
        fila, col = self.grilla.a_celda(*pose[:2])
        self.sectorizador = SectoresNBV(
            self.grilla.libre, self.grilla.transitable_base, self.grilla.res,
            (col, fila), self.cfg['cobertura_objetivo_sector'],
            self.cfg['area_minima_sector_m2'], self.cfg['ancho_minimo_sector_m'],
            self.cfg['area_residual_ignorable_m2'])
        self.sectorizador.actualizar_bloques(self.grilla.bloques_conocidos)
        if self.cfg['publicar_debug']:
            self.pub_sectores.publish(marcadores_sectores(self.mapa_msg, self.sectorizador.regiones))
        self.get_logger().info(
            f'BCD: {len(self.sectorizador.orden)} sectores alcanzables; '
            f'depuración {self.sectorizador.depuracion}')

    def mascaras_portales(self):
        """
        A las cajas/objetos (bloques conocidos y obstáculos fuera del mapa) siempre se
        les deja `margen_objeto_ruta_m`, aunque a las paredes se les deje menos.
        """
        g = self.grilla
        margen = self.cfg.get('margen_parada_portal_m', 0.22)
        estricto = g.transitable & ~dilatar(
            g.bloqueo | g.obstaculos_extra, margen / g.res)
        margen_obj = self.cfg.get('margen_objeto_ruta_m', 0.22)
        lejos_objetos = ~dilatar(g.obstaculos_extra, margen_obj / g.res) if g.obstaculos_extra.any() \
            else np.ones_like(g.transitable)
        return ((estricto, margen),
                (g.candidatable & g.transitable & lejos_objetos, g.m_cand * g.res),
                (g.transitable & lejos_objetos, g.m_trans * g.res),
                (g.transitable, g.m_trans * g.res - 1e-3))     # último recurso: rozando objetos

    def seleccionar_ruta_portales(self, pose, planificar, destino=None):
        """Compara recorridos completos antes de preferir un margen mayor.

        La holgura no justifica cambiar a un rodeo: sólo compiten las rutas
        que se mantienen cerca de la longitud mínima realmente enviada a Nav2.
        Devuelve también la máscara elegida para conservar el mismo recorrido.
        """
        candidatas = []
        g = self.grilla
        regiones = self.sectorizador.regiones if self.sectorizador is not None else None
        mascaras = list(self.mascaras_portales())
        for indice_mascara, (mascara, margen) in enumerate(mascaras):
            seguro = mascara.copy()
            ruta = planificar(seguro)
            if not ruta:
                continue
            pasos, distancia = self.guias_de_ruta(ruta, pose, destino, seguro=seguro)
            if not pasos:
                continue
            puntos = [pose[:2]]
            for punto in pasos:
                if math.dist(puntos[-1], punto[:2]) > 1e-6:
                    puntos.append(punto[:2])
            rumbos = [math.atan2(b[1] - a[1], b[0] - a[0])
                      for a, b in zip(puntos, puntos[1:]) if math.dist(a, b) >= 0.08]
            cambios = [abs(ang_norm(b - a)) for a, b in zip(rumbos, rumbos[1:])]
            giros = sum(cambio > math.radians(25) for cambio in cambios)
            giro_total = sum(cambios)
            cruces = 0
            if regiones is not None:
                ids = []
                for punto in puntos:
                    f, c = g.a_celda(*punto)
                    ids.append(int(regiones[f, c]) if g.dentro(f, c) else -1)
                cruces = sum(a != b for a, b in zip(ids, ids[1:]))
            candidatas.append((ruta, pasos, distancia, seguro, margen,
                               cruces, giros, giro_total, indice_mascara == len(mascaras) - 1))
        if not candidatas:
            return [], [], 0.0, None, 0.0
        # Una ruta que respeta el margen a las cajas gana aunque sea bastante más larga:
        # pasar rozando una caja costó varios minutos de atascos.
        con_margen = [c for c in candidatas if not c[8]]
        if con_margen:
            extra = self.cfg.get('desvio_maximo_por_objetos_m', 4.0)
            minima_con_margen = min(c[2] for c in con_margen)
            if minima_con_margen <= min(c[2] for c in candidatas) + extra:
                candidatas = con_margen
        minima = min(candidata[2] for candidata in candidatas)
        desvio = min(max(0.0, self.cfg.get('desvio_maximo_ruta_m', 0.30)),
                     minima * max(0.0, self.cfg.get('fraccion_desvio_ruta', 0.15)))
        cercanas = [candidata for candidata in candidatas
                    if candidata[2] <= minima + desvio + 1e-6]
        # Entre rutas de largo parecido manda la distancia a las paredes: antes ganaba
        # la de menos giros y, en 7 de 15 tramos, las paradas quedaban a 13 cm.
        elegida = min(cercanas, key=lambda c: (
            c[5], -c[4], c[6], round(c[7], 1), c[2]))
        return elegida[:5]

    def seleccionar_sector_pendiente(self, pose):
        """Reordena sólo los sectores pendientes desde la pose actual del robot."""
        s = self.sectorizador
        fila, col = self.grilla.a_celda(*pose[:2])
        while s.actual is not None:
            elegido = s.elegir_proximo(self.grilla.transitable, (col, fila))
            if elegido is not None:
                self.get_logger().info(
                    f'Siguiente sector {elegido}: recorrido global desde acá '
                    f'{" -> ".join(str(r) for r in s.recorrido_planificado)} -> largada '
                    f'(≈{s.metros_planificados:.1f} m)')
            anterior = s.actual
            s.avanzar_cubiertos(self.visto)
            if s.actual == anterior:
                break

    def en_sector(self, pose, radio=None):
        """¿El robot ya está en el sector actual, o a menos de `radio_entrada_sector_m`
        de él? """
        s, g = self.sectorizador, self.grilla
        if s is None or s.actual is None or pose is None:
            return False
        fila, col = g.a_celda(*pose[:2])
        if s.dentro_actual((col, fila)):
            return True
        k = int(round((self.cfg['radio_entrada_sector_m'] if radio is None else radio) / g.res))
        if k <= 0:
            return False
        f0, f1 = max(0, fila - k), min(g.alto, fila + k + 1)
        c0, c1 = max(0, col - k), min(g.ancho, col + k + 1)
        if f0 >= f1 or c0 >= c1:
            return False
        ff, cc = np.ogrid[f0:f1, c0:c1]
        disco = (ff - fila) ** 2 + (cc - col) ** 2 <= k * k
        return bool((disco & (s.regiones[f0:f1, c0:c1] == s.actual) & g.transitable[f0:f1, c0:c1]).any())

    def paso_hacia_sector(self, pose):
        """Envía una sola navegación continua por los portales hasta el sector."""
        s, g = self.sectorizador, self.grilla
        fila, col = g.a_celda(*pose[:2])
        if not g.dentro(fila, col) or not g.transitable[fila, col]:
            self.get_logger().warn('Inicio dentro del margen: me separo sin descartar sectores pendientes')
            self.cancelar_actual()
            self.goal_actual = None
            self.n_atascos += 1
            self.iniciar_escape(ELIGIENDO)
            return True
        if self.en_sector(pose):
            self.ruta_sector.clear()
            self.ruta_region = None
            return False
        if self.ruta_region != s.actual or not self.ruta_sector:
            def planificar(seguro):
                RutasExploracion.bloquear_pasos_fallidos(self, seguro, (col, fila))
                return s.ruta_entrada(
                    seguro, (col, fila), self.cfg['paso_maximo_portal_m'])
            (self.ruta_sector, pasos, distancia, self.seguro_ruta_sector,
             margen_usado) = self.seleccionar_ruta_portales(pose, planificar)
            self.ruta_region = s.actual
            if not self.ruta_sector:
                destino = None if s.actual in self.sectores_directo else self.punto_interior_sector(pose)
                if destino is not None:
                    self.sectores_directo.add(s.actual)
                    x, y, rumbo, d = destino
                    self.get_logger().warn(f'Sector {s.actual} sin ruta por portales: entro directo con Nav2 '
                                           f'hasta ({x:.2f}, {y:.2f})')
                    if self.encarar_y_enviar(x, y, rumbo, 1, d, f'portal sector {s.actual}', NAVEGANDO):
                        return True
                self.get_logger().warn(f'Sector {s.actual} sin ruta por portales')
                s.avanzar_inaccesible()
                self.ruta_region = None
                return False
            self.get_logger().info(
                f'Sector {s.actual}: trayectoria por {len(self.ruta_sector)} puntos calculados, '
                f'{sum(portal for _, portal in self.ruta_sector)} portal(es), '
                f'margen de paradas {margen_usado:.2f} m')
        else:
            pasos, distancia = self.guias_de_ruta(
                self.ruta_sector, pose, seguro=self.seguro_ruta_sector)
        if not pasos:
            self.get_logger().warn('No hay conexión libre desde el robot a la ruta del sector; recalculo tras desencajar')
            self.ruta_sector.clear()
            self.n_atascos += 1
            self.cancelar_actual()
            self.goal_actual = None
            self.iniciar_escape(ELIGIENDO)
            return True
        x, y, rumbo = pasos[-1]
        etiqueta = f'portal sector {s.actual}'
        self.encarar_y_enviar(x, y, rumbo, 1, distancia, etiqueta, NAVEGANDO, pasos)
        self.publicar_debug([])
        return True

    def punto_interior_sector(self, pose):
        """Entrada cercana y alcanzable; mejora la holgura sin cruzar todo el sector."""
        s, g = self.sectorizador, self.grilla
        fila, col = g.a_celda(*pose[:2])
        dist = bfs(g.transitable, fila, col)
        if dist is None or s.actual is None:
            return None
        cand = (s.regiones == s.actual) & g.transitable & (dist >= 0)
        if not cand.any():
            return None
        interior = cand & ~dilatar(s.regiones != s.actual, 0.15 / g.res)
        if interior.any():
            cand = interior
        minima = float(dist[cand].min())
        cercanas = cand & (dist <= minima + 0.15 / g.res)
        filas, columnas = np.nonzero(cercanas)
        i = min(range(len(filas)), key=lambda i: (
            -self.dist_pared[filas[i], columnas[i]], dist[filas[i], columnas[i]]))
        f, c = int(filas[i]), int(columnas[i])
        x, y = g.a_mundo(f, c)
        return x, y, math.atan2(y - pose[1], x - pose[0]), float(dist[f, c] * g.res)

    def guias_de_ruta(self, ruta, pose, destino=None, seguro=None):
        """Entrada, intermedio y salida de cada zona; como máximo dos uniones extra."""
        g = self.grilla
        inicio = g.a_celda(*pose[:2])[1::-1]
        regiones = (self.sectorizador.regiones if self.sectorizador is not None
                    else np.zeros_like(g.transitable, dtype=np.int32))
        guias_celdas = []
        mascaras = ([seguro & g.transitable] if seguro is not None
                    else [mascara for mascara, _margen in self.mascaras_portales()])
        for mascara in mascaras:
            guias_celdas = guias_por_zona(mascara, regiones, inicio, ruta, g.transitable)
            if guias_celdas:
                break
        if not guias_celdas:
            return [], 0.0
        # La longitud se calcula con la ruta completa, aunque Nav2 reciba pocas
        # referencias y trace por su cuenta el camino entre ellas.
        puntos_ruta = [pose[:2]] + [g.a_mundo(celda[1], celda[0]) for celda, _ in ruta]
        distancia_ruta = sum(math.dist(a, b) for a, b in zip(puntos_ruta, puntos_ruta[1:]))
        mascara_ruta = (seguro & g.transitable) if seguro is not None else g.transitable
        if self.cfg.get('guias_portal_y_medio', False):
            guias_celdas = RutasExploracion.guias_portal_y_centro(
                self, guias_celdas, mascara_ruta, destino_fijo=destino is not None)
        else:
            guias_celdas = RutasExploracion.guias_portal_medio(self, guias_celdas, mascara_ruta)
            guias_celdas = self.alejar_guias(guias_celdas, inicio, fijar_ultima=destino is not None,
                                             mascara=(seguro & g.transitable) if seguro is not None else None)
        if not guias_celdas:
            return [], 0.0
        guias = [(*g.a_mundo(celda[1], celda[0]), portal) for celda, portal in guias_celdas]
        distancia = math.dist(pose[:2], guias[0][:2])
        distancia += sum(math.dist(a[:2], b[:2]) for a, b in zip(guias, guias[1:]))
        pasos = []
        for i, (x, y, _portal) in enumerate(guias):
            if i + 1 < len(guias):
                siguiente = guias[i + 1]
                yaw = math.atan2(siguiente[1] - y, siguiente[0] - x)
            elif destino is not None:
                yaw = destino[2]
            else:
                anterior = guias[i - 1] if i else (pose[0], pose[1])
                yaw = math.atan2(y - anterior[1], x - anterior[0])
            pasos.append((x, y, yaw))
        return pasos, max(distancia, distancia_ruta)

    def holgura_m(self):
        """Metros de cada celda a la pared, zona desconocida u obstáculo (del costmap
        o bloque conocido) más cercano. Se recalcula si cambian los obstáculos."""
        g = self.grilla
        bloqueado = ~g.libre | g.obstaculos_extra
        clave = (id(g), hash(np.packbits(bloqueado).tobytes()))
        cache = getattr(self, '_holgura', None)
        if cache is None or cache[0] != clave:
            cache = (clave, _distancia_al_borde(~bloqueado) * g.res)
            self._holgura = cache
        return cache[1]

    @staticmethod
    def _bfs_en_zona(zona, origen):
        """Distancias (celdas, 4-vecinos) dentro de `zona` desde `origen` (x, y); -1 si no llega."""
        dist = np.full(zona.shape, -1, dtype=np.int32)
        if not zona[origen[1], origen[0]]:
            return dist
        dist[origen[1], origen[0]] = 0
        cola = deque([origen])
        while cola:
            x, y = cola.popleft()
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if 0 <= ny < zona.shape[0] and 0 <= nx < zona.shape[1] and zona[ny, nx] and dist[ny, nx] < 0:
                    dist[ny, nx] = dist[y, x] + 1
                    cola.append((nx, ny))
        return dist

    def guias_portal_y_centro(self, guias_celdas, mascara, destino_fijo=False):
        """Por cada zona nueva de la ruta, sólo DOS poses: el portal de entrada y el medio
        de la zona (o el destino, si es la última zona y hay un destino fijo). Nav2 traza
        por su cuenta el camino entre ellas con el costmap (paredes, inflación y objetos
        del mapa local).
        Las dos poses se corren hacia donde hay más lugar a paredes y objetos
        (`holgura_guia_m`) sin salir de la zona: si el medio cae sobre o junto a un
        objeto, se mueve al punto libre con holgura más cercano."""
        g = self.grilla
        if not guias_celdas:
            return guias_celdas
        regiones = (self.sectorizador.regiones if getattr(self, 'sectorizador', None) is not None
                    else np.zeros_like(g.transitable, dtype=np.int32))
        holgura = RutasExploracion.holgura_m(self) / g.res          # celdas a pared/objeto
        tope = self.cfg.get('holgura_guia_m', 0.30) / g.res
        radio_portal = max(1, int(round(self.cfg.get('radio_portal_guia_m', 0.35) / g.res)))
        bfs_zona = RutasExploracion._bfs_en_zona

        corridas = []                                        # [(region, [celda, ...])]
        for celda, _portal in guias_celdas:
            celda = (int(celda[0]), int(celda[1]))
            reg = int(regiones[celda[1], celda[0]])
            if corridas and corridas[-1][0] == reg:
                corridas[-1][1].append(celda)
            else:
                corridas.append((reg, [celda]))

        def entrar_a_zona(zona, celda):
            """La celda si está en la zona; si no, la celda de la zona más cercana."""
            if zona[celda[1], celda[0]]:
                return celda
            ys, xs = np.nonzero(zona)
            if not len(xs):
                return None
            k = int(np.argmin(np.hypot(xs - celda[0], ys - celda[1])))
            if math.hypot(xs[k] - celda[0], ys[k] - celda[1]) > 2 * radio_portal:
                return None
            return int(xs[k]), int(ys[k])

        def holgado_cerca(zona, celda, radio):
            """Punto de la zona a menos de `radio` celdas de camino con más holgura."""
            celda = entrar_a_zona(zona, celda)
            if celda is None:
                return None
            dist = bfs_zona(zona, celda)
            ys, xs = np.nonzero((dist >= 0) & (dist <= radio))
            puntaje = np.minimum(holgura[ys, xs], tope) * 4.0 - dist[ys, xs] * 0.5
            k = int(np.argmax(puntaje))
            return int(xs[k]), int(ys[k])

        def medio_entre(zona, a, b, respaldo):
            """Mitad del camino a->b dentro de la zona, corrida hacia donde hay lugar."""
            if b is not None and zona[a[1], a[0]] and zona[b[1], b[0]]:
                da, db = bfs_zona(zona, a), bfs_zona(zona, b)
                minimo = da[b[1], b[0]]
                if minimo > 1:
                    ok = ((da > 0) & (db > 0) & (da + db <= minimo + max(4, 0.15 * minimo))
                          & (np.abs(da - db) <= max(2, minimo / 3)))
                    ys, xs = np.nonzero(ok)
                    if len(xs):
                        puntaje = (np.minimum(holgura[ys, xs], tope) * 4.0
                                   - (da[ys, xs] + db[ys, xs] - minimo) * 0.15
                                   - np.abs(da[ys, xs] - db[ys, xs]) * 0.05)
                        k = int(np.argmax(puntaje))
                        return int(xs[k]), int(ys[k])
            return holgado_cerca(zona, respaldo, radio_portal)

        def centro_de_zona(zona, desde):
            """Medio de la parte de la zona alcanzable desde el portal, con holgura."""
            if desde is None:
                return None
            dist = bfs_zona(zona, desde)
            ys, xs = np.nonzero(dist >= 0)
            if not len(xs):
                return None
            cx, cy = float(xs.mean()), float(ys.mean())
            puntaje = np.minimum(holgura[ys, xs], tope) * 4.0 - 0.3 * np.hypot(xs - cx, ys - cy)
            k = int(np.argmax(puntaje))
            return int(xs[k]), int(ys[k])

        nuevas = []
        ultima = len(corridas) - 1
        for k, (reg, celdas) in enumerate(corridas):
            entrada, salida = celdas[0], celdas[-1]
            zona = mascara & (regiones == reg)
            final = k == ultima
            if k == 0 and not final:
                continue                    # el robot ya está en esta zona: Nav2 va al próximo portal
            if k == 0:
                # Destino en la misma zona que el robot: una sola pose.
                punto = salida if destino_fijo else (centro_de_zona(zona, entrar_a_zona(zona, entrada))
                                                     or salida)
                nuevas.append((punto, False))
                continue
            portal = holgado_cerca(zona, entrada, radio_portal) or entrada
            nuevas.append((portal, True))
            if final and destino_fijo:
                segundo = salida
            elif final:
                segundo = centro_de_zona(zona, portal)
            else:
                segundo = medio_entre(zona, portal, entrar_a_zona(zona, salida), celdas[len(celdas) // 2])
            if segundo is not None and segundo != portal:
                nuevas.append((segundo, False))
        return nuevas

    def guias_portal_medio(self, guias_celdas, mascara):
        """Rehace las guías de cada zona como: portal de entrada -> medio de la zona ->
        portal de salida, con a lo sumo UN punto de unión cuando la recta entre dos de
        ellos no es transitable. `mascara` es la de la ruta elegida (respeta el margen a
        objetos del mapa propio). Si hacen falta más giros, se conservan sólo las tres
        referencias transitables y Nav2 calcula el camino entre ellas. Nunca se
        recupera la lista completa de guías como alternativa."""
        g = self.grilla
        if len(guias_celdas) < 2:
            return guias_celdas
        regiones = (self.sectorizador.regiones if getattr(self, 'sectorizador', None) is not None
                    else np.zeros_like(g.transitable, dtype=np.int32))
        holgura = RutasExploracion.holgura_m(self) / g.res          # celdas a pared/objeto
        tope = self.cfg.get('holgura_guia_m', 0.30) / g.res

        def region(celda):
            return int(regiones[celda[1], celda[0]])

        corridas = []                                        # [(region, [(celda, portal), ...])]
        for celda, portal in guias_celdas:
            celda = tuple(celda)
            if corridas and corridas[-1][0] == region(celda):
                corridas[-1][1].append((celda, portal))
            else:
                corridas.append((region(celda), [(celda, portal)]))

        def bfs_zona(zona, origen):
            dist = np.full(zona.shape, -1, dtype=np.int32)
            if not zona[origen[1], origen[0]]:
                return dist
            dist[origen[1], origen[0]] = 0
            cola = deque([origen])
            while cola:
                x, y = cola.popleft()
                for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                    if 0 <= ny < zona.shape[0] and 0 <= nx < zona.shape[1] and zona[ny, nx] and dist[ny, nx] < 0:
                        dist[ny, nx] = dist[y, x] + 1
                        cola.append((nx, ny))
            return dist

        def union(zona, u, v):
            """Un punto P de la zona con recta libre u->P y P->v (o None)."""
            x0, x1 = sorted((u[0], v[0])); y0, y1 = sorted((u[1], v[1]))
            r = int(round(0.6 / g.res))
            ys, xs = np.nonzero(zona[max(0, y0 - r):y1 + r + 1, max(0, x0 - r):x1 + r + 1])
            xs = xs + max(0, x0 - r); ys = ys + max(0, y0 - r)
            if not len(xs):
                return None
            largo = np.hypot(xs - u[0], ys - u[1]) + np.hypot(v[0] - xs, v[1] - ys)
            castigo = 2.0 * (tope - np.minimum(holgura[ys, xs], tope))
            for k in np.argsort(largo + castigo)[:400]:
                p = (int(xs[k]), int(ys[k]))
                if segmento_transitable(mascara, u, p) and segmento_transitable(mascara, p, v):
                    return p
            return None

        def unir(zona, puntos):
            """[a, b, c...] -> guías con a lo sumo un punto de unión entre consecutivos."""
            salida = [puntos[0]]
            for u, v in zip(puntos, puntos[1:]):
                if u == v:
                    continue
                if not segmento_transitable(mascara, u, v):
                    p = union(zona, u, v)
                    if p is None:
                        return None
                    salida.append(p)
                salida.append(v)
            return salida

        nuevas = []
        for k, (reg, corrida) in enumerate(corridas):
            celdas = [c for c, _ in corrida]
            entrada, salida = celdas[0], celdas[-1]
            zona = mascara & (regiones == reg)
            puntos = [entrada]
            if zona[entrada[1], entrada[0]] and zona[salida[1], salida[0]]:
                da, db = bfs_zona(zona, entrada), bfs_zona(zona, salida)
                minimo = da[salida[1], salida[0]]
                if minimo > 1:
                    ok = ((da > 0) & (db > 0) & (da + db <= minimo + max(2, 0.1 * minimo))
                          & (np.abs(da - db) <= max(2, minimo / 3)))
                    ys, xs = np.nonzero(ok)
                    if len(xs):
                        puntaje = (np.minimum(holgura[ys, xs], tope) * 4.0
                                   - (da[ys, xs] + db[ys, xs] - minimo) * 0.15
                                   - np.abs(da[ys, xs] - db[ys, xs]) * 0.05)
                        i = int(np.argmax(puntaje))
                        puntos.append((int(xs[i]), int(ys[i])))
            if len(puntos) == 1 and len(celdas) > 2:
                largos = np.cumsum([0.] + [math.dist(a, b) for a, b in zip(celdas, celdas[1:])])
                medio = 1 + int(np.argmin(np.abs(largos[1:-1] - largos[-1] / 2)))
                puntos.append(celdas[medio])
            puntos.append(salida)
            unidas = unir(mascara if k == 0 else zona, puntos)
            if unidas is None:
                unidas = list(dict.fromkeys(puntos))  # Nav2 resuelve los giros entre las anclas.
            portal_de = {c: p for c, p in corrida}
            for c in unidas:
                es_portal = portal_de.get(c, False) or (c == salida and k < len(corridas) - 1) \
                    or (c == entrada and k > 0)
                if nuevas and nuevas[-1][0] == c:
                    nuevas[-1] = (c, nuevas[-1][1] or es_portal)
                else:
                    nuevas.append((c, es_portal))
        return nuevas

    def alejar_guias(self, guias_celdas, inicio, fijar_ultima=False, mascara=None):
        """Corre cada guía (portales y puntos intermedios) hacia donde hay más lugar
        a las paredes y obstáculos, dentro de `radio_alejar_guia_m`, sin cambiar de
        sector y con línea recta transitable desde la guía anterior y hasta la
        siguiente. Así Nav2 no pasa rozando esquinas ni marcos de puertas."""
        g = self.grilla
        trans = mascara if mascara is not None else g.transitable
        holgura = self.holgura_m()
        objetivo = self.cfg['holgura_guia_m']
        radio = max(1, int(round(self.cfg['radio_alejar_guia_m'] / g.res)))
        regiones = self.sectorizador.regiones if self.sectorizador is not None else None
        disco = [(dx, dy) for dx in range(-radio, radio + 1) for dy in range(-radio, radio + 1)
                 if 0 < dx * dx + dy * dy <= radio * radio]
        salida = []
        anterior = tuple(inicio)
        for i, (celda, portal) in enumerate(guias_celdas):
            celda = tuple(celda)
            ultima = i == len(guias_celdas) - 1
            if celda == tuple(inicio) or (ultima and fijar_ultima):
                salida.append((celda, portal))
                anterior = celda
                continue
            siguiente = tuple(guias_celdas[i + 1][0]) if not ultima else None
            x, y = celda

            def largo(c):
                d = math.hypot(c[0] - anterior[0], c[1] - anterior[1])
                if siguiente is not None:
                    d += math.hypot(siguiente[0] - c[0], siguiente[1] - c[1])
                return d * g.res
            base = largo(celda)
            mejor, puntaje_mejor = celda, min(holgura[y, x], objetivo)
            for dx, dy in disco:
                c = (x + dx, y + dy)
                if not g.dentro(c[1], c[0]) or not trans[c[1], c[0]]:
                    continue
                if regiones is not None and regiones[c[1], c[0]] != regiones[y, x]:
                    continue
                puntaje = min(holgura[c[1], c[0]], objetivo) - 0.3 * (largo(c) - base)
                if puntaje <= puntaje_mejor + 0.02:
                    continue
                if not segmento_transitable(trans, anterior, c):
                    continue
                if siguiente is not None and not segmento_transitable(trans, c, siguiente):
                    continue
                mejor, puntaje_mejor = c, puntaje
            salida.append((mejor, portal))
            anterior = mejor
        return salida

    def ruta_a_destino(self, pose, x, y, yaw):
        """También guía las vistas NBV y el regreso global alrededor de paredes."""
        g = self.grilla
        inicio = g.a_celda(*pose[:2])[1::-1]
        meta = g.a_celda(x, y)[1::-1]
        if not g.dentro(meta[1], meta[0]):
            return [], 0.0
        def planificar(seguro):
            RutasExploracion.bloquear_pasos_fallidos(self, seguro, inicio, meta)
            if not seguro[meta[1], meta[0]]:
                return []
            if self.sectorizador is not None:
                return self.sectorizador.ruta_hacia(
                    seguro, inicio, meta, self.cfg['paso_maximo_portal_m'])
            if not g.dentro(inicio[1], inicio[0]) or not seguro[inicio[1], inicio[0]]:
                return []
            return ruta_por_portales(seguro, np.zeros_like(seguro, dtype=np.int32), {},
                                     inicio, meta, g.res, self.cfg['paso_maximo_portal_m'])
        _ruta, pasos, distancia, _seguro, _margen = self.seleccionar_ruta_portales(
            pose, planificar, (x, y, yaw))
        return pasos, distancia

    def bloquear_pasos_fallidos(self, seguro, *libres, radio_m=0.25):
        """Quita de `seguro` los pasos donde ya se atascó, salvo cerca del robot o del destino."""
        g = self.grilla
        k2 = (radio_m / g.res) ** 2
        for x, y in getattr(self, 'portales_fallidos', ()):
            if not g.dentro(y, x):
                continue
            if any((x - lx) ** 2 + (y - ly) ** 2 <= k2 for lx, ly in libres):
                continue
            seguro[y, x] = False
