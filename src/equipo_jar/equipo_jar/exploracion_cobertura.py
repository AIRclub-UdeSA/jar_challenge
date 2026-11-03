import math
import time
import numpy as np
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from equipo_jar.exploracion_geometria import (
    yaw_a_quat,
    ang_norm,
    contar_huecos_pequenos,
    punto_para_mirar,
    dilatar,
    bfs,
)


class CoberturaExploracion:

    def rayos_desde(self, fila, col):
        g = self.grilla
        f = np.floor(fila + 0.5 + self.ray_dy).astype(np.int32)
        c = np.floor(col + 0.5 + self.ray_dx).astype(np.int32)
        fuera = (f < 0) | (f >= g.alto) | (c < 0) | (c >= g.ancho)
        f = np.clip(f, 0, g.alto - 1)
        c = np.clip(c, 0, g.ancho - 1)
        visible = ~np.maximum.accumulate(g.bloqueo[f, c] | g.obstaculos_extra[f, c] | fuera, axis=1)
        return f, c, visible

    def marcar_visto(self, pose):
        """Cono desde el centro óptico y rumbo real de cámara, no desde la base."""
        g = self.grilla
        fila, col = g.a_celda(pose[0], pose[1])
        if not g.dentro(fila, col):
            return
        f, c, vis = self.rayos_desde(fila, col)
        dif = np.abs(np.arctan2(np.sin(self.angulos - pose[2]), np.cos(self.angulos - pose[2])))
        sel = vis & (dif <= self.fov / 2.0)[:, None]
        self.visto[f[sel], c[sel]] = True

    def marcar_alrededor(self, x, y, radio):
        """Da por visto el piso alrededor de un objeto que no está en el mapa: la
        sombra que deja detrás es chica y no vale ir a rodearlo para mirarla."""
        g = self.grilla
        if g is None or self.visto is None:
            return
        f, c = g.a_celda(x, y)
        k = max(1, int(round(radio / g.res)))
        f0, f1 = max(0, f - k), min(g.alto, f + k + 1)
        c0, c1 = max(0, c - k), min(g.ancho, c + k + 1)
        ff, cc = np.ogrid[f0:f1, c0:c1]
        disco = (ff - f) ** 2 + (cc - c) ** 2 <= k * k
        self.visto[f0:f1, c0:c1] |= disco & g.libre[f0:f1, c0:c1]

    def cobertura(self):
        piso = self.grilla.libre & ~self.grilla.bloques_conocidos
        return float((self.visto & piso).sum()) / max(1, int(piso.sum()))

    def completar_huecos(self):
        if self.grilla is None or self.visto is None:
            return
        ahora = time.monotonic()
        if ahora - self.t_ultimo_huecos < 1.0:
            return
        self.t_ultimo_huecos = ahora
        g = self.grilla
        contar_huecos_pequenos(self.visto, g.libre & ~g.bloques_conocidos,
                               g.res, self.cfg['area_caja_m2'])

    def ganancia_en(self, fila, col):
        f, c, vis = self.rayos_desde(fila, col)
        nuevo = (vis & self.grilla.libre[f, c] & ~self.grilla.bloques_conocidos[f, c]
                 & ~self.visto[f, c])
        sectorizador = getattr(self, 'sectorizador', None)
        if sectorizador is not None:
            nuevo &= sectorizador.mascara_observacion()[f, c]
        if not nuevo.any():
            return 0.0, 0.0
        # Varios rayos atraviesan la misma celda, sobre todo cerca del robot.
        # Contar celdas únicas evita sobrevalorar rincones y vistas muy próximas.
        _, inversa = np.unique(f[nuevo] * self.grilla.ancho + c[nuevo], return_inverse=True)
        rayos = np.broadcast_to(np.arange(self.nrayos)[:, None], nuevo.shape)[nuevo]
        pesos = np.zeros(int(inversa.max()) + 1)
        np.maximum.at(pesos, inversa, np.broadcast_to(self.peso_rayo, nuevo.shape)[nuevo])
        presencia = np.zeros((self.nrayos, len(pesos)), dtype=np.int16)
        presencia[rayos, inversa] = 1
        w = self.ventana
        extendida = np.concatenate((presencia, presencia[:w - 1]), axis=0)
        acum = np.vstack((np.zeros((1, len(pesos)), dtype=np.int16),
                          np.cumsum(extendida, axis=0, dtype=np.int16)))
        visibles = acum[w:w + self.nrayos] - acum[:self.nrayos] > 0
        porciones = (visibles * pesos[None, :]).sum(axis=1)
        i = int(np.argmax(porciones))
        yaw = -math.pi + (i + (w - 1) / 2.0) * (2.0 * math.pi / self.nrayos)
        return float(porciones[i]), ang_norm(yaw)

    def elegir_vista(self, pose):
        g = self.grilla
        sectorizador = getattr(self, 'sectorizador', None)
        if (sectorizador is not None
                and not (sectorizador.mascara_observacion() & ~self.visto).any()):
            return None, []
        fila_r, col_r = g.a_celda(pose[0], pose[1])
        dist = bfs(g.transitable, fila_r, col_r)
        if dist is None:
            return None, []
        casa = self.campo_casa()
        v = self.velocidad_media()
        cam = self.pose_camara()
        desfase_camara = ang_norm(cam[2] - pose[2]) if cam is not None else 0.0
        presupuesto = self.tiempo_restante() - self.cfg['margen_regreso_s']
        paso_base = max(1, int(round(self.cfg['paso_candidatos'] / g.res)))
        intentos = [(paso_base, self.cfg['ganancia_minima'])]
        if sectorizador is not None:
            # Para terminar un sector pequeño, no abandonar las últimas vistas
            # sólo porque ninguna suma 25 celdas nuevas en la grilla gruesa.
            intentos.append((max(1, paso_base // 2), 1))
            intentos.append((1, 1))
        preferida = g.candidatable
        if hasattr(g, 'bloqueo'):
            preferida = g.candidatable & ~dilatar(
                g.bloqueo | g.obstaculos_extra,
                self.cfg.get('margen_parada_portal_m', 0.22) / g.res)
        holgura = self.holgura_m() if hasattr(self, 'cfg') and 'holgura_guia_m' in self.cfg else None
        objetivo_holgura = self.cfg.get('holgura_guia_m', 0.30)
        peso_holgura = self.cfg.get('peso_holgura_nbv', 0.6)
        mejores = []
        for paso, ganancia_minima in intentos:
            for mascara_metas in (preferida, g.candidatable):
                for fila in range(paso // 2, g.alto, paso):
                    for col in range(paso // 2, g.ancho, paso):
                        # La meta debe ser segura y alcanzable; el piso observado
                        # puede estar del otro lado de un paso que no admite el robot.
                        if not mascara_metas[fila, col] or dist[fila, col] < 0:
                            continue
                        x, y = g.a_mundo(fila, col)
                        if any(math.hypot(x - bx, y - by) < self.cfg['radio_lista_negra']
                               for bx, by in self.lista_negra):
                            continue
                        d_m = dist[fila, col] * g.res
                        # ¿Llego, miro y vuelvo a tiempo?
                        d_casa = casa[fila, col] * g.res if casa is not None and casa[fila, col] >= 0 else d_m
                        if 1.3 * (d_m + d_casa) / v + 8.0 > presupuesto:
                            continue
                        gan, yaw = self.ganancia_en(fila, col)
                        if gan < ganancia_minima:
                            continue
                        yaw = ang_norm(yaw - desfase_camara)
                        giro = abs(ang_norm(yaw - pose[2]))
                        util = (gan * math.exp(-self.cfg['lambda_distancia'] * d_m)
                                / (1.0 + self.cfg['peso_giro_nbv'] * giro / math.pi))
                        if holgura is not None:
                            # Mismo punto de vista, pero lejos de la pared: Nav2 llega
                            # sin rozar y el robot gira sin "Collision Ahead".
                            util *= (1.0 - peso_holgura) + peso_holgura * min(
                                1.0, float(holgura[fila, col]) / objetivo_holgura)
                        mejores.append((util, x, y, yaw, gan, d_m))
                if mejores:
                    break
            if mejores:
                break
        if not mejores:
            return None, []
        mejores.sort(reverse=True)
        return mejores[0], mejores[:40]

    def elegir_confirmacion(self, pose):
        """Caja roja vista pero sin reportar: punto a 0.6-1.2 m con vista libre para
        que el detector la confirme de cerca. None si no hay o no da el tiempo."""
        if (not self.cfg['confirmar_cajas'] or not self.pendientes_vision or self.t_pendientes is None
                or time.monotonic() - self.t_pendientes > 3.0):
            return None
        g = self.grilla
        fila, col = g.a_celda(pose[0], pose[1])
        dist = bfs(g.transitable, fila, col)
        if dist is None:
            return None
        for bx, by, _n in sorted(self.pendientes_vision, key=lambda p: math.hypot(p[0] - pose[0], p[1] - pose[1])):
            if not self.objetos_fuera_mapa.disponible((bx, by), time.monotonic()):
                continue
            registro = next((r for r in self.intentos_confirmar if math.hypot(r[0] - bx, r[1] - by) < 0.35), None)
            if registro is not None and registro[2] >= self.cfg['intentos_confirmacion']:
                continue
            # La caja aparece en el costmap: ese obstáculo no tapa la vista. Las
            # paredes del mapa sí (nunca se "limpian").
            vista = g.libre & ~(g.obstaculos_costmap & ~self._cerca_de(bx, by))
            punto = punto_para_mirar(g.candidatable, vista, dist, g.res, (g.ox, g.oy), (bx, by))
            if punto is None:
                continue
            x, y, yaw, d_m = punto
            if (self.tiempo_restante() - self.cfg['margen_regreso_s'] - self.cfg['tiempo_mirar_caja_s']
                    < 1.3 * d_m / self.velocidad_media() + self.segundos_para_volver(x, y)):
                continue
            if registro is None:
                registro = [bx, by, 0]
                self.intentos_confirmar.append(registro)
            registro[2] += 1
            return x, y, yaw, d_m, (bx, by)
        return None

    def _cerca_de(self, bx, by, radio=0.30):
        """Máscara de las celdas a menos de `radio` de la caja (la caja misma no es pared)."""
        g = self.grilla
        filas = (np.arange(g.alto) + 0.5) * g.res + g.oy
        cols = (np.arange(g.ancho) + 0.5) * g.res + g.ox
        return ((filas[:, None] - by) ** 2 + (cols[None, :] - bx) ** 2) < radio ** 2

    def publicar_visto(self):
        """Cobertura operativa para las estimaciones finales, también sin RViz."""
        if self.grilla is None or self.visto is None:
            return
        g = self.grilla
        m = OccupancyGrid()
        m.header.frame_id = g.frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.info.resolution = g.res
        m.info.width, m.info.height = g.ancho, g.alto
        m.info.origin.position.x, m.info.origin.position.y = g.ox, g.oy
        m.info.origin.orientation.w = 1.0
        capa = np.zeros((g.alto, g.ancho), dtype=np.int8)
        capa[self.visto & g.libre] = 60
        capa[g.bloques_conocidos] = 100
        m.data = capa.ravel().tolist()
        self.pub_visto.publish(m)

    def publicar_debug(self, top):
        if not self.cfg['publicar_debug'] or self.grilla is None:
            return
        ahora = time.monotonic()
        if not top and ahora - self.ultimo_debug < 1.0:
            return
        self.ultimo_debug = ahora
        g = self.grilla
        arr = MarkerArray()
        borrar = Marker()
        borrar.action = Marker.DELETEALL
        arr.markers.append(borrar)

        def marcador(ns, i, tipo, x, y, escala, rgba, yaw=None):
            mk = Marker()
            mk.header.frame_id = g.frame
            mk.ns, mk.id, mk.type = ns, i, tipo
            mk.pose.position.x, mk.pose.position.y = x, y
            if yaw is None:
                mk.pose.orientation.w = 1.0
            else:
                mk.pose.orientation.z, mk.pose.orientation.w = yaw_a_quat(yaw)
            mk.scale.x, mk.scale.y, mk.scale.z = escala
            mk.color.r, mk.color.g, mk.color.b, mk.color.a = rgba
            arr.markers.append(mk)
            return mk

        if top:
            umax = max(t[0] for t in top) or 1.0
            for i, (u, x, y, *_r) in enumerate(top):
                s = 0.06 + 0.10 * u / umax
                marcador('candidatos', i, Marker.SPHERE, x, y, (s, s, s), (1.0 - u / umax, u / umax, 0.2, 0.9))
        for i, (celda, portal) in enumerate(self.ruta_sector):
            wx, wy = g.a_mundo(celda[1], celda[0])
            marcador('ruta_portales', i, Marker.SPHERE, wx, wy,
                     (0.15, 0.15, 0.15), (1.0, 0.45 if portal else 0.8, 0.1, 0.9))
        for i, (celda, portal) in enumerate(self.ruta_regreso):
            wx, wy = g.a_mundo(celda[1], celda[0])
            marcador('ruta_regreso', i, Marker.SPHERE, wx, wy,
                     (0.15, 0.15, 0.15), (0.2, 0.9 if portal else 0.5, 1.0, 0.9))
        for i, (x, y) in enumerate(self.bloques_objetos):
            lado = self.cfg['lado_bloque_objeto_m']
            marcador('bloques_objetos', i, Marker.CUBE, x, y,
                     (lado, lado, 0.08), (1.0, 0.25, 0.1, 0.8))
        if self.goal_actual is not None:
            ga = self.goal_actual
            meta = ga['meta_tramo']
            marcador('goal', 0, Marker.ARROW, meta[0], meta[1],
                     (0.4, 0.07, 0.07), (1.0, 0.1, 0.9, 1.0), meta[2])
            for i, (px, py, rumbo) in enumerate(ga.get('pasos') or []):
                marcador('guias_nav2', i, Marker.ARROW, px, py,
                         (0.20, 0.035, 0.035), (1.0, 0.65, 0.0, 1.0), rumbo)
        if self.objeto_observado is not None:
            ox, oy = self.objeto_observado['punto']
            marcador('objeto_nuevo', 0, Marker.SPHERE, ox, oy,
                     (0.22, 0.22, 0.22), (1.0, 0.2, 0.2, 0.95))
        lx, ly, _ = self.largada
        marcador('largada', 0, Marker.CYLINDER, lx, ly, (0.5, 0.5, 0.02),
                 (0.1, 0.6, 1.0, 0.5 if self.largada_fija else 0.2))
        txt = marcador('estado', 0, Marker.TEXT_VIEW_FACING, g.ox + 0.3, g.oy + g.alto * g.res + 0.3,
                       (0.25, 0.25, 0.25), (1.0, 1.0, 1.0, 1.0))
        txt.text = (f'{self.estado}  cob {100 * self.cobertura():.0f}%  t={self.tiempo_transcurrido():.0f}s  '
                    f'rep {self.reportes}/{self.cfg["cantidad_victimas"]}'
                    + (f'  sector {self.sectorizador.actual} '
                       f'{100 * self.sectorizador.cobertura(self.visto):.0f}%'
                       if self.sectorizador is not None and self.sectorizador.actual is not None else ''))
        self.pub_marc.publish(arr)
