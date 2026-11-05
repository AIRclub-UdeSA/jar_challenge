import math
import time
import numpy as np
from collections import deque
from nav_msgs.msg import OccupancyGrid
from equipo_jar.exploracion_estados import (
    ESPERANDO_MAPA,
    LOCALIZANDO,
    ELIGIENDO,
    RELOCALIZANDO,
    OBSERVANDO_OBJETO,
)
from equipo_jar.exploracion_geometria import bfs, Grilla


class MapaExploracion:

    def cb_mapa(self, msg):
        nuevo = self.grilla is None or (msg.info.width, msg.info.height) != (self.grilla.ancho, self.grilla.alto)
        self.mapa_msg = msg
        self.grilla = Grilla(msg, self.cfg['margen_transito'], self.cfg['margen_candidato'])
        g = self.grilla
        for x, y in self.bloques_objetos:
            g.registrar_bloque_objeto(x, y, self.cfg['lado_bloque_objeto_m'])
        pasos = int(self.cfg['rango_vision'] / g.res)
        dist = np.arange(1, pasos + 1) * g.res
        self.ray_dx = np.cos(self.angulos)[:, None] * dist[None, :] / g.res
        self.ray_dy = np.sin(self.angulos)[:, None] * dist[None, :] / g.res
        # Una celda vista de cerca vale más que una vista en el borde del alcance:
        # de lejos el detector confirma peor (y la ubica con más error).
        lejos = max(1e-6, self.cfg['rango_vision'] - 1.0)
        self.peso_rayo = np.clip(1.0 - (1.0 - self.cfg['peso_lejos']) * (dist - 1.0) / lejos,
                                 self.cfg['peso_lejos'], 1.0)
        if nuevo or self.visto is None:
            self.visto = np.zeros((g.alto, g.ancho), dtype=bool)
            self.sectorizador = None
            self.ruta_sector = []
            self.ruta_regreso = []
            self.seguro_ruta_sector = None
            self.seguro_ruta_regreso = None
            self.ruta_region = None
            self.portales_fallidos.clear()
            # La memoria usa coordenadas de map: una nueva grilla no convierte
            # objetos conocidos en nuevos ni interrumpe la inspección en curso.
        if self.sectorizador is not None:
            self.sectorizador.actualizar_bloques(g.bloques_conocidos)
        self.dist_casa = None
        self.dist_pared = self.campo_distancia_paredes(g)
        self.publicar_mapa_amcl()
        self.publicar_mapa_objetos()
        if self.estado == ESPERANDO_MAPA:
            self.t_mapa = time.monotonic()
            self.p0_odom = self.lookup(self.cfg['odom_frame'], self.cfg['base_frame'])
            self.estado = LOCALIZANDO
            self.get_logger().info(f'/map recibido ({g.ancho}x{g.alto} @ {g.res:.2f} m). ¡Largada!')

    def cb_costmap(self, msg):
        g = self.grilla
        if (g is None or abs(msg.info.resolution - g.res) > 1e-4
                or self.estado == RELOCALIZANDO or self.loc_mala_desde is not None):
            return
        datos = np.array(msg.data, dtype=np.int16).reshape(msg.info.height, msg.info.width)
        ocup = datos >= int(self.cfg['costo_obstaculo'])
        extra = np.zeros((g.alto, g.ancho), dtype=bool)
        # Alinear las dos grillas por su origen (misma resolución).
        df = int(round((msg.info.origin.position.y - g.oy) / g.res))
        dc = int(round((msg.info.origin.position.x - g.ox) / g.res))
        f0, c0 = max(0, df), max(0, dc)
        f1, c1 = min(g.alto, df + msg.info.height), min(g.ancho, dc + msg.info.width)
        if f1 > f0 and c1 > c0:
            extra[f0:f1, c0:c1] = ocup[f0 - df:f1 - df, c0 - dc:c1 - dc]
        # Lo que ya es pared en el mapa no hace falta; queda lo NO mapeado.
        extra &= g.libre
        g.actualizar_obstaculos(extra)

    def alcanzable_desde_robot(self, transitable):
        """Celdas transitables conectadas con el robot (o con la largada si no hay pose)."""
        g = self.grilla
        try:
            pose = self.pose_robot()
        except Exception:  # noqa: BLE001  sin TF todavía
            pose = None
        x, y = pose[:2] if pose is not None else getattr(self, 'largada', (0.0, 0.0))[:2]
        f, c = g.a_celda(x, y)
        fs, cs = np.nonzero(transitable)
        if len(fs) == 0:
            return np.zeros_like(transitable)
        if not (g.dentro(f, c) and transitable[f, c]):
            k = int(np.argmin((fs - f) ** 2 + (cs - c) ** 2))
            f, c = int(fs[k]), int(cs[k])
        return bfs(transitable, f, c) >= 0

    def registrar_objeto_local(self, punto):
        """Obstáculo persistente del equipo, independiente de futuros ecos o imágenes.

        Nunca deja que un bloque cierre un paso: si con el bloque quedan celdas libres
        separadas del robot (además de las que ocupa), no se registra. 
        """
        if any(math.dist(punto, previo) < 0.45 for previo in self.bloques_objetos):
            return
        g = self.grilla
        antes_bloques = g.bloques_conocidos.copy()
        antes = self.alcanzable_desde_robot(g.transitable)
        g.registrar_bloque_objeto(*punto, self.cfg['lado_bloque_objeto_m'])
        despues = self.alcanzable_desde_robot(g.transitable)
        cortadas = int((antes & g.transitable & ~despues).sum())
        if cortadas > int(self.cfg.get('celdas_max_cortadas_por_bloque', 20)):
            g.bloques_conocidos = antes_bloques
            g.actualizar_obstaculos(g.obstaculos_costmap)
            self.get_logger().warn(
                f'Objeto en ({punto[0]:.2f}, {punto[1]:.2f}): no lo agrego como bloque, dejaría '
                f'{cortadas * g.res * g.res:.2f} m² sin camino. Nav2 lo esquiva igual si es real')
            return
        self.bloques_objetos.append(tuple(punto))
        if self.cfg.get('cerrar_huecos_objetos', False):
            self.cerrar_huecos_objeto(punto, despues)
        if self.sectorizador is not None:
            self.sectorizador.actualizar_bloques(self.grilla.bloques_conocidos)
        self.dist_casa = None
        self.publicar_mapa_objetos()
        self.get_logger().info(
            f'Bloque conocido para objeto fuera del mapa en ({punto[0]:.2f}, {punto[1]:.2f})')

    def corregir_bloque_con_lectura(self, centro, sensor):
        """Si la lectura del LiDAR cae sobre un objeto ya guardado, se usa para corregir su
        posición en el mapa local (la primera estimación tiene unos cm de error: se ve una
        sola cara y desde un solo lado). Devuelve True si la lectura era de un objeto
        guardado. Sólo lecturas cercanas (más precisas) y con buena localización."""
        bloques = getattr(self, 'bloques_objetos', [])
        if not bloques or not self.cfg.get('corregir_objetos', True):
            return False
        radio = self.cfg.get('radio_correccion_objeto_m', 0.35)
        i = min(range(len(bloques)), key=lambda k: math.dist(centro, bloques[k]))
        if math.dist(centro, bloques[i]) > radio:
            return False
        if math.dist(centro, sensor) > self.cfg.get('distancia_max_correccion_m', 1.8):
            return True
        # Un bloque fijo no puede moverse bajo una ruta que Nav2 está ejecutando.
        # Se corrige entre rutas o durante la inspección, antes de volver a planificar.
        estado = getattr(self, 'estado', None)
        if (estado is not None and estado not in (ELIGIENDO, OBSERVANDO_OBJETO)) or self.heredado_vivo():
            return True
        lecturas = self.lecturas_bloques.setdefault(i, deque(maxlen=int(self.cfg.get('lecturas_correccion', 60))))
        lecturas.append(tuple(centro))
        if len(lecturas) < int(self.cfg.get('lecturas_minimas_correccion', 6)):
            return True
        # Se evalúa cada `lecturas_por_evaluacion` lecturas nuevas, y el objeto sólo se mueve
        # si dos evaluaciones seguidas coinciden (posición estable) y a lo sumo
        # `paso_max_correccion_m` por vez. Antes saltaba 10-19 cm entre lecturas tomadas
        # desde lados distintos y el hueco junto a la caja se abría y cerraba en el plan.
        cuenta = getattr(self, 'cuenta_correcciones', None)
        if cuenta is None:
            cuenta = self.cuenta_correcciones = {}
        cuenta[i] = cuenta.get(i, 0) + 1
        if cuenta[i] % max(1, int(self.cfg.get('lecturas_por_evaluacion', 8))):
            return True
        xs = sorted(p[0] for p in lecturas); ys = sorted(p[1] for p in lecturas)
        nuevo = (xs[len(xs) // 2], ys[len(ys) // 2])
        previas = getattr(self, 'evaluaciones_bloques', None)
        if previas is None:
            previas = self.evaluaciones_bloques = {}
        anterior, previas[i] = previas.get(i), nuevo
        minimo = self.cfg.get('correccion_minima_objeto_m', 0.03)
        if anterior is None or math.dist(anterior, nuevo) > minimo:
            return True                                   # todavía no es estable
        delta = math.dist(nuevo, bloques[i])
        if delta >= minimo:
            paso = self.cfg.get('paso_max_correccion_m', 0.05)
            if delta > paso:
                nuevo = (bloques[i][0] + (nuevo[0] - bloques[i][0]) * paso / delta,
                         bloques[i][1] + (nuevo[1] - bloques[i][1]) * paso / delta)
            # Tope a la deriva acumulada: con el robot trabado contra la caja, pasos de
            # 5 cm seguidos la corrieron 25 cm hacia él (corrida 18:29) y cerraron el paso.
            origenes = getattr(self, 'origen_bloques', None)
            if origenes is None:
                origenes = self.origen_bloques = {}
            origen = origenes.setdefault(i, tuple(bloques[i]))
            tope = self.cfg.get('correccion_total_max_m', 0.10)
            lejos = math.dist(origen, nuevo)
            if lejos > tope:
                nuevo = (origen[0] + (nuevo[0] - origen[0]) * tope / lejos,
                         origen[1] + (nuevo[1] - origen[1]) * tope / lejos)
            if math.dist(nuevo, bloques[i]) < 0.01:
                return True
            self.mover_bloque(i, nuevo, math.dist(nuevo, bloques[i]))
        return True

    def mover_bloque(self, i, nuevo, delta):
        """Rehace los bloques del mapa local con el objeto `i` en su nueva posición,
        sin cerrar pasos."""
        g = self.grilla
        anterior = self.bloques_objetos[i]
        antes_bloques = g.bloques_conocidos.copy()
        alcanzable = self.alcanzable_desde_robot(g.transitable)
        self.bloques_objetos[i] = (float(nuevo[0]), float(nuevo[1]))
        g.bloques_conocidos = np.zeros_like(g.bloques_conocidos)
        for x, y in self.bloques_objetos:
            g.registrar_bloque_objeto(x, y, self.cfg['lado_bloque_objeto_m'])
        despues = self.alcanzable_desde_robot(g.transitable)
        if int((alcanzable & g.transitable & ~despues).sum()) > int(self.cfg.get('celdas_max_cortadas_por_bloque', 20)):
            self.bloques_objetos[i] = anterior
            g.bloques_conocidos = antes_bloques
            g.actualizar_obstaculos(g.obstaculos_costmap)
            return
        if self.sectorizador is not None:
            self.sectorizador.actualizar_bloques(g.bloques_conocidos)
        self.dist_casa = None
        self.publicar_mapa_objetos()
        self.get_logger().info(
            f'Corrijo el objeto del mapa local: ({anterior[0]:.2f}, {anterior[1]:.2f}) -> '
            f'({nuevo[0]:.2f}, {nuevo[1]:.2f}) ({100 * delta:.0f} cm, '
            f'{len(self.lecturas_bloques.get(i, ()))} lecturas del LiDAR)')

    def huecos_angostos_objeto(self, punto):
        """Huecos libres entre el objeto y una pared del mapa por donde el robot pasa
        con dificultad: d(objeto) + d(pared) < ancho_paso_minimo_m. Devuelve una lista
        de (ancho_minimo_m, máscara), una por hueco (componente conexa)."""
        g = self.grilla
        ancho = self.cfg.get('ancho_paso_minimo_m', 0.40)
        medio = self.cfg['lado_bloque_objeto_m'] / 2.0
        k = int(math.ceil((medio + ancho) / g.res)) + 1
        f, c = g.a_celda(*punto)
        f0, f1 = max(0, f - k), min(g.alto, f + k + 1)
        c0, c1 = max(0, c - k), min(g.ancho, c + k + 1)
        if f0 >= f1 or c0 >= c1:
            return []
        filas, cols = np.mgrid[f0:f1, c0:c1]
        xs = g.ox + (cols + 0.5) * g.res
        ys = g.oy + (filas + 0.5) * g.res
        d_obj = np.hypot(np.maximum(np.abs(xs - punto[0]) - medio, 0.0),
                         np.maximum(np.abs(ys - punto[1]) - medio, 0.0))
        paredes = getattr(self, 'dist_pared', None)
        if paredes is None:
            paredes = self.dist_pared = self.campo_distancia_paredes(g)
        suma = d_obj + paredes[f0:f1, c0:c1]
        local = (g.libre[f0:f1, c0:c1] & ~g.bloques_conocidos[f0:f1, c0:c1]
                 & (d_obj > 0) & (suma < ancho))
        # Sólo las celdas que están ENTRE la caja y la pared (la caja y la pared más
        # cercana quedan para lados opuestos); si no, el anillo alrededor de la caja
        # unía todos los huecos en uno solo y no se podía cerrar uno y dejar otro.
        pf, pc = np.nonzero(~g.libre[f0:f1, c0:c1])
        if pf.size and local.any():
            lf, lc = np.nonzero(local)
            px, py = xs[pf, pc], ys[pf, pc]
            cx, cy = xs[lf, lc], ys[lf, lc]
            cerca = np.argmin((cx[:, None] - px[None, :]) ** 2 + (cy[:, None] - py[None, :]) ** 2, axis=1)
            wx, wy = px[cerca] - cx, py[cerca] - cy
            bx = np.clip(cx, punto[0] - medio, punto[0] + medio) - cx
            by = np.clip(cy, punto[1] - medio, punto[1] + medio) - cy
            coseno = (wx * bx + wy * by) / np.maximum(np.hypot(wx, wy) * np.hypot(bx, by), 1e-9)
            local[lf[coseno > -0.5], lc[coseno > -0.5]] = False
        elif not pf.size:
            local[:] = False
        huecos, pendientes = [], set(zip(*np.nonzero(local)))
        while pendientes:
            inicio = pendientes.pop()
            grupo, cola = [inicio], [inicio]
            while cola:
                a, b = cola.pop()
                for da in (-1, 0, 1):
                    for db in (-1, 0, 1):
                        v = (a + da, b + db)
                        if v in pendientes:
                            pendientes.remove(v)
                            grupo.append(v)
                            cola.append(v)
            mascara = np.zeros_like(g.libre)
            fs = np.array([q[0] for q in grupo]); cs = np.array([q[1] for q in grupo])
            mascara[fs + f0, cs + c0] = True
            huecos.append((float(suma[fs, cs].min()), mascara))
        return sorted(huecos, key=lambda h: h[0])

    def cerrar_huecos_objeto(self, punto, alcanzable_con_caja=None):
        """En el mapa propio cierra los huecos angostos entre el objeto y las paredes,
        del más angosto al más ancho, mientras quede otro camino: si la caja está en
        un cuello de botella, Nav2 pasa por el hueco más ancho en vez de meterse por
        la rendija entre la caja y la punta de una pared (~30 cm).
        """
        g = self.grilla
        huecos = self.huecos_angostos_objeto(punto)
        if not huecos:
            return
        referencia = (alcanzable_con_caja if alcanzable_con_caja is not None
                      else self.alcanzable_desde_robot(g.transitable))
        maximo = int(self.cfg.get('celdas_max_cortadas_por_bloque', 20))
        cerrados, abiertos = [], []
        for ancho_hueco, hueco in huecos:
            antes_bloques = g.bloques_conocidos.copy()
            g.bloques_conocidos = g.bloques_conocidos | hueco
            g.actualizar_obstaculos(g.obstaculos_costmap)
            despues = self.alcanzable_desde_robot(g.transitable)
            if int((referencia & g.transitable & ~despues).sum()) > maximo:
                g.bloques_conocidos = antes_bloques      # es el paso que queda: abierto
                g.actualizar_obstaculos(g.obstaculos_costmap)
                abiertos.append(ancho_hueco)
            else:
                cerrados.append(ancho_hueco)
        if cerrados or abiertos:
            self.get_logger().info(
                f'Objeto en ({punto[0]:.2f}, {punto[1]:.2f}): cierro {len(cerrados)} hueco(s) angosto(s) '
                f'contra la pared ({", ".join(f"{100 * a:.0f} cm" for a in cerrados) or "-"}); '
                + (f'dejo abierto el de {", ".join(f"{100 * a:.0f} cm" for a in abiertos)} (sin él no hay camino)'
                   if abiertos else 'no queda ninguno abierto'))

    def publicar_mapa_objetos(self):
        """Mapa propio para el plan global de Nav2; nunca modifica /map del juez."""
        if self.mapa_msg is None:
            return
        m = OccupancyGrid()
        m.header.frame_id = self.grilla.frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.info = self.mapa_msg.info
        datos = np.array(self.mapa_msg.data, dtype=np.int8).reshape(self.grilla.alto, self.grilla.ancho)
        datos[self.grilla.bloques_conocidos] = 100
        m.data = datos.ravel().tolist()
        self.pub_mapa_objetos.publish(m)
