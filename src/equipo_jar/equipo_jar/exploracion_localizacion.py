import math
import time
import numpy as np
from collections import deque
from geometry_msgs.msg import PointStamped, PoseWithCovarianceStamped, Twist
from nav_msgs.msg import OccupancyGrid
from rclpy.time import Time
from std_srvs.srv import Empty as EmptySrv
from equipo_jar.exploracion_estados import RELOCALIZANDO
from equipo_jar.exploracion_geometria import yaw_a_quat, quat_a_yaw, ang_norm


class LocalizacionExploracion:

    def cb_amcl(self, msg):
        cov = msg.pose.covariance
        self.std_amcl = (math.sqrt(max(cov[0], cov[7], 0.0)), math.sqrt(max(cov[35], 0.0)))
        self.secuencia_amcl += 1
        self.t_ultima_pose_amcl = time.monotonic()

    @staticmethod
    def campo_distancia_paredes(g, ocup=None):
        """Distancia (m) de cada celda a la celda ocupada más cercana del mapa (BFS 8-vecinos).
        `ocup`: máscara de ocupado a usar (por defecto, lo no libre del mapa)."""
        ocup = ~g.libre if ocup is None else ocup
        dist = np.full(ocup.shape, np.inf)
        dist[ocup] = 0.0
        frente = list(zip(*np.nonzero(ocup)))
        alto, ancho = ocup.shape
        cola = deque(frente)
        while cola:
            f, c = cola.popleft()
            d0 = dist[f, c]
            for df, dc, w in ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                              (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)):
                nf, nc = f + df, c + dc
                if 0 <= nf < alto and 0 <= nc < ancho and d0 + w < dist[nf, nc] - 1e-6:
                    dist[nf, nc] = d0 + w
                    cola.append((nf, nc))
        return dist * g.res

    def mascara_objetos(self, radio=0.10):
        """Celdas ocupadas por los objetos fuera del mapa ya confirmados (disco de
        `radio` alrededor de cada uno: una caja de 20 cm)."""
        g = self.grilla
        m = np.zeros((g.alto, g.ancho), dtype=bool)
        k = max(1, int(round(radio / g.res)))
        for x, y in getattr(self, 'bloques_objetos', []):
            f, c = g.a_celda(x, y)
            f0, f1 = max(0, f - k), min(g.alto, f + k + 1)
            c0, c1 = max(0, c - k), min(g.ancho, c + k + 1)
            if f0 < f1 and c0 < c1:
                ff, cc = np.ogrid[f0:f1, c0:c1]
                m[f0:f1, c0:c1] |= (ff - f) ** 2 + (cc - c) ** 2 <= k * k
        return m

    def campo_loc(self):
        """Campo de distancias para medir/corregir la localización: paredes del mapa
        MÁS los objetos fuera del mapa ya observados. """
        objetos = tuple(getattr(self, 'bloques_objetos', []))
        cache = getattr(self, '_campo_loc', None)
        if not objetos:
            return self.dist_pared
        if cache is None or cache[0] != (id(self.grilla), objetos):
            ocup = ~self.grilla.libre | self.mascara_objetos()
            self._campo_loc = cache = ((id(self.grilla), objetos),
                                       self.campo_distancia_paredes(self.grilla, ocup))
        return cache[1]

    def publicar_mapa_amcl(self, con_objetos=True):
        """AMCL usa /mapa_amcl: el /map del juez con los objetos fuera del mapa ya
        observados agregados como ocupados, para que el filtro de partículas los
        tenga en cuenta al reubicarse."""
        if self.mapa_msg is None or not hasattr(self, 'pub_mapa_amcl'):
            return
        m = OccupancyGrid()
        m.header = self.mapa_msg.header
        m.info = self.mapa_msg.info
        datos = np.array(self.mapa_msg.data, dtype=np.int8).reshape(self.grilla.alto, self.grilla.ancho)
        n = 0
        if con_objetos and getattr(self, 'bloques_objetos', None):
            objetos = self.mascara_objetos() & self.grilla.libre
            n = len(self.bloques_objetos)
            datos = datos.copy()
            datos[objetos] = 100
        m.data = datos.ravel().tolist()
        self.pub_mapa_amcl.publish(m)
        if n:
            self.get_logger().info(f'Mapa de AMCL actualizado con {n} objeto(s) fuera del mapa')

    def medir_localizacion(self, pose):
        """Percentil 70 de la distancia de los puntos del láser a la pared más cercana del mapa.

        Bien localizado: ~1-3 cm. Si AMCL se corrió, los puntos de las paredes
        perpendiculares al corrimiento caen al costado y el p70 sube (la mediana
        no alcanzaba: con un corrimiento en una sola dirección, las paredes
        paralelas siguen calzando). El 30% más alto se descarta: son rayos que
        pegan en cosas que no están en el mapa (señuelos, víctimas).

        La distancia es CONTINUA (interpolación bilineal del campo entre centros
        de celda, menos media celda): con la distancia "por celda" el valor salía
        en escalones de 5 cm (0, 5, 10) y en el robot real (bag de la semana 10)
        un robot bien localizado ya marcaba 5 cm, sin margen para distinguir.
        """
        puntos = self.puntos_scan(alcance=self.cfg.get('rango_laser_localizacion_m', 8.0))
        if puntos is None or self.dist_pared is None:
            return None
        x, y = puntos
        if x.size < 30:
            return None
        paso = max(1, math.ceil(x.size / 120))
        x, y = x[::paso], y[::paso]
        c, s = math.cos(pose[2]), math.sin(pose[2])
        mx, my = pose[0] + c * x - s * y, pose[1] + s * x + c * y
        g = self.grilla
        u = (mx - g.ox) / g.res - 0.5
        v = (my - g.oy) / g.res - 0.5
        # Los rayos fuera del mapa también cuentan: descartarlos puede hacer
        # parecer buena una pose falsa que sólo alinea unas pocas paredes.
        uc, vc = np.clip(u, 0, g.ancho - 1), np.clip(v, 0, g.alto - 1)
        col, f = np.floor(uc).astype(int), np.floor(vc).astype(int)
        col1, f1 = np.minimum(col + 1, g.ancho - 1), np.minimum(f + 1, g.alto - 1)
        a, b = uc - col, vc - f
        D = self.campo_loc()
        d = ((1 - a) * (1 - b) * D[f, col] + a * (1 - b) * D[f, col1]
             + (1 - a) * b * D[f1, col] + a * b * D[f1, col1])
        d += g.res * np.hypot(u - uc, v - vc)
        # Una pared de borde ocupa media celda más allá de su centro: no
        # penalizar el rayo que cae en su cara exterior por el redondeo del mapa.
        d = np.clip(d - 0.5 * g.res, 0.0, 1.0)
        return float(np.percentile(d, 70))

    def actualizar_localizacion(self, pose):
        med = self.medir_localizacion(pose)
        if med is None:
            return
        # Resumen cada 15 s (para calibrar error_loc_malo en cada entorno)
        self.muestras_loc.append(med)
        if time.monotonic() - self.t_resumen_loc > 15.0 and len(self.muestras_loc) >= 10:
            m = np.array(self.muestras_loc)
            self.get_logger().info(f'[loc] láser vs mapa (p70) últimos 15 s: mediana {100 * np.median(m):.1f} cm, '
                                   f'máx {100 * m.max():.1f} cm')
            self.muestras_loc = []
            self.t_resumen_loc = time.monotonic()
        self.error_loc = med if self.error_loc is None else 0.7 * self.error_loc + 0.3 * med
        t = self.get_clock().now().nanoseconds * 1e-9
        if self.error_loc > self.cfg['error_loc_malo']:
            if self.loc_mala_desde is None:
                self.loc_mala_desde = t
                self.get_logger().warn(
                    f'Localización empeorando: el láser queda a {100 * self.error_loc:.1f} cm de las paredes '
                    f'del mapa (bien: < {100 * self.cfg["error_loc_malo"]:.0f} cm). Si sigue así '
                    f'{self.cfg["segundos_loc_mala"]:.1f} s, relocalizo.')
        else:
            if self.loc_mala_desde is not None and t - self.loc_mala_desde > 0.5:
                self.get_logger().info(f'Localización recuperada ({100 * self.error_loc:.1f} cm)')
            self.loc_mala_desde = None

    def publicar_estado_loc(self):
        """Para el detector: x = error láser-mapa (m, -1 sin medir), y = 1 relocalizando."""
        if not hasattr(self, 'pub_estado_loc'):
            return
        m = PointStamped()
        m.header.frame_id = 'map'
        m.header.stamp = self.get_clock().now().to_msg()
        m.point.x = float(self.error_loc) if self.error_loc is not None else -1.0
        m.point.y = 1.0 if self.estado == RELOCALIZANDO or self.loc_mala_desde is not None else 0.0
        self.pub_estado_loc.publish(m)

    def localizacion_mala(self):
        if self.loc_mala_desde is None:
            return False
        t = self.get_clock().now().nanoseconds * 1e-9
        return t - self.loc_mala_desde >= self.cfg['segundos_loc_mala']

    def relocalizar(self, siguiente, motivo=None):
        """Detiene la misión y corrige AMCL: primero calce local láser-mapa; si no
        alcanza, partículas alrededor de donde creía estar (pose al perderse + odometría),
        cada vez más abiertas, moviéndose de a tramos cortos para que AMCL converja."""
        self.t_ult_reloc = time.monotonic()
        self.loc_mala_desde = None
        motivo = motivo or f'láser/mapa fuera de tolerancia ({100 * (self.error_loc or 0):.1f} cm)'
        self.get_logger().warn(f'Localización perdida: {motivo}; freno y relocalizo (primero calce local; '
                               'si no alcanza, partículas alrededor de donde creo estar)')
        pose_creida = self.pose_robot()
        odom_creida = self.lookup(self.cfg['odom_frame'], self.cfg['base_frame'])
        self.publicar_mapa_amcl()           # AMCL se reubica teniendo en cuenta los objetos
        cancelacion = self.cancelar_actual()
        self.pub_cmd.publish(Twist())
        self.goal_actual = None
        self.pendiente = None
        self.escape = None
        if self.objeto_observado is not None:
            self.recordar_inspeccion(self.objeto_observado, False)
            self.limite_observacion(100.0)
        self.objeto_observado = None
        self.ruta_sector.clear()
        self.ruta_regreso.clear()
        self.ruta_region = None
        self.dir_ruta = None
        self.estado = RELOCALIZANDO
        self.reloc = dict(t0=time.monotonic(), siguiente=siguiente, fase='frenando',
                          cancelacion=cancelacion, solicitud=None, ultimo_aviso=0.0,
                          ultimo_nomotion=0.0, nomotion=None, estable_desde=None,
                          ultimo_scan=None, giro_fin=False, intentos=0,
                          pose_creida=pose_creida, odom_creida=odom_creida, mov='esperando',
                          t_mov=time.monotonic(), movs=0, ang_mov=None)
        self.hist.clear()
        self.hist_global.clear()

    def relocalizar_local(self):
        """Busca cerca (±35 cm/±17°; si no, ±70 cm/±41°) la pose donde el láser calza
        con el mapa y se la da a AMCL. True si encontró una pose clara."""
        pose = self.pose_robot()
        if pose is None:
            return False
        res = self.buscar_pose(pose, rango_xy=0.35, rango_yaw=0.30)
        if res is None or res[3] >= self.cfg['error_reloc_ok']:
            amplia = self.buscar_pose(pose, rango_xy=0.70, rango_yaw=0.72, paso_xy=0.05, paso_yaw=0.06)
            if amplia is not None and (res is None or amplia[3] < res[3]):
                res = amplia
        if res is None or res[3] >= self.cfg['error_reloc_ok'] or res[3] >= res[4] - 0.01:
            self.get_logger().warn('Relocalización local: no encontré una pose clara cerca')
            return False
        x, y, yaw, err, err0 = res
        self.publicar_pose(x, y, yaw, 0.06, 0.06)
        self.get_logger().info(
            f'Relocalización local: corrijo AMCL {100 * math.hypot(x - pose[0], y - pose[1]):.0f} cm y '
            f'{math.degrees(abs(ang_norm(yaw - pose[2]))):.0f}° (láser {100 * err0:.1f} -> {100 * err:.1f} cm)')
        return True

    def registrar_fallo_localizacion(self, siguiente):
        """El calce del láser puede dar falsa confianza en pasillos repetidos.
        Solo se relocaliza por fallos de navegación si además el láser ya empezó
        a despegarse del mapa: en un paso angosto Nav2 falla seguido aunque la
        localización esté bien, y frenar a relocalizar ahí solo pierde tiempo."""
        if self.error_loc is None or self.error_loc < 0.5 * self.cfg['error_loc_malo']:
            return False
        ahora = time.monotonic()
        self.fallos_localizacion.append(ahora)
        while (self.fallos_localizacion
               and ahora - self.fallos_localizacion[0] > self.cfg['ventana_fallos_relocalizacion_s']):
            self.fallos_localizacion.popleft()
        if (len(self.fallos_localizacion) >= self.cfg['fallos_para_relocalizar']
                and ahora - self.t_ult_reloc >= self.cfg['espera_entre_relocalizaciones']):
            self.relocalizar(siguiente, f'{len(self.fallos_localizacion)} fallos de navegación sin resolver')
            return True
        return False

    def pose_creida_ahora(self, r):
        """Donde el robot creía estar al perderse, más lo que se movió desde ahí
        según la odometría (que no se pierde: solo acumula error de a poco)."""
        p, o0 = r['pose_creida'], r['odom_creida']
        o1 = self.lookup(self.cfg['odom_frame'], self.cfg['base_frame'])
        if p is None:
            return self.pose_robot()
        if o0 is None or o1 is None:
            return p
        dx, dy = o1[0] - o0[0], o1[1] - o0[1]
        c, s_ = math.cos(-o0[2]), math.sin(-o0[2])
        lx, ly = c * dx - s_ * dy, s_ * dx + c * dy           # avance en el marco del robot al perderse
        c, s_ = math.cos(p[2]), math.sin(p[2])
        return (p[0] + c * lx - s_ * ly, p[1] + s_ * lx + c * ly, ang_norm(p[2] + o1[2] - o0[2]))

    def mover_para_relocalizar(self, r, ahora):
        """Quieto `espera_mover_reloc_s` (AMCL se actualiza igual con /request_nomotion_update);
        si no convergió, avanza un tramo corto hacia donde hay lugar y vuelve a esperar.
        Devuelve True mientras el robot está quieto (se puede validar AMCL)."""
        c = self.cfg
        if r['mov'] == 'moviendo':
            odom = self.lookup(c['odom_frame'], c['base_frame'])
            movido = math.dist(odom[:2], r['p0_mov'][:2]) if odom is not None and r['p0_mov'] else 0.0
            puntos = self.puntos_scan()
            libre = self.espacio_libre(puntos, r['ang_mov']) if puntos is not None else 0.0
            v = c['velocidad_mover_reloc']
            if (movido >= r['dist_mov'] or libre < 0.05
                    or ahora - r['t_mov'] > 2.0 + 3.0 * r['dist_mov'] / v):
                self.pub_cmd.publish(Twist())
                r.update(mov='esperando', t_mov=ahora, movs=r['movs'] + 1, estable_desde=None)
                return False
            cmd = Twist()
            cmd.linear.x, cmd.linear.y = v * math.cos(r['ang_mov']), v * math.sin(r['ang_mov'])
            self.pub_cmd.publish(cmd)
            return False
        self.pub_cmd.publish(Twist())
        if (r['estable_desde'] is None and r['movs'] < c['movimientos_reloc']
                and ahora - r['t_mov'] >= c['espera_mover_reloc_s']):
            puntos = self.puntos_scan()
            mejor = None
            if puntos is not None:
                for ang in np.radians(np.arange(0, 360, 15)):
                    libre = self.espacio_libre(puntos, float(ang))
                    puntaje = min(libre, 0.8)
                    if r['ang_mov'] is not None:       # preferir una dirección nueva
                        puntaje -= 0.2 * math.cos(float(ang) - r['ang_mov'])
                    if libre >= 0.20 and (mejor is None or puntaje > mejor[0]):
                        mejor = (puntaje, float(ang), libre)
            if mejor is None:
                r['movs'] = c['movimientos_reloc']    # sin lugar: queda quieto validando
                return True
            dist = min(c['distancia_mover_reloc_m'], mejor[2] - 0.12)
            self.get_logger().info(f'AMCL sin converger: me muevo {100 * dist:.0f} cm hacia '
                                   f'{math.degrees(mejor[1]):.0f}° para ver el lugar desde otro punto')
            r.update(mov='moviendo', ang_mov=mejor[1], dist_mov=dist, t_mov=ahora,
                     p0_mov=self.lookup(c['odom_frame'], c['base_frame']))
            return False
        return True

    def paso_relocalizacion(self):
        r = self.reloc
        if r is None:
            return
        ahora = time.monotonic()
        if r['fase'] == 'frenando':
            self.pub_cmd.publish(Twist())
            cancelacion = r['cancelacion']
            if ahora - r['t0'] < 1.0 or (cancelacion is not None and not cancelacion.done()):
                if ahora - r['t0'] >= 5.0 and ahora - r['ultimo_aviso'] >= 5.0:
                    self.get_logger().warn('Sigo esperando que termine la navegación cancelada antes de reiniciar AMCL')
                    r['ultimo_aviso'] = ahora
                return
            # 1º) Corrección LOCAL (~1 s): buscar cerca la pose donde el láser vuelve
            #     a calzar con el mapa y dársela a AMCL. Es lo que pasa casi siempre
            #     (AMCL atrasado en un giro). Reiniciar AMCL en todo el mapa tarda
            #     decenas de segundos y en un laberinto con pasillos parecidos puede
            #     converger a otro lugar: queda como último recurso.
            if not r.get('local_hecho'):
                r['local_hecho'] = True
                if self.relocalizar_local():
                    r.update(fase='confirmando_local', t_local=ahora)
                    return
            # 2º) Partículas alrededor de donde cree estar (como en la largada), no
            #     en todo el mapa: el reinicio global lo ubicaba en lugares a los que
            #     el robot no pudo haber llegado. Cada reintento abre más la nube.
            centro = self.pose_creida_ahora(r)
            if centro is None:
                if ahora - r['ultimo_aviso'] >= 5.0:
                    self.get_logger().warn('Sin pose para reubicar AMCL; navegación detenida')
                    r['ultimo_aviso'] = ahora
                return
            k = self.cfg['factor_std_reintento'] ** r['intentos']
            std_xy = min(self.cfg['std_reubicar_xy_m'] * k, self.cfg['max_std_reubicar_xy_m'])
            std_yaw = min(self.cfg['std_reubicar_yaw_rad'] * k, math.pi)
            self.publicar_pose(centro[0], centro[1], centro[2], std_xy, std_yaw)
            r.update(fase='reiniciando', centro=centro, std_reubicar=std_xy)
            self.get_logger().warn(
                f'Reparto las partículas de AMCL alrededor de donde creo estar ({centro[0]:.2f}, '
                f'{centro[1]:.2f}, {math.degrees(centro[2]):.0f}°) con σ={100 * std_xy:.0f} cm, '
                f'{math.degrees(std_yaw):.0f}°')
            return
        if r['fase'] == 'confirmando_local':
            self.pub_cmd.publish(Twist())
            if ahora - r['t_local'] < 1.5:
                return
            pose = self.pose_robot()
            error = self.medir_localizacion(pose) if pose is not None else None
            if error is not None and error <= self.cfg['error_reloc_ok']:
                self.error_loc = None
                self.loc_mala_desde = None
                self.fallos_localizacion.clear()
                self.hist.clear()
                self.hist_global.clear()
                self.get_logger().info(f'Relocalización local confirmada (láser a {100 * error:.1f} cm del mapa)')
                self.reloc = None
                self.estado = r['siguiente']
                return
            self.get_logger().warn('La corrección local no se sostuvo: reparto partículas alrededor de donde creo estar')
            r.update(fase='frenando', t0=ahora - 1.0, cancelacion=None)
            return
        if r['fase'] == 'reiniciando':
            self.pub_cmd.publish(Twist())
            r.update(fase='convergiendo', t0=ahora, secuencia_inicio=self.secuencia_amcl,
                     estable_desde=None, ultimo_scan=None, giro_fin=False,
                     intentos=r['intentos'] + 1, mov='esperando', t_mov=ahora, movs=0)
            self.std_amcl = (math.inf, math.inf)
            self.error_loc = None
            self.loc_mala_desde = None
            self.lista_negra.clear()
            self.portales_fallidos.clear()
            # Los objetos fuera del mapa ya confirmados se CONSERVAN: son referencias
            # para ubicarse (también están en el mapa de AMCL). Solo se limpia lo del
            # costmap, que pudo quedar marcado con la pose equivocada.
            self.grilla.actualizar_obstaculos(np.zeros_like(self.grilla.libre))
            if self.sectorizador is not None:
                self.sectorizador.indice_ordenado = -1
            self.dist_casa = None
            self.get_logger().warn(f'AMCL reubicado (intento {r["intentos"]}); espero convergencia con lecturas nuevas')
            return

        if (self.nomotion_cli.service_is_ready() and ahora - r['ultimo_nomotion'] >= 0.5
                and (r['nomotion'] is None or r['nomotion'].done())):
            try:
                if r['nomotion'] is not None:
                    r['nomotion'].result()
                r['nomotion'] = self.nomotion_cli.call_async(EmptySrv.Request())
            except Exception as error:  # noqa: BLE001
                r['nomotion'] = None
                if ahora - r['ultimo_aviso'] >= 5.0:
                    self.get_logger().warn(f'AMCL no aceptó la actualización sin movimiento: {error}')
                    r['ultimo_aviso'] = ahora
            r['ultimo_nomotion'] = ahora
        r['giro_fin'] = self.mover_para_relocalizar(r, ahora)
        if not r['giro_fin']:
            r['estable_desde'] = None
        else:
            pose = self.pose_robot()
            stamp = Time.from_msg(self.scan.header.stamp).nanoseconds if self.scan is not None else None
            if stamp is not None and stamp != r['ultimo_scan']:
                r['ultimo_scan'] = stamp
                error = self.medir_localizacion(pose) if pose is not None else None
                fresca = (self.secuencia_amcl >= r['secuencia_inicio'] + 3
                          and self.t_ultima_pose_amcl is not None
                          and ahora - self.t_ultima_pose_amcl <= 2.0)
                creida = self.pose_creida_ahora(r) if r.get('centro') is not None else None
                # Una pose a más de 3σ de donde creía estar no es alcanzable: no aceptarla.
                cerca = (creida is None or pose is None
                         or math.dist(pose[:2], creida[:2]) <= 3.0 * r.get('std_reubicar', 1.0))
                buena = (fresca and error is not None and error <= self.cfg['error_reloc_ok'] and cerca
                         and self.std_amcl[0] <= self.cfg['relocalizacion_std_xy_m']
                         and self.std_amcl[1] <= self.cfg['relocalizacion_std_yaw_rad'])
                if buena:
                    if r['estable_desde'] is None:
                        r['estable_desde'] = ahora
                        r['pose_estable'] = pose
                    elif (math.dist(pose[:2], r['pose_estable'][:2]) > 0.10
                          or abs(ang_norm(pose[2] - r['pose_estable'][2])) > 0.15):
                        r['estable_desde'] = ahora
                        r['pose_estable'] = pose
                else:
                    r['estable_desde'] = None
        if (r['estable_desde'] is not None
                and ahora - r['estable_desde'] >= self.cfg['confirmacion_relocalizacion_s']
                and self.t_ultima_pose_amcl is not None and ahora - self.t_ultima_pose_amcl <= 2.0
                and self.std_amcl[0] <= self.cfg['relocalizacion_std_xy_m']
                and self.std_amcl[1] <= self.cfg['relocalizacion_std_yaw_rad']
                and self.puntos_scan() is not None):
            self.error_loc = None
            self.loc_mala_desde = None
            self.fallos_localizacion.clear()
            self.reintentos_regreso = 0
            self.n_atascos = 0
            self.hist.clear()
            self.hist_global.clear()
            self.limpiar_costmaps()
            self.get_logger().info(f'AMCL convergió: σxy={self.std_amcl[0]:.2f} m, '
                                   f'σyaw={math.degrees(self.std_amcl[1]):.1f}°; recalculo la ruta')
            self.reloc = None
            self.estado = r['siguiente']
            return
        if ahora - r['t0'] >= self.cfg['timeout_relocalizacion_s']:
            r['cancelacion'] = self.cancelar_actual()
            r.update(fase='frenando', t0=ahora, estable_desde=None)
            self.pub_cmd.publish(Twist())
            self.pub_cmd.publish(Twist())
            self.get_logger().warn('AMCL todavía no converge: vuelvo a repartir las partículas, un poco más abiertas')

    def lookup(self, padre, hijo):
        try:
            t = self.tf_buffer.lookup_transform(padre, hijo, Time())
        except Exception:  # noqa: BLE001
            return None
        tr = t.transform.translation
        return tr.x, tr.y, quat_a_yaw(t.transform.rotation)

    def pose_robot(self):
        if self.grilla is None:
            return None
        return self.lookup(self.grilla.frame, self.cfg['base_frame'])

    def buscar_pose(self, centro, rango_xy=0.30, rango_yaw=0.40, paso_xy=0.03, paso_yaw=0.04,
                    prior=0.004):
        """Pose cercana a `centro` (x, y, yaw) donde el láser mejor calza con las paredes
        del mapa. Devuelve (x, y, yaw, error_medio_m, error_en_centro_m) o None."""
        puntos = self.puntos_scan(alcance=self.cfg.get('rango_laser_localizacion_m', 8.0))
        if puntos is None or self.dist_pared is None or puntos[0].size < 30:
            return None
        px, py = puntos
        paso = max(1, math.ceil(px.size / 120))
        px, py = px[::paso], py[::paso]
        g = self.grilla
        x0, y0, yaw0 = centro
        off = np.arange(-rango_xy, rango_xy + 1e-4, paso_xy)
        ox, oy = np.meshgrid(off, off)
        ox, oy = (x0 + ox).ravel(), (y0 + oy).ravel()
        def error_scan(mx, my):
            # Interpolar igual que en medir_localizacion, incluyendo todos los
            # rayos fuera del mapa y la media celda de las paredes del borde.
            u, v = (mx - g.ox) / g.res - 0.5, (my - g.oy) / g.res - 0.5
            uc, vc = np.clip(u, 0, g.ancho - 1), np.clip(v, 0, g.alto - 1)
            col, f = np.floor(uc).astype(int), np.floor(vc).astype(int)
            col1, f1 = np.minimum(col + 1, g.ancho - 1), np.minimum(f + 1, g.alto - 1)
            a, b = uc - col, vc - f
            d = ((1 - a) * (1 - b) * D[f, col] + a * (1 - b) * D[f, col1]
                 + (1 - a) * b * D[f1, col] + a * b * D[f1, col1])
            d += g.res * np.hypot(u - uc, v - vc)
            return np.clip(d - 0.5 * g.res, 0.0, 0.30).mean(axis=-1)

        D = self.campo_loc()
        c0, s0 = math.cos(yaw0), math.sin(yaw0)
        en_centro = float(error_scan(x0 + c0 * px - s0 * py, y0 + s0 * px + c0 * py))
        mejor = None
        for dyaw in np.arange(-rango_yaw, rango_yaw + 1e-4, paso_yaw):
            yaw = yaw0 + dyaw
            c, s = math.cos(yaw), math.sin(yaw)
            rx, ry = c * px - s * py, s * px + c * py
            mx = ox[:, None] + rx[None, :]
            my = oy[:, None] + ry[None, :]
            crudo = error_scan(mx, my)
            # Desempate hacia el centro: en un pasillo largo varias poses "calzan"
            # igual; preferimos la más cercana a la estimación actual.
            err = crudo + prior * (np.hypot(ox - x0, oy - y0) / rango_xy + abs(dyaw) / rango_yaw)
            i = int(np.argmin(err))
            if mejor is None or err[i] < mejor[3]:
                mejor = (float(ox[i]), float(oy[i]), float(ang_norm(yaw)), float(err[i]), float(crudo[i]))
        return mejor[0], mejor[1], mejor[2], mejor[4], en_centro

    def estimar_pose_inicial(self):
        """Pose de largada: la que mejor calza el láser con el mapa cerca de la nominal."""
        r = self.buscar_pose((self.cfg['x_inicial'], self.cfg['y_inicial'], self.cfg['yaw_inicial']))
        return None if r is None else r[:4]

    def publicar_pose(self, x, y, yaw, std_xy, std_yaw):
        m = PoseWithCovarianceStamped()
        m.header.frame_id = self.grilla.frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.pose.pose.position.x, m.pose.pose.position.y = x, y
        m.pose.pose.orientation.z, m.pose.pose.orientation.w = yaw_a_quat(yaw)
        cov = [0.0] * 36
        cov[0] = cov[7] = std_xy ** 2
        cov[35] = std_yaw ** 2
        m.pose.covariance = cov
        self.pub_init.publish(m)

    def publicar_pose_inicial(self):
        x, y, yaw = self.cfg['x_inicial'], self.cfg['y_inicial'], self.cfg['yaw_inicial']
        std_xy, std_yaw = self.cfg['std_xy'], self.cfg['std_yaw']
        if self.pose_inicial_estimada is None:
            est = self.estimar_pose_inicial()
            if est is not None:
                self.pose_inicial_estimada = est
                self.get_logger().info(
                    f'Pose de largada por láser vs. mapa: ({est[0]:.2f}, {est[1]:.2f}, '
                    f'{math.degrees(est[2]):.0f}°), error medio {100 * est[3]:.1f} cm')
        est = self.pose_inicial_estimada
        if est is not None and est[3] < 0.08:
            x, y, yaw = est[0], est[1], est[2]
            std_xy, std_yaw = 0.12, 0.12   # confiamos más: el láser ya calzó
        self.publicar_pose(x, y, yaw, std_xy, std_yaw)

    def fijar_largada(self, forzar=False):
        """Largada en el marco map = (map->odom actual) ∘ (pose de odom al largar)."""
        if self.largada_fija:
            return
        convergido = (max(self.std_amcl) < self.cfg['std_convergido']
                      and self.error_loc is not None and self.error_loc < 0.6 * self.cfg['error_loc_malo'])
        vencido = time.monotonic() - (self.t_localizado or time.monotonic()) > self.cfg['max_espera_convergencia_s']
        if not (convergido or vencido or forzar):
            return
        m_o = self.lookup(self.grilla.frame, self.cfg['odom_frame'])
        if m_o is not None and self.p0_odom is not None:
            tx, ty, th = m_o
            px, py, pth = self.p0_odom
            self.largada = (tx + math.cos(th) * px - math.sin(th) * py,
                            ty + math.sin(th) * px + math.cos(th) * py,
                            ang_norm(th + pth))
        self.largada_fija = True
        self.dist_casa = None
        self.get_logger().info(
            f'Largada estimada en ({self.largada[0]:.2f}, {self.largada[1]:.2f}) '
            f'[AMCL std={self.std_amcl[0]:.3f} m]')
