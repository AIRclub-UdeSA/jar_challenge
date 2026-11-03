import math
import time
from equipo_jar.observacion_objetos import leer_observaciones, objetos_del_scan
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Twist
from nav2_msgs.msg import SpeedLimit
from rclpy.time import Time
from equipo_jar.exploracion_estados import (
    ESPERANDO_MAPA,
    LOCALIZANDO,
    ACTIVANDO_NAV2,
    ELIGIENDO,
    RELOCALIZANDO,
    OBSERVANDO_OBJETO,
    TERMINADO,
)
from equipo_jar.exploracion_geometria import ang_norm, punto_para_mirar, dilatar, bfs


class ObjetosExploracion:

    def cb_imagen(self, msg):
        if msg.header.frame_id and msg.width > 0 and msg.height > 0 and msg.data:
            stamp = Time.from_msg(msg.header.stamp).nanoseconds
            if stamp <= self.ultimo_stamp_imagen_camara:
                return
            self.ultimo_stamp_imagen_camara = stamp
            self.frame_camara = msg.header.frame_id
            self.t_ultima_imagen = time.monotonic()

    def cb_info_camara(self, msg):
        if msg.header.frame_id and msg.width > 0 and msg.k[0] > 0:
            self.frame_camara = msg.header.frame_id
            self.fov = 2 * math.atan(msg.width / (2 * msg.k[0]))
            self.ventana = max(1, int(round(self.fov / (2 * math.pi / self.nrayos))))

    def cb_pendientes(self, msg):
        self.pendientes_vision = [(p.position.x, p.position.y, p.position.z) for p in msg.poses]
        self.t_pendientes = time.monotonic()

    def cb_observaciones(self, msg):
        if self.grilla is None:
            return
        observaciones = leer_observaciones(msg.data, self.grilla.frame)
        if observaciones is None or observaciones['stamp_ns'] <= self.ultimo_stamp_vision:
            return
        edad = (self.get_clock().now().nanoseconds - observaciones['stamp_ns']) / 1e9
        if not 0.0 <= edad <= 0.75:
            return
        self.ultimo_stamp_vision = observaciones['stamp_ns']
        self.observaciones_vision = dict(observaciones, recibido=time.monotonic())
        self.objetos_fuera_mapa.actualizar_vision(observaciones['objetos'], time.monotonic())

    def cb_scan(self, msg):
        self.scan = msg
        self.scans_objetos.append((msg, self.vel_giro))
        self.actualizar_objetos_lidar()

    def actualizar_objetos_lidar(self):
        """Procesa el barrido más reciente cuya TF ya llegó, sin esperas bloqueantes."""
        ahora = time.monotonic()
        if (self.grilla is None or not self.cfg['inspeccionar_objetos']
                or self.estado in (ESPERANDO_MAPA, LOCALIZANDO, ACTIVANDO_NAV2, RELOCALIZANDO, TERMINADO)
                or self.loc_mala_desde is not None or self.vel_giro > 0.5
                or ahora - self.t_scan_objetos < 0.2):
            return
        elegido = None
        for msg, giro in reversed(self.scans_objetos):
            stamp = Time.from_msg(msg.header.stamp).nanoseconds
            edad = (self.get_clock().now().nanoseconds - stamp) / 1e9
            if (not msg.header.frame_id or not stamp or stamp <= self.ultimo_stamp_scan_objetos
                    or not 0 <= edad <= self.cfg['timeout_scan_s'] or giro > 0.5):
                continue
            try:
                tf = self.tf_buffer.lookup_transform(self.grilla.frame, msg.header.frame_id,
                                                     Time.from_msg(msg.header.stamp)).transform
            except Exception:  # noqa: BLE001  conservar el scan hasta que llegue su TF
                continue
            elegido = msg, stamp, tf
            break
        if elegido is None:
            return
        msg, stamp, tf = elegido
        q, p = tf.rotation, tf.translation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self.t_scan_objetos = ahora
        self.ultimo_stamp_scan_objetos = stamp
        centros = objetos_del_scan(msg, (p.x, p.y, yaw), self.grilla, self.dist_pared,
            self.cfg['separacion_pared_objeto_m'], self.cfg['rango_vision'],
            profundidad=self.cfg.get('profundidad_objeto_m', 0.12))
        centros = [c for c in centros if self.linea_libre_al_objeto((p.x, p.y), c)]
        centros = [c for c in centros if not self.corregir_bloque_con_lectura(c, (p.x, p.y))]
        self.objetos_fuera_mapa.actualizar(centros, ahora)

    def pose_camara(self):
        """Posición y rumbo del eje óptico +Z en el marco del mapa."""
        if self.grilla is None or self.frame_camara is None:
            return None
        try:
            t = self.tf_buffer.lookup_transform(self.grilla.frame, self.frame_camara, Time())
        except Exception:  # noqa: BLE001
            return None
        p, q = t.transform.translation, t.transform.rotation
        eje_x = 2 * (q.x * q.z + q.w * q.y)
        eje_y = 2 * (q.y * q.z - q.w * q.x)
        if math.hypot(eje_x, eje_y) < 0.2:
            return None
        return p.x, p.y, math.atan2(eje_y, eje_x)

    def linea_libre_al_objeto(self, origen, objeto):
        """Descarta un obstáculo que está detrás de una pared del mapa."""
        g = self.grilla
        dx, dy = objeto[0] - origen[0], objeto[1] - origen[1]
        d = math.hypot(dx, dy)
        recorrido = max(0.0, d - 0.12)
        pasos = max(1, int(math.ceil(recorrido / (g.res / 2))))
        for i in range(pasos + 1):
            fraccion = i * recorrido / pasos / max(d, 1e-6)
            f, c = g.a_celda(origen[0] + fraccion * dx, origen[1] + fraccion * dy)
            if not g.dentro(f, c) or not g.libre[f, c]:
                return False
        return True

    def considerar_objeto_nuevo(self, pose):
        if not self.cfg['inspeccionar_objetos'] or self.objeto_observado is not None:
            return False
        cam = self.pose_camara() or pose
        if cam is None:
            return False
        ahora = time.monotonic()
        candidatos = sorted(self.objetos_fuera_mapa.confirmados(ahora),
                            key=lambda p: math.dist(p, cam[:2]))
        for objeto in candidatos:
            distancia = math.dist(objeto, cam[:2])
            if not (0.25 <= distancia <= self.cfg['rango_vision']):
                continue
            if not self.linea_libre_al_objeto(cam[:2], objeto):
                continue
            cancelacion = self.cancelar_actual()
            self.goal_actual = None
            self.pendiente = None
            visual = self.observacion_visual(objeto)
            self.objeto_observado = dict(punto=objeto, inicio=ahora,
                puntos=[objeto], id_visual=visual['id'] if visual else None,
                desde=None, quieto_desde=None, quieto_stamp_ns=None, ultimo_frame=None,
                imagenes=0, positivas=0, color=visual['color'] if visual else None,
                fase='frenando', reubicaciones=0, giros=0, cancelacion=cancelacion)
            self.estado = OBSERVANDO_OBJETO
            self.pub_cmd.publish(Twist())
            self.get_logger().info(
                f'Objeto fuera del mapa en ({objeto[0]:.2f}, {objeto[1]:.2f}): '
                f'mirada breve de {self.cfg["tiempo_observacion_objeto_s"]:.1f} s; '
                'usaré LiDAR y la cámara si está disponible')
            return True
        return False

    def observacion_visual(self, punto):
        mensaje = self.observaciones_vision
        if mensaje is None or time.monotonic() - mensaje['recibido'] > 0.5:
            return None
        objetos = [o for o in mensaje['objetos'] if math.dist(punto, (o['x'], o['y'])) < 0.40]
        return min(objetos, key=lambda o: math.dist(punto, (o['x'], o['y'])), default=None)

    def limite_observacion(self, porcentaje):
        m = SpeedLimit()
        m.header.stamp = self.get_clock().now().to_msg()
        m.percentage = True
        m.speed_limit = float(porcentaje)
        self.pub_vel.publish(m)
        self.ultimo_limite = porcentaje

    def posicion_observacion(self, pose, objeto):
        """Piso alcanzable, distancia útil y vista libre de paredes y otros objetos."""
        g = self.grilla
        # El objeto tampoco puede atravesarse si todavía no apareció en el costmap.
        objetivo = self._cerca_de(*objeto, radio=self.cfg['lado_bloque_objeto_m'] / 2)
        seguro = g.transitable_base & ~dilatar(g.obstaculos_extra | objetivo, g.m_trans)
        dist = bfs(seguro, *g.a_celda(*pose[:2]))
        if dist is None:
            return None
        vista = g.libre & ~(g.obstaculos_extra & ~self._cerca_de(*objeto))
        return punto_para_mirar(g.candidatable & seguro, vista, dist, g.res, (g.ox, g.oy), objeto,
            d_min=self.cfg['distancia_observacion_min_m'], d_max=self.cfg['distancia_observacion_max_m'],
            d_ideal=self.cfg['distancia_observacion_ideal_m'], fin_vista=0.25)

    def recordar_inspeccion(self, o, observado):
        self.objetos_fuera_mapa.finalizar(o['punto'], time.monotonic(), observado,
            aliases=o['puntos'], id_visual=o['id_visual'])
        punto = o['punto']
        # El color de una pared nunca habilita un bloque que la geometría rechaza.
        f, c = self.grilla.a_celda(*punto)
        campo = getattr(self, 'dist_pared', None)
        if not self.grilla.dentro(f, c):
            return
        d = float(campo[f, c]) if campo is not None else float('inf')
        if d < self.cfg.get('distancia_pared_bloque_m', 0.20):
            self.get_logger().info(
                f'Objeto en ({punto[0]:.2f}, {punto[1]:.2f}) a '
                f'{100 * d:.0f} cm de una pared del mapa: no lo bloqueo (probable eco de la pared)')
            return
        self.registrar_objeto_local(punto)

    def terminar_observacion_objeto(self, observado, motivo):
        o = self.objeto_observado
        if o is None:
            return
        cancelacion = self.cancelar_actual()
        self.goal_actual = None
        # No reanudar una ruta hasta el resultado final de la acción anterior.
        self.cancelacion_cobertura = (cancelacion, float('inf')) if cancelacion is not None else None
        self.pub_cmd.publish(Twist())
        self.limite_observacion(100.0)
        self.recordar_inspeccion(o, observado)
        self.limpiar_costmaps()
        self.ruta_sector.clear()
        self.ruta_region = None
        if observado:
            self.marcar_alrededor(*o['punto'], self.cfg['radio_visto_objeto_m'])
        self.objeto_observado = None
        self.pendiente = None
        self.escape = None
        self.giro_pendiente = None
        self.estado = ELIGIENDO
        self.get_logger().info(f'Observación de objeto: {motivo}; guardado en el mapa local, no lo vuelvo a analizar')

    def paso_observacion_objeto(self):
        """Una mirada corta; la ausencia de confirmación visual nunca repite el objeto."""
        o = self.objeto_observado
        if o is None:
            return
        ahora = time.monotonic()
        if ahora - o['inicio'] > self.cfg['timeout_observacion_objeto_s']:
            self.terminar_observacion_objeto(False, 'conservo la posición del LiDAR sin insistir con la cámara')
            return
        if o['fase'] == 'acercando':
            self.limite_observacion(self.cfg['porcentaje_vel_observacion'])
            return  # Nav2 sigue usando LiDAR aunque dejen de llegar imágenes.
        self.pub_cmd.publish(Twist())
        cancelacion = o['cancelacion']
        if ahora - o['inicio'] < 0.3 or (cancelacion is not None and not cancelacion.done()):
            return
        if math.hypot(*self.vel_base) > 0.05 or self.vel_giro > 0.15:
            o.update(desde=None, quieto_desde=None, imagenes=0, positivas=0)
            return
        if o['quieto_desde'] is None:
            o['quieto_desde'] = ahora
            o['quieto_stamp_ns'] = self.get_clock().now().nanoseconds
        if ahora - o['quieto_desde'] < 0.2:
            return
        cam = self.pose_camara()
        camara_activa = (cam is not None and self.t_ultima_imagen is not None
                         and ahora - self.t_ultima_imagen <= 0.75)
        punto_lidar = self.objetos_fuera_mapa.mas_cercano(o['punto'], ahora)
        if punto_lidar is not None:
            o['punto'] = punto_lidar
            if all(math.dist(punto_lidar, p) >= 0.10 for p in o['puntos']):
                o['puntos'].append(punto_lidar)
        visual = self.observacion_visual(o['punto']) if camara_activa else None
        if visual is not None:
            # La cámara añade atributos sin desplazar la posición confirmada por /scan.
            o['color'], o['id_visual'] = visual['color'], visual['id']
        punto = o['punto']
        referencia = cam or self.pose_robot()
        error = 0.0
        en_cuadro = False
        if referencia is not None:
            error = ang_norm(math.atan2(punto[1] - referencia[1], punto[0] - referencia[0]) - referencia[2])
            mensaje = self.observaciones_vision
            # La mancha corrige el rumbo solo si la foto se tomó después de frenar.
            if visual is not None and mensaje['stamp_ns'] > o['quieto_stamp_ns']:
                error = visual['error_angular']
            tolerancia = min(math.radians(self.cfg['tolerancia_centrar_objeto_deg']), self.fov / 2 - 0.1)
            max_giros = 2 if o.get('giro_resultado') == GoalStatus.STATUS_SUCCEEDED else 1
            if abs(error) > tolerancia and o['giros'] < max_giros:
                if not self.spin_cli.server_is_ready():
                    self.recuperar_nav2(siguiente=OBSERVANDO_OBJETO)
                    return
                o['giros'] += 1
                o.update(desde=None, quieto_desde=None, imagenes=0, positivas=0)
                self.get_logger().info(f'[objeto] Apunto al objeto del LiDAR: giro {math.degrees(error):.0f}° '
                                       f'con el eje {self.frame_camara or "frontal del robot"}')
                self.girar(error, OBSERVANDO_OBJETO)
                return
            en_cuadro = cam is not None and abs(error) <= tolerancia
        if camara_activa:
            distancia = visual['distancia'] if visual else math.dist(punto, cam[:2])
            if (o['reubicaciones'] == 0 and not self.cfg['distancia_observacion_min_m'] <= distancia
                    <= self.cfg['distancia_observacion_max_m']):
                o['reubicaciones'] = 1
                pose = self.pose_robot()
                posicion = self.posicion_observacion(pose, punto) if pose is not None else None
                if posicion is not None:
                    x, y, yaw, d_m = posicion
                    if (d_m <= 0.60 and self.tiempo_restante() > self.segundos_para_volver(x, y)
                            + self.cfg['margen_regreso_s'] + self.cfg['tiempo_observacion_objeto_s'] + d_m / 0.10):
                        yaw = ang_norm(yaw - ang_norm(cam[2] - pose[2]))
                        o.update(fase='acercando', desde=None, quieto_desde=None, imagenes=0, positivas=0)
                        self.limite_observacion(self.cfg['porcentaje_vel_observacion'])
                        if not self.enviar_goal(x, y, yaw, 1.0, d_m, etiqueta='observar objeto'):
                            self.terminar_observacion_objeto(False, 'Nav2 no disponible; conservo el obstáculo del LiDAR')
                        return
        if o['desde'] is None:
            o['desde'] = ahora
        mensaje = self.observaciones_vision
        if (mensaje is not None and ahora - mensaje['recibido'] <= 0.5
                and mensaje['stamp_ns'] > o['quieto_stamp_ns'] + 200_000_000
                and mensaje['stamp_ns'] != o['ultimo_frame']):
            o['ultimo_frame'] = mensaje['stamp_ns']
            o['imagenes'] += 1
            if visual is not None:
                o['positivas'] += 1
        if ahora - o['desde'] >= self.cfg['tiempo_observacion_objeto_s']:
            con_color = (visual is not None and visual['color'] != 'desconocido'
                         and o['positivas'] >= self.cfg['imagenes_observacion_minimas']
                         and abs(error) <= math.radians(self.cfg['tolerancia_centrar_objeto_deg']))
            mirada = (en_cuadro and camara_activa
                      and self.t_ultima_imagen > o['quieto_desde'] + 0.2)
            o['color_confirmado'] = bool(mirada and con_color)
            detalle = (f'objeto {o["color"]} visto con cámara y LiDAR' if mirada and con_color
                       else 'mirada completada hacia el objeto; sin color confirmado' if mirada
                       else 'posición del LiDAR guardada; no hubo una mirada de cámara válida')
            self.terminar_observacion_objeto(mirada, detalle)
