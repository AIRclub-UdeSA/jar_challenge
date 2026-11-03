"""Nodo de percepción (ADR-0006): cajas rojas por color + LiDAR, en `map`.

Por cada imagen (con un tope de frecuencia): blobs rojos → ventana de rumbos →
puntos del LiDAR en esa ventana → centro de la caja en `map` → tracker.
Publica los objetivos en /jar/objetivos (JSON, lo lee la misión) y marcadores
para RViz en /jar/objetivos/marcadores. Mientras el robot gira rápido no
detecta: la cámara y el LiDAR quedan desfasados.

Además lleva los candidatos (candidatos.py): todo objeto del LiDAR que el mapa no tiene, del
tamaño de una caja. Cada uno se inspecciona cuando la cámara lo tiene de frente y queda como
víctima o descartado; la lista va a /jar/candidatos y el explorador manda a mirar los que
faltan.
"""
import json
import math

import numpy as np
import rclpy
from scipy import ndimage
from cv_bridge import CvBridge
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, LaserScan
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from jar_percepcion.candidatos import (Candidatos, ConfigCandidatos, confirmar_con_candidatos,
                                       en_vista, lugar_vacio, objetos_del_scan,
                                       vista_concluyente)
from jar_percepcion.color import ConfigColor, segmentar_rojo, toca_borde
from jar_percepcion.fusion import (alto_coherente, estimar_centro, puntos_en_ventana,
                                   rotacion_de_cuaternion, rumbo_en_laser, transformar,
                                   ventana_angular)
from jar_percepcion.tracker import ConfigTracker, Tracker

QOS_MAPA = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                      durability=DurabilityPolicy.TRANSIENT_LOCAL)


def yaw_de(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def segundos(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


class NodoPercepcion(Node):
    def __init__(self):
        super().__init__('percepcion')
        self.cfg_color = ConfigColor()
        for nombre, valor in vars(self.cfg_color).items():
            setattr(self.cfg_color, nombre, self.declare_parameter(nombre, valor).value)
        cfg_tracker = ConfigTracker()
        for nombre, valor in vars(cfg_tracker).items():
            setattr(cfg_tracker, nombre, self.declare_parameter(nombre, valor).value)
        p = self.declare_parameter
        self.profundidad = p('profundidad_media_caja_m', 0.125).value
        self.alcance = p('alcance_deteccion_m', 2.5).value
        self.recorte = p('recorte_ventana', 0.15).value
        self.salto = p('salto_cluster_m', 0.15).value
        self.min_puntos = p('puntos_min_lidar', 3).value
        # El rojo tiene que estar sobre el objeto: el más cercano en la ventana del blob ocupa al
        # menos esta fracción de sus rayos; si no, el rojo asoma detrás (0: como antes).
        self.fraccion_rojo_sobre_objeto = p('fraccion_rojo_sobre_objeto', 0.0).value
        # El rojo tiene que medir lo que mide una caja a la distancia del objeto donde se lo ubica
        # (fusion.alto_coherente): si mide menos de esta fracción, está detrás (0: como antes).
        self.alto_rojo_min_frac = p('alto_rojo_min_frac', 0.0).value
        self.alto_caja = p('alto_caja_m', 0.30).value
        self.giro_max = p('giro_max_deteccion', 0.4).value
        self.desfase_max = p('desfase_max_scan_s', 0.3).value
        self.fraccion_pared_max = p('fraccion_max_en_pared', 0.5).value
        self.periodo_min = 1.0 / p('frecuencia_max_hz', 5.0).value
        self.marco_mapa = p('marco_mapa', 'map').value
        # Candidato víctima sin objetivo rojo que lo confirme: se ofrece a la misión con la
        # posición del LiDAR (ver confirmar_con_candidatos).
        self.sumar_candidatos = p('reportar_candidatos_lidar', False).value
        # "No es rojo" si en la dirección del candidato no hay ningún píxel rojo (blobs del
        # borde incluidos), a cualquier distancia de `en_vista`: sin el mínimo de 0.7 m, que
        # con la tolerancia de Nav2 dejaba al robot quieto frente al objeto sin poder votar.
        self.no_rojo_por_pixeles = p('no_rojo_por_pixeles', False).value
        # Dudosos: los candidatos se publican apenas reciben un voto (la misión espera la
        # decisión mirando) y los que quedan sin decidir con algún voto rojo (o sin ningún voto)
        # se ofrecen para los cupos que sobren al cierre (ver confirmar_con_candidatos).
        self.seguir_dudosos = p('seguir_dudosos', False).value

        # Política de detección de candidatos (candidatos.py), como `candidatos.<campo>`.
        cfg_candidatos = ConfigCandidatos()
        for nombre, valor in vars(cfg_candidatos).items():
            setattr(cfg_candidatos, nombre, p(f'candidatos.{nombre}', valor).value)

        self.tracker = Tracker(cfg_tracker)
        self.candidatos = Candidatos(cfg_candidatos)
        self.distancia_pared = None   # m a la pared del mapa más cercana, por celda
        self.bridge = CvBridge()
        self.mapa = None
        self.mapa_arr = None
        self.camara = None
        self.scan = None
        self.giro = 0.0
        self.velocidad = 0.0
        self.ultimo_voto = {}         # id de candidato -> último motivo de voto registrado
        self.r_laser_optico = None
        self.t_ultima = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(OccupancyGrid, '/map', self.al_mapa, QOS_MAPA)
        self.create_subscription(Odometry, '/odom', self.al_odom, 10)
        self.create_subscription(LaserScan, '/scan', self.al_scan, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, '/cam_1/color/camera_info', self.al_camara,
                                 qos_profile_sensor_data)
        self.create_subscription(Image, '/cam_1/color/image_raw', self.al_imagen,
                                 qos_profile_sensor_data)
        self.pub = self.create_publisher(String, '/jar/objetivos', 10)
        self.pub_marcas = self.create_publisher(MarkerArray, '/jar/objetivos/marcadores', 10)
        self.pub_candidatos = self.create_publisher(String, '/jar/candidatos', 10)
        self.pub_marcas_c = self.create_publisher(MarkerArray, '/jar/candidatos/marcadores', 10)
        self.create_timer(1.0, self.publicar)

    def al_mapa(self, mapa):
        self.mapa = mapa.info
        self.mapa_arr = np.asarray(mapa.data, dtype=np.int16).reshape(mapa.info.height,
                                                                      mapa.info.width)
        self.distancia_pared = ndimage.distance_transform_edt(
            (self.mapa_arr < 50) & (self.mapa_arr >= 0)) * mapa.info.resolution

    def al_odom(self, msg):
        self.giro = msg.twist.twist.angular.z
        self.velocidad = math.hypot(msg.twist.twist.linear.x, msg.twist.twist.linear.y)

    def al_scan(self, msg):
        self.scan = msg

    def al_camara(self, msg):
        self.camara = msg

    def transformada(self, destino, origen):
        try:
            return self.tf_buffer.lookup_transform(destino, origen, rclpy.time.Time())
        except Exception:  # noqa: BLE001 - TF todavía no disponible
            return None

    def en_pared(self, puntos):
        """Fracción de puntos que caen en celdas ocupadas del mapa."""
        info = self.mapa
        j = np.floor((puntos[:, 0] - info.origin.position.x) / info.resolution).astype(int)
        i = np.floor((puntos[:, 1] - info.origin.position.y) / info.resolution).astype(int)
        dentro = (i >= 0) & (i < info.height) & (j >= 0) & (j < info.width)
        ocupados = np.zeros(len(puntos), dtype=bool)
        ocupados[dentro] = self.mapa_arr[i[dentro], j[dentro]] >= 50
        return float(np.mean(ocupados | ~dentro))

    def a_pared(self, puntos):
        """Distancia (m) de cada punto (en `map`) a la pared del mapa más cercana."""
        info = self.mapa
        j = np.floor((puntos[:, 0] - info.origin.position.x) / info.resolution).astype(int)
        i = np.floor((puntos[:, 1] - info.origin.position.y) / info.resolution).astype(int)
        dentro = (i >= 0) & (i < info.height) & (j >= 0) & (j < info.width)
        d = np.zeros(len(puntos))
        d[dentro] = self.distancia_pared[i[dentro], j[dentro]]
        return d

    def inspeccionar(self, scan, laser_en_mapa, ventanas, rojos, rojos_lejanos, fx, cx, ancho,
                     ventanas_crudas=None, ventanas_borde=()):
        """Candidatos: los suma desde el scan y vota los que la cámara tiene de frente.

        `rojos`: posiciones (en `map`) de las cajas rojas ubicadas en esta imagen. Un candidato
        es rojo si una cae en su lugar, no si sólo hay rojo en su dirección: desde una esquina
        de pared, una caja roja más lejos en la misma dirección lo hacía pasar por víctima.
        `rojos_lejanos`: también las de más lejos (posición aproximada), que no votan: sólo
        confirman que un objeto lejano del LiDAR es real (política de detección).
        `ventanas_crudas` (con `no_rojo_por_pixeles`): las de todos los blobs rojos, también
        los cortados por el borde; si se dan, "rojo en la dirección" se mira en ellas.
        `ventanas_borde`: las de los blobs cortados por el borde. Con el robot quieto y el
        candidato de frente, rojo cortado en su dirección es una caja roja demasiado cerca para
        ubicarla: se marca `rojo_cortado` y el explorador busca un punto de vista más lejos.
        """
        cfg = self.candidatos.config
        r = np.asarray(scan.ranges, dtype=float)
        ang = scan.angle_min + scan.angle_increment * np.arange(len(r))
        ok = (np.isfinite(r) & (r >= scan.range_min)
              & (r <= min(scan.range_max, cfg.alcance_posible_m + cfg.profundidad_m)))
        indices = np.flatnonzero(ok)
        puntos = np.column_stack((r[ok] * np.cos(ang[ok]), r[ok] * np.sin(ang[ok])))
        vistos = set()
        if len(puntos):
            for obj in objetos_del_scan(transformar(puntos, *laser_en_mapa), self.a_pared,
                                        laser_en_mapa[:2], cfg, rangos=r, indices=indices,
                                        range_min=scan.range_min):
                cand = self.candidatos.observar(obj, laser_en_mapa[:2], rojos_lejanos)
                if cand is not None:
                    vistos.add(cand.id)
        for quedo, se_fue in self.candidatos.unir():
            self.get_logger().info(f'candidato {se_fue} unido al {quedo} (la misma caja)')
        for cand in self.candidatos.marcar_rojo_lejano(rojos_lejanos):
            self.get_logger().info(f'candidato {cand.id} en ({cand.x:.2f}, {cand.y:.2f}): '
                                   'rojo de lejos, a inspeccionar')
        rumbo_camara = rumbo_en_laser(cx, fx, cx, self.r_laser_optico)
        medio_fov = math.atan(ancho / (2.0 * fx))

        def rango_en_rumbo(a):
            k = int(round((a - scan.angle_min) / scan.angle_increment)) % len(r)
            return r[k] if np.isfinite(r[k]) else None
        for cand in list(self.candidatos.lista):
            if (cand.id not in vistos
                    and lugar_vacio((cand.x, cand.y), laser_en_mapa, rango_en_rumbo,
                                    self.candidatos.config)
                    and self.candidatos.ausente(cand)):
                self.get_logger().info(f'candidato {cand.id} en ({cand.x:.2f}, {cand.y:.2f}): '
                                       'borrado (de cerca no hay nada)')
                continue
        hubo_voto = False
        for cand in self.candidatos.lista:
            if cand.estado == 'victima' or not self.candidatos.listos(cand):
                continue
            visto, rumbo = en_vista((cand.x, cand.y), laser_en_mapa, rumbo_camara, medio_fov,
                                    rango_en_rumbo, self.candidatos.config)
            d = math.hypot(cand.x - laser_en_mapa[0], cand.y - laser_en_mapa[1])
            relativo = math.atan2(math.sin(rumbo - rumbo_camara), math.cos(rumbo - rumbo_camara))
            if not visto:
                if d <= cfg.distancia_max_m:
                    if d < cfg.distancia_min_m:
                        motivo = 'sin voto: muy cerca'
                        if abs(relativo) <= 0.8 * medio_fov:
                            self._marcar_rojo_cortado(cand, rumbo, d, ventanas_borde)
                    elif abs(relativo) > 0.8 * medio_fov:
                        motivo = 'sin voto: fuera de la imagen'
                    else:
                        motivo = f'sin voto: ocluido (LiDAR {rango_en_rumbo(rumbo)})'
                    self._log_voto(cand, motivo, d, relativo)
                continue
            en_direccion = any(abs(math.atan2(math.sin(rumbo - c), math.cos(rumbo - c)))
                               <= s + 0.05 for c, s in (ventanas if ventanas_crudas is None
                                                        else ventanas_crudas))
            radio = self.candidatos.config.radio_union_m
            en_lugar = any(math.hypot(rx - cand.x, ry - cand.y) < radio for rx, ry in rojos)
            # Rojo en su lugar: sí. Sin rojo en su dirección: no. Rojo en la dirección pero en
            # otro lugar (o sin ubicar): no vota.
            estado_previo = cand.estado
            concluyente = vista_concluyente((cand.x, cand.y), laser_en_mapa, rumbo_camara,
                                            medio_fov, self.candidatos.config,
                                            con_distancia=not self.no_rojo_por_pixeles)
            if en_lugar:
                voto = True
            elif (en_direccion or self.velocidad > 0.03 or abs(self.giro) > 0.1
                  or not concluyente):
                # "No es rojo" sólo con el robot posicionado: quieto, de frente y cerca.
                voto = None
            else:
                voto = False
            self.candidatos.inspeccionar(cand, voto)
            hubo_voto = hubo_voto or voto is not None
            if voto is not True:
                self._marcar_rojo_cortado(cand, rumbo, d, ventanas_borde)
            if voto is True:
                motivo = 'rojo en su lugar'
            elif voto is False:
                motivo = 'no rojo'
            else:
                motivo = 'sin voto: ' + ', '.join(
                    m for m, si in (('rojo en la dirección', en_direccion),
                                    (f'moviéndose ({self.velocidad:.2f} m/s)',
                                     self.velocidad > 0.03),
                                    (f'girando ({self.giro:.2f} rad/s)', abs(self.giro) > 0.1),
                                    ('no concluyente (lejos, cerca o descentrado)',
                                     not concluyente)) if si)
            self._log_voto(cand, motivo, d, relativo)
            if cand.estado != estado_previo:
                self.get_logger().info(f'candidato {cand.id} en ({cand.x:.2f}, {cand.y:.2f}): '
                                       f'{cand.estado}')
        return hubo_voto

    def _marcar_rojo_cortado(self, cand, rumbo, d, ventanas_borde):
        """Quieto y con rojo cortado por el borde en la dirección del candidato: una caja roja
        demasiado cerca para ubicarla. Se marca para que el explorador la mire de más lejos."""
        if (cand.rojo_cortado or self.velocidad > 0.03 or abs(self.giro) > 0.1
                or not any(abs(math.atan2(math.sin(rumbo - c), math.cos(rumbo - c))) <= s + 0.05
                           for c, s in ventanas_borde)):
            return
        cand.rojo_cortado = True
        self.get_logger().info(f'candidato {cand.id} en ({cand.x:.2f}, {cand.y:.2f}): '
                               f'rojo cortado por el borde a {d:.2f} m, alejarse')

    def _log_voto(self, cand, motivo, d, relativo):
        """Por qué votó (o no) un candidato, cuando cambia: para ver por qué no se descarta."""
        clave = (motivo.split(' (')[0], cand.rojo, cand.no_rojo, cand.estado)
        if self.ultimo_voto.get(cand.id) == clave:
            return
        self.ultimo_voto[cand.id] = clave
        self.get_logger().info(
            f'candidato {cand.id} en ({cand.x:.2f}, {cand.y:.2f}): {motivo} | rojo={cand.rojo} '
            f'no_rojo={cand.no_rojo} d={d:.2f} m desvío={relativo:+.2f} rad')

    def al_imagen(self, msg):
        if self.mapa is None or self.camara is None or self.scan is None:
            return
        t = segundos(msg.header.stamp)
        if self.t_ultima is not None and t - self.t_ultima < self.periodo_min:
            return
        self.t_ultima = t
        if abs(self.giro) > self.giro_max:
            return
        if abs(segundos(self.scan.header.stamp) - t) > self.desfase_max:
            return
        scan = self.scan
        if self.r_laser_optico is None:
            tf = self.transformada(scan.header.frame_id, msg.header.frame_id)
            if tf is None:
                return
            q = tf.transform.rotation
            self.r_laser_optico = rotacion_de_cuaternion(q.x, q.y, q.z, q.w)
        tf_mapa = self.transformada(self.marco_mapa, scan.header.frame_id)
        if tf_mapa is None:
            return
        tr = tf_mapa.transform.translation
        laser_en_mapa = (tr.x, tr.y, yaw_de(tf_mapa.transform.rotation))

        rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        fx, cx, fy = self.camara.k[0], self.camara.k[2], self.camara.k[4]
        nuevos = 0
        todos = segmentar_rojo(rgb, self.cfg_color, descartar_borde=False)
        al_borde = [toca_borde(b.u_min, b.u_max, rgb.shape[1], self.cfg_color) for b in todos]
        blobs = [b for b, borde in zip(todos, al_borde) if not borde]
        ventanas = [ventana_angular(b.u_min, b.u_max, fx, cx, self.r_laser_optico, 0.0)
                    for b in blobs]
        # Cortados por el borde: no ubican la caja, pero dicen que en esa dirección hay rojo.
        ventanas_borde = [ventana_angular(b.u_min, b.u_max, fx, cx, self.r_laser_optico, 0.0)
                          for b, borde in zip(todos, al_borde) if borde]
        rojos, rojos_lejanos = [], []
        alcance_lejano = max(self.alcance, self.candidatos.config.alcance_posible_m)
        for blob in blobs:
            centro, semiancho = ventana_angular(blob.u_min, blob.u_max, fx, cx,
                                                self.r_laser_optico, self.recorte)
            puntos = puntos_en_ventana(scan.ranges, scan.angle_min, scan.angle_increment,
                                       scan.range_min, scan.range_max, centro, semiancho,
                                       alcance_max=alcance_lejano, salto=self.salto,
                                       min_puntos=self.min_puntos,
                                       fraccion_cerca_min=self.fraccion_rojo_sobre_objeto)
            if puntos is None:
                continue
            cara = float(np.mean(np.hypot(puntos[:, 0], puntos[:, 1])))
            if not alto_coherente(blob.v_min, blob.v_max, rgb.shape[0], fy, cara,
                                  self.alto_caja, self.alto_rojo_min_frac):
                self.get_logger().info(
                    f'rojo detrás: {blob.v_max - blob.v_min + 1} px de alto, una caja a '
                    f'{cara:.2f} m mediría {fy * self.alto_caja / cara:.0f}; no se ubica',
                    throttle_duration_sec=2.0)
                continue
            en_mapa = transformar(puntos, *laser_en_mapa)
            if self.en_pared(en_mapa) > self.fraccion_pared_max:
                continue
            x, y = estimar_centro(en_mapa, laser_en_mapa[:2], self.profundidad)
            rojos_lejanos.append((x, y))
            # Más lejos que `alcance_deteccion_m` la posición es mala (~0.4 m): no va al
            # tracker (que da la posición del reporte) ni vota; sólo confirma candidatos.
            if math.hypot(x - laser_en_mapa[0], y - laser_en_mapa[1]) > self.alcance:
                continue
            rojos.append((x, y))
            obj = self.tracker.agregar(x, y, desde=laser_en_mapa[:2])
            if obj is not None:
                nuevos += 1
                if obj.n == 1 or (obj.confirmado and obj.n == self.tracker.config.observaciones_min):
                    self.get_logger().info(
                        f'objetivo {obj.id}: ({obj.x:.2f}, {obj.y:.2f}) n={obj.n} '
                        f'{"CONFIRMADO" if obj.confirmado else ""}')
        ventanas_crudas = ventanas + ventanas_borde if self.no_rojo_por_pixeles else None
        voto = self.inspeccionar(scan, laser_en_mapa, ventanas, rojos, rojos_lejanos, fx, cx,
                                 rgb.shape[1], ventanas_crudas=ventanas_crudas,
                                 ventanas_borde=ventanas_borde)
        if nuevos or (voto and self.seguir_dudosos):
            self.publicar()

    def publicar(self):
        candidatos = self.candidatos.resumen()
        resumen = confirmar_con_candidatos(self.tracker.resumen(), candidatos,
                                           self.candidatos.config.radio_union_m,
                                           sumar_candidatos=self.sumar_candidatos,
                                           sumar_dudosos=self.seguir_dudosos)
        self.pub.publish(String(data=json.dumps(resumen)))
        self.pub_candidatos.publish(String(data=json.dumps(candidatos)))
        # Candidatos en RViz: amarillo posible, gris sin inspeccionar (naranja si la cámara vio
        # rojo de lejos), azul descartado, rojo víctima.
        colores = {'posible': (0.9, 0.9, 0.3), 'sin_inspeccionar': (0.6, 0.6, 0.6),
                   'descartado': (0.1, 0.3, 1.0), 'victima': (1.0, 0.0, 0.0)}
        # Primero se borran todas: un candidato fantasma borrado no tiene que quedar en RViz.
        marcas_c = MarkerArray(markers=[Marker(action=Marker.DELETEALL)])
        for cand in candidatos:
            m = Marker()
            m.header.frame_id = self.marco_mapa
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns, m.id, m.type, m.action = 'candidatos', cand['id'], Marker.CUBE, Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = cand['x'], cand['y'], 0.35
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = 0.18
            m.scale.z = 0.05
            m.color.r, m.color.g, m.color.b = colores.get(cand['estado'], (1.0, 1.0, 1.0))
            if cand['estado'] == 'sin_inspeccionar' and cand['rojo_lejano']:
                m.color.r, m.color.g, m.color.b = 1.0, 0.5, 0.0
            m.color.a = 0.9
            marcas_c.markers.append(m)
            t = Marker()
            t.header = m.header
            t.ns, t.id, t.type, t.action = 'candidatos_texto', cand['id'], Marker.TEXT_VIEW_FACING, \
                Marker.ADD
            t.pose.position.x, t.pose.position.y, t.pose.position.z = cand['x'], cand['y'], 0.5
            t.pose.orientation.w = 1.0
            t.scale.z = 0.12
            t.color.r = t.color.g = t.color.b = t.color.a = 1.0
            t.text = f"{cand['id']} r{cand['rojo']}/n{cand['no_rojo']}"
            marcas_c.markers.append(t)
        self.pub_marcas_c.publish(marcas_c)
        marcas = MarkerArray()
        for o in resumen:
            m = Marker()
            m.header.frame_id = self.marco_mapa
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns, m.id, m.type, m.action = 'objetivos', o['id'], Marker.CYLINDER, Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = o['x'], o['y'], 0.15
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = 0.25
            m.scale.z = 0.3
            m.color.r, m.color.g, m.color.b = (1.0, 0.0, 0.0) if o['confirmado'] else (1.0, 0.6, 0.0)
            m.color.a = 0.8
            marcas.markers.append(m)
        self.pub_marcas.publish(marcas)


def main(args=None):
    rclpy.init(args=args)
    nodo = NodoPercepcion()
    try:
        rclpy.spin(nodo)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        nodo.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
