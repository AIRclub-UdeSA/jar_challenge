"""Adaptador ROS de M3. Sin mando de movimiento ni acceso a verdad del simulador."""
from dataclasses import fields
from collections import OrderedDict
import json
import math
import time

import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from nav_msgs.msg import OccupancyGrid
from rclpy.executors import ExternalShutdownException
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Empty, String
from tf2_ros import Buffer, TransformException, TransformListener

from equipo_jar.navegacion.grilla import grilla_desde_occupancy
from equipo_jar.percepcion import (ParametrosPercepcion, RegistroVictimas, detectar_cajas,
                                  hay_region_roja, leer_imagen, leer_nube, mascara_roja)
from equipo_jar.reportes import armar_reporte
from equipo_jar.vistas_percepcion import espacio_libre, proponer_vistas

QOS_ESTADO = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                        durability=DurabilityPolicy.TRANSIENT_LOCAL)


class NodoPercepcion(Node):
    def __init__(self, **kwargs):
        super().__init__('percepcion', **kwargs)
        self.declare_parameter('cantidad_victimas', 5)
        self.declare_parameter('frecuencia_hz', 2.)
        self.declare_parameter('frescura_s', .8)
        self.declare_parameter('rafaga_s', 2.)
        self.declare_parameter('habilitada', True)
        self.declare_parameter('solo_observar', False)
        self.declare_parameter('asumir_base_cuadrada', False)
        self.declare_parameter('vista_separacion_m', .9)
        self.declare_parameter('vista_despeje_m', .30)
        self.declare_parameter('plazo_respaldo_s',270.)
        defaults = ParametrosPercepcion()
        for f in fields(defaults):
            self.declare_parameter(f.name, getattr(defaults, f.name))
        self.p = ParametrosPercepcion(**{f.name: self.valor(f.name) for f in fields(defaults)})
        for n in ('frecuencia_hz', 'frescura_s', 'rafaga_s', 'vista_separacion_m', 'vista_despeje_m','plazo_respaldo_s'):
            if not math.isfinite(self.valor(n)) or self.valor(n) <= 0:
                raise ValueError('Tiempo/frecuencia de percepción inválido')
        self.registro = RegistroVictimas(self.valor('cantidad_victimas'), self.p)
        self.mapa = None
        self.terminada = False
        self.inicio_mapa_s = None
        self.regresando = False
        self.localizacion = {}
        self.t_localizacion = -math.inf
        self.ultima_imagen = -math.inf
        self.ultima_nube = -math.inf
        self.sub_nube = None
        self.nube_pendiente = None
        self.observacion_pendiente = None
        self.fin_rafaga = -math.inf
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(Empty, '/done', self.al_done, 10)
        self.create_subscription(String,'/equipo/mision_estado',self.al_mision,QOS_ESTADO)
        self.pub_reportes = self.create_publisher(PointStamped, '/victimas', 10)
        self.pub_estado = self.create_publisher(String, '/equipo/percepcion_estado', QOS_ESTADO)
        self.pub_observacion = self.create_publisher(String, '/equipo/observacion_visual', 10)
        self.info_camara = None
        self.historial_rgb = OrderedDict()
        self.create_subscription(CameraInfo, '/cam_1/color/camera_info',
                                 self.al_info_camara, qos_profile_sensor_data)
        self.create_subscription(OccupancyGrid, '/map', self.al_mapa, QOS_ESTADO)
        self.create_subscription(String, '/equipo/localizacion_estado', self.al_localizacion,
                                 QOS_ESTADO)
        self.create_subscription(Image, '/cam_1/color/image_raw', self.al_imagen,
                                 qos_profile_sensor_data)
        self.create_timer(.2, self.cerrar_rafaga,
                          clock=Clock(clock_type=ClockType.STEADY_TIME))
        self.estado('ESPERANDO_MAPA')

    def valor(self, name):
        return self.get_parameter(name).value

    def estado(self, reason, **details):
        self.pub_estado.publish(String(data=json.dumps(dict(
            motivo=reason, reportes=self.registro.reportes,
            candidatos=len(self.registro.victimas),
            pendientes_reportes=len(self.registro.pendientes()),
            reportes_respaldo=sum(v.reportada_por_respaldo for v in self.registro.victimas),
            reportadas=[dict(x=float(v.centro[0]), y=float(v.centro[1]),
                            error_estimado_m=v.error_m,respaldo=v.reportada_por_respaldo)
                        for v in self.registro.victimas if v.reportada], **details))))

    def al_mapa(self, msg):
        if msg.header.frame_id != 'map':
            self.estado('MARCO_MAPA_INVALIDO')
            return
        q = msg.info.origin.orientation
        angle = math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
        self.mapa = grilla_desde_occupancy(
            msg.data, msg.info.width, msg.info.height, msg.info.resolution,
            msg.info.origin.position.x, msg.info.origin.position.y, angle)
        self.espacio_vistas = espacio_libre(self.mapa)
        if self.inicio_mapa_s is None:
            self.inicio_mapa_s=self.get_clock().now().nanoseconds/1e9

    def publicar_reportes(self, reports):
        for victim in reports:
            self.pub_reportes.publish(armar_reporte(
                *victim.centro, self.get_clock().now().to_msg()))
            self.get_logger().info(f'Víctima {victim.identificador} reportada en map; '
                f'estimado {victim.error_m:.3f} m, respaldo={victim.reportada_por_respaldo}; '
                f'intentos {self.registro.reportes}/{self.registro.cantidad}')

    def reportar_respaldo(self, motivo):
        if (self.terminada or self.mapa is None or not self.valor('habilitada')
                or self.valor('solo_observar')):
            return
        reports=self.registro.reportar_pendientes()
        self.publicar_reportes(reports)
        self.estado('RESPALDO_PUBLICADO',causa_respaldo=motivo,
                    publicados=[v.identificador for v in reports])

    def al_mision(self,msg):
        try:datos=json.loads(msg.data)
        except (ValueError,TypeError):return
        if isinstance(datos,dict) and datos.get('estado') in ('REGRESANDO','FINALIZADA'):
            self.regresando=True
            if self.registro.pendientes():self.reportar_respaldo('REGRESO')

    def al_localizacion(self, msg):
        try:
            self.localizacion = json.loads(msg.data)
            self.t_localizacion = time.monotonic()
        except (ValueError, TypeError):
            self.localizacion = {}

    def al_done(self, mensaje):
        self.terminada = True
        self.historial_rgb.clear()
        self.nube_pendiente = None
        self.observacion_pendiente = None
        self.fin_rafaga = -math.inf

    def disponible(self):
        return (not self.terminada and self.valor('habilitada') and self.mapa is not None
                and self.registro.reportes < self.registro.cantidad
                and self.localizacion.get('utilizable') is True
                and time.monotonic()-self.t_localizacion <= self.valor('frescura_s'))

    def fresca(self, stamp):
        delta = (self.get_clock().now()-Time.from_msg(stamp)).nanoseconds/1e9
        return -.05 <= delta <= self.valor('frescura_s')

    def al_info_camara(self, msg):
        if msg.width > 0 and math.isfinite(msg.k[0]) and msg.k[0] > 0:
            self.info_camara = msg

    def publicar_observacion(self, header, roja):
        """Evidencia RGB para M4: TF exacto y óptica calibrada, sin movimiento."""
        if not self.disponible() or not self.fresca(header.stamp):
            return
        info = self.info_camara
        if info is None or info.header.frame_id != header.frame_id:
            return
        try:
            tf = self.tf_buffer.lookup_transform('map', header.frame_id,
                                                  Time.from_msg(header.stamp))
        except TransformException:
            # Conservar sólo metadatos; TF exacto puede llegar después del RGB.
            self.observacion_pendiente = (header, roja)
            return
        q = tf.transform.rotation
        norm = math.sqrt(q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w)
        if not math.isfinite(norm) or norm < 1e-9:
            return
        x, y, z, w = q.x/norm, q.y/norm, q.z/norm, q.w/norm
        # Eje +Z óptico: delante de la cámara; no es el yaw del frame óptico.
        forward_x, forward_y = 2*(x*z+y*w), 2*(y*z-x*w)
        if math.hypot(forward_x, forward_y) < .5:
            return
        t = tf.transform.translation
        pose = [t.x, t.y, math.atan2(forward_y, forward_x)]
        if not all(math.isfinite(v) for v in pose):
            return
        self.observacion_pendiente = None
        regions=[]
        cached=self.historial_rgb.get((header.stamp.sec,header.stamp.nanosec))
        if roja and cached is not None:
            from equipo_jar.percepcion_cara import regiones_desde_mascara
            for region in regiones_desde_mascara(mascara_roja(leer_imagen(cached),self.p),self.p.pixeles_min):
                regions.append(dict(desvio_rad=-math.atan((region['left']+(region['width']-1)/2-info.k[2])/info.k[0]),
                    area_relativa=region['width']*region['height']/(cached.width*cached.height),
                    calidad_visual=region['width']/max(1,region['height']),
                    semiancho_rad=math.atan(region['width']/(2*info.k[0]))))
        self.pub_observacion.publish(String(data=json.dumps(dict(
            estampa_s=header.stamp.sec+header.stamp.nanosec/1e9,
            frame_id=header.frame_id, pose_camara=pose, campo_horizontal_rad=2*math.atan(info.width/(2*info.k[0])),
            region_roja=bool(roja),regiones_rojas=regions))))

    def al_imagen(self, msg):
        if not self.terminada:
            key=(msg.header.stamp.sec,msg.header.stamp.nanosec)
            self.historial_rgb[key]=msg
            while len(self.historial_rgb)>30:self.historial_rgb.popitem(last=False)
        now = time.monotonic()
        if now-self.ultima_imagen < 1/self.valor('frecuencia_hz') or not self.disponible():
            return
        self.ultima_imagen = now
        if not self.fresca(msg.header.stamp):
            self.estado('IMAGEN_VENCIDA')
            return
        try:
            candidate = hay_region_roja(leer_imagen(msg), self.p)
        except ValueError as error:
            self.estado('IMAGEN_INVALIDA', detalle=str(error))
            return
        self.publicar_observacion(msg.header, candidate)
        if candidate and self.sub_nube is None:
            # Suscripción sólo por ráfagas: la nube completa puede elevar latencia.
            self.sub_nube = self.create_subscription(
                PointCloud2, '/cam_1/depth/color/points', self.al_nube, qos_profile_sensor_data)
            self.fin_rafaga = now+self.valor('rafaga_s')
            self.estado('ESPERANDO_RGBD')
        elif not candidate:
            self.estado('SIN_REGION_ROJA')

    def cerrar_rafaga(self):
        if self.inicio_mapa_s is not None and self.registro.pendientes():
            elapsed=self.get_clock().now().nanoseconds/1e9-self.inicio_mapa_s
            if self.regresando or elapsed>=self.valor('plazo_respaldo_s'):
                self.reportar_respaldo('REGRESO' if self.regresando else 'PLAZO_BONO')
        visual, self.observacion_pendiente = self.observacion_pendiente, None
        if visual is not None:
            self.publicar_observacion(*visual)
        if self.sub_nube is not None and (
                time.monotonic() >= self.fin_rafaga or not self.disponible()):
            self.destroy_subscription(self.sub_nube)
            self.sub_nube = None
        pending, self.nube_pendiente = self.nube_pendiente, None
        if pending is not None:
            self.al_nube(pending)

    def contorno_rgb(self, stamp):
        msg=self.historial_rgb.get((stamp.sec,stamp.nanosec))
        info=self.info_camara
        if msg is None or info is None or info.header.frame_id!=msg.header.frame_id:return None
        if info.width!=msg.width or info.height!=msg.height:return None
        from equipo_jar.percepcion_cara import regiones_desde_mascara
        try:
            rgb=leer_imagen(msg)
            regions=regiones_desde_mascara(mascara_roja(rgb,self.p),self.p.pixeles_min)
            tf=self.tf_buffer.lookup_transform('map',msg.header.frame_id,Time.from_msg(stamp)).transform
        except (ValueError,TransformException):return None
        t,q=tf.translation,tf.rotation
        return dict(k=list(info.k),translation=[t.x,t.y,t.z],quaternion=[q.x,q.y,q.z,q.w],
                    regions=regions,width=msg.width,height=msg.height)

    def al_nube(self, msg):
        now = time.monotonic()
        if not self.disponible() or now-self.ultima_nube < 1/self.valor('frecuencia_hz'):
            return
        # La nube tiene RGB propio registrado: no se empareja con otra imagen
        # por índice de píxel, porque puede estar desorganizada o diezmada.
        if not self.fresca(msg.header.stamp):
            self.estado('NUBE_VENCIDA')
            return
        try:
            tf = self.tf_buffer.lookup_transform('map', msg.header.frame_id,
                                                  Time.from_msg(msg.header.stamp))
            t, q = tf.transform.translation, tf.transform.rotation
        except TransformException as error:
            # TF puede llegar unos milisegundos después de la nube. Reintentar
            # sin bloquear el ejecutor; la misma regla de frescura limita espera.
            self.nube_pendiente = msg
            self.estado('ESPERANDO_TF_RGBD', detalle=str(error))
            return
        self.nube_pendiente = None
        self.ultima_nube = now
        try:
            xyz, rgb = leer_nube(msg)
            diagnostic = {}
            detections = detectar_cajas(
                xyz, rgb, (t.x, t.y, t.z), (q.x, q.y, q.z, q.w),
                self.localizacion['std_xy_m'], self.localizacion['std_yaw_rad'],
                self.p, self.mapa, diagnostic,
                asumir_base_cuadrada=self.valor('asumir_base_cuadrada'),
                observacion_rgb=self.contorno_rgb(msg.header.stamp))
        except (ValueError, TypeError, KeyError) as error:
            self.estado('RGBD_NO_UTILIZABLE', detalle=str(error))
            return
        stamp = msg.header.stamp.sec+msg.header.stamp.nanosec/1e9
        reports = [] if self.valor('solo_observar') else self.registro.observar(detections, stamp)
        self.publicar_reportes(reports)
        self.estado('RGBD_PROCESADA', detecciones=[dict(x=d.x, y=d.y,
            error_estimado_m=d.error_estimado_m, lados_m=d.lados_m, puntos=d.puntos,
            superficie=d.superficie,calidad_preferida=d.error_estimado_m<=self.p.error_objetivo_m)
            for d in detections], publicados=[v.identificador for v in reports],
            necesita_otra_vista=bool(diagnostic['vistas_incompletas']),
            vistas_sugeridas=proponer_vistas(diagnostic['vistas_incompletas'],
                (t.x, t.y), self.mapa, self.espacio_vistas,
                self.valor('vista_separacion_m'), self.valor('vista_despeje_m')),
            diagnostico=diagnostic,
            procesamiento_ms=(time.monotonic()-now)*1000)
        # Una muestra por ráfaga; la próxima imagen decidirá si pedir otra.
        self.fin_rafaga = now


def main(args=None):
    rclpy.init(args=args)
    node = NodoPercepcion()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        import signal
        # El primer SIGINT ya detuvo spin; no interrumpir otra vez la limpieza.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            node.destroy_node()
        except (KeyboardInterrupt, ExternalShutdownException):
            pass
        try:
            rclpy.try_shutdown()
        except (KeyboardInterrupt, ExternalShutdownException):
            pass
        finally:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
