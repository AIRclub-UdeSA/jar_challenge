"""Coordina la localización inicial y conserva las interfaces oficiales."""
from collections import deque
from dataclasses import fields
import json
import math
import time

import numpy as np

import rclpy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.clock import Clock, ClockType
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Empty, String
from std_srvs.srv import Empty as SolicitudSinMovimiento
from tf2_ros import Buffer, TransformException, TransformListener

from equipo_jar.coordinador_mision import CoordinadorMision
from equipo_jar.mision import ParametrosMision
from equipo_jar.localizacion_activa import (LocalizacionActiva, Observacion, concordancia_mapa,
                                            concordancia_mapa_simple,EvaluadorMapa)
from equipo_jar.navegacion.costos import ParametrosCosto
from equipo_jar.navegacion.grilla import grilla_desde_occupancy, refinar_para_obstaculos
from equipo_jar.navegacion.guardia import (huella_desde_parametros,puntos_desde_scan,STOPPING_RESERVE_M)
from equipo_jar.navegacion.ida_vuelta import EntradaNavegacion, IdaVuelta, ParametrosIdaVuelta
from equipo_jar.navegacion.metas import elegir_meta_prueba
from equipo_jar.navegacion.seguimiento import ParametrosSeguimiento
from equipo_jar.navegacion.guardia import huella_convexa_x3
from equipo_jar.navegacion.tipos import Pose2D
from equipo_jar.seguridad_giro import comprobar_giro

QOS_MAPA = QoSProfile(
    depth=1, reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL)


# Solo residuos numéricos de varianzas teóricamente nulas (m² o rad²).
# No modifica los umbrales de calidad ni el mensaje publicado por AMCL.
TOLERANCIA_REDONDEO_COVARIANZA = 1e-12
MODOS_NAVEGACION = ('inactivo', 'ida_vuelta_prueba', 'meta_externa', 'mision')


def yaw(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y ** 2 + q.z ** 2))


class NodoEquipo(Node):
    def __init__(self, **kwargs):
        super().__init__('nodo_equipo', **kwargs)
        valores = {
            'frecuencia_control_hz': 10.0, 'pausar_control_sim': False,
            'scan_topic': '/scan', 'frescura_pose_s': 3.0,
            'plazo_recuperacion_s': 15.0, 'max_recuperaciones': 2,
            'recuperacion_activa_post_lista': False,
            'espera_recuperacion_activa_s': 3.0, 'giro_recuperacion_rad': math.radians(35),
            'max_intentos_recuperacion_activa': 3,
            'recuperacion_std_xy_m': .30, 'recuperacion_std_yaw_rad': .35,
            'velocidad_lineal_max': 0.30, 'velocidad_angular_max': 1.0,
            'cantidad_victimas': 5, 'localizacion_activa': True,
            'giro_velocidad_rad_s': 0.15, 'giro_objetivo_rad': math.radians(20),
            'max_giros': 3, 'presupuesto_localizacion_s': 35.0,
            'plazo_giro_s': 12.0, 'plazo_verificacion_s': 5.0,
            'huella_x_min_m': -0.19, 'huella_x_max_m': 0.13,
            'huella_semi_ancho_m': 0.12, 'margen_giro_m': 0.10,
            'reserva_detencion_giro_rad': 0.12, 'max_std_xy_m': 0.15,
            'max_std_yaw_rad': 0.15, 'concordancia_min': 0.80,
            'concordancia_min_supervision': 0.70,
            'estabilidad_xy_m': 0.03, 'estabilidad_yaw_rad': 0.035,
            'frescura_sensores_s': 0.6, 'tolerancia_mapa_m': 0.10,
            'min_rayos_validos': 30, 'muestra_rayos': 120,
            'deficit_obstaculo_m': 0.40, 'plazo_falta_datos_s': 2.0,
            'plazo_falla_calidad_s': 1.0,
            # M2: navegación hacia una meta y regreso (valores generales, sin coordenadas).
            'modo_navegacion': 'inactivo', 'prueba_en_seco': False,
            'nav_alcance_blando_m': 0.25, 'nav_peso_costo': 5.0,
            'nav_velocidad_max': 0.30, 'nav_aceleracion_max': 0.4,
            'nav_distancia_mira_m': 0.20, 'nav_tolerancia_meta_m': 0.03,
            'nav_ganancia_yaw': 1.5, 'nav_velocidad_angular_max': 0.5,
            'nav_espacio_lento_m': 0.25, 'nav_espacio_rapido_m': 0.45,
            'nav_velocidad_cerca_pared': 0.10, 'nav_dt_max_s': 0.2,
            'nav_plazo_detenido_s': 20.0, 'nav_espera_bloqueo_s': 1.0,
            'nav_max_replanificaciones': 5, 'nav_plazo_bloqueo_total_s': 30.0,
            'nav_pausa_en_meta_s': 0.5, 'nav_tolerancia_largada_m': 0.05,
            'nav_desaceleracion': 0.5, 'nav_latencia_s': 0.15,
            'nav_desaceleracion_angular': 0.5,
            'nav_vida_obstaculo_s': 600.0,
            'nav_margen_m': .04, 'nav_huella_convexa': True, 'nav_chasis_semi_ancho_m': .070,
            'nav_rueda_x_m': .080, 'nav_rueda_y_m': .0845, 'nav_rueda_radio_m': .033,
            'nav_rueda_semi_ancho_visual_m': .018,
            'meta_distancia_min_m': 1.5, 'meta_distancia_max_m': 3.0,
            'meta_espacio_min_m': 0.35,
        }
        for campo in fields(ParametrosMision):
            valores['mision_'+campo.name] = getattr(ParametrosMision(), campo.name)
        for nombre, valor in valores.items():
            self.declare_parameter(nombre, valor)
        self.cantidad_victimas = self.get_parameter('cantidad_victimas').value
        self.control = LocalizacionActiva(
            velocidad=self.valor('giro_velocidad_rad_s'),
            giro=self.valor('giro_objetivo_rad'), max_giros=self.valor('max_giros'),
            presupuesto=self.valor('presupuesto_localizacion_s'),
            plazo_giro=self.valor('plazo_giro_s'),
            plazo_verificacion=self.valor('plazo_verificacion_s'),
            max_std_xy=self.valor('max_std_xy_m'),
            max_std_yaw=self.valor('max_std_yaw_rad'),
            concordancia_min=self.valor('concordancia_min'),
            concordancia_min_supervision=self.valor('concordancia_min_supervision'),
            estabilidad_xy=self.valor('estabilidad_xy_m'),
            estabilidad_yaw=self.valor('estabilidad_yaw_rad'),
            plazo_recuperacion=self.valor('plazo_recuperacion_s'),
            max_recuperaciones=self.valor('max_recuperaciones'),
            plazo_falta_datos=self.valor('plazo_falta_datos_s'),
            plazo_falla_calidad=self.valor('plazo_falla_calidad_s'),
            recuperacion_activa=self.valor('recuperacion_activa_post_lista'),
            espera_recuperacion_activa=self.valor('espera_recuperacion_activa_s'),
            giro_recuperacion=self.valor('giro_recuperacion_rad'),
            max_intentos_activos=self.valor('max_intentos_recuperacion_activa'))
        for nombre in ('recuperacion_std_xy_m', 'recuperacion_std_yaw_rad'):
            if not math.isfinite(self.valor(nombre)) or self.valor(nombre) <= 0:
                raise ValueError('Covarianza de reinicialización local inválida')
        self.reinicios_publicados = 0
        self.pub_reinicio_local = self.create_publisher(
            PoseWithCovarianceStamped, '/equipo/initialpose', 10)
        if (not math.isfinite(self.valor('reserva_detencion_giro_rad'))
                or self.valor('reserva_detencion_giro_rad') <= 0):
            raise ValueError('Reserva angular de detención inválida')
        # Validar antes del mapa: una configuración geométrica inválida no mueve.
        comprobar_giro([],self.valor('huella_x_min_m'),self.valor('huella_x_max_m'),
                       self.valor('huella_semi_ancho_m'),
                       self.valor('giro_objetivo_rad')+self.valor('reserva_detencion_giro_rad'),
                       self.valor('margen_giro_m'))
        if self.valor('modo_navegacion') not in MODOS_NAVEGACION:
            raise ValueError('modo_navegacion desconocido: ' + str(self.valor('modo_navegacion')))
        self.huella = huella_desde_parametros(
            self.valor('huella_x_min_m'), self.valor('huella_x_max_m'),
            self.valor('huella_semi_ancho_m'), self.valor('nav_margen_m'))
        if self.valor('nav_huella_convexa'):
            self.huella=huella_convexa_x3(
                self.valor('huella_x_min_m'),self.valor('huella_x_max_m'),
                self.valor('nav_chasis_semi_ancho_m'),self.valor('nav_rueda_x_m'),
                self.valor('nav_rueda_y_m'),self.valor('nav_rueda_radio_m'),
                self.valor('nav_margen_m'),self.valor('nav_rueda_semi_ancho_visual_m'))
        self.navegacion = self.mapa_navegacion = self.meta_pendiente = None
        self.mision = None
        self.t_mapa_reloj = None
        self.t_fin_quieto = None
        self.puntos_base = self.puntos_mapa = self.ruta_publicada = None
        self.motivo_meta = ''
        self.duraciones = deque(maxlen=100)
        self.numero_ciclo=0
        self.mapa = self.scan = self.odom = self.pose = None
        self.historial_scan = deque(maxlen=8)
        self.secuencia_pose = 0
        self.diagnostico_observacion = {'codigo': 'SIN_EVALUAR'}
        self.fallo_inicio = None
        self.ultimo_estado = None
        self.ultima_peticion_scan = 0.0
        self.peticion_scan = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.sub_mapa = self.create_subscription(
            OccupancyGrid, '/map', self.al_recibir_mapa, QOS_MAPA)
        self.create_subscription(LaserScan, self.valor('scan_topic'), self.al_scan, qos_profile_sensor_data)
        self.create_subscription(Odometry, '/odom', self.al_odom, 10)
        self.create_subscription(
            PoseWithCovarianceStamped, '/equipo/amcl_pose', self.al_pose, QOS_MAPA)
        self.create_subscription(
            String, '/equipo/inicializacion_estado', self.al_inicio, QOS_MAPA)
        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_done = self.create_publisher(Empty, '/done', 10)
        self.pub_estado = self.create_publisher(String, '/equipo/localizacion_estado', QOS_MAPA)
        self.pub_navegacion = self.create_publisher(
            String, '/equipo/navegacion_estado', QOS_MAPA)
        self.pub_mision = self.create_publisher(String, '/equipo/mision_estado', QOS_MAPA)
        if self.valor('modo_navegacion') == 'mision':
            self.create_subscription(String, '/equipo/observacion_visual', self.al_visual_mision, 10)
            self.create_subscription(String, '/equipo/percepcion_estado', self.al_percepcion_mision, QOS_MAPA)
        self.pub_ruta = self.create_publisher(Path, '/equipo/ruta', QOS_MAPA)
        self.pub_propuesto = self.create_publisher(Twist, '/equipo/cmd_propuesto', 10)
        if self.valor('modo_navegacion') == 'meta_externa':
            self.create_subscription(PoseStamped, '/equipo/meta', self.al_meta, 10)
        self.nomotion = self.create_client(
            SolicitudSinMovimiento, '/equipo/request_nomotion_update')
        self.create_timer(
            1.0 / self.valor('frecuencia_control_hz'), self.ciclo_de_control,
            clock=Clock(clock_type=ClockType.STEADY_TIME))
        self.get_logger().info(
            f'Hay que encontrar {self.cantidad_victimas} víctimas. Esperando el mapa en /map...')

    def valor(self, nombre):
        return self.get_parameter(nombre).value

    def reloj_mision(self):
        # Simulación sigue /clock; robot real usa tiempo transcurrido monotónico.
        return (self.get_clock().now().nanoseconds/1e9 if self.valor('use_sim_time')
                else time.monotonic())

    def al_recibir_mapa(self, mapa):
        if self.mapa_navegacion is None:
            self.mapa_navegacion = mapa  # la navegación usa solo el primer /map
            self.t_mapa_reloj = self.reloj_mision()
        if self.mapa is None:
            self.get_logger().info(
                f'Mapa recibido: {mapa.info.width}x{mapa.info.height} celdas, '
                f'resolución {mapa.info.resolution:.3f} m')
        self.mapa = mapa
        self.evaluador_calidad=EvaluadorMapa(mapa)

    def al_scan(self, mensaje):
        if self.scan is not None:
            nueva = (mensaje.header.stamp.sec, mensaje.header.stamp.nanosec)
            anterior = (self.scan.header.stamp.sec, self.scan.header.stamp.nanosec)
            if nueva <= anterior:
                return
        self.scan = mensaje
        self.t_scan = time.monotonic()
        self.historial_scan.append((mensaje, self.t_scan))

    def al_odom(self, mensaje):
        self.odom = mensaje
        self.t_odom = time.monotonic()

    def al_pose(self, mensaje):
        if self.pose is not None:
            nueva = (mensaje.header.stamp.sec, mensaje.header.stamp.nanosec)
            anterior = (self.pose.header.stamp.sec, self.pose.header.stamp.nanosec)
            if nueva <= anterior:
                return
        self.pose = mensaje
        self.t_pose = time.monotonic()
        self.secuencia_pose += 1

    def al_inicio(self, mensaje):
        if mensaje.data.startswith('FALLO'):
            self.fallo_inicio = mensaje.data

    def observacion(self, ahora):
        self.puntos_base = self.puntos_mapa = None
        nombres = ('mapa', 'scan', 'odom', 'pose')
        faltantes = [nombre for nombre in nombres if getattr(self, nombre) is None]
        self.diagnostico_observacion = {'codigo': 'OK'}
        if faltantes:
            self.diagnostico_observacion = {'codigo': 'DATOS_AUSENTES', 'faltantes': faltantes}
            return None
        self.diagnostico_observacion.update(
            edad_scan_s=ahora-self.t_scan, edad_odom_s=ahora-self.t_odom,
            edad_pose_s=ahora-self.t_pose,
            scan_stamp_s=self.scan.header.stamp.sec+self.scan.header.stamp.nanosec/1e9,
            pose_stamp_s=self.pose.header.stamp.sec+self.pose.header.stamp.nanosec/1e9)
        if max(ahora - self.t_scan, ahora - self.t_odom) > self.valor('frescura_sensores_s'):
            self.diagnostico_observacion['codigo'] = 'SENSORES_VENCIDOS'
            return None
        if ahora - self.t_pose > self.valor('frescura_pose_s'):
            self.diagnostico_observacion['codigo'] = 'POSE_VENCIDA'
            return None
        pose = self.pose.pose.pose
        cov = self.pose.pose.covariance
        if self.pose.header.frame_id != 'map':
            self.diagnostico_observacion['codigo'] = 'MARCO_POSE_INVALIDO'
            return None
        numeros = (pose.position.x, pose.position.y, yaw(pose.orientation),
                   yaw(self.odom.pose.pose.orientation), cov[0], cov[7], cov[35])
        if (not all(math.isfinite(v) for v in numeros)
                or min(cov[0], cov[7], cov[35]) < -TOLERANCIA_REDONDEO_COVARIANZA):
            self.diagnostico_observacion.update(
                codigo='POSE_INVALIDA',
                valores=[v if math.isfinite(v) else repr(v) for v in numeros])
            return None
        varianzas = tuple(max(0.0, v) for v in (cov[0], cov[7], cov[35]))
        if min(cov[0], cov[7], cov[35]) < 0:
            self.diagnostico_observacion['redondeo_covarianza'] = {
                'original': [cov[0], cov[7], cov[35]],
                'normalizada': list(varianzas)}
        rangos_actuales = np.asarray(self.scan.ranges,dtype=float)
        indices_actuales = np.flatnonzero(np.isfinite(rangos_actuales)
            & (rangos_actuales>=self.scan.range_min) & (rangos_actuales<self.scan.range_max))
        actuales = rangos_actuales[indices_actuales]
        if len(actuales) < self.valor('min_rayos_validos'):
            self.diagnostico_observacion.update(codigo='SCAN_INSUFICIENTE', rayos_validos=len(actuales))
            return None
        # TF y sensores llegan por canales distintos. No extrapolar ni convertir
        # el recibo de un scan nuevo en frescura de uno retenido.
        scan = tf = None
        detalle = 'Ningún scan vigente tiene TF a su estampa'
        for candidato, recibo in reversed(self.historial_scan):
            if ahora - recibo > self.valor('frescura_sensores_s'):
                continue
            try:
                transformacion = self.tf_buffer.lookup_transform(
                    'map', candidato.header.frame_id, Time.from_msg(candidato.header.stamp))
            except TransformException as error:
                if scan is None:
                    detalle = str(error)
                continue
            scan, tf = candidato, transformacion
            self.diagnostico_observacion.update(
                scan_tf_stamp_s=scan.header.stamp.sec+scan.header.stamp.nanosec/1e9,
                edad_scan_tf_s=ahora-recibo,
                scan_tf_retenido=scan is not self.scan)
            break
        if scan is None:
            self.diagnostico_observacion.update(codigo='TF_NO_DISPONIBLE', detalle=detalle)
            return None
        if scan is self.scan:
            indices, rangos = indices_actuales, rangos_actuales
        else:
            rangos = np.asarray(scan.ranges,dtype=float)
            indices = np.flatnonzero(np.isfinite(rangos)
                & (rangos>=scan.range_min) & (rangos<scan.range_max))
        if len(indices) < self.valor('min_rayos_validos'):
            self.diagnostico_observacion.update(codigo='SCAN_INSUFICIENTE', rayos_validos=len(indices))
            return None
        elegidos = indices[::max(1,len(indices)//self.valor('muestra_rayos'))]
        muestra = list(zip(elegidos.tolist(),rangos[elegidos].tolist()))
        t = tf.transform
        laser_pose = (t.translation.x, t.translation.y, yaw(t.rotation))
        detalle_calidad = {}
        evaluator=getattr(self,'evaluador_calidad',None)
        if evaluator is None:
            coincidencia,cantidad=concordancia_mapa(
                self.mapa,muestra,scan.angle_min,scan.angle_increment,laser_pose,
                self.valor('tolerancia_mapa_m'),self.valor('deficit_obstaculo_m'),
                diagnostico=detalle_calidad)
        else:
            (coincidencia,cantidad),detalle_calidad=evaluator.calcular(
                muestra,scan.angle_min,scan.angle_increment,laser_pose,
                self.valor('tolerancia_mapa_m'),self.valor('deficit_obstaculo_m'))
            detalle_calidad.update(calidad_cache_aciertos=getattr(evaluator,'hits',0),
                                   calidad_cache_calculos=getattr(evaluator,'misses',0))
        # Criterio viejo solo para diagnóstico y comparación en los ensayos.
        if 'concordancia_simple' not in detalle_calidad:
            detalle_calidad['concordancia_simple'] = concordancia_mapa_simple(
                self.mapa, muestra, scan.angle_min, scan.angle_increment, laser_pose,
                self.valor('tolerancia_mapa_m'))[0]
        self.diagnostico_observacion.update(detalle_calidad)
        self.diagnostico_observacion['rayos_muestra'] = len(muestra)
        if cantidad < self.valor('min_rayos_validos'):
            self.diagnostico_observacion.update(codigo='SCAN_INSUFICIENTE', rayos_validos=cantidad)
            return None
        guardia = {'seguro':None,'motivo':'M2 comprueba su propio barrido de movimiento'}
        # Tras la entrega a M2, M1 solo supervisa calidad. No solicita giros.
        # La navegación transforma su scan y comprueba traslación y frenado angular.
        if self.navegacion is None or self.control.estado != 'LISTA':
            try:
                laser_base = self.tf_buffer.lookup_transform(
                    'base_footprint', self.scan.header.frame_id,
                    Time.from_msg(self.scan.header.stamp)).transform
            except TransformException as error:
                self.diagnostico_observacion.update(
                    codigo='TF_NO_DISPONIBLE',etapa='guardia_giro',detalle=str(error))
                return None
            puntos = puntos_desde_scan(
                self.scan.ranges, self.scan.angle_min, self.scan.angle_increment,
                self.scan.range_min, self.scan.range_max,
                (laser_base.translation.x, laser_base.translation.y, yaw(laser_base.rotation)))
            puntos_mapa = puntos_desde_scan(
                scan.ranges, scan.angle_min, scan.angle_increment, scan.range_min,
                scan.range_max, laser_pose)
            restante=self.control.giro
            if self.control.estado=='GIRANDO':
                delta=math.atan2(math.sin(numeros[3]-self.control.yaw_inicial),
                                 math.cos(numeros[3]-self.control.yaw_inicial))
                restante=max(0.,self.control.giro-delta)
            direccion = 1
            if (getattr(self.control, 'recuperacion_activa', False)
                    and self.control.secuencia_lista >= 0):
                restante, direccion = self.control.giro_de_guardia(numeros[3])
            puntos_guardia = puntos if direccion > 0 else [(x,-y) for x,y in puntos]
            guardia=comprobar_giro(
                puntos_guardia,self.valor('huella_x_min_m'),self.valor('huella_x_max_m'),
                self.valor('huella_semi_ancho_m'),
                restante+self.valor('reserva_detencion_giro_rad'),self.valor('margen_giro_m'))
            guardia['direccion'] = direccion
            if direccion < 0:
                for campo in ('bloqueantes_m','poligono_barrido_m'):
                    if campo in guardia:guardia[campo]=[(x,-y) for x,y in guardia[campo]]
                left,right,bottom,top=guardia['limites_m']
                guardia['limites_m']=(left,right,-top,-bottom)
            self.diagnostico_observacion['guardia_giro']=guardia
            self.puntos_base, self.puntos_mapa = puntos, puntos_mapa
        self.diagnostico_observacion['rayos_validos'] = cantidad
        return Observacion(
            numeros[3], numeros[:3], self.secuencia_pose,
            math.sqrt(max(varianzas[:2])), math.sqrt(varianzas[2]),
            coincidencia, float(actuales.min()), guardia['seguro'])

    def reinicializar_recuperacion(self, observacion):
        """La idea de reinicialización local amplia fue aportada por ramossofia."""
        numero = getattr(self.control, 'reinicios_locales', 0)
        if numero <= getattr(self, 'reinicios_publicados', 0) or observacion is None:
            return
        x,y,a = observacion.pose
        if not all(math.isfinite(v) for v in (x,y,a)):
            return
        mensaje = PoseWithCovarianceStamped()
        mensaje.header.frame_id = 'map'
        mensaje.header.stamp = self.get_clock().now().to_msg()
        mensaje.pose.pose.position.x = x
        mensaje.pose.pose.position.y = y
        mensaje.pose.pose.orientation.z = math.sin(a/2)
        mensaje.pose.pose.orientation.w = math.cos(a/2)
        mensaje.pose.covariance[0] = mensaje.pose.covariance[7] = self.valor('recuperacion_std_xy_m')**2
        mensaje.pose.covariance[35] = self.valor('recuperacion_std_yaw_rad')**2
        self.pub_reinicio_local.publish(mensaje)
        self.reinicios_publicados = numero
        self.get_logger().info(f'Recuperación activa: reinicio local AMCL {numero}')

    def solicitar_observacion_quieta(self, ahora):
        if self.peticion_scan is not None:
            if self.peticion_scan.done() or ahora - self.ultima_peticion_scan > 1.0:
                self.peticion_scan.cancel()
                self.peticion_scan = None
            else:
                return
        if ahora - self.ultima_peticion_scan >= 0.4 and self.nomotion.service_is_ready():
            self.peticion_scan = self.nomotion.call_async(SolicitudSinMovimiento.Request())
            self.ultima_peticion_scan = ahora

    def al_percepcion_mision(self, mensaje):
        if self.mision is None:
            return
        try:
            datos = json.loads(mensaje.data)
            if isinstance(datos, dict):
                self.mision.percepcion(datos)
        except (ValueError, TypeError):
            return

    def al_visual_mision(self, mensaje):
        if self.mision is None or not self.control.utilizable:
            return
        try:
            datos = json.loads(mensaje.data)
            edad = self.get_clock().now().nanoseconds/1e9-float(datos['estampa_s'])
            if not -.05 <= edad <= .8:
                return
            tf = self.tf_buffer.lookup_transform(
                'base_footprint', datos['frame_id'], Time()).transform
            q = tf.rotation
            norm = math.sqrt(q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w)
            if not math.isfinite(norm) or norm < 1e-9:
                return
            x, y, z, w = q.x/norm, q.y/norm, q.z/norm, q.w/norm
            giro = math.atan2(2*(y*z-x*w), 2*(x*z+y*w))
            self.mision.observar(datos, time.monotonic(),
                (tf.translation.x, tf.translation.y, giro),
                self.navegacion.capa.ocupadas(time.monotonic()))
        except (KeyError, ValueError, TypeError, TransformException):
            return

    def al_meta(self, mensaje):
        if mensaje.header.frame_id not in ('map', ''):
            self.get_logger().warning('Meta ignorada: marco distinto de map')
            return
        meta = (mensaje.pose.position.x, mensaje.pose.position.y)
        if not all(math.isfinite(v) for v in meta):
            self.get_logger().warning('Meta ignorada: coordenadas no finitas')
            return
        if self.navegacion is None:
            self.meta_pendiente = meta
        elif self.navegacion.estado in ('ESPERANDO_LISTA', 'ESPERANDO_META'):
            self.navegacion.fijar_meta(meta)

    def crear_navegacion(self):
        m = self.mapa_navegacion
        o = m.info.origin
        grilla = grilla_desde_occupancy(
            m.data, m.info.width, m.info.height, m.info.resolution,
            o.position.x, o.position.y, yaw(o.orientation))
        grilla = refinar_para_obstaculos(grilla)
        v = self.valor
        costos = ParametrosCosto(
            min(v('huella_x_max_m'), -v('huella_x_min_m'), v('huella_semi_ancho_m')),
            self.huella['radius_m'], self.huella['margin_m'], v('nav_alcance_blando_m'),
            v('nav_peso_costo'))
        seguimiento = ParametrosSeguimiento(
            v('nav_velocidad_max'), v('nav_aceleracion_max'), v('nav_distancia_mira_m'),
            max(v('nav_tolerancia_meta_m'), math.sqrt(2) * m.info.resolution),
            v('nav_ganancia_yaw'), v('nav_velocidad_angular_max'),
            v('nav_espacio_lento_m'), v('nav_espacio_rapido_m'),
            v('nav_velocidad_cerca_pared'), v('nav_dt_max_s'))
        ida_vuelta = ParametrosIdaVuelta(
            v('nav_plazo_detenido_s'), v('nav_espera_bloqueo_s'),
            v('nav_max_replanificaciones'), v('nav_plazo_bloqueo_total_s'),
            v('nav_pausa_en_meta_s'), v('nav_tolerancia_largada_m'), v('nav_desaceleracion'),
            v('nav_latencia_s'), v('nav_vida_obstaculo_s'), v('tolerancia_mapa_m'),
            v('nav_desaceleracion_angular'),
            regreso_automatico=v('modo_navegacion') != 'mision')
        self.navegacion = IdaVuelta(grilla, costos, seguimiento, self.huella, ida_vuelta)
        self.navegacion.habilitar_planificacion_asincrona()
        if v('modo_navegacion') == 'mision':
            parametros = ParametrosMision(**{f.name: v('mision_'+f.name)
                                           for f in fields(ParametrosMision)})
            self.mision = CoordinadorMision(self.navegacion, self.t_mapa_reloj, parametros)
        if self.meta_pendiente is not None:
            self.navegacion.fijar_meta(self.meta_pendiente)

    def paso_navegacion(self, ahora, observacion):
        """Comando de navegación con M1 en LISTA; None si no corresponde navegar."""
        modo = self.valor('modo_navegacion')
        if modo == 'inactivo' or self.mapa_navegacion is None:
            return None
        if self.navegacion is None:
            if self.control.estado != 'LISTA':
                return None
            self.crear_navegacion()
        nav = self.navegacion
        if self.control.estado!='LISTA' and nav.estado not in ('EN_LARGADA','FALLO'):
            nav.invalidar_plan_por_localizacion()
            if self.control.estado=='FALLO':nav.cancelar('Localización M1 en FALLO')
        datos = self.datos_navegacion(ahora, observacion)
        pose, puntos_base, puntos_mapa = datos if datos is not None else (None, None, None)
        lista = self.control.utilizable and observacion is not None
        laser_info=getattr(self,'observacion_laser_nav',None) if datos is not None else None
        origen,incremento,estampa=laser_info if laser_info is not None else (None,None,None)
        entrada = EntradaNavegacion(ahora, pose, lista,
                                    puntos_base, puntos_mapa,
                                    observacion.std_yaw if observacion is not None else None,
                                    origen,incremento,estampa)
        if (modo == 'ida_vuelta_prueba' and nav.meta is None and lista and pose is not None
                and nav.estado not in ('FALLO','EN_LARGADA')):
            meta = elegir_meta_prueba(
                nav.grilla, nav.costos, nav.distancia, (pose.x, pose.y),
                self.valor('meta_distancia_min_m'), self.valor('meta_distancia_max_m'),
                max(self.valor('meta_espacio_min_m'),
                    self.huella['radius_m']+self.huella['margin_m']+
                    STOPPING_RESERVE_M+nav.grilla.resolucion/2))
            if meta is None:
                self.motivo_meta = 'Sin meta de prueba alcanzable con espacio de frenado'
                nav.cancelar(self.motivo_meta)
            else:
                nav.fijar_meta(meta)
        c = nav.paso(entrada)
        if self.mision is not None:
            self.mision.paso(self.reloj_mision(), ahora,
                             entrada, self.cantidad_victimas)
            self.pub_mision.publish(String(data=json.dumps(self.mision.resumen())))
            if (self.mision.estado in ('FALLO_RELOJ', 'FINALIZADA')
                    or nav.solicitud_regreso):
                c = type(c).cero()
        if nav.ruta is not self.ruta_publicada:
            self.publicar_ruta(nav.ruta)
        if self.control.estado!='LISTA' and nav.estado not in ('EN_LARGADA','FALLO'):
            # M1 conserva su guardia de giro: M2 pausa traslación y le cede
            # la recuperación. Un resultado terminal sigue publicando cero.
            return None
        cmd = Twist()
        cmd.linear.x, cmd.linear.y, cmd.angular.z = c.vx, c.vy, c.wz
        return cmd

    def datos_navegacion(self, ahora, observacion):
        """Pose continua por odometría y scan reproyectado a la misma estampa.

        AMCL aporta la corrección map<-odom, no una pose retenida entre sus
        actualizaciones. El LiDAR se compensa en odom entre captura y control.
        No se extrapola TF ni se renueva el recibo original del scan.
        """
        self.observacion_laser_nav=None
        if observacion is None:
            return None
        try:
            movimiento = self.tf_buffer.lookup_transform('odom', 'base_footprint', Time())
            if self.odom is None or ahora-self.t_odom > self.valor('frescura_sensores_s'):
                return None
            odom_s = self.odom.header.stamp.sec+self.odom.header.stamp.nanosec/1e9
            tf_s = movimiento.header.stamp.sec+movimiento.header.stamp.nanosec/1e9
            if abs(odom_s-tf_s) > self.valor('frescura_sensores_s'):
                self.diagnostico_observacion['nav_tf_error'] = 'TF de odometría fuera de vigencia'
                return None
            correccion = self.tf_buffer.lookup_transform('map', 'odom', Time()).transform
            t = movimiento.transform
            giro = yaw(correccion.rotation)
            c, s = math.cos(giro), math.sin(giro)
            pose = Pose2D(correccion.translation.x + c*t.translation.x - s*t.translation.y,
                          correccion.translation.y + s*t.translation.x + c*t.translation.y,
                          math.atan2(math.sin(giro+yaw(t.rotation)),
                                     math.cos(giro+yaw(t.rotation))))
            for scan, recibido in reversed(self.historial_scan):
                if ahora-recibido > self.valor('frescura_sensores_s'):
                    continue
                try:
                    laser = self.tf_buffer.lookup_transform_full(
                        'base_footprint', Time.from_msg(movimiento.header.stamp),
                        scan.header.frame_id, Time.from_msg(scan.header.stamp), 'odom').transform
                except TransformException:
                    continue
                puntos = puntos_desde_scan(
                    scan.ranges, scan.angle_min, scan.angle_increment,
                    scan.range_min, scan.range_max,
                    (laser.translation.x, laser.translation.y, yaw(laser.rotation)))
                cp, sp = math.cos(pose.yaw), math.sin(pose.yaw)
                mapa = [(pose.x+cp*x-sp*y, pose.y+sp*x+cp*y) for x,y in puntos]
                self.diagnostico_observacion['nav_scan_stamp_s'] = (
                    scan.header.stamp.sec+scan.header.stamp.nanosec/1e9)
                self.diagnostico_observacion['nav_control_stamp_s'] = (
                    movimiento.header.stamp.sec+movimiento.header.stamp.nanosec/1e9)
                self.observacion_laser_nav=(
                    (pose.x+cp*laser.translation.x-sp*laser.translation.y,
                     pose.y+sp*laser.translation.x+cp*laser.translation.y),
                    abs(scan.angle_increment),
                    (scan.header.stamp.sec,scan.header.stamp.nanosec))
                return pose, puntos, mapa
        except TransformException as error:
            self.diagnostico_observacion['nav_tf_error'] = str(error)
        return None

    def publicar_ruta(self, ruta):
        camino = Path()
        camino.header.frame_id = 'map'
        camino.header.stamp = self.get_clock().now().to_msg()
        for x, y in (ruta.puntos if ruta is not None else []):
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.pose.position.x, pose.pose.position.y = float(x), float(y)
            pose.pose.orientation.w = 1.0
            camino.poses.append(pose)
        self.pub_ruta.publish(camino)
        self.ruta_publicada = ruta

    def publicar_navegacion(self):
        duraciones = sorted(self.duraciones)
        estado = self.navegacion.resumen()
        estado.update(
            ciclo_numero=self.numero_ciclo,
            ciclo_ms=1000*self.duraciones[-1] if self.duraciones else None,
            modo=self.valor('modo_navegacion'), prueba_en_seco=self.valor('prueba_en_seco'),
            motivo_meta=self.motivo_meta,
            ciclo_p95_ms=1000 * duraciones[int(0.95 * (len(duraciones) - 1))]
            if duraciones else None)
        self.pub_navegacion.publish(String(data=json.dumps(estado)))

    def ciclo_de_control(self):
        if self.mapa is None:
            return
        if self.mision is not None and self.mision.done:
            self.pub_cmd_vel.publish(Twist())
            return
        if self.valor('use_sim_time') and self.valor('pausar_control_sim'):
            return
        inicio = time.perf_counter()
        ahora = time.monotonic()
        observacion = self.observacion(ahora)
        cmd = Twist()
        if self.valor('localizacion_activa'):
            if self.fallo_inicio:
                self.control.fallar(self.fallo_inicio)
            cmd.angular.z = min(
                self.valor('velocidad_angular_max'),
                self.control.paso(ahora, True, observacion))
        self.reinicializar_recuperacion(observacion)
        navegacion = self.paso_navegacion(ahora, observacion)
        giro_recuperacion = (getattr(self.control,'giro_activo',False)
                              and self.control.estado == 'GIRANDO')
        if navegacion is not None and not giro_recuperacion:
            cmd = navegacion
        if (self.valor('localizacion_activa')
                and getattr(self.control,'secuencia_lista',-1) >= 0
                and not self.control.utilizable):
            cmd.linear.x = cmd.linear.y = 0.0
            if not giro_recuperacion:cmd.angular.z = 0.0
        quieto = cmd.linear.x == cmd.linear.y == cmd.angular.z == 0.0
        # Actualización sin movimiento de AMCL solo con el robot detenido.
        if (self.valor('localizacion_activa') and self.control.estado in (
                'VERIFICANDO', 'VERIFICANDO_QUIETO', 'RECUPERANDO', 'LISTA', 'GIRANDO')
                and (quieto or self.control.estado not in ('LISTA', 'GIRANDO'))):
            self.solicitar_observacion_quieta(ahora)
        if self.valor('prueba_en_seco'):
            self.pub_propuesto.publish(cmd)
            cmd = Twist()
        self.pub_cmd_vel.publish(cmd)
        if self.mision is not None and self.mision.finalizar and not self.mision.done:
            p = self.navegacion.ultima_pose
            largada = self.navegacion.largada
            velocidad = self.odom.twist.twist if self.odom is not None else None
            detenido = (velocidad is not None and math.hypot(
                velocidad.linear.x, velocidad.linear.y) < .02
                and abs(velocidad.angular.z) < .03)
            posicion = (p is not None and largada is not None and
                        math.dist((p.x, p.y), largada[:2]) <= .15)
            if (detenido and posicion and self.control.utilizable
                    and observacion is not None and quieto):
                if self.t_fin_quieto is None:
                    self.t_fin_quieto = ahora
                # M3 vacía candidatos al recibir REGRESANDO/FINALIZADA.
                # Dar tiempo al acuse antes de /done, incluso estando en origen.
                reportes_listos=(self.mision.pendientes_reportes==0 or
                                 ahora-self.t_fin_quieto>=2.5)
                if ahora-self.t_fin_quieto >= .5 and reportes_listos and not self.valor('prueba_en_seco'):
                    self.pub_done.publish(Empty())
                    self.mision.done = True
                    self.pub_mision.publish(String(data=json.dumps(self.mision.resumen())))
            else:
                self.t_fin_quieto = None
        estado = {
            'estado': self.control.estado, 'motivo': self.control.motivo,
            'lista': self.control.estado == 'LISTA' and observacion is not None,
            'utilizable': self.control.utilizable and observacion is not None,
            'concordancia_simple': self.diagnostico_observacion.get('concordancia_simple'),
            'giros': self.control.giros, 'recuperaciones': self.control.recuperaciones,
            'reinicios_locales': getattr(self.control,'reinicios_locales',0),
            'reinicios_publicados': getattr(self,'reinicios_publicados',0),
            'recuperacion_activa': getattr(self.control,'giro_activo',False),
            'pose_amcl': list(observacion.pose) if observacion else None,
            'diagnostico_observacion': self.diagnostico_observacion,
            'concordancia': observacion.concordancia if observacion else None,
            'std_xy_m': observacion.std_xy if observacion else None,
            'std_yaw_rad': observacion.std_yaw if observacion else None}
        self.pub_estado.publish(String(data=json.dumps(estado)))
        self.duraciones.append(time.perf_counter() - inicio)
        self.numero_ciclo+=1
        if self.navegacion is not None:
            self.publicar_navegacion()
        if estado['estado'] != self.ultimo_estado:
            self.ultimo_estado = estado['estado']
            self.get_logger().info(
                f'Localización: {estado["estado"]} {estado["motivo"]} '
                f'diagnóstico={json.dumps(self.diagnostico_observacion)}')

    def destroy_node(self):
        mision = getattr(self, 'mision', None)
        if mision is not None:
            mision.cerrar()
        nav=getattr(self,'navegacion',None)
        cerrar=getattr(nav,'cerrar',None)
        if cerrar is not None:cerrar()
        return super().destroy_node()

    def terminar(self):
        self.pub_cmd_vel.publish(Twist())
        self.pub_done.publish(Empty())


def main(args=None):
    rclpy.init(args=args)
    nodo = NodoEquipo()
    try:
        rclpy.spin(nodo)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        import signal
        # El primer SIGINT ya detuvo spin; no interrumpir otra vez la limpieza.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            nodo.destroy_node()
        except (KeyboardInterrupt, ExternalShutdownException):
            pass
        try:
            rclpy.try_shutdown()
        except (KeyboardInterrupt, ExternalShutdownException):
            pass
        finally:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
