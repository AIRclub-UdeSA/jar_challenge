"""Nodo de misión (ADR-0008): orquesta la corrida y es el único que reporta.

Envuelve a `maquina_mision.Mision`: junta las entradas (pose por TF, odometría,
objetivos de percepción, goal de exploración, estado de Nav2), le pide las
acciones y las ejecuta. No mueve el robot directo: pide velocidad al
supervisor por /jar/cmd_vel_mision y manda goals a Nav2.
"""
import json
import math

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Point, PointStamped, PoseStamped, Twist
from nav2_msgs.action import NavigateToPose
from nav2_msgs.srv import ClearEntireCostmap
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.action import ActionClient
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import LaserScan, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Empty, Header, String
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from equipo_jar.huecos import (ConfigHuecos, RegistroHuecos, cuello_en_camino, huecos_del_scan,
                               pasable)
from equipo_jar.maquina_mision import Config, Entradas, Estado, Mision
from equipo_jar.paso_angosto import ConfigPaso
from equipo_jar.reportes import armar_reporte
from jar_supervisor.logica import direccion_mas_libre, giro_para_salir, scan_a_puntos

QOS_MAPA = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                      durability=DurabilityPolicy.TRANSIENT_LOCAL)


def yaw_de(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class NodoMision(Node):
    def __init__(self):
        super().__init__('mision')
        # Cantidad de víctimas a buscar (N), que llega como argumento del launch (victimas:=N).
        self.cupos = self.declare_parameter('cantidad_victimas', 5).value
        radio_union = self.declare_parameter('radio_union_objetivo_m', 0.35).value
        config = Config()
        for nombre, valor in vars(config).items():
            setattr(config, nombre, self.declare_parameter(nombre, valor).value)
        # La maniobra de paso angosto (paso_angosto.py), como `paso.<campo>` en params.yaml.
        config_paso = ConfigPaso()
        for nombre, valor in vars(config_paso).items():
            setattr(config_paso, nombre, self.declare_parameter(f'paso.{nombre}', valor).value)
        # Huecos medidos con el LiDAR (huecos.py, r9): por ahora sólo se miden, se publican en
        # /jar/huecos para RViz y se avisan en el log; la misión no los usa todavía.
        self.medir_huecos = self.declare_parameter('medir_huecos', False).value
        config_huecos = ConfigHuecos()
        for nombre, valor in vars(config_huecos).items():
            setattr(config_huecos, nombre, self.declare_parameter(f'huecos.{nombre}', valor).value)
        self.huecos = RegistroHuecos(config_huecos)
        # Girando, el scan sale torcido (los rayos son de instantes distintos): no se mide.
        self.giro_max_huecos = self.declare_parameter('giro_max_huecos', 0.3).value
        self.giro_odom = 0.0
        frecuencia = self.declare_parameter('frecuencia_hz', 10.0).value
        self.marco_mapa = self.declare_parameter('marco_mapa', 'map').value
        self.marco_robot = self.declare_parameter('marco_robot', 'base_footprint').value
        self.umbral_quieto_lineal = self.declare_parameter('umbral_quieto_lineal', 0.02).value
        # El rectángulo del robot, como el del supervisor, para buscar hacia dónde escapar.
        self.medio_largo = self.declare_parameter('medio_largo_robot_m', 0.16).value
        self.medio_ancho = self.declare_parameter('medio_ancho_robot_m', 0.11).value
        # El mismo margen que el supervisor: el escape va hacia donde el supervisor lo deja pasar.
        self.margen = self.declare_parameter('margen_m', 0.02).value
        self.margen_giro = self.declare_parameter('margen_giro_m', 0.005).value   # = supervisor

        self.mision = Mision(config, self.cupos, radio_union, config_paso)
        self.mapa_libre = None        # (libre, resolución, origen) del mapa estático
        self.costmap = None           # (costos, resolución, origen) del costmap global de Nav2
        self.mision.buscar_cuello = self.buscar_cuello
        self.t_mapa = None
        self.yaw_odom = None
        self.quieto_desde = None
        self.objetivos = []
        self.candidatos = []
        self.exploracion = {}
        self.obstaculo = None
        self.salida = None
        self.giro_salida = 0
        self.puntos = None            # LiDAR en el marco del robot, para el paso angosto
        self.plan = None              # último plan de Nav2 en `map`, para el paso angosto
        self.bloqueos = []            # huecos angostos tapados, en `map` (se republican siempre)
        self.tf_laser = None
        self.estado_previo = None
        self.reportes_emitidos = 0

        # Nav2: un goal por vez. Los resultados de goals cancelados se ignoran.
        self.goal_handle = None
        self.esperando_aceptacion = False
        self.nav_resultado = None
        self.id_goal = 0

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.nav = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        self.create_subscription(OccupancyGrid, '/map', self.al_mapa, QOS_MAPA)
        self.giro_frenado_desde = self.giro_frenado_ultimo = None
        if config.escapar_giro_frenado:
            self.create_subscription(String, '/jar/supervisor/estado', self.al_supervisor, 10)
        if config.medir_antes_de_descartar:
            self.create_subscription(OccupancyGrid, '/global_costmap/costmap', self.al_costmap,
                                     QOS_MAPA)
        self.create_subscription(Odometry, '/odom', self.al_odom, 10)
        self.create_subscription(String, '/jar/objetivos', self.al_objetivos, 10)
        self.create_subscription(String, '/jar/exploracion', self.al_exploracion, 10)
        self.create_subscription(String, '/jar/candidatos', self.al_candidatos, 10)
        self.create_subscription(Path, '/plan', self.al_plan, 10)
        self.create_subscription(LaserScan, '/scan', self.al_scan, qos_profile_sensor_data)
        self.pub_victimas = self.create_publisher(PointStamped, '/victimas', 10)
        self.pub_done = self.create_publisher(Empty, '/done', 10)
        self.pub_cmd = self.create_publisher(Twist, '/jar/cmd_vel_mision', 10)
        self.pub_descartar = self.create_publisher(PointStamped, '/jar/exploracion/descartar', 10)
        self.pub_visitado = self.create_publisher(PoseStamped, '/jar/exploracion/visitado', 10)
        self.pub_bloqueos = self.create_publisher(PointCloud2, '/jar/bloqueos', 1)
        self.create_timer(1.0, self.publicar_bloqueos)
        # El costmap global marca las cajas (que no están en el mapa) con la pose de AMCL de
        # cada scan: con la pose oscilando ~10 cm las marcas se acumulan y "estiran" la caja
        # hasta cerrar pasos por donde el robot entra. Limpiarlo seguido deja sólo lo que se
        # ve ahora (el mapa estático queda y los tapones se republican cada segundo).
        self.limpiar_global = self.create_client(
            ClearEntireCostmap, '/global_costmap/clear_entirely_global_costmap')
        periodo = self.declare_parameter('periodo_limpieza_costmap_s', 4.0).value
        if periodo > 0:
            self.create_timer(periodo, self.limpiar_costmap_global)
        self.pub_estado = self.create_publisher(String, '/jar/mision/estado', 10)
        self.pub_goal = self.create_publisher(PoseStamped, '/jar/mision/goal', 10)
        if self.medir_huecos:
            self.pub_huecos = self.create_publisher(MarkerArray, '/jar/huecos', 1)
            self.create_timer(1.0, self.publicar_huecos)
        self.create_timer(1.0 / frecuencia, self.ciclo)
        self.get_logger().info(
            f'Hay que encontrar {self.cupos} víctimas. Esperando el mapa en /map...')

    # Callbacks ----------------------------------------------------------------
    def al_mapa(self, mapa):
        info = mapa.info
        libre = np.asarray(mapa.data, dtype=np.int16).reshape(info.height, info.width) == 0
        self.mapa_libre = (libre, info.resolution,
                           (info.origin.position.x, info.origin.position.y))
        if self.t_mapa is None:
            self.t_mapa = self.get_clock().now()
            self.get_logger().info(
                f'Mapa recibido: {mapa.info.width}x{mapa.info.height} celdas, '
                f'resolución {mapa.info.resolution:.3f} m. Largada.')

    def al_odom(self, msg):
        self.yaw_odom = yaw_de(msg.pose.pose.orientation)
        v = msg.twist.twist
        self.giro_odom = v.angular.z
        # Quieto = no se traslada. Girar en el lugar (las recuperaciones de Nav2) no es avanzar:
        # si contara como movimiento, un robot trabado que gira no se detectaría nunca.
        quieto = math.hypot(v.linear.x, v.linear.y) < self.umbral_quieto_lineal
        ahora = self.get_clock().now()
        if not quieto:
            self.quieto_desde = None
        elif self.quieto_desde is None:
            self.quieto_desde = ahora

    def al_scan(self, msg):
        """Puntos del LiDAR (marco del robot) y el más cercano: para escape y paso angosto."""
        if self.tf_laser is None:
            try:
                tf = self.tf_buffer.lookup_transform(self.marco_robot, msg.header.frame_id,
                                                     rclpy.time.Time())
            except Exception:  # noqa: BLE001 - TF todavía no disponible
                return
            self.tf_laser = (tf.transform.translation.x, tf.transform.translation.y,
                             yaw_de(tf.transform.rotation))
        puntos = scan_a_puntos(msg.ranges, msg.angle_min, msg.angle_increment,
                               msg.range_min, msg.range_max, *self.tf_laser)
        # Los mismos puntos que usa el supervisor, también los que quedan dentro del rectángulo
        # (el robot rozando una caja): si la maniobra o el escape no los ven, piden ir justo
        # hacia ahí, el supervisor los frena a cero y el robot queda quieto hasta que el juez
        # corta por inmovilidad.
        self.puntos = puntos
        if self.medir_huecos:
            self.medir_huecos_en(msg)
        if len(puntos) == 0:
            self.obstaculo = self.salida = None
            self.giro_salida = 0
            return
        k = int(np.argmin(np.hypot(puntos[:, 0], puntos[:, 1])))
        self.obstaculo = (float(puntos[k, 0]), float(puntos[k, 1]))
        d = math.hypot(*self.obstaculo)
        lejos = None if d < 1e-6 else (-self.obstaculo[0] / d, -self.obstaculo[1] / d)
        self.salida = direccion_mas_libre(puntos, self.medio_largo + self.margen,
                                          self.medio_ancho + self.margen, preferida=lejos,
                                          margen=self.margen)
        self.giro_salida = giro_para_salir(puntos, self.medio_largo + self.margen,
                                           self.medio_ancho + self.margen, self.margen,
                                           self.margen_giro)

    def al_supervisor(self, msg):
        try:
            frenado = bool(json.loads(msg.data).get('giro_frenado'))
        except ValueError:
            return
        # Nav2 pide girar a ratos (entre medio replanifica y no pide nada): el frenado se da
        # por terminado recién tras 1 s seguido sin frenado.
        ahora = self.get_clock().now()
        if frenado:
            self.giro_frenado_ultimo = ahora
            if self.giro_frenado_desde is None:
                self.giro_frenado_desde = ahora
        elif (self.giro_frenado_desde is not None
              and (ahora - self.giro_frenado_ultimo).nanoseconds * 1e-9 > 1.0):
            self.giro_frenado_desde = None

    def al_costmap(self, msg):
        info = msg.info
        self.costmap = (np.asarray(msg.data, dtype=np.int16).reshape(info.height, info.width),
                        info.resolution, (info.origin.position.x, info.origin.position.y))

    def cerrado_en_costmap(self, x, y):
        """¿Nav2 tiene cerrado (x, y)? 99 = inscripto, 100 = obstáculo: el centro del robot no
        puede estar ahí."""
        costos, res, (ox, oy) = self.costmap
        i, j = int((y - oy) / res), int((x - ox) / res)
        return 0 <= i < costos.shape[0] and 0 <= j < costos.shape[1] and costos[i, j] >= 99

    def buscar_cuello(self, pose, goal):
        """Para la misión (medir antes de descartar): el cuello del camino en el mapa estático."""
        if self.mapa_libre is None:
            return None
        libre, res, origen = self.mapa_libre
        return cuello_en_camino(libre, res, origen, pose[:2], goal[:2],
                                despeje_min=self.medio_ancho,
                                bloqueado=None if self.costmap is None
                                else self.cerrado_en_costmap)

    def huecos_pasables(self):
        if not (self.medir_huecos and self.mision.config.cruzar_huecos):
            return []
        return [{'a': h.borde_a, 'b': h.borde_b, 'ancho': h.ancho}
                for h in self.huecos.medidos() if pasable(h.ancho, self.huecos.config)]

    def medir_huecos_en(self, msg):
        pose = self.pose()
        if pose is None or self.t_mapa is None or abs(self.giro_odom) > self.giro_max_huecos:
            return
        huecos = huecos_del_scan(msg.ranges, msg.angle_min, msg.angle_increment, msg.range_min,
                                 msg.range_max, self.tf_laser, self.huecos.config)
        for h, cambio in self.huecos.agregar(huecos, pose):
            estado = 'pasable' if pasable(h.ancho, self.huecos.config) else 'no pasable'
            self.get_logger().info(
                f'hueco {h.id} en ({h.centro[0]:.2f}, {h.centro[1]:.2f}): {h.ancho:.2f} m, '
                f'{estado}' + ('' if cambio == 'medido' else ' (cambió)'))

    def publicar_huecos(self):
        """Los huecos medidos en RViz: verde pasable, rojo no, con el ancho."""
        marcas = MarkerArray()
        borrar = Marker()
        borrar.action = Marker.DELETEALL
        marcas.markers.append(borrar)
        ahora = self.get_clock().now().to_msg()
        for h in self.huecos.medidos():
            ok = pasable(h.ancho, self.huecos.config)
            for tipo, k in ((Marker.LINE_LIST, 0), (Marker.TEXT_VIEW_FACING, 1)):
                m = Marker()
                m.header.frame_id, m.header.stamp = self.marco_mapa, ahora
                m.ns, m.id, m.type, m.action = 'huecos', 2 * h.id + k, tipo, Marker.ADD
                m.pose.orientation.w = 1.0
                m.color.r, m.color.g, m.color.b, m.color.a = (0.1, 0.9, 0.1, 1.0) if ok else \
                    (0.9, 0.1, 0.1, 1.0)
                if tipo == Marker.LINE_LIST:
                    m.scale.x = 0.03
                    m.points = [Point(x=h.borde_a[0], y=h.borde_a[1], z=0.05),
                                Point(x=h.borde_b[0], y=h.borde_b[1], z=0.05)]
                else:
                    m.scale.z = 0.12
                    m.pose.position.x, m.pose.position.y, m.pose.position.z = \
                        h.centro[0], h.centro[1], 0.25
                    m.text = f'{h.ancho:.2f} m'
                marcas.markers.append(m)
        self.pub_huecos.publish(marcas)

    def limpiar_costmap_global(self):
        if self.t_mapa is not None and self.limpiar_global.service_is_ready():
            self.limpiar_global.call_async(ClearEntireCostmap.Request())

    def publicar_bloqueos(self):
        """Los huecos tapados, para la capa bloqueos_layer del costmap global (nav2.yaml)."""
        if not self.bloqueos:
            return
        cabecera = Header(frame_id=self.marco_mapa, stamp=self.get_clock().now().to_msg())
        self.pub_bloqueos.publish(point_cloud2.create_cloud_xyz32(
            cabecera, [(x, y, 0.1) for x, y in self.bloqueos]))

    def al_plan(self, msg):
        if msg.header.frame_id in ('', self.marco_mapa):
            self.plan = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]

    def al_objetivos(self, msg):
        try:
            self.objetivos = json.loads(msg.data)
        except ValueError:
            self.get_logger().warning('objetivos con JSON inválido')

    def al_candidatos(self, msg):
        try:
            self.candidatos = json.loads(msg.data)
        except ValueError:
            self.get_logger().warning('candidatos con JSON inválido')

    def al_exploracion(self, msg):
        try:
            self.exploracion = json.loads(msg.data)
        except ValueError:
            self.get_logger().warning('exploración con JSON inválido')

    # Nav2 ---------------------------------------------------------------------
    def nav_libre(self):
        return (self.nav.server_is_ready() and self.goal_handle is None
                and not self.esperando_aceptacion)

    def enviar_goal(self, x, y, yaw):
        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = self.marco_mapa
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x, goal.pose.pose.position.y = float(x), float(y)
        goal.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.pose.orientation.w = math.cos(yaw / 2.0)
        self.pub_goal.publish(goal.pose)
        self.id_goal += 1
        mi_id = self.id_goal
        self.esperando_aceptacion = True
        futuro = self.nav.send_goal_async(
            goal, feedback_callback=lambda f: self.al_feedback(f, mi_id))
        futuro.add_done_callback(lambda f: self.al_aceptar(f, mi_id))
        self.get_logger().info(f'goal {mi_id}: ({x:.2f}, {y:.2f}, {yaw:.2f})')

    def al_aceptar(self, futuro, mi_id):
        handle = futuro.result()
        if mi_id != self.id_goal:          # se canceló mientras esperaba
            if handle.accepted:
                handle.cancel_goal_async()
            return
        self.esperando_aceptacion = False
        if not handle.accepted:
            self.nav_resultado = 'falla'
            return
        self.goal_handle = handle
        handle.get_result_async().add_done_callback(lambda f: self.al_resultado(f, mi_id))

    def al_feedback(self, feedback, mi_id):
        maximo = self.mision.config.max_recuperaciones_nav
        n = feedback.feedback.number_of_recoveries
        # Sólo explorando: volviendo, cada falla gasta un intento de regreso y conviene que Nav2
        # insista.
        if (maximo <= 0 or mi_id != self.id_goal or n < maximo
                or self.mision.estado != Estado.EXPLORANDO):
            return
        self.get_logger().info(f'goal {mi_id}: {n} recuperaciones de Nav2, se cancela como falla')
        self.cancelar_goal()
        self.nav_resultado = 'falla'

    def al_resultado(self, futuro, mi_id):
        if mi_id != self.id_goal:
            return
        self.goal_handle = None
        estado = futuro.result().status
        self.nav_resultado = 'ok' if estado == GoalStatus.STATUS_SUCCEEDED else 'falla'

    def cancelar_goal(self):
        self.id_goal += 1                  # invalida los callbacks pendientes
        self.esperando_aceptacion = False
        if self.goal_handle is not None:
            self.goal_handle.cancel_goal_async()
        self.goal_handle = None

    # Ciclo --------------------------------------------------------------------
    def pose(self, marco=None):
        try:
            tf = self.tf_buffer.lookup_transform(marco or self.marco_mapa, self.marco_robot,
                                                 rclpy.time.Time())
        except Exception:  # noqa: BLE001 - sin localización todavía
            return None
        tr = tf.transform.translation
        return (tr.x, tr.y, yaw_de(tf.transform.rotation))

    def ciclo(self):
        ahora = self.get_clock().now()
        t = None if self.t_mapa is None else (ahora - self.t_mapa).nanoseconds * 1e-9
        quieto_s = 0.0 if self.quieto_desde is None else \
            (ahora - self.quieto_desde).nanoseconds * 1e-9
        goal = self.exploracion.get('goal')
        entradas = Entradas(
            t=t, pose=self.pose(), yaw_odom=self.yaw_odom, nav_libre=self.nav_libre(),
            nav_resultado=self.nav_resultado,
            goal_exploracion=tuple(goal) if goal else None,
            goal_es_inspeccion=self.exploracion.get('inspeccion') is not None,
            inspeccion_id=self.exploracion.get('inspeccion'), candidatos=self.candidatos,
            exploracion_terminada=bool(self.exploracion.get('terminado', False)),
            objetivos=self.objetivos, quieto_s=quieto_s, obstaculo=self.obstaculo,
            salida=self.salida, giro_salida=self.giro_salida, plan=self.plan, puntos=self.puntos,
            pose_odom=self.pose('odom'), huecos=self.huecos_pasables(),
            giro_frenado_s=0.0 if self.giro_frenado_desde is None
            else (ahora - self.giro_frenado_desde).nanoseconds * 1e-9)
        self.nav_resultado = None
        a = self.mision.paso(entradas)
        self.ejecutar(a)
        if self.mision.estado != self.estado_previo:
            self.get_logger().info(f'estado: {self.mision.estado.value} {self.mision.motivo}')
            self.estado_previo = self.mision.estado
        if t is not None:
            self.pub_estado.publish(String(data=json.dumps({
                't': round(t, 1), 'estado': self.mision.estado.value,
                'motivo': self.mision.motivo, 'reportes': self.reportes_emitidos,
                'cobertura': self.exploracion.get('cobertura')})))

    def ejecutar(self, a):
        for aviso in a.avisos:
            self.get_logger().info(aviso)
        if a.liberar_tapones:
            self.bloqueos = []            # se dejan de republicar y se borran del costmap ya
            self.limpiar_costmap_global()
        if a.bloquear:
            self.bloqueos.extend(a.bloquear)
            self.publicar_bloqueos()
        if a.cancelar:
            self.cancelar_goal()
        if a.descartar is not None:
            p = PointStamped()
            p.header.frame_id = self.marco_mapa
            p.point.x, p.point.y = float(a.descartar[0]), float(a.descartar[1])
            self.pub_descartar.publish(p)
        if a.visitado is not None:
            v = PoseStamped()
            v.header.frame_id = self.marco_mapa
            v.pose.position.x, v.pose.position.y = float(a.visitado[0]), float(a.visitado[1])
            v.pose.orientation.z = math.sin(a.visitado[2] / 2.0)
            v.pose.orientation.w = math.cos(a.visitado[2] / 2.0)
            self.pub_visitado.publish(v)
        if a.goal is not None:
            self.enviar_goal(*a.goal)
        for x, y in a.reportes:
            try:
                self.pub_victimas.publish(armar_reporte(x, y, self.get_clock().now().to_msg()))
            except ValueError as e:
                self.get_logger().error(f'reporte descartado: {e}')
                continue
            self.reportes_emitidos += 1
            self.get_logger().info(
                f'REPORTE {self.reportes_emitidos}/{self.cupos}: ({x:.2f}, {y:.2f})')
        if a.cmd is not None:
            cmd = Twist()
            cmd.linear.x, cmd.linear.y, cmd.angular.z = (float(v) for v in a.cmd)
            self.pub_cmd.publish(cmd)
        if a.done:
            self.pub_done.publish(Empty())
            self.get_logger().info(f'/done publicado ({self.mision.motivo})')


def main(args=None):
    rclpy.init(args=args)
    nodo = NodoMision()
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
