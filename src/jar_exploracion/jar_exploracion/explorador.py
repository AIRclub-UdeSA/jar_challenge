"""Nodo explorador (ADR-0005): marca lo que vio la cámara y propone el próximo goal.

Antes que la cobertura, manda a mirar de frente los candidatos que la percepción todavía no
clasificó (/jar/candidatos): cada objeto que no está en el mapa puede ser una víctima.
Lee también el costmap global de Nav2: los obstáculos que el mapa no tiene (cubos
de color) tapan la cámara y cortan caminos para la elección del goal.
Publica en /jar/exploracion un JSON {"goal": [x, y, yaw] | null, "cobertura": f,
"terminado": bool} y, para RViz, la grilla de lo visto en /jar/exploracion/visto.
La misión toma el goal cuando Nav2 queda libre; los goals que fallan llegan por
/jar/exploracion/descartar y no se vuelven a proponer; los goals a los que la misión
llegó y miró llegan por /jar/exploracion/visitado: se da por visto lo que prometían y
tampoco se repiten.
"""
import json
import math

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped, PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

from jar_exploracion.cobertura import Grilla
from jar_exploracion.inspecciones import Inspecciones

QOS_MAPA = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                      durability=DurabilityPolicy.TRANSIENT_LOCAL)


def yaw_de(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class NodoExplorador(Node):
    def __init__(self):
        super().__init__('explorador')
        p = self.declare_parameter
        self.resolucion_gruesa = p('resolucion_gruesa_m', 0.10).value
        self.alcance = p('alcance_camara_m', 1.8).value
        self.fov = p('fov_camara_rad', 1.0).value       # se reemplaza con camera_info
        self.giro_max_marcado = p('giro_max_marcado', 0.4).value
        self.despeje_min = p('despeje_min_m', 0.20).value
        self.despeje_goal = p('despeje_goal_m', 0.35).value
        self.separacion = p('separacion_candidatos_m', 0.30).value
        self.costo_distancia = p('costo_distancia', 30.0).value
        self.ganancia_minima = p('ganancia_minima', 15.0).value
        self.radio_local = p('radio_local_m', 1.5).value
        self.ganancia_local = p('ganancia_local', 10.0).value
        self.fraccion_local = p('fraccion_local', 0.5).value
        self.radio_descarte = p('radio_descarte_m', 0.30).value
        self.intentos_inspeccion = p('intentos_inspeccion', 2).value
        # Rojo cortado por el borde (la percepción lo marca): la caja está demasiado cerca para
        # ubicarla; se la vuelve a mirar desde más lejos, con un intento más.
        self.alejar_rojo_cortado = p('alejar_rojo_cortado', False).value
        self.inspeccion_lejos = (p('inspeccion_lejos_min_m', 1.0).value,
                                 p('inspeccion_lejos_max_m', 1.5).value)
        self.cortados = set()             # ids de candidatos con rojo cortado ya atendidos
        # Mantener la pose de una inspección mientras siga sirviendo, en vez de buscarla de
        # nuevo cada segundo desde donde está el robot (saltaba de un lado al otro de un hueco).
        self.mantener_inspeccion = p('mantener_inspeccion', False).value
        # Una inspección a la que sólo se llega por un hueco angosto (a menos de despeje_amplio_m
        # de las paredes) cuesta penalidad_paso_m de más: si se puede, se mira desde este lado.
        self.despeje_amplio = p('despeje_amplio_m', 0.22).value
        self.penalidad_paso = p('penalidad_paso_m', 0.0).value
        # Despeje mínimo de una pose de inspección. Antes no se pasaba y valía el de la función
        # (0.25), menos que el de la cobertura (despeje_goal_m): en laberinto_puertas una pose
        # quedó en la boca de la puerta de D, pegada a su caja, y el robot se trabó ahí.
        self.despeje_inspeccion = p('despeje_inspeccion_m', 0.25).value
        # Dudosos (inspecciones.py): una visita sin votos no gasta el intento, una con votos sin
        # decidir no repite el punto de vista, y al terminar la cobertura los que no se
        # decidieron se miran una vez más.
        self.seguir_dudosos = p('seguir_dudosos', False).value
        visitas_sin_voto_max = p('visitas_sin_voto_max', 2).value
        # Con seguir_dudosos: un descarte lejos del candidato no gasta el intento, y un dudoso que
        # queda cerca del robot se mira de paso.
        descarte_lejos_m = p('descarte_lejos_m', 1.0).value
        descartes_lejos_max = p('descartes_lejos_max', 3).value
        dudoso_de_paso_m = p('dudoso_de_paso_m', 1.0).value
        # Planificación por zonas (zonas.py): el próximo goal sale de la zona sin ver que
        # minimiza el tiempo esperado hasta encontrar las víctimas, no del greedy por goal.
        self.planificar_zonas = p('planificar_zonas', False).value
        self.cantidad_victimas = p('cantidad_victimas', 5).value      # del launch (victimas:=N)
        self.vel_plan = p('zonas.vel_m_s', 0.15).value               # viaje entre zonas
        self.cobertura_plan = p('zonas.cobertura_m2_s', 0.15).value   # m² vistos por segundo
        self.celdas_min_zona = p('zonas.celdas_min', 20).value
        self.max_zonas = p('zonas.max_zonas', 7).value
        self.histeresis_zona = p('zonas.histeresis', 0.1).value
        self.peso_posible = p('zonas.peso_posible', 100.0).value
        self.marco_mapa = p('marco_mapa', 'map').value
        self.marco_robot = p('marco_robot', 'base_footprint').value
        frecuencia_marcado = p('frecuencia_marcado_hz', 5.0).value
        frecuencia_goal = p('frecuencia_goal_hz', 1.0).value

        self.grilla = None
        self.mapa = None
        self.giro = 0.0
        self.descartados = []
        self.costmap = None
        self.terminado_avisado = False
        self.candidatos = []              # de la percepción: [{'id', 'x', 'y', 'estado'}]
        self.inspecciones = Inspecciones(self.intentos_inspeccion, self.seguir_dudosos,
                                         visitas_sin_voto_max, descarte_lejos_m,
                                         descartes_lejos_max, dudoso_de_paso_m)
        self.inspeccion = None            # (goal, id) de la última inspección propuesta
        self.inspeccion_fija = None       # (id, goal) que se mantiene (mantener_inspeccion)
        self.ultima_decision = None
        self.fov_de_camara = False

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(OccupancyGrid, '/map', self.al_mapa, QOS_MAPA)
        self.create_subscription(OccupancyGrid, '/global_costmap/costmap', self.al_costmap,
                                 QOS_MAPA)
        self.create_subscription(Odometry, '/odom', self.al_odom, 10)
        self.create_subscription(CameraInfo, '/cam_1/color/camera_info', self.al_camara,
                                 qos_profile_sensor_data)
        self.create_subscription(PointStamped, '/jar/exploracion/descartar', self.al_descartar, 10)
        self.create_subscription(PoseStamped, '/jar/exploracion/visitado', self.al_visitado, 10)
        self.create_subscription(String, '/jar/candidatos', self.al_candidatos, 10)
        self.pub = self.create_publisher(String, '/jar/exploracion', 10)
        self.pub_visto = self.create_publisher(OccupancyGrid, '/jar/exploracion/visto', 1)
        self.create_timer(1.0 / frecuencia_marcado, self.marcar)
        self.create_timer(1.0 / frecuencia_goal, self.proponer)

    def al_mapa(self, mapa):
        if self.grilla is not None:
            return
        info = mapa.info
        self.mapa = mapa
        self.grilla = Grilla(mapa.data, info.width, info.height, info.resolution,
                             info.origin.position.x, info.origin.position.y,
                             resolucion_gruesa=self.resolucion_gruesa)
        self.get_logger().info(f'grilla de cobertura {self.grilla.forma} a {self.grilla.res:.2f} m')

    def al_costmap(self, msg):
        self.costmap = msg
        if self.grilla is None:
            return
        info = msg.info
        self.grilla.actualizar_costmap(msg.data, info.width, info.height, info.resolution,
                                       info.origin.position.x, info.origin.position.y)

    def al_odom(self, msg):
        self.giro = msg.twist.twist.angular.z

    def al_camara(self, msg):
        if not self.fov_de_camara and msg.k[0] > 0:
            self.fov = 2.0 * math.atan(msg.width / (2.0 * msg.k[0]))
            self.fov_de_camara = True
            self.get_logger().info(f'FOV horizontal de la cámara: {self.fov:.2f} rad')

    def registrar_decision(self, pose, tipo, goal, detalle):
        """Una línea en el log cuando cambia el goal propuesto: por qué se eligió ese lugar."""
        clave = (tipo, round(goal[0], 1), round(goal[1], 1))
        if clave == self.ultima_decision:
            return
        self.ultima_decision = clave
        detalle = {k: v for k, v in detalle.items() if k not in ('navegables',)}
        self.get_logger().info(
            f'decisión: {tipo} → ({goal[0]:.2f}, {goal[1]:.2f}) desde ({pose[0]:.2f}, '
            f'{pose[1]:.2f}) {json.dumps(detalle, ensure_ascii=False)}')

    def al_candidatos(self, msg):
        try:
            self.candidatos = json.loads(msg.data)
        except ValueError:
            self.get_logger().warning('candidatos con JSON inválido')

    def al_visitado(self, msg):
        if self.inspeccion is not None:
            (gx, gy, _), ident = self.inspeccion
            if math.hypot(msg.pose.position.x - gx, msg.pose.position.y - gy) < 0.3:
                self.inspecciones.visitado(ident, self.candidatos, (gx, gy))
                self.inspeccion_fija = None   # llegó: el próximo intento elige de nuevo
        if self.grilla is not None:
            p = msg.pose
            self.grilla.llegada(p.position.x, p.position.y, yaw_de(p.orientation),
                                self.fov, self.alcance)

    def al_descartar(self, msg):
        self.descartados.append((msg.point.x, msg.point.y))
        # Una inspección que la misión no pudo alcanzar también cuenta como intento: si no, se
        # proponía el mismo candidato desde otra pose (a más del radio de descarte) sin parar.
        if self.inspeccion is not None:
            (gx, gy, _), ident = self.inspeccion
            if math.hypot(msg.point.x - gx, msg.point.y - gy) < 0.3:
                pose = self.pose()
                self.inspecciones.descartado(ident, self.candidatos,
                                             None if pose is None else pose[:2])
                self.inspeccion_fija = None

    def pose(self):
        try:
            tf = self.tf_buffer.lookup_transform(self.marco_mapa, self.marco_robot,
                                                 rclpy.time.Time())
        except Exception:  # noqa: BLE001 - sin localización todavía
            return None
        return (tf.transform.translation.x, tf.transform.translation.y,
                yaw_de(tf.transform.rotation))

    def marcar(self):
        if self.grilla is None:
            return
        pose = self.pose()
        # Girando rápido la percepción no detecta: no se cuenta como visto.
        if pose is None or abs(self.giro) > self.giro_max_marcado:
            return
        self.grilla.marcar_visto(pose[0], pose[1], pose[2], self.fov, self.alcance)

    def proponer(self):
        if self.grilla is None:
            return
        pose = self.pose()
        if pose is None:
            return
        g = self.grilla
        rangos = {}
        if self.alejar_rojo_cortado:
            for c in self.candidatos:
                if c.get('estado') == 'sin_inspeccionar' and c.get('rojo_cortado'):
                    rangos[c['id']] = self.inspeccion_lejos
                    if c['id'] not in self.cortados:
                        # Un intento más: el anterior terminó demasiado cerca.
                        self.cortados.add(c['id'])
                        self.inspecciones.uno_mas(c['id'])
        de_paso = self.inspecciones.de_paso(self.candidatos, pose[:2])
        if de_paso:
            self.get_logger().info(f'dudosos cerca: se miran de paso {de_paso}')
        pendientes = self.inspecciones.pendientes(self.candidatos)
        insp, ident, sin_pose = g.mejor_inspeccion(
            pose[0], pose[1], pendientes, despeje_min=self.despeje_min,
            descartados=self.descartados + self.inspecciones.puntos_sin_decidir,
            radio_descarte=self.radio_descarte, rangos=rangos,
            fija=self.inspeccion_fija if self.mantener_inspeccion else None,
            despeje_amplio=self.despeje_amplio, penalidad_paso=self.penalidad_paso,
            despeje_goal=self.despeje_inspeccion)
        for k in sin_pose:
            self.inspecciones.agotar(k)
        self.inspeccion = None if insp is None else (insp, ident)
        self.inspeccion_fija = None if insp is None else (ident, insp)
        if insp is not None:
            self.registrar_decision(pose, 'inspección', insp, {'candidato': ident})
            self.pub.publish(String(data=json.dumps({
                'goal': [round(v, 3) for v in insp], 'inspeccion': ident,
                'cobertura': round(g.cobertura(pose[0], pose[1]), 3), 'terminado': False})))
            self.publicar_visto()
            return
        kw = dict(despeje_min=self.despeje_min, separacion=self.separacion,
                  costo_distancia=self.costo_distancia, descartados=self.descartados,
                  radio_descarte=self.radio_descarte, ganancia_minima=self.ganancia_minima,
                  despeje_goal=self.despeje_goal)
        goal = None
        if self.planificar_zonas:
            posibles = [(c['x'], c['y']) for c in self.candidatos if c.get('estado') == 'posible']
            zona = g.zona_objetivo(pose[0], pose[1], despeje_min=self.despeje_min,
                                   vel_m_s=self.vel_plan, cobertura_m2_s=self.cobertura_plan,
                                   celdas_min=self.celdas_min_zona, max_zonas=self.max_zonas,
                                   histeresis=self.histeresis_zona, posibles=posibles,
                                   peso_posible=self.peso_posible,
                                   victimas=self.cantidad_victimas - sum(
                                       c.get('estado') == 'victima' for c in self.candidatos))
            if zona is not None:
                goal = g.mejor_goal(pose[0], pose[1], self.fov, self.alcance, zona=zona, **kw)
        if goal is None:
            goal = g.mejor_goal(pose[0], pose[1], self.fov, self.alcance,
                                radio_local=self.radio_local if self.radio_local > 0 else None,
                                ganancia_local=self.ganancia_local,
                                fraccion_local=self.fraccion_local, **kw)
        cobertura = g.cobertura(pose[0], pose[1])
        if goal is None:
            revisar = self.inspecciones.revisar_dudosos(self.candidatos)
            if revisar:
                # Antes de darse por terminado: otra mirada a los que no se decidieron.
                self.get_logger().info(f'cobertura terminada: se vuelven a mirar los dudosos '
                                       f'{revisar}')
                return
        if goal is not None:
            diag = getattr(g, 'diagnostico', {})
            tipo = ('cobertura por zonas' if 'plan' in diag
                    else 'cobertura local' if diag.get('local') else 'cobertura global')
            self.registrar_decision(pose, tipo, goal, diag)
        if goal is None and not self.terminado_avisado:
            # Por qué no queda nada: para depurar una exploración que termina antes de tiempo.
            self.terminado_avisado = True
            self.get_logger().info(
                f'sin goals desde ({pose[0]:.2f}, {pose[1]:.2f}): cobertura {cobertura:.3f}, '
                f'{json.dumps(getattr(g, "diagnostico", {}))}, '
                f'descartados {len(self.descartados)}, visitados {len(g.visitados)}')
            if self.costmap is not None:
                np.savez_compressed('/tmp/explorador_terminado.npz',
                                    costmap=np.array(self.costmap.data, dtype=np.int8),
                                    info=[self.costmap.info.width, self.costmap.info.height,
                                          self.costmap.info.resolution,
                                          self.costmap.info.origin.position.x,
                                          self.costmap.info.origin.position.y],
                                    visto=g.visto, pose=pose)
        elif goal is not None:
            self.terminado_avisado = False
        self.pub.publish(String(data=json.dumps({
            'goal': None if goal is None else [round(v, 3) for v in goal[:3]],
            'ganancia': None if goal is None else round(goal[3], 1),
            'cobertura': round(cobertura, 3),
            'terminado': goal is None,
        })))
        self.publicar_visto()

    def publicar_visto(self):
        g = self.grilla
        msg = OccupancyGrid()
        msg.header.frame_id = self.marco_mapa
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.info.resolution = float(g.res)
        msg.info.width, msg.info.height = g.forma[1], g.forma[0]
        msg.info.origin.position.x, msg.info.origin.position.y = g.origen
        msg.info.origin.orientation.w = 1.0
        celdas = np.where(g.ocupado, 100, np.where(g.visto, 0, 50)).astype(np.int8)
        msg.data = celdas.ravel().tolist()
        self.pub_visto.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    nodo = NodoExplorador()
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
