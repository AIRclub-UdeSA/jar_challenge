"""Le da a AMCL la pose inicial del robot: una gaussiana centrada en el origen del mapa.

El robot arranca cerca del origen (0, 0, 0) del mapa publicado, no exactamente ahí (§1). AMCL
reparte sus partículas según la pose inicial y su covarianza; con covarianza cero (lo que hace el
parámetro `set_initial_pose` de AMCL) todas arrancarían en el mismo punto. Por eso la mandamos
nosotros por /initialpose, con una incertidumbre que sale de params/params.yaml.

Se manda una vez por período, después de recibir /map, hasta que AMCL publica la transformación
map -> odom (señal de que ya se localizó). /initialpose es un topic interno: solo lo usan nuestros
nodos.

Después le pide a AMCL unas cuantas actualizaciones sin movimiento (servicio
request_nomotion_update). AMCL solo corrige la pose cuando el robot se movió `update_min_d` o
`update_min_a`: si el robot arrancó corrido del origen y está quieto, la pose queda mal hasta que
se mueve. Probado con el bag del robot real arrancando a 0,29 m del origen: sin esto, la pose
quedó corrida esos 0,29 m hasta que el robot avanzó; con esto, se corrige antes de moverse.
"""
import math

from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from std_srvs.srv import Empty
from tf2_ros import Buffer, TransformListener

from equipo_jar.ejecucion import main_de
from equipo_jar.localizacion import covarianza_pose_inicial, cuaternion_de_yaw

# /map lo publica el juez con esta QoS (docs/QOS.md).
QOS_MAPA = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


class NodoPoseInicial(Node):
    def __init__(self):
        super().__init__('nodo_pose_inicial')
        self.declare_parameter('sigma_xy_inicial_m', 0.15)
        self.declare_parameter('sigma_yaw_inicial_deg', 10.0)
        self.declare_parameter('periodo_pose_inicial_s', 1.0)
        self.declare_parameter('frame_mapa', 'map')
        self.declare_parameter('frame_odom', 'odom')
        self.declare_parameter('actualizaciones_sin_movimiento', 10)
        self.declare_parameter('periodo_actualizacion_s', 0.3)

        self.mapa_recibido = False
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.sub_mapa = self.create_subscription(
            OccupancyGrid, '/map', self.al_recibir_mapa, QOS_MAPA)
        self.pub_pose = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
        self.cliente_sin_movimiento = self.create_client(Empty, 'request_nomotion_update')
        self.actualizaciones_pedidas = 0
        self.timer = self.create_timer(
            self.get_parameter('periodo_pose_inicial_s').value, self.ciclo)

    def al_recibir_mapa(self, _mapa):
        self.mapa_recibido = True

    def ciclo(self):
        if not self.mapa_recibido:
            return  # sin mapa AMCL no puede localizarse
        mapa = self.get_parameter('frame_mapa').value
        odom = self.get_parameter('frame_odom').value
        if self.tf_buffer.can_transform(mapa, odom, Time()):
            self.get_logger().info(f'AMCL publica {mapa} -> {odom}: localizado')
            self.timer.cancel()
            self.timer = self.create_timer(
                self.get_parameter('periodo_actualizacion_s').value,
                self.pedir_actualizacion_sin_movimiento)
            return
        if self.pub_pose.get_subscription_count() == 0:
            return  # AMCL todavía no arrancó: un mensaje ahora se perdería
        self.pub_pose.publish(self.pose_inicial(mapa))
        self.get_logger().info('Pose inicial enviada a AMCL')

    def pedir_actualizacion_sin_movimiento(self):
        """Que AMCL compare el scan con el mapa aunque el robot no se haya movido."""
        if self.actualizaciones_pedidas >= self.get_parameter(
                'actualizaciones_sin_movimiento').value:
            self.timer.cancel()
            return
        if not self.cliente_sin_movimiento.service_is_ready():
            return
        self.cliente_sin_movimiento.call_async(Empty.Request())
        self.actualizaciones_pedidas += 1

    def pose_inicial(self, frame_mapa):
        """El origen del mapa, con la incertidumbre de params/params.yaml."""
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = frame_mapa
        msg.header.stamp = self.get_clock().now().to_msg()
        q = msg.pose.pose.orientation
        q.x, q.y, q.z, q.w = cuaternion_de_yaw(0.0)
        msg.pose.covariance = covarianza_pose_inicial(
            self.get_parameter('sigma_xy_inicial_m').value,
            math.radians(self.get_parameter('sigma_yaw_inicial_deg').value))
        return msg


def main():
    main_de(NodoPoseInicial)  # una excepción en un callback no mata el nodo (ver ejecucion.py)


if __name__ == '__main__':
    main()
