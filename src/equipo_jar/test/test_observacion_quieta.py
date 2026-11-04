"""El pedido de actualización quieta debe llegar al servidor interno."""
import time

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.executors import SingleThreadedExecutor
from std_srvs.srv import Empty

from equipo_jar.nodo_equipo import NodoEquipo, QOS_MAPA


def test_solicita_observacion_quieta_y_recibe_pose_nueva():
    rclpy.init()
    node = NodoEquipo()
    peer = rclpy.create_node('amcl_quieto_prueba')
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    executor.add_node(peer)
    calls = []
    publisher = peer.create_publisher(
        PoseWithCovarianceStamped, '/equipo/amcl_pose', QOS_MAPA)

    def update(_request, response):
        calls.append(time.monotonic())
        message = PoseWithCovarianceStamped()
        message.header.frame_id = 'map'
        message.header.stamp = peer.get_clock().now().to_msg()
        message.pose.pose.orientation.w = 1.0
        publisher.publish(message)
        return response

    peer.create_service(Empty, '/equipo/request_nomotion_update', update)
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and node.secuencia_pose == 0:
            node.solicitar_observacion_quieta(time.monotonic())
            executor.spin_once(timeout_sec=0.05)
        assert calls
        assert node.secuencia_pose > 0
        assert node.pose.header.frame_id == 'map'
    finally:
        executor.shutdown()
        node.destroy_node()
        peer.destroy_node()
        rclpy.shutdown()


def test_giro_detenido_por_pose_vencida_solicita_refresco_sin_habilitar_movimiento():
    from rclpy.parameter import Parameter
    rclpy.init();node=NodoEquipo()
    try:
        node.set_parameters([Parameter('localizacion_activa',value=True)])
        node.mapa=object()
        node.control.estado='GIRANDO'
        node.observacion=lambda ahora:None
        node.paso_navegacion=lambda ahora,observacion:None
        node.control.paso=lambda *args:0.
        calls=[]
        node.solicitar_observacion_quieta=lambda ahora:calls.append(ahora)
        node.ciclo_de_control()
        assert len(calls)==1
        assert node.control.estado=='GIRANDO'
        # Mientras se está ordenando giro, no pedir actualización quieta.
        node.control.paso=lambda *args:.15
        node.ciclo_de_control()
        assert len(calls)==1
    finally:
        node.destroy_node();rclpy.shutdown()
