"""Una respuesta perdida no debe bloquear ni repetir una transición aplicada."""
import time

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from lifecycle_msgs.msg import State, Transition
from lifecycle_msgs.srv import ChangeState, GetState
from nav_msgs.msg import OccupancyGrid
from rclpy.executors import SingleThreadedExecutor
from rclpy.parameter import Parameter
from rclpy.task import Future

from equipo_jar.inicializar_localizacion import InicializarLocalizacion, QOS_RETENIDO


def test_reconsulta_estado_tras_respuesta_de_configuracion_perdida(monkeypatch):
    rclpy.init()
    node = InicializarLocalizacion(parameter_overrides=[
        Parameter('rpc_timeout_s', value=0.15),
        Parameter('startup_timeout_s', value=8.0)])
    peer = rclpy.create_node('amcl_prueba')
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    executor.add_node(peer)
    state = {'id': State.PRIMARY_STATE_UNCONFIGURED}
    transitions = []
    lost = []
    poses = []

    def get_state(_request, response):
        response.current_state.id = state['id']
        return response

    peer.create_service(GetState, '/amcl/get_state', get_state)
    peer.create_service(ChangeState, '/amcl/change_state', lambda _, response: response)

    def change(request):
        transitions.append(request.transition.id)
        future = Future()
        if request.transition.id == Transition.TRANSITION_CONFIGURE:
            state['id'] = State.PRIMARY_STATE_INACTIVE
            lost.append(future)  # Se aplicó, pero nunca llega la respuesta.
        else:
            state['id'] = State.PRIMARY_STATE_ACTIVE
            future.set_result(ChangeState.Response(success=True))
        return future

    monkeypatch.setattr(node.transiciones, 'call_async', change)
    peer.create_subscription(
        PoseWithCovarianceStamped, '/equipo/initialpose', poses.append, 10)
    publisher = peer.create_publisher(OccupancyGrid, '/map', QOS_RETENIDO)
    publisher.publish(OccupancyGrid())
    try:
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline and not poses:
            executor.spin_once(timeout_sec=0.05)
        assert len(poses) == 1
        assert transitions == [
            Transition.TRANSITION_CONFIGURE, Transition.TRANSITION_ACTIVATE]
        assert lost[0].cancelled()
    finally:
        executor.shutdown()
        node.destroy_node()
        peer.destroy_node()
        rclpy.shutdown()
