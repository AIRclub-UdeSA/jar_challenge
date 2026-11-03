"""La pausa de desarrollo solo opera en simulación y después del mapa."""
import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.parameter import Parameter

from equipo_jar.nodo_equipo import NodoEquipo


def test_pausa_no_desactiva_control_real_ni_rompe_espera_mapa():
    rclpy.init()
    node = NodoEquipo()

    class Capture:
        def __init__(self):
            self.messages = []

        def publish(self, message):
            self.messages.append(message)

    capture = Capture()
    node.pub_cmd_vel = capture
    try:
        node.ciclo_de_control()
        assert not capture.messages
        node.al_recibir_mapa(OccupancyGrid())
        node.set_parameters([Parameter('pausar_control_sim', value=True)])
        node.ciclo_de_control()
        assert len(capture.messages) == 1  # El robot real sigue bajo control.
        node.set_parameters([Parameter('use_sim_time', value=True)])
        node.ciclo_de_control()
        assert len(capture.messages) == 1  # Simulación: no competir con el ensayo.
        node.set_parameters([Parameter('pausar_control_sim', value=False)])
        node.ciclo_de_control()
        assert len(capture.messages) == 2
    finally:
        node.destroy_node()
        rclpy.shutdown()
