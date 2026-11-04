"""El parámetro ROS de supervisión no modifica el umbral de adquisición inicial."""
import rclpy
from rclpy.parameter import Parameter
from equipo_jar.nodo_equipo import NodoEquipo


def test_parametro_ros_llega_a_control_con_inicial_estricta():
    rclpy.init()
    node = None
    try:
        node = NodoEquipo(parameter_overrides=[Parameter('concordancia_min_supervision', value=.72)])
        assert node.control.concordancia_min_supervision == .72
        assert node.control.concordancia_min == .80
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()
