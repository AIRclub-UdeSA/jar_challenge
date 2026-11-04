"""El nodo configura la histéresis de M1 y publica si la localización es utilizable."""
import json

import rclpy
from nav_msgs.msg import OccupancyGrid

from equipo_jar.nodo_equipo import NodoEquipo


class Captura:
    def __init__(self):
        self.mensajes = []

    def publish(self, mensaje):
        self.mensajes.append(mensaje)


def test_parametros_de_histeresis_llegan_al_control():
    rclpy.init()
    node = NodoEquipo()
    try:
        assert node.control.plazo_falta_datos == 2.0
        assert node.control.plazo_falla_calidad == 1.0
        assert node.valor('deficit_obstaculo_m') == 0.40
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_estado_publica_utilizable_y_ambas_concordancias():
    rclpy.init()
    node = NodoEquipo()
    estado = Captura()
    node.pub_estado = estado
    node.pub_cmd_vel = Captura()
    try:
        node.al_recibir_mapa(OccupancyGrid())
        node.ciclo_de_control()
        publicado = json.loads(estado.mensajes[-1].data)
        assert publicado['utilizable'] is False
        assert 'concordancia_simple' in publicado
    finally:
        node.destroy_node()
        rclpy.shutdown()
