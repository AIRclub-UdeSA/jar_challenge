"""Comprueba los requisitos de largada e inicialización de AMCL."""
import time

import pytest
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from lifecycle_msgs.msg import State
from lifecycle_msgs.srv import GetState
from nav_msgs.msg import OccupancyGrid
from rclpy.executors import SingleThreadedExecutor

from equipo_jar.inicializar_localizacion import InicializarLocalizacion, QOS_RETENIDO


def test_inicializacion_espera_mapa_y_activacion_y_no_se_repite():
    rclpy.init()
    inicializador = InicializarLocalizacion()
    observador = rclpy.create_node('prueba_inicializacion')
    executor = SingleThreadedExecutor()
    executor.add_node(inicializador)
    executor.add_node(observador)
    mensajes = []
    estado = {'activo': False}

    def consultar(_request, response):
        response.current_state.id = (
            State.PRIMARY_STATE_ACTIVE if estado['activo']
            else State.PRIMARY_STATE_INACTIVE)
        return response

    servicio = observador.create_service(GetState, '/amcl/get_state', consultar)
    sub = observador.create_subscription(
        PoseWithCovarianceStamped, '/equipo/initialpose', mensajes.append, 10)
    mapa_pub = observador.create_publisher(OccupancyGrid, '/map', QOS_RETENIDO)

    def avanzar(duracion, hasta=None):
        limite = time.monotonic() + duracion
        while time.monotonic() < limite:
            executor.spin_once(timeout_sec=0.05)
            if hasta is not None and hasta():
                return
        if hasta is not None:
            assert hasta(), 'No se cumplió la condición dentro del plazo'

    try:
        # Incluso con AMCL activo, sin mapa no se permite inicializar.
        estado['activo'] = True
        avanzar(1.5)
        assert not mensajes

        estado['activo'] = False
        mapa = OccupancyGrid()
        mapa.header.frame_id = 'map'
        mapa_pub.publish(mapa)
        avanzar(3.0, lambda: inicializador.mapa_recibido)
        avanzar(1.5)
        assert not mensajes

        estado['activo'] = True
        avanzar(5.0, lambda: len(mensajes) == 1)
        pose = mensajes[0]
        assert pose.header.frame_id == 'map'
        assert pose.pose.pose.orientation.w == 1.0
        assert pose.pose.covariance[0] == pytest.approx(0.09)
        assert pose.pose.covariance[7] == pytest.approx(0.09)
        assert pose.pose.covariance[35] == pytest.approx(0.1225)
        mapa_pub.publish(mapa)
        avanzar(1.5)
        assert len(mensajes) == 1
    finally:
        observador.destroy_subscription(sub)
        observador.destroy_service(servicio)
        executor.shutdown()
        inicializador.destroy_node()
        observador.destroy_node()
        rclpy.shutdown()
