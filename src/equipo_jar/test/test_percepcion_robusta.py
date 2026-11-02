"""Una imagen que cv_bridge no puede convertir no deja a percepción sin cámara el resto de la
corrida: se descarta y se sigue con la próxima."""
from equipo_jar.nodo_percepcion import NodoPercepcion
import pytest
import rclpy
from sensor_msgs.msg import Image


@pytest.fixture
def nodo():
    rclpy.init()
    n = NodoPercepcion()
    yield n
    n.destroy_node()
    rclpy.try_shutdown()


def imagen(datos, encoding='rgb8', ancho=4, alto=3):
    msg = Image()
    msg.encoding = encoding
    msg.width, msg.height = ancho, alto
    msg.step = ancho * 3
    msg.data = list(datos)
    return msg


@pytest.mark.parametrize('msg', [
    imagen([0] * 5),                        # datos cortados
    imagen([0] * 36, encoding='raro'),      # encoding que cv_bridge no conoce
])
def test_una_imagen_que_hace_fallar_a_cv_bridge_se_descarta(nodo, msg):
    nodo.al_recibir_imagen(msg)  # no lanza la excepción
    assert nodo.imagen is None and nodo.stamp_imagen is None


def test_despues_de_una_imagen_rota_se_usa_la_siguiente(nodo):
    nodo.al_recibir_imagen(imagen([0] * 5))
    nodo.al_recibir_imagen(imagen([255, 0, 0] * 12))
    assert nodo.imagen.shape == (3, 4, 3)
    assert tuple(nodo.imagen[0, 0]) == (0, 0, 255)  # rgb8 pasado a bgr8
