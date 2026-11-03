"""Una imagen que cv_bridge no puede convertir no deja a percepción sin cámara el resto de la
corrida: se descarta y se sigue con la próxima. Un /map que no se puede armar tampoco la deja sin
mapa: se arma en paso() y se reintenta."""
from equipo_jar import nodo_percepcion
from equipo_jar.nodo_percepcion import NodoPercepcion
from nav_msgs.msg import OccupancyGrid
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


def mapa(ancho=10, alto=8, datos=None):
    msg = OccupancyGrid()
    msg.header.frame_id = 'map'
    msg.info.width, msg.info.height, msg.info.resolution = ancho, alto, 0.05
    msg.info.origin.orientation.w = 1.0
    msg.data = [0] * (ancho * alto) if datos is None else datos
    return msg


def test_el_mapa_se_arma_en_el_ciclo_y_no_en_el_callback(nodo):
    nodo.al_recibir_mapa(mapa())
    assert nodo.mapa is None  # el callback solo guarda el mensaje
    nodo.ciclo()
    assert (nodo.mapa.alto, nodo.mapa.ancho) == (8, 10)
    assert nodo.mensaje_mapa is None


def test_un_mapa_que_no_se_puede_armar_no_lanza_en_el_callback(nodo):
    nodo.al_recibir_mapa(mapa(datos=[0] * 5))  # datos cortados: no coinciden con 10 x 8
    nodo.ciclo()  # el error lo atrapa ciclo()
    assert nodo.mapa is None


def test_si_armar_el_mapa_falla_se_reintenta_con_el_mismo_mensaje(nodo, monkeypatch):
    original = nodo_percepcion.GrillaMapa.desde_mensaje
    llamadas = []

    def falla_la_primera_vez(msg):
        llamadas.append(msg)
        if len(llamadas) == 1:
            raise MemoryError('falla pasajera')
        return original(msg)
    monkeypatch.setattr(nodo_percepcion.GrillaMapa, 'desde_mensaje', falla_la_primera_vez)

    nodo.al_recibir_mapa(mapa())  # /map llega una sola vez
    nodo.ciclo()
    assert nodo.mapa is None
    nodo.ciclo()
    assert nodo.mapa is not None and nodo.info_mapa.width == 10
    assert len(llamadas) == 2 and llamadas[0] is llamadas[1]
