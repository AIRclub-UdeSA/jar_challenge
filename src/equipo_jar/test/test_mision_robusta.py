import math
from types import SimpleNamespace

from action_msgs.msg import GoalStatus
import equipo_jar.nodo_mision as modulo_mision
from equipo_jar.nodo_mision import COBERTURA, NodoMision
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import OccupancyGrid
import pytest
import rclpy
from rclpy.task import Future


@pytest.fixture
def nodo():
    rclpy.init()
    n = NodoMision()
    yield n
    n.destroy_node()
    rclpy.try_shutdown()


def test_un_error_en_el_ciclo_no_mata_el_nodo_y_frena(nodo):
    frenos = []
    nodo.pub_cmd.publish = frenos.append

    def falla():
        raise TypeError('map -> odom no disponible')
    nodo.paso = falla
    nodo.ciclo()  # no tiene que lanzar la excepción
    assert len(frenos) == 1
    assert frenos[0].linear.x == 0.0 and frenos[0].angular.z == 0.0


def test_el_ciclo_llama_a_paso(nodo):
    llamadas = []
    nodo.paso = lambda: llamadas.append(1)
    nodo.ciclo()
    assert llamadas == [1]


@pytest.mark.parametrize('x, y', [(math.nan, 1.0), (1.0, math.inf), (-math.inf, math.nan)])
def test_una_caja_no_finita_no_gasta_un_reporte(nodo, x, y):
    publicados = []
    nodo.pub_victimas.publish = publicados.append
    nodo.t_inicio = nodo.ahora()
    nodo.registrar_caja(x, y)
    assert publicados == []
    assert nodo.reportes == 0
    assert nodo.confirmadas == []


def test_una_caja_valida_se_reporta_una_sola_vez(nodo):
    publicados = []
    nodo.pub_victimas.publish = publicados.append
    nodo.t_inicio = nodo.ahora()
    nodo.registrar_caja(1.0, 2.0)
    nodo.registrar_caja(1.05, 2.0)  # la misma caja
    assert len(publicados) == 1
    assert isinstance(publicados[0], PointStamped)
    assert nodo.reportes == 1


def mapa_libre(lado=20):
    mapa = OccupancyGrid()
    mapa.info.width = mapa.info.height = lado
    mapa.info.resolution = 0.05
    mapa.data = [0] * (lado * lado)
    return mapa


def test_un_mapa_que_hace_fallar_a_cobertura_no_mata_a_mision_y_se_reintenta(nodo, monkeypatch):
    """/map llega una sola vez: si la cobertura fallaba adentro del callback, el nodo moría o el
    robot no arrancaba nunca. Ahora el callback solo guarda el mensaje y paso() lo reintenta."""
    intentos = []
    cobertura_real = modulo_mision.Cobertura

    def falla_la_primera_vez(*args, **kwargs):
        intentos.append(1)
        if len(intentos) == 1:
            raise ValueError('un mapa raro')
        return cobertura_real(*args, **kwargs)
    monkeypatch.setattr(modulo_mision, 'Cobertura', falla_la_primera_vez)
    nodo.pub_cmd.publish = lambda _cmd: None
    nodo.al_recibir_mapa(mapa_libre())
    assert nodo.cobertura is None and intentos == []  # el callback no arma nada
    nodo.ciclo()  # falla, pero no lanza la excepción
    assert nodo.cobertura is None
    nodo.ciclo()  # se reintenta con el mismo mensaje
    assert nodo.cobertura is not None and len(intentos) == 2


def futuro_fallido():
    futuro = Future()
    futuro.set_exception(RuntimeError('el servidor de acciones se cayó'))
    return futuro


def test_un_future_de_nav2_fallido_al_mandar_la_meta_cuenta_como_rechazo(nodo):
    nodo.al_aceptar(futuro_fallido(), nodo.num_meta)  # no lanza la excepción
    assert nodo.meta_rechazada  # supervisar_meta la suelta y reintenta en 1 s


def test_un_future_de_nav2_fallido_de_una_meta_vieja_no_toca_la_actual(nodo):
    nodo.al_aceptar(futuro_fallido(), nodo.num_meta - 1)
    nodo.al_terminar(futuro_fallido(), nodo.num_meta - 1)
    assert not nodo.meta_rechazada and nodo.resultado_meta is None


def test_un_future_de_nav2_fallido_al_terminar_hace_abandonar_la_meta(nodo):
    nodo.al_terminar(futuro_fallido(), nodo.num_meta)  # no lanza la excepción
    # Cualquier resultado que no sea SUCCEEDED hace que supervisar_meta abandone la meta, en vez
    # de esperar el tiempo límite con el robot quieto.
    assert nodo.resultado_meta == GoalStatus.STATUS_UNKNOWN


def test_un_error_que_se_repite_en_paso_suelta_la_meta_el_escape_y_la_inspeccion(nodo):
    nodo.pub_cmd.publish = lambda _cmd: None
    nodo.cobertura = SimpleNamespace(lista_negra=set())
    nodo.meta, nodo.k_meta = (1.0, 2.0, None, COBERTURA), 7
    nodo.escape = object()
    nodo.t_mirar_hasta = 99.0

    def falla():
        raise IndexError('estado inconsistente')
    nodo.paso = falla
    for _ in range(nodo.ciclos_para_resetear - 1):
        nodo.ciclo()
    assert nodo.meta is not None  # todavía no pasaron los 2 s
    nodo.ciclo()
    assert nodo.meta is None
    assert 7 in nodo.cobertura.lista_negra  # no se vuelve a elegir la misma meta
    assert nodo.escape is None and nodo.t_mirar_hasta == 0.0
    assert nodo.ciclos_con_error == 0


def test_un_ciclo_que_anda_reinicia_la_cuenta_de_errores(nodo):
    nodo.pub_cmd.publish = lambda _cmd: None

    def falla():
        raise IndexError('pasajero')
    nodo.paso = falla
    nodo.ciclo()
    nodo.ciclo()
    assert nodo.ciclos_con_error == 2
    nodo.paso = lambda: None
    nodo.ciclo()
    assert nodo.ciclos_con_error == 0
