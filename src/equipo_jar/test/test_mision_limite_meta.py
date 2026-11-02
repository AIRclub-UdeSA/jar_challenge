"""nodo_mision le da a cada meta un tiempo según el largo de su camino
(cobertura.tiempo_limite_meta), y no la corta antes: con 45 s fijos se abandonaban metas lejanas
mientras el robot iba llegando."""
import numpy as np
import pytest
import rclpy
from rclpy.parameter import Parameter

from equipo_jar.cobertura import Cobertura
from equipo_jar.nodo_mision import COBERTURA, INSPECCION, NodoMision, REGRESO

RES = 0.05


def sala_con_pared():
    """Sala de 4 x 3 m con una pared en x = 2, abierta arriba: de un lado al otro, se rodea."""
    grilla = np.zeros((int(3.0 / RES), int(4.0 / RES)), dtype=np.int16)
    grilla[0, :] = grilla[-1, :] = 100
    grilla[:, 0] = grilla[:, -1] = 100
    grilla[:int(2.2 / RES), int(2.0 / RES)] = 100
    return grilla


@pytest.fixture
def nodo():
    rclpy.init()
    n = NodoMision()
    n.cobertura = Cobertura(sala_con_pared(), RES, 0.0, 0.0, (1.0, 1.0), 0.17, 0.30)
    n.t_inicio = n.ahora()
    n.pose = (1.0, 1.0, 0.0)
    n.pub_cmd.publish = lambda cmd: None
    yield n
    n.destroy_node()
    rclpy.try_shutdown()


def test_una_meta_cercana_tiene_el_tiempo_minimo(nodo):
    nodo.ir(1.5, 1.0, None, REGRESO, distancia=0.5)
    assert nodo.limite_meta == pytest.approx(45.0)


def test_una_meta_lejana_tiene_tiempo_segun_su_camino(nodo):
    nodo.set_parameters([Parameter('velocidad_min_meta_mps', value=0.05)])
    nodo.ir(3.0, 1.0, None, REGRESO, distancia=4.0)
    assert nodo.limite_meta == pytest.approx(80.0)


def test_sin_largo_de_camino_tiene_el_minimo(nodo):
    nodo.ir(3.0, 1.0, None, REGRESO)
    assert nodo.limite_meta == pytest.approx(45.0)


def test_no_abandona_una_meta_lejana_antes_de_su_limite(nodo):
    nodo.ir(3.0, 1.0, None, REGRESO, distancia=10.0)      # 100 s, a 0,10 m/s
    nodo.t_meta = nodo.ahora() - 60.0
    nodo.supervisar_meta(nodo.ahora())
    assert nodo.meta is not None
    nodo.t_meta = nodo.ahora() - 101.0
    nodo.supervisar_meta(nodo.ahora())
    assert nodo.meta is None


def test_la_cobertura_elegida_lleva_el_largo_de_su_camino(nodo):
    nodo.set_parameters([Parameter('velocidad_min_meta_mps', value=0.01)])
    assert nodo.elegir_meta()
    assert nodo.meta[3] == COBERTURA
    esperado = nodo.cobertura.distancia_a(nodo.cobertura.distancias_desde(1.0, 1.0),
                                          *nodo.meta[:2])
    assert nodo.distancia_meta == pytest.approx(esperado)
    assert nodo.limite_meta == pytest.approx(max(45.0, esperado / 0.01))


def test_la_inspeccion_lleva_el_largo_de_su_camino(nodo):
    nodo.pendientes = [(3.0, 1.0)]                         # del otro lado de la pared
    assert nodo.elegir_meta()
    assert nodo.meta[3] == INSPECCION
    assert nodo.distancia_meta is not None and nodo.distancia_meta > 2.0


def test_el_regreso_lleva_el_largo_de_su_camino(nodo):
    nodo.estado = REGRESO
    nodo.largada = (3.0, 1.0)
    nodo.planificar(nodo.ahora())
    assert nodo.meta[3] == REGRESO
    assert nodo.distancia_meta > 3.5                       # rodea la pared: no son los 2 m en recta


def test_al_retomar_la_meta_despues_de_un_escape_conserva_su_limite(nodo):
    nodo.ir(3.0, 1.0, None, REGRESO, distancia=10.0)
    nodo.atender_traba(nodo.ahora())                       # primera traba: se retoma después
    assert nodo.meta is None and nodo.escape is not None
    nodo.scan = None                                       # sin lidar, el escape termina enseguida
    nodo.seguir_escape(nodo.ahora())
    assert nodo.meta is not None and nodo.meta[3] == REGRESO
    assert nodo.distancia_meta == 10.0
    assert nodo.limite_meta == pytest.approx(100.0)
