"""nodo_mision no va a mirar lo que la cámara ya vio sin rojo (memoria_camara.py)."""
import numpy as np
import pytest
import rclpy
from rclpy.parameter import Parameter

from equipo_jar.cobertura import Cobertura
from equipo_jar.memoria_camara import ROJO
from equipo_jar.nodo_mision import COBERTURA, INSPECCION, NodoMision

RES = 0.05


def sala(ancho_m=4.0, alto_m=3.0):
    """Una sala vacía con paredes alrededor; el origen del mapa es la esquina."""
    grilla = np.zeros((int(alto_m / RES), int(ancho_m / RES)), dtype=np.int16)
    grilla[0, :] = grilla[-1, :] = 100
    grilla[:, 0] = grilla[:, -1] = 100
    return grilla


def caja(cx, cy, lado=0.2):
    s = np.arange(-lado / 2, lado / 2, RES)
    xx, yy = np.meshgrid(cx + s, cy + s)
    return np.array(xx.ravel()), np.array(yy.ravel())


@pytest.fixture
def nodo():
    rclpy.init()
    n = NodoMision()
    n.cobertura = Cobertura(sala(), RES, 0.0, 0.0, (2.0, 1.5), 0.17, 0.30)
    n.t_inicio = n.ahora()
    n.pose = (0.6, 1.5, 0.0)
    n.vistas_camara = np.zeros((n.cobertura.alto, n.cobertura.ancho), dtype=np.int16)
    n.pub_cmd.publish = lambda cmd: None
    yield n
    n.destroy_node()
    rclpy.try_shutdown()


def un_objeto(n, x=2.0, y=1.5):
    """El lidar ve un objeto desde dos lados: devuelve lo que da objetos_nuevos()."""
    wx, wy = caja(x, y)
    for pose in ((0.5, 1.5, 0.0), (3.5, 1.5, 0.0)):
        for _ in range(10):
            n.cobertura.acumular_lidar(pose, wx, wy)
    return n.cobertura.objetos_nuevos()


def la_camara_lo_ve_sin_rojo(n):
    n.vistas_camara[n.cobertura.cuenta > 0] = 5


def la_camara_ve_rojo(n, x=2.0, y=1.5):
    n.vistas_camara[n.cobertura.celda(x, y)] = ROJO


def test_lo_que_la_camara_no_vio_queda_para_inspeccionar(nodo):
    nodo.anotar_objetos(un_objeto(nodo), nodo.ahora())
    assert len(nodo.pendientes) == 1 and nodo.descartados == []


def test_lo_que_la_camara_ya_vio_sin_rojo_no_se_inspecciona(nodo):
    objetos = un_objeto(nodo)
    la_camara_lo_ve_sin_rojo(nodo)
    nodo.anotar_objetos(objetos, nodo.ahora())
    assert nodo.pendientes == [] and len(nodo.descartados) == 1
    nodo.anotar_objetos(objetos, nodo.ahora())  # el mismo objeto otra vez: ya se conoce
    assert len(nodo.descartados) == 1


def test_si_la_camara_lo_ve_despues_se_descarta_y_si_aparece_rojo_vuelve(nodo):
    nodo.anotar_objetos(un_objeto(nodo), nodo.ahora())
    la_camara_lo_ve_sin_rojo(nodo)
    nodo.revisar_objetos(nodo.ahora())
    assert nodo.pendientes == [] and len(nodo.descartados) == 1
    la_camara_ve_rojo(nodo)
    nodo.revisar_objetos(nodo.ahora())
    assert len(nodo.pendientes) == 1 and nodo.descartados == []


def test_con_el_descarte_apagado_se_inspecciona_todo(nodo):
    nodo.set_parameters([Parameter('descartar_con_camara', value=False)])
    objetos = un_objeto(nodo)
    la_camara_lo_ve_sin_rojo(nodo)
    nodo.anotar_objetos(objetos, nodo.ahora())
    assert len(nodo.pendientes) == 1


def test_sin_la_grilla_de_la_camara_se_inspecciona_todo(nodo):
    nodo.vistas_camara = None
    nodo.anotar_objetos(un_objeto(nodo), nodo.ahora())
    assert len(nodo.pendientes) == 1


def test_deja_de_ir_a_mirar_lo_que_la_camara_vio_sin_rojo(nodo):
    un_objeto(nodo)
    nodo.meta = (1.0, 1.5, 0.0, INSPECCION)
    nodo.objeto_meta = (2.0, 1.5)
    nodo.t_meta = nodo.ahora()
    nodo.supervisar_meta(nodo.ahora())
    assert nodo.meta is not None            # todavía no lo vio: sigue yendo
    la_camara_lo_ve_sin_rojo(nodo)
    nodo.supervisar_meta(nodo.ahora())
    assert nodo.meta is None


def test_deja_de_mirar_cuando_ya_es_una_caja_confirmada(nodo):
    nodo.objeto_mirando = (2.0, 1.5)
    nodo.yaw_mirar = 0.0
    nodo.t_mirar_hasta = nodo.ahora() + 4.0
    nodo.mirar(nodo.ahora())
    assert nodo.t_mirar_hasta > 0.0         # sigue mirando
    nodo.confirmadas.append((2.05, 1.5))
    nodo.mirar(nodo.ahora())
    assert nodo.t_mirar_hasta == 0.0 and nodo.objeto_mirando is None


def test_deja_de_mirar_cuando_la_camara_lo_vio_sin_rojo(nodo):
    un_objeto(nodo)
    nodo.objeto_mirando = (2.0, 1.5)
    nodo.yaw_mirar = 0.0
    nodo.t_mirar_hasta = nodo.ahora() + 4.0
    la_camara_lo_ve_sin_rojo(nodo)
    nodo.mirar(nodo.ahora())
    assert nodo.t_mirar_hasta == 0.0


def test_no_elige_inspeccionar_lo_que_la_camara_vio_sin_rojo(nodo):
    nodo.pendientes = list(un_objeto(nodo))
    la_camara_lo_ve_sin_rojo(nodo)
    metas = []
    nodo.ir = lambda *args, **kwargs: metas.append(args[3])
    assert nodo.elegir_meta()
    assert metas == [COBERTURA]             # va a cubrir, no a inspeccionar
    assert nodo.pendientes == [] and len(nodo.descartados) == 1


def test_elige_inspeccionar_lo_que_la_camara_no_vio(nodo):
    nodo.pendientes = list(un_objeto(nodo))
    metas = []
    nodo.ir = lambda *args, **kwargs: metas.append((args[3], kwargs.get('objeto')))
    assert nodo.elegir_meta()
    assert metas[0][0] == INSPECCION and metas[0][1] == pytest.approx((2.0, 1.5), abs=0.1)
