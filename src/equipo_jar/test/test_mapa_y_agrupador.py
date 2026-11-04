import math

import numpy as np
import pytest

from equipo_jar.agrupador import AgrupadorCajas
from equipo_jar.geometria import componer, de_robot_a_mapa, diferencia, inversa
from equipo_jar.mapa import GrillaMapa


# --- Geometría -----------------------------------------------------------------------------

def test_componer_e_inversa():
    a = (1.0, 2.0, math.pi / 2)
    assert componer(a, inversa(a)) == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)
    # Parado en (1, 2) mirando a +y, un punto 1 m adelante está en (1, 3).
    assert de_robot_a_mapa(a, 1.0, 0.0) == pytest.approx((1.0, 3.0))


def test_diferencia():
    d_m, d_rad = diferencia((0.0, 0.0, 3.1), (3.0, 4.0, -3.1))
    assert d_m == pytest.approx(5.0)
    assert d_rad == pytest.approx(2 * math.pi - 6.2)  # pasando por ±pi, no 6,2 rad


# --- Mapa ----------------------------------------------------------------------------------

def mapa_con_pared():
    """4 x 2 m con resolución 0,1 m, origen en (-1, -1). Pared vertical en x = 1 (columna 20)."""
    ancho, alto = 40, 20
    datos = [0] * (ancho * alto)
    for fila in range(alto):
        datos[fila * ancho + 20] = 100
    datos[0] = -1  # una celda desconocida en la esquina
    return GrillaMapa(ancho, alto, 0.1, -1.0, -1.0, 0.0, datos)


def test_celdas_libres_ocupadas_y_afuera():
    m = mapa_con_pared()
    assert m.libre(0.0, 0.0)
    assert m.ocupada(1.05, 0.0)
    assert not m.libre(-0.95, -0.95)          # desconocida: ni libre ni ocupada
    assert not m.ocupada(-0.95, -0.95)
    assert m.celda(5.0, 0.0) is None          # afuera del mapa
    assert not m.libre(5.0, 0.0)


@pytest.mark.parametrize('yaw', [0.0, math.pi / 2])
def test_celdas_de_muchos_puntos_coinciden_con_celda(yaw):
    m = mapa_con_pared() if yaw == 0.0 else GrillaMapa(20, 20, 0.1, 0.5, -0.5, yaw, [0] * 400)
    xs = np.array([0.0, 1.05, -0.95, 5.0, -1.2, 0.33, -0.7])
    ys = np.array([0.0, 0.0, -0.95, 0.0, 0.3, 0.71, 0.9])
    filas, columnas, adentro = m.celdas_de(xs, ys)
    for k in range(len(xs)):
        esperada = m.celda(xs[k], ys[k])
        assert adentro[k] == (esperada is not None)
        if esperada is not None:
            assert (filas[k], columnas[k]) == esperada


def test_hay_libre_cerca():
    m = mapa_con_pared()
    assert m.hay_libre_cerca(1.05, 0.0, 0.10)       # en la pared, pero pegado a celdas libres
    assert not m.hay_libre_cerca(10.0, 10.0, 0.10)  # afuera del laberinto


def test_linea_de_vista():
    m = mapa_con_pared()
    assert m.linea_de_vista((-0.5, 0.0), (0.8, 0.0), 0.1, 0.1)
    assert not m.linea_de_vista((-0.5, 0.0), (2.0, 0.0), 0.1, 0.1)  # cruza la pared
    # Si la pared está en los últimos cm (la caja contra la pared), no cuenta.
    assert m.linea_de_vista((-0.5, 0.0), (1.1, 0.0), 0.1, 0.2)


def test_mapa_con_origen_girado():
    # El mismo mapa, girado 90°: la pared queda horizontal.
    m = GrillaMapa(40, 20, 0.1, 0.0, 0.0, math.pi / 2, mapa_con_pared().celdas.ravel())
    assert m.ocupada(-0.05, 2.05)


# --- Agrupador -----------------------------------------------------------------------------

def agrupador(**kw):
    args = dict(confirmaciones=5, radio_grupo_m=0.3, dispersion_max_m=0.15, olvido_s=3.0,
                radio_duplicado_m=0.6)
    args.update(kw)
    return AgrupadorCajas(**args)


def test_confirma_con_suficientes_vistas_y_devuelve_la_mediana():
    a = agrupador()
    vistas = [(2.0, 1.0), (2.05, 1.02), (1.97, 0.99), (2.02, 1.01)]
    assert all(a.agregar(p, i * 0.1) is None for i, p in enumerate(vistas))
    centro = a.agregar((2.01, 1.0), 0.4)
    assert centro == pytest.approx((2.01, 1.0))


def test_una_vista_mala_en_el_medio_no_borra_las_buenas():
    a = agrupador()
    for i in range(3):
        a.agregar((2.0, 1.0), i * 0.1)
    assert a.agregar((3.5, -1.0), 0.3) is None   # un error lejos: grupo propio
    a.agregar((2.0, 1.0), 0.4)
    assert a.agregar((2.0, 1.0), 0.5) == pytest.approx((2.0, 1.0))


def test_la_mediana_no_se_mueve_por_una_vista_corrida():
    a = agrupador(confirmaciones=6)
    for i, p in enumerate([(2.0, 1.0)] * 4 + [(2.25, 1.0)]):
        a.agregar(p, i * 0.1)
    assert a.agregar((2.0, 1.0), 0.6) == pytest.approx((2.0, 1.0))


def test_una_caja_confirmada_no_se_vuelve_a_confirmar():
    a = agrupador(confirmaciones=2)
    a.agregar((2.0, 1.0), 0.0)
    assert a.agregar((2.0, 1.0), 0.1) is not None
    for i in range(10):
        assert a.agregar((2.3, 1.2), 1.0 + i * 0.1) is None  # a 0,36 m: la misma caja


def test_las_vistas_viejas_se_olvidan():
    a = agrupador()
    for i in range(4):
        a.agregar((2.0, 1.0), i * 0.1)
    assert a.agregar((2.0, 1.0), 10.0) is None  # pasaron 10 s: empieza de nuevo
    assert len(a.grupos) == 1 and len(a.grupos[0].puntos) == 1


def test_vistas_desparramadas_no_confirman():
    a = agrupador(dispersion_max_m=0.05)
    for i, p in enumerate([(2.0, 1.0), (2.2, 1.0), (1.85, 1.0), (2.0, 1.2), (2.0, 0.85)]):
        assert a.agregar(p, i * 0.1) is None
