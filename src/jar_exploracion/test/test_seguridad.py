"""Regresiones geométricas y de control para pasillos y obstáculos."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from jar_exploracion.planificador import a_estrella
from jar_exploracion.seguridad import (
    buscar_desvio, celda_libre_cercana, distancia_huella, escape_seguro, espacio_transitable,
    puntos_scan, regular_traslacion)


@pytest.mark.parametrize('ancho_pasillo, admisible', [
    (0.28, False), (0.32, False), (0.38, True), (0.5, True)])
def test_planificar_pasillo_segun_huella(ancho_pasillo, admisible):
    """Admite pasos que caben con margen y rechaza pasos físicamente insuficientes."""
    res = 0.01
    filas = round(ancho_pasillo / res)
    datos = np.zeros((filas + 2, 260), dtype=int)
    datos[0, :] = datos[-1, :] = 100
    libres = espacio_transitable(datos, res, 0.37, 0.24, 0.05)
    camino = a_estrella(libres, (filas // 2, 30), (filas // 2, 230))
    assert (camino is not None) == admisible


def test_huella_orientacion_y_bordes():
    """La orientación importa: no presupone giros en un pasillo estrecho."""
    datos = np.zeros((260, 40), dtype=int)
    datos[:, 0] = datos[:, -1] = 100
    lateral = espacio_transitable(datos, 0.01, 0.37, 0.24, 0.05, 0.)
    alineado = espacio_transitable(datos, 0.01, 0.37, 0.24, 0.05, math.pi / 2)
    assert not lateral[130, 20]
    assert alineado[130, 20]
    assert not alineado[0, 20]  # El exterior nunca se interpreta como espacio libre.
    datos[130, 20] = -1
    assert not espacio_transitable(datos, 0.01, 0.37, 0.24, 0.05, math.pi / 2)[130, 20]


def paredes(ancho=0.38):
    """Impactos de dos paredes paralelas en el marco de la base."""
    x = np.linspace(-1., 1., 161)
    return np.column_stack((np.r_[x, x], np.r_[np.full(len(x), ancho / 2),
                                               np.full(len(x), -ancho / 2)]))


def test_avanzar_sin_repulsion_ni_dependencia_de_numero_rayos():
    """Las paredes laterales no cancelan el avance ni lo empujan de lado."""
    puntos = paredes()
    for nube in (puntos, np.repeat(puntos, 5, axis=0)):
        (vx, vy), motivo = regular_traslacion(
            (0.3, 0.), nube, 0.37, 0.24, 0.05, 0.3, lambda x, y: True)
        assert 0.05 < vx <= 0.10
        assert vy == 0.
        assert motivo == 'avance'


def test_obstaculo_frontal_y_orden_que_corta_esquina():
    """Frena ante colisión frontal y no permite que el comando abandone el mapa seguro."""
    (vx, vy), _ = regular_traslacion(
        (0.3, 0.), np.array([[0.22, 0.]]), 0.37, 0.24, 0.05, 0.3, lambda x, y: True)
    assert vx == vy == 0.
    velocidad, motivo = regular_traslacion(
        (0.3, 0.1), paredes(0.8), 0.37, 0.24, 0.05, 0.3, lambda x, y: x == 0)
    assert velocidad == (0., 0.)
    assert motivo == 'obstaculo_o_margen'


def test_no_salta_obstaculo_entre_muestras():
    """El barrido detecta un impacto fuera de la pose inicial."""
    puntos = np.array([[0.4, 0.]])
    (vx, _), _ = regular_traslacion(
        (0.3, 0.), puntos, 0.37, 0.24, 0.05, 0.3, lambda x, y: True)
    assert vx < 0.3
    assert 0.185 + 0.05 + vx * 0.8 <= 0.4 + 1e-6


@pytest.mark.parametrize('valores', [[], [math.nan], [-math.inf], [0.0], [-1.0]])
def test_scan_sin_evidencia_frena(valores):
    """Un sensor sin datos válidos no demuestra que no haya obstáculos."""
    scan = SimpleNamespace(ranges=valores, range_min=0.05, range_max=10.,
                           angle_min=0., angle_increment=0.1)
    assert puntos_scan(scan) is None
    assert regular_traslacion(
        (0.3, 0.), None, 0.37, 0.24, 0.05, 0.3, lambda x, y: True)[0] == (0., 0.)


def test_scan_transformado_y_sin_retorno():
    """Respeta montaje y giro del láser; infinito positivo significa sin retorno."""
    scan = SimpleNamespace(ranges=[1., math.nan, math.inf], range_min=0.05,
                           range_max=10., angle_min=0., angle_increment=0.1)
    puntos = puntos_scan(scan, 0.043, 0., math.pi)
    assert puntos.shape == (1, 2)
    assert puntos[0] == pytest.approx((-0.957, 0.))
    scan.ranges = [math.inf]
    assert puntos_scan(scan).shape == (0, 2)


def test_distancia_caja_y_validacion():
    """Distingue impactos dentro, fuera y sobre la huella."""
    distancias = distancia_huella(np.array([[0., 0.], [0.2, 0.], [0.1, 0.1]]), 0.2, 0.2)
    assert distancias == pytest.approx([-0.1, 0.1, 0.0])
    with pytest.raises(ValueError):
        espacio_transitable(np.zeros((4, 4)), 0., 0.37, 0.24, 0.05)


def test_recorrido_cerrado_por_pasillo():
    """El integrador recorre dos metros sin tocar las paredes de un paso de 38 cm."""
    posicion = np.array([0., 0.])
    for _ in range(240):
        restante = 2.0 - posicion[0]
        velocidad, _ = regular_traslacion(
            (min(0.3, restante * 0.6), 0.), paredes(), 0.37, 0.24, 0.05, 0.3,
            lambda x, y: abs(posicion[1] + y) <= 0.38 / 2 - 0.12 - 0.05)
        posicion += np.array(velocidad) * 0.1
        assert abs(posicion[1]) + 0.12 + 0.05 < 0.38 / 2
    assert posicion[0] > 1.95


def test_subdivision_encuentra_centro_sin_adelgazar_paredes():
    """El subpíxel permite centrar el chasis en un paso que cae entre centros originales."""
    datos = np.zeros((10, 60), dtype=int)
    datos[0, :] = datos[-1, :] = 100  # 40 cm de área libre a 5 cm/píxel.
    original = espacio_transitable(datos, 0.05, 0.37, 0.24, 0.056)
    assert not original[:, 30].any()
    fino = np.repeat(np.repeat(datos, 2, axis=0), 2, axis=1)
    subdividido = espacio_transitable(fino, 0.025, 0.37, 0.24, 0.056)
    assert subdividido[9, 60] or subdividido[10, 60]
    assert a_estrella(subdividido, (9, 20), (9, 100)) is not None
    assert np.count_nonzero(fino == 100) * 0.025**2 == pytest.approx(
        np.count_nonzero(datos == 100) * 0.05**2)


def test_celda_libre_cercana():
    libre = np.zeros((10, 10), dtype=bool)
    libre[2, 8] = libre[9, 9] = True
    assert celda_libre_cercana(libre, (2, 5), 4) == (2, 8)
    assert celda_libre_cercana(libre, (2, 5), 2) is None
    assert celda_libre_cercana(np.zeros((5, 5), dtype=bool), (2, 2), 3) is None


def test_escape_seguro_no_acerca_el_chasis_al_scan():
    """Alejarse de una pared vale; ir hacia ella, no."""
    pared = np.array([[0.0, -0.13], [0.1, -0.13], [-0.1, -0.13]])
    assert escape_seguro(pared, 0.37, 0.24, (0.0, 0.02))
    assert not escape_seguro(pared, 0.37, 0.24, (0.0, -0.02))
    assert escape_seguro(np.zeros((0, 2)), 0.37, 0.24, (0.02, 0.0))


def test_desvio_esquiva_una_pared_que_corta_el_avance():
    """Avanzando en +x con una esquina adelante a la izquierda, se desvía a la derecha."""
    # Ya a 1.5 cm del frente (menos que el margen de 2 cm).
    esquina = np.array([[x, y] for x in np.arange(0.20, 0.6, 0.01) for y in (0.10, 0.13)])
    bloqueada = regular_traslacion((0.1, 0.0), esquina, 0.37, 0.24, 0.02, 0.1,
                                   lambda dx, dy: True)
    assert bloqueada[1] == 'obstaculo_o_margen'
    desvio = buscar_desvio((0.1, 0.0), esquina, 0.37, 0.24, 0.02, 0.1)
    assert desvio is not None
    (vx, vy), grados = desvio
    assert grados < 0 and vy < 0 and vx >= -1e-9


def test_sin_desvio_si_esta_encerrado():
    # Caja apenas mayor que la huella: cualquier movimiento acerca algún lado.
    lados = np.linspace(-1, 1, 41)
    pared = np.array([[0.21 * u, 0.14 * w] for u in lados for w in (-1, 1)]
                     + [[0.21 * w, 0.14 * u] for u in lados for w in (-1, 1)])
    assert buscar_desvio((0.1, 0.0), pared, 0.37, 0.24, 0.02, 0.1) is None
