"""Guardia de barrido de huella (portada del banco de M1) y límite de velocidad."""
import math

import pytest

from equipo_jar.navegacion.guardia import (STOPPING_RESERVE_M, comprobar_barrido,
                                           huella_desde_parametros, limitar_velocidad,
                                           puntos_desde_scan)

H = huella_desde_parametros(-0.19, 0.13, 0.12, 0.10)
FONDO = [(3.0, 0.0)] * 30
DECEL, LATENCIA = 0.5, 0.15


def pared_frontal(x):
    return [(x, -0.3 + 0.6 * i / 40) for i in range(41)]


def test_huella_desde_parametros():
    assert sorted(H['corners_base_m']) == [(-0.19, -0.12), (-0.19, 0.12),
                                           (0.13, -0.12), (0.13, 0.12)]
    assert H['margin_m'] == 0.10
    assert H['radius_m'] == pytest.approx(math.hypot(0.19, 0.12))


@pytest.mark.parametrize('valores', [(0.1, -0.1, 0.12, 0.1), (-0.19, 0.13, 0.0, 0.1),
                                     (-0.19, 0.13, 0.12, -0.1),
                                     (-0.19, float('nan'), 0.12, 0.1)])
def test_huella_invalida_falla(valores):
    with pytest.raises(ValueError):
        huella_desde_parametros(*valores)


def test_pared_lateral_a_32cm_no_ocupa_el_barrido():
    for x in (0.02, 0.06, 0.2, 0.5):
        assert comprobar_barrido(FONDO + [(x, 0.32)], 0.0, 0.4, H)['clear']


def test_obstaculo_frontal_dentro_del_recorrido_detiene():
    assert not comprobar_barrido(FONDO + [(0.4, 0.0)], 0.0, 0.4, H)['clear']


def test_obstaculo_lateral_proximo_detiene():
    assert not comprobar_barrido(FONDO + [(0.1, 0.20)], 0.0, 0.4, H)['clear']


def test_obstaculo_trasero_no_se_ignora_al_volver():
    puntos = FONDO + [(-0.45, 0.0)]
    assert comprobar_barrido(puntos, 0.0, 0.3, H)['clear']
    assert not comprobar_barrido(puntos, 0.0, -0.3, H)['clear']


def test_diagonal_barre_huella_rotada():
    assert not comprobar_barrido(FONDO + [(0.4, 0.4)], math.pi / 4, 0.5, H)['clear']


def test_obstaculos_laterales_entre_extremos_del_barrido_detienen():
    assert not comprobar_barrido(FONDO + [(0.2, 0.16), (0.3, 0.18)], 0.0, 0.4, H)['clear']


def test_pared_a_0_35_m_escala_0_30_y_deja_pasar_0_10():
    puntos = FONDO + pared_frontal(0.13 + 0.35)
    vx, vy, diag = limitar_velocidad(puntos, 0.30, 0.0, H, DECEL, LATENCIA)
    assert 0.0 < vx < 0.30 and vy == 0.0 and not diag['bloqueado']
    vx, vy, _ = limitar_velocidad(puntos, 0.10, 0.0, H, DECEL, LATENCIA)
    assert (vx, vy) == (0.10, 0.0)


def test_pared_muy_cerca_bloquea():
    vx, vy, diag = limitar_velocidad(FONDO + pared_frontal(0.13 + 0.15), 0.30, 0.0, H,
                                     DECEL, LATENCIA)
    assert (vx, vy) == (0.0, 0.0) and diag['bloqueado']


def test_pared_frontal_no_bloquea_avance_lateral():
    puntos = FONDO + pared_frontal(0.13 + 0.15)
    vx, vy, diag = limitar_velocidad(puntos, 0.0, 0.20, H, DECEL, LATENCIA)
    assert (vx, vy) == (0.0, 0.20) and not diag['bloqueado']


def test_pared_lateral_no_bloquea_avance():
    puntos = FONDO + [(x / 10, 0.32) for x in range(-5, 15)]
    assert limitar_velocidad(puntos, 0.30, 0.0, H, DECEL, LATENCIA)[:2] == (0.30, 0.0)


def test_comando_nulo_no_bloquea():
    assert limitar_velocidad([], 0.0, 0.0, H, DECEL, LATENCIA)[:2] == (0.0, 0.0)
    assert not limitar_velocidad([], 0.0, 0.0, H, DECEL, LATENCIA)[2]['bloqueado']


@pytest.mark.parametrize('puntos', [[], [(1.0, 0.0)] * 29, FONDO + [(float('nan'), 0.0)]])
def test_datos_insuficientes_o_nan_bloquean(puntos):
    vx, vy, diag = limitar_velocidad(puntos, 0.2, 0.0, H, DECEL, LATENCIA)
    assert (vx, vy) == (0.0, 0.0) and diag['bloqueado']


def test_reserva_de_detencion_es_la_del_banco():
    assert STOPPING_RESERVE_M == 0.10


def test_rayos_inf_se_descartan_antes_de_la_guardia():
    rangos = [float('inf')] * 300 + [2.0] * 60 + [float('nan'), 0.01, 12.0]
    puntos = puntos_desde_scan(rangos, -math.pi, 2 * math.pi / 363, 0.05, 12.0,
                               (0.0, 0.0, 0.0))
    assert len(puntos) == 60
    assert all(math.isfinite(v) for p in puntos for v in p)
    assert not limitar_velocidad(puntos, 0.1, 0.0, H, DECEL, LATENCIA)[2]['bloqueado']


def test_puntos_desde_scan_aplica_montaje_del_laser():
    puntos = puntos_desde_scan([1.0], 0.0, 0.1, 0.05, 12.0, (0.1, 0.0, math.pi / 2))
    assert puntos[0] == pytest.approx((0.1, 1.0))
