import math

import numpy as np
import pytest

from equipo_jar.escape import (
    Cuerpo, despeje_actual, DetectorTraba, distancia_al_cuerpo, elegir_direccion, Escape,
    puntos_en_robot, recorrido_libre)

CUERPO = Cuerpo(frente=0.13, atras=0.195, semiancho=0.125)
MARGEN = 0.02
ALCANCE = 0.25


def pared(x=None, y=None):
    """Puntos de una pared recta de 2 m en x=cte o y=cte (marco del robot)."""
    s = np.arange(-1.0, 1.0, 0.02)
    if x is not None:
        return np.full_like(s, x), s
    return s, np.full_like(s, y)


def juntar(*paredes):
    return (np.concatenate([p[0] for p in paredes]), np.concatenate([p[1] for p in paredes]))


def test_lidar_montado_al_reves():
    # Girado 180° y corrido 0,043 m: un rayo al ángulo 0 mira hacia atrás del robot.
    px, py = puntos_en_robot([1.0], 0.0, 0.01, 0.05, 3.5, (0.043, 0.0, -1.0, 0.0, 0.0, -1.0))
    assert px[0] == pytest.approx(-0.957)
    assert py[0] == pytest.approx(0.0, abs=1e-9)


def test_descarta_lecturas_invalidas():
    px, _ = puntos_en_robot(
        [math.inf, math.nan, 0.01, 9.0, 1.0], 0.0, 0.1, 0.05, 3.5, (0, 0, 1, 0, 0, 1))
    assert len(px) == 1


def test_distancia_al_cuerpo_es_asimetrica():
    d = distancia_al_cuerpo(np.array([0.23, -0.295, 0.0]), np.array([0.0, 0.0, 0.225]), CUERPO)
    assert d == pytest.approx([0.10, 0.10, 0.10])


def test_un_punto_dentro_del_cuerpo_no_cuenta():
    px, py = np.array([0.0, 0.5]), np.array([0.0, 0.0])
    assert despeje_actual(px, py, CUERPO) == pytest.approx(0.37)


def test_no_se_puede_avanzar_contra_la_pared():
    px, py = pared(x=0.16)
    assert recorrido_libre(px, py, 1.0, 0.0, CUERPO, MARGEN, ALCANCE) <= 0.01


def test_estando_mas_cerca_que_el_margen_puede_alejarse():
    px, py = pared(x=0.14)
    assert recorrido_libre(px, py, -1.0, 0.0, CUERPO, MARGEN, ALCANCE) == pytest.approx(ALCANCE)


def test_con_pared_al_costado_se_aleja_de_costado():
    d = elegir_direccion(*pared(y=0.16), CUERPO, MARGEN, ALCANCE)
    assert d.uy == pytest.approx(-1.0)
    assert d.despeje > 0.25


def test_con_pared_adelante_retrocede():
    d = elegir_direccion(*pared(x=0.16), CUERPO, MARGEN, ALCANCE)
    assert d.ux == pytest.approx(-1.0)


def test_en_una_esquina_sale_en_diagonal():
    d = elegir_direccion(*juntar(pared(x=0.16), pared(y=0.16)), CUERPO, MARGEN, ALCANCE)
    assert d.ux < -0.5 and d.uy < -0.5


def test_en_un_pasillo_angosto_queda_centrado():
    # 0,03 m de un lado y 0,20 m del otro: lo mejor es ir hasta el medio, no contra la otra pared.
    d = elegir_direccion(*juntar(pared(y=0.155), pared(y=-0.325)), CUERPO, MARGEN, ALCANCE)
    assert d.uy < -0.5
    assert d.despeje == pytest.approx(0.115, abs=0.01)
    assert d.libre > d.avance


def test_sin_obstaculos_no_falla():
    assert elegir_direccion(np.array([]), np.array([]), CUERPO, MARGEN, ALCANCE) is not None


def simular(px, py, ahora=0.0, duracion=6.0, dt=0.1):
    """Corre un escape moviendo los puntos como si el robot avanzara."""
    escape = Escape(CUERPO, 0.10, 0.15, MARGEN, ALCANCE, duracion, ahora)
    pasos = 0
    while True:
        comando = escape.comando(px, py, ahora)
        if comando is None:
            return px, py, pasos
        px, py = px - comando[0] * dt, py - comando[1] * dt
        ahora += dt
        pasos += 1


def test_el_escape_termina_despejado():
    px, py, pasos = simular(*pared(y=0.16))
    assert despeje_actual(px, py, CUERPO) >= 0.15
    assert pasos > 0


def test_el_escape_nunca_acerca_al_robot():
    # Pasillo donde no se llega al despeje objetivo: tiene que quedar en el medio y terminar.
    px, py, _ = simular(*juntar(pared(y=0.155), pared(y=-0.325)))
    assert despeje_actual(px, py, CUERPO) >= 0.05


def test_ya_despejado_no_se_mueve():
    escape = Escape(CUERPO, 0.10, 0.15, MARGEN, ALCANCE, 6.0, 0.0)
    assert escape.comando(*pared(y=0.40), 0.0) is None


def test_el_escape_respeta_el_tiempo_maximo():
    escape = Escape(CUERPO, 0.10, 0.15, MARGEN, ALCANCE, 6.0, 0.0)
    assert escape.comando(*pared(y=0.16), 6.1) is None


def test_traba_tarda_lo_que_tiene_que_tardar():
    t = DetectorTraba(0.08, 0.3, 6.0, 0.15, 15.0)
    assert not t.actualizar(0.0, 0.0, 0.0, 0.0)
    assert not t.actualizar(0.0, 0.0, 0.0, 5.0)
    assert t.actualizar(0.0, 0.0, 0.0, 6.1)


def test_moverse_reinicia_la_traba():
    t = DetectorTraba(0.08, 0.3, 6.0, 0.15, 15.0)
    t.actualizar(0.0, 0.0, 0.0, 0.0)
    assert not t.actualizar(0.2, 0.0, 0.0, 5.0)
    assert not t.actualizar(0.2, 0.0, 0.0, 10.0)


def test_girar_sin_avanzar_termina_siendo_traba():
    t = DetectorTraba(0.08, 0.3, 6.0, 0.15, 15.0)
    for segundo in range(16):
        assert not t.actualizar(0.0, 0.0, segundo * 1.0, float(segundo))
    assert t.actualizar(0.0, 0.0, 16.0, 16.0)


def test_girar_tambien_cuenta_como_moverse():
    t = DetectorTraba(0.08, 0.3, 6.0, 0.15, 15.0)
    t.actualizar(0.0, 0.0, 0.0, 0.0)
    assert not t.actualizar(0.0, 0.0, 0.5, 7.0)
