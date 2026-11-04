"""Pruebas del giro de 360° por pasos sin ejecutar ROS."""

import math

from jar_exploracion.mirada import (MiradaPorPasos, eje_mas_cercano, giro_seguro, normalizar,
                                    radio_de_giro)

DT = 0.1


def simular(mirada, yaw, ahora=0.0, maximo_s=200.0, eficiencia=1.0):
    """Integra la velocidad ordenada; `eficiencia` < 1 imita ruedas que patinan."""
    girado = 0.0
    pausas = 0
    quieto_antes = False
    while mirada.activa and ahora < maximo_s:
        w = mirada.actualizar(yaw, ahora)
        quieto = w == 0.0 and mirada.activa
        pausas += quieto and not quieto_antes
        quieto_antes = quieto
        yaw = normalizar(yaw + eficiencia * w * DT)
        girado += eficiencia * w * DT
        ahora += DT
    return yaw, girado, pausas, ahora


def test_vuelta_completa_termina_alineada_con_el_mapa():
    """Da la vuelta con una pausa por paso y termina en el múltiplo de 90° más cercano."""
    mirada = MiradaPorPasos(pasos=8, pausa_s=0.6, velocidad=0.6)
    mirada.iniciar(1.0, 0.0)
    yaw, girado, pausas, duracion = simular(mirada, 1.0)
    assert mirada.resultado == 'completa'
    assert abs(normalizar(yaw - math.pi / 2)) <= 0.02
    assert abs(girado - (2 * math.pi + math.pi / 2 - 1.0)) < 0.1
    assert pausas == 7
    assert duracion < 25.0


def test_eje_mas_cercano():
    assert eje_mas_cercano(0.3) == 0.0
    assert math.isclose(eje_mas_cercano(1.0), math.pi / 2)
    assert math.isclose(abs(eje_mas_cercano(3.0)), math.pi)
    assert math.isclose(eje_mas_cercano(-0.9), -math.pi / 2)


def test_por_tiempo_se_alinea_en_vez_de_quedar_en_diagonal():
    """Con ruedas que patinan y no completa la vuelta, igual termina alineado."""
    mirada = MiradaPorPasos(pasos=8, pausa_s=0.6, velocidad=0.6, timeout_s=8.0)
    mirada.iniciar(0.0, 0.0)
    yaw, _, _, _ = simular(mirada, 0.0, eficiencia=0.3)
    assert mirada.resultado == 'alineada_por_tiempo'
    assert abs(normalizar(yaw - eje_mas_cercano(yaw))) <= 0.02


def test_cruza_pi_sin_volver_atras():
    """Arrancando cerca de pi, cada paso sigue en sentido antihorario."""
    mirada = MiradaPorPasos(pasos=4, pausa_s=0.2, velocidad=0.6)
    mirada.iniciar(3.0, 0.0)
    yaw, girado, _, _ = simular(mirada, 3.0)
    # Una vuelta más lo que falta para alinearse en pi.
    assert abs(girado - (2 * math.pi + math.pi - 3.0)) < 0.1
    assert abs(normalizar(yaw - math.pi)) <= 0.02


def test_abandona_si_no_puede_girar():
    mirada = MiradaPorPasos(timeout_s=1.0, timeout_alineacion_s=1.0)
    mirada.iniciar(0.3, 0.0)
    # El robot no gira (por ejemplo, trabado): se abandona tras ambos plazos.
    for i in range(30):
        mirada.actualizar(0.3, i * DT)
    assert not mirada.activa
    assert mirada.resultado == 'abandonada'
    assert mirada.actualizar(0.3, 5.0) == 0.0


def test_giro_seguro():
    radio = radio_de_giro(0.37, 0.24, 0.05)
    assert math.isclose(radio, math.hypot(0.185, 0.12) + 0.05)
    assert giro_seguro([(1.0, 0.0)], 0.5, radio)
    assert not giro_seguro([(0.2, 0.1)], 0.5, radio)   # algo dentro del círculo
    assert not giro_seguro([], 0.2, radio)              # pasillo angosto
    assert not giro_seguro([], None, radio)             # fuera del mapa
    assert giro_seguro([], 0.5, radio)
