import math

import pytest

from equipo_jar.evasion import (
    distancia_en_sector, giro_para_seguir_pared, normalizar_angulo, sentido_de_giro)

# Un lidar de 360 rayos, uno por grado, desde -pi. Con frente_deg=180 el frente del robot
# cae en el rayo del ángulo pi (que es el mismo que -pi: el rayo 0).
INCREMENTO = math.radians(1.0)
ANGULO_MIN = -math.pi


def scan(**obstaculos):
    """Un scan libre (inf en todos lados) con obstáculos en algunos rayos: g<rayo>=<distancia>."""
    rangos = [math.inf] * 360
    for rayo, distancia in obstaculos.items():
        rangos[int(rayo.removeprefix('g'))] = distancia
    return rangos


def en_sector(rangos, centro_deg, ancho_deg=40.0, frente_deg=180.0):
    return distancia_en_sector(rangos, ANGULO_MIN, INCREMENTO, frente_deg, centro_deg, ancho_deg)


def test_sin_obstaculos_no_ve_nada():
    assert en_sector(scan(), 0.0) == math.inf


def test_obstaculo_adelante():
    # El rayo 0 es -pi, lo mismo que +pi: el frente cuando frente_deg=180.
    assert en_sector(scan(g0=0.4), 0.0) == pytest.approx(0.4)


def test_toma_el_mas_cercano_del_sector():
    assert en_sector(scan(g0=0.9, g5=0.4, g355=1.5), 0.0) == pytest.approx(0.4)


def test_obstaculo_atras_no_cuenta():
    # El rayo 180 apunta a 0 rad: justo al lado opuesto del frente.
    assert en_sector(scan(g180=0.2), 0.0) == math.inf


def test_obstaculo_fuera_del_sector_no_cuenta():
    # Sector de 40 grados: 20 a cada lado. Un rayo a 45 grados del frente no entra.
    assert en_sector(scan(g45=0.2), 0.0) == math.inf


def test_izquierda_y_derecha():
    # Con el frente en 180, el rayo 90 queda 90 grados a la izquierda y el 270, a la derecha.
    rangos = scan(g90=0.7, g270=0.5)
    assert en_sector(rangos, 90.0) == pytest.approx(0.7)
    assert en_sector(rangos, -90.0) == pytest.approx(0.5)


def test_rayos_sin_retorno_o_fuera_de_rango_se_ignoran():
    rangos = [math.nan] * 360
    rangos[0] = 0.02  # más cerca que range_min: ruido
    assert distancia_en_sector(rangos, ANGULO_MIN, INCREMENTO, 180.0, 0.0, 40.0,
                               rango_min=0.1, rango_max=10.0) == math.inf


def test_el_frente_se_configura():
    # Si el lidar estuviera montado mirando al frente, el frente sería el ángulo 0.
    assert en_sector(scan(g180=0.3), 0.0, frente_deg=0.0) == pytest.approx(0.3)


def test_normalizar_angulo():
    assert normalizar_angulo(3 * math.pi) == pytest.approx(math.pi)
    assert normalizar_angulo(-1.5 * math.pi) == pytest.approx(0.5 * math.pi)


def giro(distancia, velocidad=0.0):
    return giro_para_seguir_pared(distancia, velocidad, objetivo=0.5, ganancia=2.0,
                                  amortiguacion=4.0, giro_max=0.8, pared_perdida=1.2)


def test_a_la_distancia_justa_y_quieto_no_gira():
    assert giro(0.5) == pytest.approx(0.0)


def test_pared_lejos_gira_a_la_derecha_y_cerca_a_la_izquierda():
    assert giro(0.7) < 0
    assert giro(0.3) > 0


def test_si_ya_se_esta_acercando_corrige_menos():
    # A 0,6 m de la pared (de más), el mismo error pide menos giro si ya se está acercando
    # que si se está alejando: eso es lo que evita el zigzag.
    acercandose, alejandose = giro(0.6, velocidad=-0.1), giro(0.6, velocidad=0.1)
    assert abs(acercandose) < abs(alejandose)


def test_el_giro_tiene_un_maximo():
    assert giro(0.0) == pytest.approx(0.8)    # error de 0,5 m: pediría 1,0 rad/s
    assert giro(1.15) == pytest.approx(-0.8)  # error de 0,65 m: pediría -1,3 rad/s


def test_pared_perdida_gira_a_la_derecha_para_encontrarla():
    assert giro(5.0) == pytest.approx(-0.8)
    assert giro(math.inf) == pytest.approx(-0.8)


def test_con_lugar_a_la_derecha_se_dobla_a_la_derecha():
    assert sentido_de_giro(hay_lugar_a_la_derecha=True) == -1


def test_con_pared_a_la_derecha_se_gira_a_la_izquierda():
    assert sentido_de_giro(hay_lugar_a_la_derecha=False) == 1
