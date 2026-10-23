import math

import numpy as np
import pytest

from equipo_jar.deteccion import (
    angulo_de_columna, buscar_caja_roja, cerca_de_alguno, coinciden, distancia_a_la_caja,
    distancia_por_altura, mascara_rojo, promedio, punto_en_el_plano, yaw_de_cuaternion)

SATURACION, VALOR, AREA, BORDE = 120, 80, 100, 5
ROJO = (0, 0, 220)   # BGR
AZUL = (220, 0, 0)


def imagen(color_bgr, x0=100, x1=140):
    """Imagen gris de 200x300 con un rectángulo de `color_bgr` entre las columnas x0 y x1."""
    img = np.full((200, 300, 3), 128, dtype=np.uint8)
    img[60:140, x0:x1] = color_bgr
    return img


def buscar(img):
    return buscar_caja_roja(img, SATURACION, VALOR, AREA, BORDE)


def test_encuentra_la_caja_roja():
    columna, ancho, alto = buscar(imagen(ROJO))
    assert columna == pytest.approx(120.0, abs=1.0)
    assert ancho == pytest.approx(40, abs=2)
    assert alto == pytest.approx(80, abs=2)


def test_ignora_un_distractor_azul():
    assert buscar(imagen(AZUL)) is None


def test_ignora_manchas_chicas():
    img = np.full((200, 300, 3), 128, dtype=np.uint8)
    img[60:66, 100:106] = ROJO  # 6x6 píxeles: menos que el área mínima
    assert buscar(img) is None


def test_ignora_una_caja_cortada_por_el_borde():
    assert buscar(imagen(ROJO, 0, 40)) is None
    assert buscar(imagen(ROJO, 270, 300)) is None


def test_la_mascara_marca_solo_el_rojo():
    mascara = mascara_rojo(imagen(ROJO), SATURACION, VALOR)
    assert mascara[100, 120] == 255
    assert mascara[10, 10] == 0


def test_angulo_de_columna():
    # En el centro de la imagen, de frente; a la izquierda del centro, ángulo positivo.
    assert angulo_de_columna(150.0, fx=300.0, cx=150.0) == pytest.approx(0.0)
    assert angulo_de_columna(0.0, fx=300.0, cx=150.0) == pytest.approx(math.atan2(150, 300))
    assert angulo_de_columna(300.0, fx=300.0, cx=150.0) < 0


INCREMENTO = math.radians(1.0)
ANGULO_MIN = -math.pi


def lidar(**rayos):
    """Scan libre (3 m) con algunos rayos distintos: g<rayo>=<distancia>. Frente en el rayo 0."""
    rangos = [3.0] * 360
    for rayo, d in rayos.items():
        rangos[int(rayo.removeprefix('g'))] = d
    return rangos


def a_la_caja(rangos, centro_deg=0.0, ancho_deg=6.0):
    return distancia_a_la_caja(rangos, ANGULO_MIN, INCREMENTO, 180.0, centro_deg, ancho_deg)


def test_distancia_a_la_caja_es_la_mediana_del_tramo():
    # Caja a 1,2 m, pero el borde de una pared asoma en un rayo del sector a 0,5 m: no cuenta.
    rangos = lidar(**{f'g{r}': 1.2 for r in (358, 359, 0, 1, 2)}, g3=0.5)
    assert a_la_caja(rangos, ancho_deg=10.0) == pytest.approx(1.2)


def test_distancia_a_la_caja_en_otra_direccion():
    # El sector se mide desde el frente: 20 grados a la izquierda es el rayo 20.
    rangos = lidar(**{f'g{r}': 1.0 for r in range(16, 25)})
    assert a_la_caja(rangos, centro_deg=20.0) == pytest.approx(1.0)


def test_distancia_a_la_caja_sin_rayos_validos():
    assert a_la_caja([math.inf] * 360) is None


def test_distancia_por_altura():
    # Con fy = 270 px, una caja de 0,30 m que ocupa 81 px de alto está a 1 m.
    assert distancia_por_altura(81, fy=270.0, alto_caja_m=0.30) == pytest.approx(1.0)


def test_el_lidar_y_la_imagen_tienen_que_coincidir():
    assert coinciden(1.0, 1.1, tolerancia_relativa=0.25)
    # El lidar ve la esquina de una pared a 1 m, pero la caja está a 1,65 m: no coinciden.
    assert not coinciden(1.0, 1.65, tolerancia_relativa=0.25)


def test_punto_en_el_plano():
    # Robot en (1, 2) mirando a +y, caja a 2 m justo al frente.
    x, y = punto_en_el_plano(1.0, 2.0, math.pi / 2, 0.0, 2.0)
    assert (x, y) == pytest.approx((1.0, 4.0))
    # La misma caja vista 90 grados a la izquierda del robot queda hacia -x.
    x, y = punto_en_el_plano(1.0, 2.0, math.pi / 2, math.pi / 2, 2.0)
    assert (x, y) == pytest.approx((-1.0, 2.0))


def test_yaw_de_cuaternion():
    # Giro de 90 grados alrededor de z.
    s = math.sin(math.pi / 4)
    assert yaw_de_cuaternion(0.0, 0.0, s, s) == pytest.approx(math.pi / 2)


def test_duplicados_y_promedio():
    assert cerca_de_alguno((1.0, 1.0), [(1.2, 1.0)], 0.5)
    assert not cerca_de_alguno((1.0, 1.0), [(3.0, 1.0)], 0.5)
    assert not cerca_de_alguno((1.0, 1.0), [], 0.5)
    assert promedio([(0.0, 0.0), (2.0, 4.0)]) == pytest.approx((1.0, 2.0))
