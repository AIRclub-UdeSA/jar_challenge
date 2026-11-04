import math

import numpy as np
import pytest

from jar_percepcion.color import ConfigColor, segmentar_rojo, toca_borde
from jar_percepcion.fusion import (alto_coherente, estimar_centro, puntos_en_ventana,
                                   rumbo_en_laser, transformar, ventana_angular)
from jar_percepcion.tracker import ConfigTracker, Tracker

# Cámara óptica (z adelante, x derecha, y abajo) en un robot con el láser girado 180°,
# como en el ROSMASTER X3: el frente del robot es el rumbo ±π del láser.
R_BASE_OPTICO = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
R_LASER_BASE = np.diag([-1.0, -1.0, 1.0])
R_LASER_OPTICO = R_LASER_BASE @ R_BASE_OPTICO
FX, CX = 271.8, 159.5


def imagen(fondo=(90, 90, 90)):
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    img[:] = fondo
    return img


def test_detecta_rojo_e_ignora_otros_colores():
    img = imagen()
    img[100:150, 140:180] = (200, 20, 20)     # rojo
    img[100:150, 40:80] = (20, 20, 200)       # azul
    img[100:150, 240:280] = (220, 200, 20)    # amarillo
    blobs = segmentar_rojo(img, ConfigColor())
    assert len(blobs) == 1
    b = blobs[0]
    assert (b.u_min, b.u_max) == (140, 179)
    assert b.area == 50 * 40


def test_rojo_oscuro_del_otro_extremo_del_tono_tambien_cuenta():
    img = imagen()
    img[60:90, 100:140] = (180, 10, 60)       # magenta rojizo: H ~ 170 en OpenCV
    assert len(segmentar_rojo(img, ConfigColor())) == 1


def test_descarta_blobs_cortados_por_el_borde_y_chicos():
    img = imagen()
    img[100:150, 0:30] = (200, 20, 20)
    img[10:15, 200:205] = (200, 20, 20)
    assert segmentar_rojo(img, ConfigColor()) == []


def test_sin_descartar_borde_una_caja_roja_de_cerca_cuenta_como_rojo():
    # De cerca la caja roja llena la imagen y toca los bordes: para ubicarla no sirve, pero
    # para el voto "no es rojo" tiene que contar como rojo.
    img = imagen()
    img[:, :] = (200, 20, 20)
    assert segmentar_rojo(img, ConfigColor()) == []
    assert len(segmentar_rojo(img, ConfigColor(), descartar_borde=False)) == 1


def test_toca_borde_solo_los_blobs_cortados_a_los_lados():
    cfg = ConfigColor()
    assert toca_borde(0, 40, 320, cfg) and toca_borde(280, 319, 320, cfg)
    assert not toca_borde(100, 200, 320, cfg)


def test_rumbo_del_centro_de_la_imagen_es_el_frente_del_robot():
    assert abs(abs(rumbo_en_laser(CX, FX, CX, R_LASER_OPTICO)) - math.pi) < 1e-9
    # Una columna a la derecha de la imagen queda a la derecha del robot (y negativa en base),
    # que en el láser girado es un rumbo un poco menor que π.
    a = rumbo_en_laser(CX + 50, FX, CX, R_LASER_OPTICO)
    assert 0 < a < math.pi


def test_ventana_que_cruza_el_corte_de_pi():
    centro, semiancho = ventana_angular(140, 179, FX, CX, R_LASER_OPTICO, recorte=0.0)
    assert abs(abs(centro) - math.pi) < 0.05
    assert semiancho == pytest.approx(math.atan(20.5 / FX) * 2 / 2, rel=0.05)


def test_puntos_en_ventana_a_traves_del_corte():
    n = 1080
    inc = 2 * math.pi / n
    ranges = np.full(n, 3.0)
    ranges[:5] = 1.0            # rumbos cerca de -π
    ranges[-5:] = 1.02          # rumbos cerca de +π
    ranges[5] = math.inf
    pts = puntos_en_ventana(ranges, -math.pi, inc, 0.05, 12.0, centro=math.pi, semiancho=0.05)
    assert pts is not None and len(pts) == 10
    assert np.all(pts[:, 0] < -0.9)          # todos atrás del láser = adelante del robot


def test_rojo_asomando_detras_de_una_punta_de_pared_no_se_ubica_en_la_punta():
    n = 360
    inc = 2 * math.pi / n
    ranges = np.full(n, 3.0)
    ranges[176:182] = 2.0       # la víctima, de frente (el rayo 180 es el rumbo 0)
    ranges[182:185] = 1.0       # la punta de una pared tapa el costado
    args = (ranges, -math.pi, inc, 0.05, 12.0, 0.0)
    pts = puntos_en_ventana(*args, semiancho=0.07, min_puntos=2)
    assert pts is not None and np.allclose(np.hypot(*pts.T), 1.0)   # antes: la punta
    assert puntos_en_ventana(*args, semiancho=0.07, min_puntos=2, fraccion_cerca_min=0.6) is None
    caja = puntos_en_ventana(*args, semiancho=0.03, min_puntos=2, fraccion_cerca_min=0.6)
    assert caja is not None and np.allclose(np.hypot(*caja.T), 2.0)  # sólo la caja: se ubica


def test_puntos_en_ventana_sin_retorno_no_detecta():
    ranges = np.full(360, math.inf)
    assert puntos_en_ventana(ranges, -math.pi, 2 * math.pi / 360, 0.05, 12.0, 0.0, 0.2) is None


def test_estimar_centro_corre_media_caja_hacia_atras_de_la_cara():
    cara = np.array([[1.0, -0.1], [1.0, 0.0], [1.0, 0.1]])
    assert estimar_centro(cara, (0.0, 0.0), 0.125) == pytest.approx((1.125, 0.0))
    pts = transformar([[1.0, 0.0]], 2.0, 1.0, math.pi / 2)
    np.testing.assert_allclose(pts, [[2.0, 2.0]], atol=1e-12)


def test_tracker_confirma_con_vistas_distintas():
    t = Tracker(ConfigTracker(observaciones_min=3, separacion_vistas_min_m=0.3))
    for _ in range(5):
        t.agregar(2.0, 1.0, desde=(0.0, 0.0))
    assert not t.objetivos[0].confirmado          # siempre desde el mismo lugar
    t.agregar(2.02, 1.01, desde=(0.5, 0.0))
    (obj,) = t.resumen()
    assert obj['confirmado'] and obj['n'] == 6
    assert obj['x'] == pytest.approx(2.003, abs=1e-3)


def test_tracker_separa_objetivos_lejanos_y_no_confirma_dispersos():
    t = Tracker(ConfigTracker(observaciones_min=3, dispersion_max_m=0.05,
                              separacion_vistas_min_m=0.0))
    t.agregar(0.0, 0.0, desde=(0, 0))
    t.agregar(2.0, 0.0, desde=(0, 0))
    assert len(t.objetivos) == 2
    for dx in (0.2, -0.2, 0.25):
        t.agregar(dx, 0.0, desde=(1, 1))
    assert not t.objetivos[0].confirmado
    assert t.agregar(math.nan, 0.0, desde=(0, 0)) is None


def test_un_rojo_mas_chico_que_una_caja_a_esa_distancia_esta_detras():
    fy, alto = 271.8, 240
    # Caja de 0.30 m a 0.63 m: ~129 px. V1 a 2.1 m, vista por el hueco: ~39 px.
    assert alto_coherente(50, 178, alto, fy, 0.63, 0.30, 0.5)
    assert not alto_coherente(100, 138, alto, fy, 0.63, 0.30, 0.5)
    assert alto_coherente(100, 138, alto, fy, 2.1, 0.30, 0.5)        # a su distancia, sí
    assert alto_coherente(0, 60, alto, fy, 0.63, 0.30, 0.5)          # cortado arriba: no se mide
    assert alto_coherente(100, 138, alto, fy, 0.63, 0.30, 0.0)       # apagado
