import math

import numpy as np
import pytest

from equipo_jar.huecos import ConfigHuecos, RegistroHuecos, huecos_del_scan, pasable

N = 720
INC = 2 * math.pi / N
C = ConfigHuecos()


def scan_de(segmentos, origen=(0.0, 0.0), alcance=8.0):
    """Rangos de un LiDAR en `origen` (mirando a +x) contra segmentos ((x0, y0), (x1, y1))."""
    ox, oy = origen
    rangos = np.full(N, np.inf)
    for k in range(N):
        a = -math.pi + k * INC
        dx, dy = math.cos(a), math.sin(a)
        for (x0, y0), (x1, y1) in segmentos:
            ex, ey = x1 - x0, y1 - y0
            den = dx * ey - dy * ex
            if abs(den) < 1e-12:
                continue
            t = ((x0 - ox) * ey - (y0 - oy) * ex) / den
            u = ((x0 - ox) * dy - (y0 - oy) * dx) / den
            if t > 0 and 0 <= u <= 1 and t < rangos[k]:
                rangos[k] = t
    rangos[rangos > alcance] = np.inf
    return rangos


def puerta(ancho, x=1.0, fondo=3.0):
    """Pared en x con un hueco de `ancho` centrado en y = 0 y una pared de fondo."""
    m = ancho / 2
    return [((x, -3.0), (x, -m)), ((x, m), (x, 3.0)), ((fondo, -3.0), (fondo, 3.0))]


def medir(segmentos, origen=(0.0, 0.0)):
    return huecos_del_scan(scan_de(segmentos, origen), -math.pi, INC, 0.05, 12.0)


def test_mide_el_hueco_de_la_puerta_de_e_y_es_pasable():
    huecos = medir(puerta(0.36))
    assert len(huecos) == 1
    h = huecos[0]
    assert h.ancho == pytest.approx(0.36, abs=0.02)
    assert h.centro[0] == pytest.approx(1.0, abs=0.02) and abs(h.centro[1]) < 0.02
    assert h.normal()[0] == pytest.approx(1.0, abs=0.01)     # hacia el otro lado
    assert pasable(h.ancho)


def test_un_hueco_mas_angosto_que_el_robot_no_es_pasable():
    huecos = medir(puerta(0.25))
    assert len(huecos) == 1 and not pasable(huecos[0].ancho)


def test_de_costado_mide_lo_mismo():
    # Desde 0.8 m al costado: los extremos visibles no son los más cercanos entre sí.
    huecos = medir(puerta(0.36), origen=(0.0, 0.8))
    assert len(huecos) == 1
    assert huecos[0].ancho == pytest.approx(0.36, abs=0.03)


def test_una_pared_entera_o_un_hueco_sin_fondo_no_son_huecos():
    assert medir([((1.0, -3.0), (1.0, 3.0))]) == []
    # Sin lugar detrás (el fondo pegado): no se puede cruzar.
    assert medir(puerta(0.36, fondo=1.15)) == []


def test_con_el_laser_girado_180_como_en_el_rosmaster():
    rangos = np.roll(scan_de(puerta(0.36)), N // 2)  # el rayo 0 del láser mira atrás del robot
    huecos = huecos_del_scan(rangos, -math.pi, INC, 0.05, 12.0, tf_laser=(0.0, 0.0, math.pi))
    assert len(huecos) == 1
    assert huecos[0].centro[0] == pytest.approx(1.0, abs=0.02)
    assert huecos[0].ancho == pytest.approx(0.36, abs=0.02)


def test_lejos_no_se_mide():
    assert medir(puerta(0.36, x=2.5, fondo=4.0)) == []


def test_el_registro_junta_las_mediciones_y_avisa_una_vez():
    reg = RegistroHuecos(ConfigHuecos(mediciones_min=3))
    avisos = []
    for k in range(4):
        huecos = medir(puerta(0.36))
        avisos += reg.agregar(huecos, pose=(5.0, 2.0, math.pi / 2))
    assert len(reg.lista) == 1
    h = reg.lista[0]
    assert [c for _, c in avisos] == ['medido']
    assert h.centro == pytest.approx((5.0, 3.0), abs=0.03)    # 1 m adelante, mirando a +y
    assert h.ancho == pytest.approx(0.36, abs=0.02)
    assert len(reg.medidos()) == 1


def caja(cx, cy, lado):
    m = lado / 2
    esq = [(cx - m, cy - m), (cx + m, cy - m), (cx + m, cy + m), (cx - m, cy + m)]
    return [(esq[k], esq[(k + 1) % 4]) for k in range(4)]


def test_puerta_de_e_con_la_caja_gris_adentro():
    # Como en laberinto_puertas, girado: pared en x = 1 con una abertura de 0.9 m (y de -0.45 a
    # 0.45) y una caja de 0.35 m que deja 0.36 m de un lado y 0.19 m del otro.
    pared = [((1.0, -3.0), (1.0, -0.45)), ((1.0, 0.45), (1.0, 3.0)), ((3.0, -3.0), (3.0, 3.0))]
    centro_caja = -0.45 + 0.36 + 0.175
    escena = pared + caja(1.0, centro_caja, 0.35)
    # De frente a cada hueco se ve la cara de la caja que lo cierra: mide bien.
    # (Los centros quedan en el marco del robot: y del mundo = y del robot + y del origen.)
    izq = [h for h in medir(escena, origen=(0.0, -0.27)) if h.centro[1] - 0.27 < -0.1]
    der = [h for h in medir(escena, origen=(0.0, 0.355)) if h.centro[1] + 0.355 > 0.1]
    assert izq[0].ancho == pytest.approx(0.36, abs=0.02) and pasable(izq[0].ancho)
    assert der[0].ancho == pytest.approx(0.19, abs=0.02) and not pasable(der[0].ancho)
    # De frente a la caja, no: lo más angosto está en sus caras laterales, que no se ven.
    de_frente = sorted(h.ancho for h in medir(escena, origen=(0.0, 0.1)))
    assert de_frente[0] > 0.19 + 0.05 and de_frente[1] > 0.36 + 0.03


def test_el_registro_se_queda_con_las_medidas_chicas():
    reg = RegistroHuecos(ConfigHuecos(mediciones_min=3, cuantil=0.25))
    for ancho in (0.41, 0.40, 0.36, 0.37):          # acercándose de costado, después de frente
        h = huecos_del_scan(scan_de(puerta(ancho)), -math.pi, INC, 0.05, 12.0)
        reg.agregar(h, pose=(0.0, 0.0, 0.0))
    assert reg.lista[0].ancho == pytest.approx(0.367, abs=0.01)


def test_el_cuello_del_camino_es_la_puerta_del_mapa_estatico():
    from equipo_jar.huecos import cuello_en_camino
    res = 0.05
    libre = np.ones((60, 80), dtype=bool)                 # 4 × 3 m, origen en (0, 0)
    libre[[0, -1], :] = libre[:, [0, -1]] = False
    libre[:, 40] = False                                  # pared en x = 2.0
    libre[25:43, 40] = True                               # puerta de 0.9 m: y de 1.25 a 2.15
    cuello = cuello_en_camino(libre, res, (0.0, 0.0), (1.0, 1.0), (3.0, 1.0))
    assert cuello is not None
    (x, y), (dx, dy) = cuello
    assert dx > 0.9                                       # hacia el goal
    assert x == pytest.approx(2.0, abs=0.1)
    assert y == pytest.approx(1.70, abs=0.08)              # el medio de la abertura, no el marco
    # Sin pared en el medio no hay cuello que medir.
    abierto = np.ones((60, 80), dtype=bool)
    abierto[[0, -1], :] = abierto[:, [0, -1]] = False
    assert cuello_en_camino(abierto, res, (0.0, 0.0), (1.0, 1.5), (3.0, 1.5)) is None


def test_con_el_costmap_el_cuello_es_la_puerta_cerrada():
    from equipo_jar.huecos import cuello_en_camino
    res = 0.05
    # Pasillo de 0.9 m (y de 0.05 a 0.95) que lleva a una puerta de 0.9 m en x = 3.0; Nav2
    # tiene cerrada sólo la puerta (una caja adentro).
    libre = np.zeros((60, 100), dtype=bool)
    libre[1:19, 1:60] = True                             # pasillo
    libre[1:59, 61:99] = True                            # habitación
    libre[1:19, 60] = True                               # puerta

    def caja(x, y):
        return 2.9 <= x <= 3.1 and 0.3 <= y <= 0.7
    (x, _), (dx, _) = cuello_en_camino(libre, res, (0.0, 0.0), (0.5, 0.5), (4.0, 2.0),
                                       bloqueado=caja)
    assert x == pytest.approx(3.0, abs=0.1) and dx > 0.9
    # Sin nada cerrado en el camino no hay cuello que medir.
    assert cuello_en_camino(libre, res, (0.0, 0.0), (0.5, 0.5), (4.0, 2.0),
                            bloqueado=lambda x, y: False) is None
