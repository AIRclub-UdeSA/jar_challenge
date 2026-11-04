import math

import numpy as np
import pytest

from jar_supervisor.logica import (Config, Estado, Supervisor, direccion_mas_libre,
                                   giro_para_salir, hueco_en_direccion, matriz_de_cuaternion, scan_a_puntos)

SIN_RAMPA = dict(acel_lineal_max=1e9, acel_angular_max=1e9)


def supervisor(**kw):
    s = Supervisor(Config(**{**SIN_RAMPA, **kw}), ['mision', 'nav'])
    s.mapa_recibido()
    s.comando(-1.0)  # primer ciclo: fija el instante previo de la rampa
    return s


def test_cero_antes_del_mapa_y_despues_de_detener():
    s = Supervisor(Config(**SIN_RAMPA), ['mision', 'nav'])
    s.pedir('nav', 0.2, 0.0, 0.0, t=0.0)
    assert s.comando(0.0) == (0.0, 0.0, 0.0)
    s.mapa_recibido()
    assert s.comando(0.1)[0] == pytest.approx(0.2)
    s.detener()
    s.pedir('nav', 0.2, 0.0, 0.0, t=0.2)
    assert s.comando(0.2) == (0.0, 0.0, 0.0)
    s.mapa_recibido()  # un mapa repetido no lo reactiva
    assert s.estado == Estado.DETENIDO


def test_prioridad_y_vencimiento():
    s = supervisor()
    s.pedir('nav', 0.1, 0.0, 0.0, t=0.0)
    s.pedir('mision', 0.0, 0.0, 0.5, t=0.0)
    assert s.comando(0.1) == pytest.approx((0.0, 0.0, 0.5))
    # La misión deja de pedir: después de vencer, manda Nav2.
    s.pedir('nav', 0.1, 0.0, 0.0, t=0.35)
    assert s.comando(0.4) == pytest.approx((0.1, 0.0, 0.0))
    # Nadie renueva: cero.
    assert s.comando(1.0) == (0.0, 0.0, 0.0)


def test_limites_de_velocidad():
    s = supervisor()
    s.pedir('nav', 5.0, -5.0, 9.0, t=0.0)
    assert s.comando(0.0) == pytest.approx((0.30, -0.20, 1.00))


def test_freno_por_obstaculo_adelante():
    s = supervisor()
    # Pared a 0.22 m del centro: hueco libre 0.04 m, por debajo del freno.
    s.actualizar_obstaculos([[0.22, 0.0], [0.22, 0.05]])
    s.pedir('nav', 0.2, 0.0, 0.3, t=0.0)
    vx, vy, wz = s.comando(0.0)
    assert vx == pytest.approx(0.0)
    assert wz == pytest.approx(0.3)  # girar en el lugar sigue permitido


def test_freno_escala_en_zona_lenta_y_no_frena_hacia_atras():
    s = supervisor()
    s.actualizar_obstaculos([[0.18 + 0.155, 0.0]])   # hueco 0.155: mitad de la zona lenta
    s.pedir('nav', 0.2, 0.0, 0.0, t=0.0)
    assert s.comando(0.0)[0] == pytest.approx(0.1, abs=1e-6)
    s.pedir('nav', -0.2, 0.0, 0.0, t=0.1)
    assert s.comando(0.1)[0] == pytest.approx(-0.2)


def test_obstaculo_de_costado_no_frena_avance():
    s = supervisor()
    s.actualizar_obstaculos([[0.10, 0.40]])
    s.pedir('nav', 0.2, 0.0, 0.0, t=0.0)
    assert s.comando(0.0)[0] == pytest.approx(0.2)
    # Pero sí el desplazamiento lateral hacia él.
    s.pedir('nav', 0.0, 0.2, 0.0, t=0.1)
    assert s.hueco_libre(0.0, 0.2) == pytest.approx(0.40 - 0.13)   # medio ancho + margen


def test_rampa_de_aceleracion_y_frenado_inmediato():
    s = Supervisor(Config(acel_lineal_max=1.0), ['nav'])
    s.mapa_recibido()
    s.comando(0.0)
    s.pedir('nav', 0.3, 0.0, 0.0, t=0.0)
    assert s.comando(0.1)[0] == pytest.approx(0.1)
    assert s.comando(0.2)[0] == pytest.approx(0.2)
    s.pedir('nav', 0.0, 0.0, 0.0, t=0.2)
    assert s.comando(0.25)[0] == pytest.approx(0.0)


def test_fuente_desconocida():
    with pytest.raises(ValueError):
        supervisor().pedir('teclado', 0.1, 0.0, 0.0, t=0.0)


def test_scan_a_puntos_con_laser_girado():
    # Láser montado con 180° de yaw: su rayo de 180° apunta hacia adelante del robot.
    pts = scan_a_puntos([1.0, math.inf, 2.0], angle_min=math.pi, angle_increment=-math.pi / 2,
                        range_min=0.05, range_max=12.0, tx=0.04, yaw=math.pi)
    assert pts.shape == (2, 2)
    np.testing.assert_allclose(pts[0], [1.04, 0.0], atol=1e-9)
    np.testing.assert_allclose(pts[1], [-1.96, 0.0], atol=1e-9)


def test_direccion_mas_libre_sale_de_una_esquina():
    # Pared atrás (x = -0.2) y pared a la izquierda (y = 0.22): el punto más cercano está
    # atrás, pero lejos de él (adelante) también se libera; a la derecha-adelante, más.
    atras = [(-0.2, y) for y in np.linspace(-0.5, 0.5, 21)]
    izquierda = [(x, 0.22) for x in np.linspace(-0.2, 1.0, 25)]
    adelante = [(0.25, y) for y in np.linspace(-0.05, 0.22, 10)]   # punta de pared enfrente
    puntos = np.array(atras + izquierda + adelante)
    d = direccion_mas_libre(puntos, 0.18, 0.13, preferida=(1.0, 0.0))
    assert d[1] < -0.5                                  # sale por la derecha, que está libre


def test_direccion_mas_libre_desempata_con_la_preferida():
    d = direccion_mas_libre(np.zeros((0, 2)), 0.18, 0.13, preferida=(0.0, 1.0))
    assert d == pytest.approx((0.0, 1.0), abs=1e-9)


def test_caja_al_costado_no_frena_el_avance_del_rectangulo():
    # La esquina de una caja a 0.17 m al costado: el círculo de 0.18 frenaba; el robot pasa.
    s = supervisor()
    s.actualizar_obstaculos([[0.05, 0.17], [0.30, 0.17]])
    s.pedir('nav', 0.2, 0.0, 0.0, t=0.0)
    assert s.comando(0.0)[0] == pytest.approx(0.2)


def test_rectangulo_frena_en_diagonal_contra_la_esquina():
    s = supervisor()
    s.actualizar_obstaculos([[0.30, 0.20]])
    # En diagonal toca cuando el frente (x = 0.18) llega a x = 0.30: 0.12·√2 de avance.
    h = s.hueco_libre(1.0, 1.0)
    assert h == pytest.approx(0.12 * math.sqrt(2), abs=1e-6)
    assert s.hueco_libre(-1.0, 0.0) == math.inf          # hacia atrás no lo toca


def test_punto_ya_adentro_frena_hacia_el_y_deja_alejarse():
    s = supervisor()
    s.actualizar_obstaculos([[0.17, 0.0]])               # dentro del margen, adelante
    assert s.hueco_libre(0.2, 0.0) == 0.0
    assert s.hueco_libre(-0.2, 0.0) == math.inf


def test_pared_en_el_margen_del_costado_no_frena_el_avance_paralelo():
    # Pared a 0.12 m del centro: fuera del cuerpo (0.11) pero dentro del margen (0.13).
    s = supervisor()
    s.actualizar_obstaculos([(x, 0.12) for x in np.linspace(-0.6, 0.6, 61)])
    assert s.hueco_libre(1.0, 0.0) == math.inf
    assert s.hueco_libre(-1.0, 0.0) == math.inf
    assert s.hueco_libre(0.0, 1.0) == 0.0                # hacia la pared, no
    assert s.hueco_libre(0.0, -1.0) == math.inf


def test_encajado_en_un_hueco_justo_puede_salir_hacia_adelante_o_atras():
    # Regresión 3 (maze_4, laberinto_s4, maze_1): el robot quedó en un hueco con roces en las
    # cuatro esquinas; ninguna dirección tenía lugar y el juez lo cortó por inmovilidad.
    s = supervisor()
    paredes = [(x, y) for x in np.linspace(-0.4, 0.4, 41) for y in (0.12, -0.125)]
    s.actualizar_obstaculos(paredes)
    assert s.hueco_libre(1.0, 0.0) == math.inf
    assert s.hueco_libre(-1.0, 0.0) == math.inf
    assert direccion_mas_libre(paredes, 0.18, 0.13, preferida=(0.0, 1.0),
                               margen=0.02)[1] == pytest.approx(0.0, abs=1e-9)


def test_esquina_de_caja_adentro_del_cuerpo_frena_solo_por_su_cara():
    # Un punto 3 cm dentro del costado (ruido o caja rozando): no deja ir hacia ese costado,
    # pero sí avanzar o retroceder.
    s = supervisor()
    s.actualizar_obstaculos([[0.05, 0.10]])
    assert s.hueco_libre(0.0, 1.0) == 0.0
    assert s.hueco_libre(1.0, 0.0) > 0.06
    assert s.hueco_libre(-1.0, 0.0) == math.inf


def cruzado_en_hueco(yaw, ancho=0.40):
    """Paredes de un pasillo de `ancho` (eje y del mundo) vistas por un robot girado `yaw`."""
    c, s = math.cos(yaw), math.sin(yaw)
    puntos = []
    for x_mundo in (-ancho / 2, ancho / 2):
        for y_mundo in np.linspace(-0.6, 0.6, 121):
            puntos.append((c * x_mundo + s * y_mundo, -s * x_mundo + c * y_mundo))
    return np.array(puntos)


def test_cruzado_en_diagonal_en_un_hueco_gira_hacia_el_eje_del_pasillo():
    # maze_4 (regresión 3): cruzado a ~49° del pasillo, dos esquinas casi tocando, ninguna
    # traslación con lugar; girar hacia el eje del pasillo (yaw 90°) abre lugar.
    puntos = cruzado_en_hueco(math.radians(139.0))
    assert max(hueco_en_direccion(puntos, math.cos(a), math.sin(a), 0.18, 0.13, 0.02)
               for a in np.linspace(0, 2 * math.pi, 16, endpoint=False)) < 0.06
    assert giro_para_salir(puntos, 0.18, 0.13, 0.02, 0.005) == -1
    assert giro_para_salir(cruzado_en_hueco(math.radians(41.0)), 0.18, 0.13, 0.02, 0.005) == 1


def test_con_lugar_para_trasladarse_no_gira():
    assert giro_para_salir(cruzado_en_hueco(math.radians(90.0)), 0.18, 0.13, 0.02, 0.005) == 0


def test_giro_pegado_a_una_pared_se_frena_y_la_traslacion_sigue():
    # maze_3: girando en un rincón contra la pared, las ruedas patinaban y AMCL se perdía.
    s = supervisor()
    pared = [(x, -0.105) for x in np.linspace(-0.4, 0.4, 41)]   # tocando el costado
    s.actualizar_obstaculos(pared)
    s.pedir('nav', 0.0, 0.1, 1.0, t=0.0)                         # se aleja y gira
    vx, vy, wz = s.comando(0.0)
    assert wz == 0.0 and vy > 0 and s.diag['giro_frenado']


def test_giro_con_lugar_no_se_frena():
    s = supervisor()
    s.actualizar_obstaculos([(x, -0.45) for x in np.linspace(-0.4, 0.4, 41)])
    s.pedir('nav', 0.0, 0.0, 1.0, t=0.0)
    assert s.comando(0.0)[2] == pytest.approx(1.0) and not s.diag['giro_frenado']


def test_giro_cerca_pero_sin_tocar_no_se_frena():
    # A 3 cm del costado: girar unos grados no toca; no hay que frenarlo (lugares angostos).
    s = supervisor()
    s.actualizar_obstaculos([(x, -0.14) for x in np.linspace(-0.12, 0.12, 13)])
    s.pedir('nav', 0.0, 0.0, 0.3, t=0.0)
    assert s.comando(0.0)[2] == pytest.approx(0.3)


def test_si_las_ruedas_giran_y_la_imu_no_se_cortan_los_giros():
    # maze_3: trabado en un rincón, la odometría decía 1 rad/s y el robot no giraba.
    s = supervisor()
    s.actualizar_giros(giro_odom=1.0, giro_imu=0.0)
    s.pedir('nav', 0.0, 0.0, 1.0, t=0.0)
    assert s.comando(0.0)[2] == pytest.approx(1.0)       # todavía no: puede ser un instante
    s.pedir('nav', 0.0, 0.0, 1.0, t=0.5)
    assert s.comando(0.5)[2] == 0.0 and s.diag['patina']
    # Unos segundos cortado aunque deje de patinar; después vuelve a girar.
    s.actualizar_giros(giro_odom=0.0, giro_imu=0.0)
    s.pedir('nav', 0.0, 0.0, 1.0, t=1.5)
    assert s.comando(1.5)[2] == 0.0
    s.pedir('nav', 0.0, 0.0, 1.0, t=3.0)
    assert s.comando(3.0)[2] == pytest.approx(1.0)


def test_ruedas_e_imu_de_acuerdo_no_es_patinar():
    s = supervisor()
    s.actualizar_giros(giro_odom=0.8, giro_imu=0.75)
    for k in range(10):
        s.pedir('nav', 0.0, 0.0, 0.8, t=0.1 * k)
        assert s.comando(0.1 * k)[2] == pytest.approx(0.8)


def test_imu_dada_vuelta_mide_el_giro_con_el_signo_correcto_en_el_robot():
    # Montaje del simulador: rpy = (0, pi, pi/2). El eje z de la IMU apunta para abajo: sin
    # pasar al marco del robot, cada giro parecía patinamiento.
    import math as m
    cr, sr = m.cos(0.0), m.sin(0.0)
    cp, sp = m.cos(m.pi / 2), m.sin(m.pi / 2)
    cy, sy = m.cos(m.pi / 4), m.sin(m.pi / 4)
    q = (sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
         cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy)
    r = matriz_de_cuaternion(*q)
    giro_en_imu = r.T @ np.array([0.0, 0.0, 1.0])      # el robot gira +1 rad/s
    assert giro_en_imu[2] == pytest.approx(-1.0, abs=1e-6)
    assert (r @ giro_en_imu)[2] == pytest.approx(1.0, abs=1e-6)
