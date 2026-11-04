import math

import numpy as np
import pytest

from jar_supervisor.logica import Config as ConfigSupervisor
from jar_supervisor.logica import Supervisor

from equipo_jar.paso_angosto import (
    ConfigPaso, a_marco_robot, ancho_en_plan, avance_en_plan, comando, descentrado, giro_libre,
    indice_actual, tapon)

C = ConfigPaso()


def contorno(cx, cy, sx, sy, paso=0.01):
    """Puntos del borde de un rectángulo, como los vería el LiDAR."""
    xs = np.arange(cx - sx / 2, cx + sx / 2 + 1e-9, paso)
    ys = np.arange(cy - sy / 2, cy + sy / 2 + 1e-9, paso)
    return np.vstack([np.column_stack((xs, np.full_like(xs, cy - sy / 2))),
                      np.column_stack((xs, np.full_like(xs, cy + sy / 2))),
                      np.column_stack((np.full_like(ys, cx - sx / 2), ys)),
                      np.column_stack((np.full_like(ys, cx + sx / 2), ys))])


# El pasillo de practica/mundos/pasillo_cajas: paredes en y = ±0.45 y el cubo A3 en (3.0, 0.05).
PAREDES = np.vstack([contorno(2.5, 0.5, 6.0, 0.1), contorno(2.5, -0.5, 6.0, 0.1)])
CUBO_A3 = contorno(3.0, 0.05, 0.25, 0.25)
MUNDO = np.vstack([PAREDES, CUBO_A3])


def toca(pose, puntos, medio_largo=0.16, medio_ancho=0.11):
    p = a_marco_robot(puntos, pose)
    return bool(np.any((np.abs(p[:, 0]) < medio_largo) & (np.abs(p[:, 1]) < medio_ancho)))


def test_a_marco_robot():
    p = a_marco_robot([(1.0, 1.0)], (1.0, 0.0, math.pi / 2))
    assert p[0] == pytest.approx((1.0, 0.0), abs=1e-9)


def test_centrado_en_un_hueco_simetrico_avanza_derecho():
    puntos = np.vstack([contorno(0.0, 0.25, 1.0, 0.05), contorno(0.0, -0.25, 1.0, 0.05)])
    plan = [(x, 0.0) for x in np.arange(0.0, 1.0, 0.05)]
    vx, vy, wz = comando(plan, puntos)
    assert vx == pytest.approx(C.vel_m_s) and abs(vy) < 1e-6 and abs(wz) < 1e-6


def test_descentrado_lo_corre_hacia_el_lado_libre():
    # Pared derecha a 0.15 m, izquierda a 0.30: tiene que correrse a la izquierda.
    puntos = np.vstack([contorno(0.0, 0.325, 1.0, 0.05), contorno(0.0, -0.175, 1.0, 0.05)])
    assert descentrado(puntos, 0.0, C) == pytest.approx(0.075, abs=0.01)
    plan = [(x, 0.0) for x in np.arange(0.0, 1.0, 0.05)]
    vx, vy, _ = comando(plan, puntos)
    assert vy > 0.03 and vx > 0


def test_no_gira_si_el_rectangulo_girado_toca():
    # Pared a 0.13 m al costado: derecho entra, girado no.
    pared = contorno(0.0, -0.14, 1.0, 0.02)
    assert not giro_libre(pared, 0.3, 0.17, 0.12)
    assert giro_libre(np.zeros((0, 2)), 0.3, 0.17, 0.12)
    plan = [(0.3 * k, 0.3 * k) for k in range(5)]          # pide girar 45°
    assert comando(plan, pared)[2] == 0.0


def test_avance_en_plan():
    plan = [(x, 0.0) for x in np.arange(-1.0, 1.0, 0.1)]
    assert avance_en_plan(plan) == pytest.approx(1.0, abs=0.06)


def recorrer(mundo, plan_map, pose, x_meta, hacia_menos_x=False):
    """Mueve el robot con la maniobra y el supervisor real hasta pasar `x_meta`."""
    pose = np.array(pose, dtype=float)
    # El supervisor real, con los parámetros de competencia (params.yaml), entre la maniobra
    # y las ruedas.
    sup = Supervisor(ConfigSupervisor(distancia_freno_m=0.03, distancia_lenta_m=0.12), ['mision'])
    sup.mapa_recibido()
    dt = 0.1
    for k in range(1200):
        puntos = a_marco_robot(mundo, pose)
        puntos = puntos[np.hypot(puntos[:, 0], puntos[:, 1]) < 3.0]
        sup.actualizar_obstaculos(puntos)
        sup.pedir('mision', *comando(a_marco_robot(plan_map, pose), puntos), t=k * dt)
        vx, vy, wz = sup.comando(k * dt)
        c, s = math.cos(pose[2]), math.sin(pose[2])
        pose = pose + np.array([(c * vx - s * vy) * dt, (s * vx + c * vy) * dt, wz * dt])
        assert not toca(pose, mundo), f'tocó en {pose}'
        if (pose[0] < x_meta) if hacia_menos_x else (pose[0] > x_meta):
            break
    assert (pose[0] < x_meta) if hacia_menos_x else (pose[0] > x_meta), f'quedó en {pose}'


@pytest.mark.parametrize('error_amcl', [0.0, 0.08, -0.08])
def test_pasa_el_hueco_de_a3_sin_tocar_aunque_el_plan_este_corrido(error_amcl):
    # Plan como el de Nav2: entra en S por debajo de A3 (centro del hueco en y = -0.26),
    # corrido `error_amcl` en y respecto del hueco real (lo que mide mal AMCL).
    xs = np.arange(2.3, 3.8, 0.05)
    ys = np.interp(xs, [2.3, 2.7, 3.3, 3.8], [0.0, -0.26, -0.26, 0.0]) + error_amcl
    recorrer(MUNDO, np.column_stack((xs, ys)), (2.35, 0.0, 0.0), 3.3)


# Hueco diagonal de ~0.33 m entre la punta de una pared que baja del techo (hasta y = 0.05)
# y un cubo abajo a su derecha: el que trababa al robot junto a la caja roja de maze_1.
PUNTA = contorno(4.0, 0.25, 0.1, 0.4)
CUBO_A4 = contorno(4.40, -0.32, 0.25, 0.25)
DIAGONAL = np.vstack([PAREDES, PUNTA, CUBO_A4])


@pytest.mark.parametrize('error_amcl', [0.0, 0.06, -0.06])
def test_pasa_el_hueco_diagonal_sin_tocar(error_amcl):
    xs = np.arange(3.4, 5.2, 0.05)
    ys = np.interp(xs, [3.4, 3.9, 4.15, 4.45, 5.2], [-0.2, -0.22, -0.05, 0.12, 0.12]) + error_amcl
    recorrer(DIAGONAL, np.column_stack((xs, ys)), (3.45, -0.2, 0.0), 4.75)


def test_se_saltea_el_rulo_del_plan_diferencial():
    # Plan que sale para adelante (+x), da la vuelta y vuelve pasando por el robot hacia -x:
    # el mecanum no necesita el rulo, sigue desde donde el plan vuelve a pasar.
    ida = [(0.05 * k, 0.0) for k in range(0, 12)]
    vuelta = [(0.55 - 0.05 * k, 0.15) for k in range(0, 30)]
    plan = ida + vuelta
    i = indice_actual(np.array(plan))
    assert i >= len(ida)
    vx, _, _ = comando(plan, np.zeros((0, 2)))
    assert vx < 0                                      # va directo hacia -x


def test_vuelve_por_a3_aunque_el_plan_de_nav2_haga_un_rulo():
    # Lo que pasó en Gazebo: el robot del lado este de A3 mirando al este; el plan diferencial
    # sale al este, da la vuelta (radio ~0.25) y vuelve al oeste por el hueco (y = -0.26).
    ida = [(3.25 + 0.05 * k, -0.30) for k in range(10)]
    giro = [(3.7 + 0.2 * math.sin(a), -0.10 - 0.2 * math.cos(a))
            for a in np.linspace(0, math.pi, 12)]
    vuelta = [(3.7 - 0.05 * k, -0.26 + (0.26 if k > 30 else 0.0) * (k - 30) / 10)
              for k in range(1, 40)]
    plan = np.array(ida + giro + vuelta)
    recorrer(MUNDO, plan, (3.25, -0.30, 0.0), 2.70, hacia_menos_x=True)


def test_ancho_en_plan_mide_el_hueco_por_donde_pasa_el_plan():
    # A3: el plan pasa por debajo del cubo, por el hueco de 0.375 m.
    xs = np.arange(2.3, 3.8, 0.05)
    ys = np.interp(xs, [2.3, 2.7, 3.3, 3.8], [0.0, -0.26, -0.26, 0.0])
    pose = (2.55, -0.15, 0.0)
    ancho, _ = ancho_en_plan(a_marco_robot(np.column_stack((xs, ys)), pose),
                             a_marco_robot(MUNDO, pose))
    assert ancho == pytest.approx(0.375, abs=0.03)


def test_caja_en_el_medio_con_lugar_al_costado_no_es_angosto():
    # A1: cubo pegado arriba, el plan pasa por abajo con 0.495 m.
    cubo = contorno(1.0, 0.17, 0.25, 0.25)
    mundo = np.vstack([PAREDES, cubo])
    xs = np.arange(0.5, 1.6, 0.05)
    plan = np.column_stack((xs, np.full_like(xs, -0.2)))
    pose = (0.7, -0.2, 0.0)
    ancho, _ = ancho_en_plan(a_marco_robot(plan, pose), a_marco_robot(mundo, pose))
    assert ancho == pytest.approx(0.495, abs=0.03)


def test_tapon_cubre_el_plan_adelante():
    plan = [(0.05 * k, 0.0) for k in range(41)]
    puntos = tapon(plan, (0.0, 0.0, 0.0), hasta=0.8, medio_ancho=0.15)
    xs = [p[0] for p in puntos]
    assert min(xs) > 0 and max(xs) <= 0.85 and len(puntos) > 50


def test_no_elige_una_direccion_que_el_supervisor_frenaria():
    # Una pared a 6 cm del frente (más el margen): el supervisor frena a cero por debajo de
    # distancia_freno_m (0.06); la maniobra no tiene que pedir ir para ahí.
    from equipo_jar.paso_angosto import ConfigPaso, direccion_con_lugar
    c = ConfigPaso()
    frente = c.medio_largo_m + c.margen_supervisor_m + 0.06
    ys = np.arange(-0.5, 0.5, 0.02)
    puntos = np.column_stack((np.full_like(ys, frente), ys))
    vx, vy = direccion_con_lugar(0.08, 0.0, puntos, c)
    assert not (vx > 0 and abs(vy) < 1e-6)


def test_no_gira_adentro_de_un_hueco_si_queda_cruzado():
    # maze_4: pasillo de 0.43 m (paredes en y = ±0.215) recorrido de frente; el plan dobla 90° a
    # la izquierda enseguida. Girar ahí lo dejaba cruzado sin poder trasladarse: no gira, avanza.
    paredes = np.array([(x, y) for x in np.linspace(-0.6, 0.6, 121) for y in (0.215, -0.215)])
    plan = [(0.0, 0.0), (0.05, 0.0), (0.10, 0.0), (0.10, 0.05), (0.10, 0.10), (0.10, 0.15),
            (0.10, 0.20), (0.10, 0.25), (0.10, 0.30)]
    vx, vy, wz = comando(plan, paredes)
    assert wz == 0.0


def test_gira_si_despues_del_giro_sigue_habiendo_lugar():
    lejos = np.array([(x, y) for x in np.linspace(-1.0, 1.0, 101) for y in (0.8, -0.8)])
    plan = [(0.05 * k, 0.05 * k) for k in range(12)]
    assert comando(plan, lejos)[2] > 0
