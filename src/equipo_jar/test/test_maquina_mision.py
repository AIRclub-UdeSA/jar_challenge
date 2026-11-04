import math

import pytest

from equipo_jar.maquina_mision import Acciones, Config, Entradas, Estado, Mision


def mision(cupos=2, **kw):
    return Mision(Config(**kw), cupos=cupos, radio_union=0.35)


def localizada(m, pose=(0.05, -0.02, 0.1)):
    """Lleva la misión hasta EXPLORANDO con una vuelta completa."""
    m.paso(Entradas(t=0.0, pose=pose, yaw_odom=0.0))
    for i in range(1, 70):
        m.paso(Entradas(t=0.1 * i, pose=pose, yaw_odom=0.1 * i))
        if m.estado == Estado.EXPLORANDO:
            return 0.1 * i
    raise AssertionError('no terminó de localizar')


def test_nada_antes_del_mapa():
    m = mision()
    a = m.paso(Entradas(t=None, objetivos=[{'x': 1, 'y': 1, 'n': 9, 'confirmado': True}]))
    assert a.cmd is None and a.goal is None and a.reportes == [] and not a.done
    assert m.estado == Estado.ESPERANDO_MAPA


def test_localiza_girando_y_guarda_la_largada():
    m = mision()
    a = m.paso(Entradas(t=0.0, pose=(0.05, -0.02, 0.1), yaw_odom=0.0))
    assert m.estado == Estado.LOCALIZANDO
    assert a.cmd[2] > 0 and a.cmd[0] == 0 and a.cmd[1] == 0
    localizada(m)
    assert m.largada == (0.05, -0.02, 0.1)


def test_localizacion_tiene_tiempo_maximo():
    m = mision(tiempo_max_localizacion_s=5.0)
    m.paso(Entradas(t=0.0, yaw_odom=0.0))
    m.paso(Entradas(t=5.0, yaw_odom=0.0))
    assert m.estado == Estado.EXPLORANDO
    assert m.largada == (0.0, 0.0, 0.0)


def test_explora_orienta_y_pide_el_siguiente_goal():
    m = mision()
    t = localizada(m)
    a = m.paso(Entradas(t=t + 0.1, pose=(0, 0, 0), goal_exploracion=(1.0, 0.0, 1.5)))
    assert a.goal == (1.0, 0.0, 1.5)
    a = m.paso(Entradas(t=t + 5, pose=(1, 0, 0), nav_resultado='ok'))
    assert m.estado == Estado.ORIENTANDO
    a = m.paso(Entradas(t=t + 5.1, pose=(1, 0, 0)))
    assert a.cmd[2] == pytest.approx(0.8)   # recortado a vel_giro_max
    a = m.paso(Entradas(t=t + 7, pose=(1, 0, 1.5)))
    assert m.estado == Estado.EXPLORANDO and a.cmd == (0.0, 0.0, 0.0)


def test_goal_fallido_se_descarta_y_no_se_repite():
    m = mision()
    t = localizada(m)
    m.paso(Entradas(t=t + 0.1, pose=(0, 0, 0), goal_exploracion=(2.0, 0.0, 0.0)))
    a = m.paso(Entradas(t=t + 3, pose=(0, 0, 0), nav_resultado='falla',
                        goal_exploracion=(2.0, 0.0, 0.0)))
    assert a.descartar == (2.0, 0.0) and a.goal is None
    a = m.paso(Entradas(t=t + 3.1, pose=(0, 0, 0), goal_exploracion=(2.1, 0.0, 0.0)))
    assert a.goal is None
    a = m.paso(Entradas(t=t + 3.2, pose=(0, 0, 0), goal_exploracion=(0.0, 2.0, 0.0)))
    assert a.goal == (0.0, 2.0, 0.0)


def test_robot_trabado_cancela_el_goal():
    m = mision()
    t = localizada(m)
    m.paso(Entradas(t=t + 0.1, pose=(0, 0, 0), goal_exploracion=(2.0, 0.0, 0.0)))
    a = m.paso(Entradas(t=t + 30, pose=(0, 0, 0), nav_libre=False, quieto_s=25.0))
    assert a.cancelar and a.descartar == (2.0, 0.0)


def test_reporta_al_confirmar_y_vuelve_con_los_cupos_completos():
    m = mision(cupos=2)
    t = localizada(m)
    confirmados = [{'x': 1, 'y': 1, 'n': 6, 'confirmado': True},
                   {'x': 2, 'y': 2, 'n': 6, 'confirmado': True}]
    m.paso(Entradas(t=t + 0.1, pose=(0, 0, 0), goal_exploracion=(3.0, 0.0, 0.0)))
    a = m.paso(Entradas(t=t + 10, pose=(1, 0, 0), nav_libre=False, objetivos=confirmados))
    assert a.reportes == [(1, 1), (2, 2)]
    assert a.cancelar and m.estado == Estado.REGRESANDO   # en el mismo ciclo
    a = m.paso(Entradas(t=t + 10.5, pose=(1, 0, 0), nav_libre=True))
    assert a.goal == m.largada
    m.paso(Entradas(t=t + 20, pose=(0.1, 0.0, 0.0), nav_resultado='ok'))
    assert m.estado == Estado.AJUSTANDO
    a = m.paso(Entradas(t=t + 20.1, pose=(0.1, 0.0, 0.0)))
    assert a.cmd[0] < 0 and not a.done
    a = m.paso(Entradas(t=t + 22, pose=m.largada))
    assert a.done and m.estado == Estado.TERMINADO
    # Terminada, no hace nada más.
    assert m.paso(Entradas(t=t + 23, pose=(0, 0, 0))).cmd is None


def test_reserva_de_tiempo_para_volver_y_cupos_sobrantes():
    m = mision(cupos=3, tiempo_max_goal_s=1000.0)
    t0 = localizada(m, pose=(0, 0, 0))
    m.paso(Entradas(t=t0 + 1, pose=(0, 0, 0), goal_exploracion=(3.0, 0.0, 0.0)))
    # A 3 m de la largada: volver se estima en 3 · 1.5 / 0.15 + 30 = 60 s.
    a = m.paso(Entradas(t=535.0, pose=(3, 0, 0), nav_libre=False))
    assert m.estado == Estado.EXPLORANDO
    a = m.paso(Entradas(t=545.0, pose=(3, 0, 0), nav_libre=False))
    assert m.estado == Estado.REGRESANDO and a.cancelar
    candidatos = [{'x': 2, 'y': 1, 'n': 2, 'confirmado': False}]
    m.paso(Entradas(t=546.0, pose=(3, 0, 0)))
    a = m.paso(Entradas(t=591.0, pose=(1, 0, 0), nav_libre=False, objetivos=candidatos))
    assert a.done and a.reportes == [(2, 1)]


def test_regreso_fallido_reintenta_y_despues_cierra():
    m = mision()
    t = localizada(m, pose=(0, 0, 0))
    m.paso(Entradas(t=t + 1, pose=(0, 0, 0), exploracion_terminada=True))
    assert m.estado == Estado.REGRESANDO
    assert m.paso(Entradas(t=t + 2, pose=(2, 0, 0))).goal == (0, 0, 0)
    assert m.paso(Entradas(t=t + 9, pose=(2, 0, 0), nav_resultado='falla')).goal == (0, 0, 0)
    a = m.paso(Entradas(t=t + 15, pose=(2, 0, 0), nav_resultado='falla'))
    assert a.done


def test_desatasco_si_no_hay_goal_y_el_robot_esta_quieto():
    m = mision()
    t = localizada(m)
    a = m.paso(Entradas(t=t + 25, pose=(0, 0, 0), quieto_s=21.0))
    assert a.cmd[2] > 0
    a = m.paso(Entradas(t=t + 29, pose=(0, 0, 0), quieto_s=0.0))
    assert a.cmd is None


def test_ajuste_fino_en_el_marco_del_robot():
    m = mision()
    localizada(m, pose=(0, 0, 0))
    m.estado, m.t_estado = Estado.AJUSTANDO, 100.0
    # Robot en (0.04, 0.1) mirando a +y: la largada le queda atrás (-x del robot) y a la derecha.
    a = m.paso(Entradas(t=100.1, pose=(0.04, 0.1, math.pi / 2)))
    vx, vy, _ = a.cmd
    assert vx < 0 and vy > 0
    assert math.hypot(vx, vy) <= 0.1 + 1e-9


def test_goal_fallido_con_obstaculo_cerca_escapa_y_despues_sigue():
    m = mision()
    t = localizada(m)
    m.paso(Entradas(t=t + 0.1, pose=(0, 0, 0), goal_exploracion=(2.0, 0.0, 0.0)))
    # Nav2 falla con una pared a 0.12 m adelante a la izquierda: descarta el goal.
    cerca = (0.1, 0.07)
    a = m.paso(Entradas(t=t + 3, pose=(0, 0, 0), nav_resultado='falla', obstaculo=cerca))
    assert a.descartar == (2.0, 0.0)
    # Ciclo siguiente: se aleja del obstáculo (atrás y a la derecha) en vez de pedir goal.
    a = m.paso(Entradas(t=t + 3.1, pose=(0, 0, 0), obstaculo=cerca,
                        goal_exploracion=(0.0, 2.0, 0.0)))
    vx, vy, wz = a.cmd
    assert a.goal is None and vx < 0 and vy < 0 and wz == 0.0
    assert math.hypot(vx, vy) == pytest.approx(0.08)
    # Ya lejos: frena y recién ahí pide el próximo goal.
    a = m.paso(Entradas(t=t + 5, pose=(0, 0, 0), obstaculo=(0.4, 0.0),
                        goal_exploracion=(0.0, 2.0, 0.0)))
    assert a.cmd == (0.0, 0.0, 0.0) and a.goal == (0.0, 2.0, 0.0)


def test_goal_fallido_sin_obstaculo_cerca_no_escapa():
    m = mision()
    t = localizada(m)
    m.paso(Entradas(t=t + 0.1, pose=(0, 0, 0), goal_exploracion=(2.0, 0.0, 0.0)))
    m.paso(Entradas(t=t + 3, pose=(0, 0, 0), nav_resultado='falla', obstaculo=(0.6, 0.0)))
    a = m.paso(Entradas(t=t + 3.1, pose=(0, 0, 0), obstaculo=(0.6, 0.0),
                        goal_exploracion=(0.0, 2.0, 0.0)))
    assert a.cmd is None and a.goal == (0.0, 2.0, 0.0)


def test_escape_tiene_tiempo_maximo():
    m = mision(tiempo_max_escape_s=2.0)
    t = localizada(m)
    m.paso(Entradas(t=t + 0.1, pose=(0, 0, 0), goal_exploracion=(2.0, 0.0, 0.0)))
    m.paso(Entradas(t=t + 3, pose=(0, 0, 0), nav_resultado='falla', obstaculo=(0.1, 0.0)))
    assert m.paso(Entradas(t=t + 4, pose=(0, 0, 0), obstaculo=(0.1, 0.0))).cmd[0] < 0
    a = m.paso(Entradas(t=t + 5.1, pose=(0, 0, 0), obstaculo=(0.1, 0.0)))
    assert a.cmd == (0.0, 0.0, 0.0)


def test_trabado_sin_goal_y_pegado_a_algo_escapa_en_vez_de_girar():
    m = mision()
    t = localizada(m)
    m.paso(Entradas(t=t + 25, pose=(0, 0, 0), quieto_s=21.0, obstaculo=(0.0, 0.1)))
    a = m.paso(Entradas(t=t + 25.1, pose=(0, 0, 0), quieto_s=21.0, obstaculo=(0.0, 0.1)))
    vx, vy, wz = a.cmd
    assert wz == 0.0 and vy < 0


def test_regreso_fallido_pegado_a_algo_escapa_y_reintenta():
    m = mision()
    t = localizada(m, pose=(0, 0, 0))
    m.paso(Entradas(t=t + 1, pose=(0, 0, 0), exploracion_terminada=True))
    m.paso(Entradas(t=t + 2, pose=(2, 0, 0)))
    a = m.paso(Entradas(t=t + 9, pose=(2, 0, 0), nav_resultado='falla', obstaculo=(-0.1, 0.0)))
    assert a.goal is None
    a = m.paso(Entradas(t=t + 9.1, pose=(2, 0, 0), obstaculo=(-0.1, 0.0)))
    assert a.cmd[0] > 0 and a.goal is None
    a = m.paso(Entradas(t=t + 11, pose=(2, 0, 0), obstaculo=(-0.5, 0.0)))
    assert a.goal == (0, 0, 0)


def test_escape_va_hacia_la_salida_libre_y_no_contra_la_otra_pared():
    m = mision()
    t = localizada(m)
    m.paso(Entradas(t=t + 0.1, pose=(0, 0, 0), goal_exploracion=(2.0, 0.0, 0.0)))
    cerca, salida = (-0.2, 0.0), (0.0, -1.0)            # lo más cercano atrás; libre a la derecha
    m.paso(Entradas(t=t + 3, pose=(0, 0, 0), nav_resultado='falla', obstaculo=cerca,
                    salida=salida))
    a = m.paso(Entradas(t=t + 3.1, pose=(0, 0, 0), obstaculo=cerca, salida=salida))
    assert a.cmd == pytest.approx((0.0, -0.08, 0.0))


def test_escape_cruzado_en_un_hueco_gira_en_vez_de_trasladarse():
    # maze_4: ninguna traslación tenía lugar; el escape pedía avanzar y el supervisor lo frenaba.
    m = mision()
    t = localizada(m)
    m.paso(Entradas(t=t + 0.1, pose=(0, 0, 0), goal_exploracion=(2.0, 0.0, 0.0)))
    cerca, salida = (0.17, 0.12), (1.0, 0.0)
    m.paso(Entradas(t=t + 3, pose=(0, 0, 0), nav_resultado='falla', obstaculo=cerca,
                    salida=salida, giro_salida=-1))
    a = m.paso(Entradas(t=t + 3.1, pose=(0, 0, 0), obstaculo=cerca, salida=salida,
                        giro_salida=-1))
    assert a.cmd == pytest.approx((0.0, 0.0, -0.3))
    a = m.paso(Entradas(t=t + 3.2, pose=(0, 0, 0), obstaculo=cerca, salida=salida))
    assert a.cmd == pytest.approx((0.08, 0.0, 0.0))     # ya hay lugar: se traslada


def test_al_terminar_de_orientarse_avisa_el_goal_visitado():
    m = mision()
    t = localizada(m)
    m.paso(Entradas(t=t + 0.1, pose=(0, 0, 0), goal_exploracion=(1.0, 0.0, 0.5)))
    a = m.paso(Entradas(t=t + 5, pose=(0.9, 0.05, 0.0), nav_resultado='ok'))
    assert m.estado == Estado.ORIENTANDO and a.visitado is None
    a = m.paso(Entradas(t=t + 6, pose=(0.9, 0.05, 0.45)))   # ya orientado
    assert m.estado == Estado.EXPLORANDO and a.visitado == (1.0, 0.0, 0.5)
    a = m.paso(Entradas(t=t + 6.1, pose=(0.9, 0.05, 0.45)))
    assert a.visitado is None                              # una sola vez


def pasillo_con_caja():
    """Puntos del LiDAR (marco del robot en el origen): un paso de 0.38 m (paredes a ±0.19)."""
    import numpy as np
    xs = np.arange(-0.5, 1.5, 0.02)
    return np.vstack([np.column_stack((xs, np.full_like(xs, 0.19))),
                      np.column_stack((xs, np.full_like(xs, -0.19)))])


def test_trabado_frente_a_un_hueco_la_mision_cruza_y_devuelve_el_goal():
    m = mision(usar_paso_angosto=True)
    t = localizada(m)
    goal = (2.0, 0.0, 0.0)
    m.paso(Entradas(t=t + 0.1, pose=(0, 0, 0), goal_exploracion=goal))
    plan = [(0.05 * k, 0.0) for k in range(41)]
    puntos = pasillo_con_caja()
    # Quieto 6 s con el plan de Nav2 y una pared a 0.30: arranca el paso angosto.
    a = m.paso(Entradas(t=t + 7, pose=(0, 0, 0), quieto_s=6.0, obstaculo=(0.0, 0.30),
                        plan=plan, puntos=puntos))
    assert a.cancelar and m.maniobra is not None and any('inicio' in x for x in a.avisos)
    a = m.paso(Entradas(t=t + 7.1, pose=(0, 0, 0), obstaculo=(0.0, 0.30), plan=plan,
                        puntos=puntos))
    vx, vy, wz = a.cmd
    assert vx > 0 and abs(vy) < 0.01 and a.goal is None
    # Cruzó 0.8 m pero sigue en lo angosto (pared a 0.30): todavía no suelta.
    a = m.paso(Entradas(t=t + 20, pose=(0.8, 0, 0), obstaculo=(0.0, 0.30), plan=plan,
                        puntos=puntos))
    assert m.maniobra is not None
    # Ya afuera (lo más cercano a 0.5 m): le devuelve el goal a Nav2.
    a = m.paso(Entradas(t=t + 21, pose=(0.9, 0, 0), obstaculo=(0.0, 0.50), plan=plan,
                        puntos=puntos))
    assert m.maniobra is None and a.goal == goal and any('cruzado' in x for x in a.avisos)


def test_paso_angosto_que_no_avanza_se_abandona_y_descarta_el_goal():
    m = mision(usar_paso_angosto=True)
    t = localizada(m)
    goal = (2.0, 0.0, 0.0)
    m.paso(Entradas(t=t + 0.1, pose=(0, 0, 0), goal_exploracion=goal))
    plan = [(0.05 * k, 0.0) for k in range(41)]
    puntos = pasillo_con_caja()
    m.paso(Entradas(t=t + 7, pose=(0, 0, 0), quieto_s=6.0, obstaculo=(0.0, 0.30),
                    plan=plan, puntos=puntos))
    a = m.paso(Entradas(t=t + 60, pose=(0.1, 0, 0), obstaculo=(0.0, 0.30), plan=plan,
                        puntos=puntos))
    assert m.maniobra is None and a.descartar == goal[:2]
    # Hasta 3 maniobras por goal; después, no más.
    e = Entradas(t=t + 71, pose=(0.1, 0, 0), obstaculo=(0.0, 0.3), plan=plan, puntos=puntos)
    assert m._iniciar_paso(e, Acciones(), goal) and m._iniciar_paso(e, Acciones(), goal)
    assert not m._iniciar_paso(e, Acciones(), goal)


def maniobra_con_plan_corto(m, t, goal):
    """Arranca un paso angosto con un plan de Nav2 de 3 puntos (0.10 m)."""
    m.paso(Entradas(t=t + 0.1, pose=(0, 0, 0), goal_exploracion=goal))
    plan = [(0.0, 0.0), (0.05, 0.0), (0.10, 0.0)]
    m.paso(Entradas(t=t + 7, pose=(0, 0, 0), quieto_s=6.0, obstaculo=(0.0, 0.30),
                    plan=plan, puntos=pasillo_con_caja()))
    assert m.maniobra is not None
    return m.paso(Entradas(t=t + 7.1, pose=(0, 0, 0), obstaculo=(0.0, 0.30), plan=plan,
                           puntos=pasillo_con_caja()))


def test_paso_angosto_que_cruza_sin_avanzar_descarta_el_goal():
    m = mision(usar_paso_angosto=True, descartar_paso_sin_avance=True)
    t = localizada(m)
    goal = (2.0, 0.0, 0.0)
    a = maniobra_con_plan_corto(m, t, goal)
    assert m.maniobra is None and a.goal is None and a.descartar == goal[:2]
    assert not m.lugares_fallidos                       # el hueco no se tapa


def test_sin_el_flag_el_paso_que_cruza_sin_avanzar_devuelve_el_goal():
    m = mision(usar_paso_angosto=True)
    t = localizada(m)
    goal = (2.0, 0.0, 0.0)
    a = maniobra_con_plan_corto(m, t, goal)
    assert m.maniobra is None and a.goal == goal and a.descartar is None


def test_sin_plan_no_hay_paso_angosto_y_sigue_el_escape():
    m = mision()
    t = localizada(m)
    m.paso(Entradas(t=t + 0.1, pose=(0, 0, 0), goal_exploracion=(2.0, 0.0, 0.0)))
    a = m.paso(Entradas(t=t + 7, pose=(0, 0, 0), quieto_s=6.0, obstaculo=(0.0, 0.30)))
    assert m.maniobra is None and not a.cancelar


def test_paso_angosto_sin_progreso_se_rinde_antes():
    m = mision(usar_paso_angosto=True)
    t = localizada(m)
    goal = (2.0, 0.0, 0.0)
    m.paso(Entradas(t=t + 0.1, pose=(0, 0, 0), goal_exploracion=goal))
    plan = [(0.05 * k, 0.0) for k in range(41)]
    puntos = pasillo_con_caja()
    m.paso(Entradas(t=t + 7, pose=(0, 0, 0), quieto_s=6.0, obstaculo=(0.0, 0.30),
                    plan=plan, puntos=puntos))
    a = m.paso(Entradas(t=t + 15, pose=(0.02, 0, 0), obstaculo=(0.0, 0.30), plan=plan,
                        puntos=puntos))
    assert m.maniobra is not None                         # 8 s: todavía espera
    a = m.paso(Entradas(t=t + 23, pose=(0.02, 0, 0), obstaculo=(0.0, 0.30), plan=plan,
                        puntos=puntos))
    assert m.maniobra is None and a.descartar == goal[:2]  # 16 s sin avanzar: se rinde


def pasillo(ancho):
    """Puntos del LiDAR (robot en el origen mirando a +x) de un pasillo de `ancho` m."""
    import numpy as np
    xs = np.arange(-0.5, 1.5, 0.02)
    return np.vstack([np.column_stack((xs, np.full_like(xs, ancho / 2))),
                      np.column_stack((xs, np.full_like(xs, -ancho / 2)))])


def trabado_en(m, ancho, t):
    plan = [(0.05 * k, 0.0) for k in range(41)]
    return m.paso(Entradas(t=t, pose=(0, 0, 0), quieto_s=6.0, obstaculo=(0.0, ancho / 2),
                           plan=plan, puntos=pasillo(ancho)))


def test_hueco_imposible_se_bloquea_y_se_descarta():
    m = mision()
    t = localizada(m)
    goal = (2.0, 0.0, 0.0)
    m.paso(Entradas(t=t + 0.1, pose=(0, 0, 0), goal_exploracion=goal))
    a = trabado_en(m, 0.30, t + 7)
    assert a.bloquear and a.cancelar and a.descartar == goal[:2]
    assert m.maniobra is None                                     # no lo pelea
    assert all(0.0 <= x <= 1.25 and abs(y) <= 0.16 for x, y in a.bloquear)
    assert any('bloqueado' in x for x in a.avisos)


def test_hueco_ancho_no_se_bloquea_ni_se_maniobra():
    m = mision()
    t = localizada(m)
    m.paso(Entradas(t=t + 0.1, pose=(0, 0, 0), goal_exploracion=(2.0, 0.0, 0.0)))
    a = trabado_en(m, 0.60, t + 7)
    assert not a.bloquear and m.maniobra is None


def test_hueco_justo_trabado_lo_cruza_con_la_maniobra():
    m = mision()
    t = localizada(m)
    m.paso(Entradas(t=t + 0.1, pose=(0, 0, 0), goal_exploracion=(2.0, 0.0, 0.0)))
    a = trabado_en(m, 0.38, t + 7)
    assert not a.bloquear and m.maniobra is not None


def test_con_la_maniobra_apagada_un_hueco_justo_se_bloquea():
    m = mision(usar_paso_angosto=False)
    t = localizada(m)
    m.paso(Entradas(t=t + 0.1, pose=(0, 0, 0), goal_exploracion=(2.0, 0.0, 0.0)))
    a = trabado_en(m, 0.38, t + 7)
    assert a.bloquear and m.maniobra is None


def test_con_paso_angosto_activado_lo_cruza_en_vez_de_bloquearlo():
    m = mision(usar_paso_angosto=True)
    t = localizada(m)
    m.paso(Entradas(t=t + 0.1, pose=(0, 0, 0), goal_exploracion=(2.0, 0.0, 0.0)))
    a = trabado_en(m, 0.38, t + 7)
    assert not a.bloquear and m.maniobra is not None


def test_hueco_imposible_al_volver_bloquea_y_replanifica_la_vuelta():
    m = mision()
    t = localizada(m)
    m.paso(Entradas(t=t + 0.1, pose=(0, 0, 0), goal_exploracion=(2.0, 0.0, 0.0)))
    m._iniciar_regreso(Entradas(t=t + 1, pose=(0, 0, 0)), Acciones(), 'prueba')
    m.paso(Entradas(t=t + 1.1, pose=(0, 0, 0), nav_libre=True))
    assert m.regreso_enviado
    a = trabado_en(m, 0.30, t + 8)
    assert a.bloquear and not m.regreso_enviado


def test_ancho_increible_no_se_bloquea_con_el_minimo_creible():
    # El plan corrido por AMCL pisa una pared: el LiDAR "ve" un hueco de 0.04 m.
    for creible, bloquea in ((0.0, True), (0.20, False)):
        m = mision(ancho_minimo_creible_m=creible)
        t = localizada(m)
        m.paso(Entradas(t=t + 0.1, pose=(0, 0, 0), goal_exploracion=(2.0, 0.0, 0.0)))
        a = trabado_en(m, 0.04, t + 7)
        assert bool(a.bloquear) == bloquea
        plan = [(0.05 * k, 0.0) for k in range(41)]
        m2 = mision(ancho_minimo_creible_m=creible)
        t2 = localizada(m2)
        m2.paso(Entradas(t=t2 + 0.1, pose=(0, 0, 0), goal_exploracion=(2.0, 0.0, 0.0)))
        a2 = m2.paso(Entradas(t=t2 + 2, pose=(0, 0, 0), plan=plan,
                              puntos=pasillo_que_se_angosta(0.5, 0.04), obstaculo=(0.5, 0.02)))
        assert bool(a2.bloquear) == bloquea


def test_no_repetir_tapon_no_tapa_dos_veces_el_mismo_lugar():
    m = mision(no_repetir_tapon=True)
    t = localizada(m)
    m.paso(Entradas(t=t + 0.1, pose=(0, 0, 0), goal_exploracion=(2.0, 0.0, 0.0)))
    m._iniciar_regreso(Entradas(t=t + 1, pose=(0, 0, 0)), Acciones(), 'prueba')
    m.paso(Entradas(t=t + 1.1, pose=(0, 0, 0), nav_libre=True))
    assert trabado_en(m, 0.30, t + 8).bloquear
    m.paso(Entradas(t=t + 20, pose=(0, 0, 0), nav_libre=True))
    assert not trabado_en(m, 0.30, t + 40).bloquear


def test_regreso_imposible_con_tapones_los_libera_una_vez_y_reintenta():
    for liberar in (False, True):
        m = mision(liberar_tapones_al_volver=liberar, intentos_regreso=1)
        t = localizada(m)
        m.paso(Entradas(t=t + 0.1, pose=(0, 0, 0), goal_exploracion=(2.0, 0.0, 0.0)))
        assert trabado_en(m, 0.30, t + 7).bloquear                  # un tapón puesto
        m._iniciar_regreso(Entradas(t=t + 8, pose=(2.0, 0, 0)), Acciones(), 'prueba')
        m.paso(Entradas(t=t + 8.1, pose=(2.0, 0, 0), nav_libre=True))
        a = m.paso(Entradas(t=t + 30, pose=(2.0, 0, 0), nav_resultado='falla'))
        if not liberar:
            assert m.estado == Estado.TERMINADO
            continue
        assert a.liberar_tapones and m.estado == Estado.REGRESANDO and not m.regreso_enviado
        m.paso(Entradas(t=t + 31, pose=(2.0, 0, 0), nav_libre=True))
        assert m.regreso_enviado                                     # reintenta la vuelta
        a = m.paso(Entradas(t=t + 60, pose=(2.0, 0, 0), nav_resultado='falla'))
        assert not a.liberar_tapones and m.estado == Estado.TERMINADO  # sólo una vez


def pasillo_que_se_angosta(desde_x, ancho):
    """Pasillo de 0.9 m que a partir de `desde_x` (marco del robot) se angosta a `ancho`."""
    import numpy as np
    xs = np.arange(-0.5, 2.0, 0.02)
    medio = np.where(xs < desde_x, 0.45, ancho / 2)
    return np.vstack([np.column_stack((xs, medio)), np.column_stack((xs, -medio))])


def test_hueco_imposible_adelante_se_bloquea_antes_de_entrar():
    m = mision()
    t = localizada(m)
    goal = (2.0, 0.0, 0.0)
    m.paso(Entradas(t=t + 0.1, pose=(0, 0, 0), goal_exploracion=goal))
    plan = [(0.05 * k, 0.0) for k in range(41)]
    # Andando (no trabado): a 0.5 m el pasillo se angosta a 0.30.
    a = m.paso(Entradas(t=t + 2, pose=(0, 0, 0), plan=plan,
                        puntos=pasillo_que_se_angosta(0.5, 0.30), obstaculo=(0.5, 0.15)))
    assert a.bloquear and a.descartar == goal[:2] and a.cancelar
    assert min(x for x, _ in a.bloquear) > 0.3                   # el tapón, en el hueco
    assert any('antes de entrar' in x for x in a.avisos)


def test_pasillo_normal_adelante_no_se_bloquea():
    m = mision()
    t = localizada(m)
    m.paso(Entradas(t=t + 0.1, pose=(0, 0, 0), goal_exploracion=(2.0, 0.0, 0.0)))
    plan = [(0.05 * k, 0.0) for k in range(41)]
    a = m.paso(Entradas(t=t + 2, pose=(0, 0, 0), plan=plan,
                        puntos=pasillo_que_se_angosta(0.5, 0.80), obstaculo=(0.5, 0.4)))
    assert not a.bloquear and not a.cancelar


def test_el_mismo_hueco_no_se_bloquea_dos_veces():
    m = mision()
    t = localizada(m)
    m.paso(Entradas(t=t + 0.1, pose=(0, 0, 0), goal_exploracion=(2.0, 0.0, 0.0)))
    plan = [(0.05 * k, 0.0) for k in range(41)]
    puntos = pasillo_que_se_angosta(0.5, 0.30)
    assert m.paso(Entradas(t=t + 2, pose=(0, 0, 0), plan=plan, puntos=puntos,
                           obstaculo=(0.5, 0.15))).bloquear
    m.paso(Entradas(t=t + 3, pose=(0, 0, 0), goal_exploracion=(2.0, 0.3, 0.0)))
    a = m.paso(Entradas(t=t + 6, pose=(0, 0, 0), plan=plan, puntos=puntos,
                        obstaculo=(0.5, 0.15)))
    assert not a.bloquear                         # el plan viejo todavía pasa por el tapón


def test_el_giro_de_localizacion_no_cuenta_como_trabado_para_el_primer_goal():
    # Quieto (sin trasladarse) desde antes, por el giro de localización: el goal recién
    # mandado no se descarta.
    m = mision()
    t = localizada(m)
    m.paso(Entradas(t=t + 0.1, pose=(0, 0, 0), goal_exploracion=(2.0, 0.0, 0.0),
                    quieto_s=30.0))
    a = m.paso(Entradas(t=t + 2, pose=(0, 0, 0), quieto_s=32.0, obstaculo=(0.0, 0.3)))
    assert a.descartar is None and not a.cancelar and m.goal_actual is not None
    # Pero si el goal mismo no avanza, sí.
    a = m.paso(Entradas(t=t + 25, pose=(0, 0, 0), quieto_s=55.0))
    assert a.descartar == (2.0, 0.0)


def test_hueco_justo_adelante_arranca_la_maniobra_al_acercarse():
    m = mision()
    t = localizada(m)
    m.paso(Entradas(t=t + 0.1, pose=(0, 0, 0), goal_exploracion=(2.0, 0.0, 0.0)))
    plan = [(0.05 * k, 0.0) for k in range(41)]
    # A 0.8 m todavía no: sigue Nav2.
    a = m.paso(Entradas(t=t + 2, pose=(0, 0, 0), plan=plan,
                        puntos=pasillo_que_se_angosta(0.8, 0.40), obstaculo=(0.8, 0.2)))
    assert m.maniobra is None and not a.cancelar
    # A 0.4 m: la misión toma el control.
    a = m.paso(Entradas(t=t + 3, pose=(0, 0, 0), plan=plan,
                        puntos=pasillo_que_se_angosta(0.4, 0.40), obstaculo=(0.4, 0.2)))
    assert m.maniobra is not None and a.cancelar and not a.bloquear


def test_la_maniobra_sigue_el_plan_en_odom_aunque_amcl_salte():
    m = mision()
    t = localizada(m)
    m.paso(Entradas(t=t + 0.1, pose=(0, 0, 0), goal_exploracion=(2.0, 0.0, 0.0)))
    plan = [(0.05 * k, 0.0) for k in range(41)]
    puntos = pasillo_que_se_angosta(0.4, 0.40)
    m.paso(Entradas(t=t + 3, pose=(0, 0, 0), pose_odom=(5.0, 5.0, 0.0), plan=plan,
                    puntos=puntos, obstaculo=(0.4, 0.2)))
    assert m.maniobra['marco'] == 'odom'
    # AMCL salta 0.4 m hacia adelante; la odometría no: el comando sigue hacia +x.
    a = m.paso(Entradas(t=t + 3.1, pose=(0.4, 0, 0), pose_odom=(5.0, 5.0, 0.0), plan=plan,
                        puntos=puntos, obstaculo=(0.4, 0.2)))
    assert a.cmd[0] > 0 and m.maniobra is not None


def test_la_cobertura_terminada_no_corta_una_maniobra_en_curso():
    m = mision()
    t = localizada(m)
    m.paso(Entradas(t=t + 0.1, pose=(0, 0, 0), goal_exploracion=(2.0, 0.0, 0.0)))
    plan = [(0.05 * k, 0.0) for k in range(41)]
    puntos = pasillo_que_se_angosta(0.4, 0.40)
    m.paso(Entradas(t=t + 3, pose=(0, 0, 0), plan=plan, puntos=puntos, obstaculo=(0.4, 0.2)))
    assert m.maniobra is not None
    m.paso(Entradas(t=t + 3.1, pose=(0, 0, 0), plan=plan, puntos=puntos, obstaculo=(0.4, 0.2),
                    exploracion_terminada=True))
    assert m.estado == Estado.EXPLORANDO and m.maniobra is not None


def test_goal_justo_en_el_borde_del_descarte_se_acepta():
    m = mision()
    t = localizada(m)
    m.paso(Entradas(t=t + 0.1, pose=(0, 0, 0), goal_exploracion=(2.0, 0.0, 0.0)))
    m.paso(Entradas(t=t + 3, pose=(0, 0, 0), nav_resultado='falla'))
    a = m.paso(Entradas(t=t + 3.1, pose=(0, 0, 0), goal_exploracion=(2.3, 0.0, 0.0)))
    assert a.goal == (2.3, 0.0, 0.0)              # a 0.30: el explorador lo propone, se acepta


def test_mucho_tiempo_sin_goal_usable_vuelve_a_casa():
    m = mision()
    t = localizada(m)
    m.paso(Entradas(t=t + 0.1, pose=(0, 0, 0), goal_exploracion=(2.0, 0.0, 0.0)))
    m.paso(Entradas(t=t + 3, pose=(0, 0, 0), nav_resultado='falla'))
    # El explorador insiste con algo descartado: no hay goal usable.
    for k in range(25):
        m.paso(Entradas(t=t + 4 + k, pose=(0, 0, 0), goal_exploracion=(2.05, 0.0, 0.0)))
    assert m.estado == Estado.REGRESANDO and m.motivo == 'sin goals útiles'


def test_en_una_inspeccion_se_queda_quieto_mirando_antes_de_seguir():
    m = mision()
    t = localizada(m)
    m.paso(Entradas(t=t + 0.1, pose=(0, 0, 0), goal_exploracion=(1.0, 0.0, 0.5),
                    goal_es_inspeccion=True))
    m.paso(Entradas(t=t + 5, pose=(1.0, 0.0, 0.0), nav_resultado='ok'))
    assert m.estado == Estado.ORIENTANDO
    a = m.paso(Entradas(t=t + 6, pose=(1.0, 0.0, 0.5)))       # ya orientado: mira
    assert m.estado == Estado.ORIENTANDO and a.cmd == (0.0, 0.0, 0.0)
    a = m.paso(Entradas(t=t + 7.6, pose=(1.0, 0.0, 0.5)))     # pasaron 1.5 s: sigue
    assert m.estado == Estado.EXPLORANDO and a.visitado == (1.0, 0.0, 0.5)


def inspeccion_orientada(m, t, ident=7):
    """Manda a mirar el candidato `ident` desde (1, 0) y lo deja orientado (t + 6)."""
    m.paso(Entradas(t=t + 0.1, pose=(0, 0, 0), goal_exploracion=(1.0, 0.0, 0.5),
                    goal_es_inspeccion=True, inspeccion_id=ident))
    m.paso(Entradas(t=t + 5, pose=(1.0, 0.0, 0.0), nav_resultado='ok'))
    return m.paso(Entradas(t=t + 6, pose=(1.0, 0.0, 0.5),
                           candidatos=[{'id': ident, 'estado': 'sin_inspeccionar'}]))


def test_esperar_decision_se_queda_mirando_hasta_que_se_decide():
    m = mision(esperar_decision=True, espera_max_decision_s=4.0)
    t = localizada(m)
    inspeccion_orientada(m, t)
    sin_decidir = [{'id': 7, 'estado': 'sin_inspeccionar'}]
    a = m.paso(Entradas(t=t + 8, pose=(1.0, 0.0, 0.5), candidatos=sin_decidir))
    assert m.estado == Estado.ORIENTANDO and a.cmd == (0.0, 0.0, 0.0)   # más que la pausa
    a = m.paso(Entradas(t=t + 8.5, pose=(1.0, 0.0, 0.5),
                        candidatos=[{'id': 7, 'estado': 'descartado'}]))
    assert m.estado == Estado.EXPLORANDO and a.visitado == (1.0, 0.0, 0.5)


def test_esperar_decision_tiene_un_maximo():
    m = mision(esperar_decision=True, espera_max_decision_s=4.0)
    t = localizada(m)
    inspeccion_orientada(m, t)
    sin_decidir = [{'id': 7, 'estado': 'sin_inspeccionar'}]
    m.paso(Entradas(t=t + 9.9, pose=(1.0, 0.0, 0.5), candidatos=sin_decidir))
    assert m.estado == Estado.ORIENTANDO
    a = m.paso(Entradas(t=t + 10.1, pose=(1.0, 0.0, 0.5), candidatos=sin_decidir))
    assert m.estado == Estado.EXPLORANDO and a.visitado == (1.0, 0.0, 0.5)


def test_esperar_decision_no_espera_si_ya_estaba_decidido_o_se_borro():
    m = mision(esperar_decision=True)
    t = localizada(m)
    inspeccion_orientada(m, t)
    m.paso(Entradas(t=t + 6.1, pose=(1.0, 0.0, 0.5), candidatos=[]))   # borrado
    assert m.estado == Estado.EXPLORANDO


def test_esperar_decision_no_toma_la_inspeccion_vieja_del_candidato_ya_decidido():
    # laberinto_puertas: salió apenas se descartó y tomó la inspección del mismo candidato
    # que el explorador todavía no había reemplazado.
    m = mision(esperar_decision=True)
    t = localizada(m)
    inspeccion_orientada(m, t)
    decidido = [{'id': 7, 'estado': 'descartado'}]
    m.paso(Entradas(t=t + 6.5, pose=(1.0, 0.0, 0.5), candidatos=decidido))
    assert m.estado == Estado.EXPLORANDO
    a = m.paso(Entradas(t=t + 6.6, pose=(1.0, 0.0, 0.5), goal_exploracion=(1.2, 0.5, 0.4),
                        goal_es_inspeccion=True, inspeccion_id=7, candidatos=decidido))
    assert a.goal is None
    a = m.paso(Entradas(t=t + 7.0, pose=(1.0, 0.0, 0.5), goal_exploracion=(3.0, 0.0, 0.0),
                        candidatos=decidido))
    assert a.goal == (3.0, 0.0, 0.0)


def test_sin_esperar_decision_la_pausa_es_la_de_siempre():
    m = mision()
    t = localizada(m)
    inspeccion_orientada(m, t)
    m.paso(Entradas(t=t + 7.6, pose=(1.0, 0.0, 0.5),
                    candidatos=[{'id': 7, 'estado': 'sin_inspeccionar'}]))
    assert m.estado == Estado.EXPLORANDO


def test_un_goal_de_cobertura_no_espera():
    m = mision()
    t = localizada(m)
    m.paso(Entradas(t=t + 0.1, pose=(0, 0, 0), goal_exploracion=(1.0, 0.0, 0.5)))
    m.paso(Entradas(t=t + 5, pose=(1.0, 0.0, 0.0), nav_resultado='ok'))
    m.paso(Entradas(t=t + 6, pose=(1.0, 0.0, 0.5)))
    assert m.estado == Estado.EXPLORANDO


def test_si_el_giro_no_avanza_deja_de_orientarse_enseguida():
    m = mision()
    t = localizada(m)
    m.paso(Entradas(t=t + 0.1, pose=(0, 0, 0), goal_exploracion=(1.0, 0.0, 1.5)))
    m.paso(Entradas(t=t + 5, pose=(1.0, 0.0, 0.0), nav_resultado='ok'))
    assert m.estado == Estado.ORIENTANDO
    m.paso(Entradas(t=t + 5.5, pose=(1.0, 0.0, 0.0)))        # el giro no avanza (frenado)
    a = m.paso(Entradas(t=t + 7.2, pose=(1.0, 0.0, 0.0)))    # 1.7 s sin avanzar: sigue
    assert m.estado == Estado.EXPLORANDO and a.visitado == (1.0, 0.0, 1.5)


def maniobra_sin_avance(m, t, goal, pose=(0, 0, 0)):
    """Arranca una maniobra trabado frente a un paso de 0.38 m y la deja rendirse sin avanzar."""
    plan = [(pose[0] + 0.05 * k, pose[1]) for k in range(41)]
    puntos = pasillo_con_caja()
    m.paso(Entradas(t=t, pose=pose, goal_exploracion=goal))
    m.paso(Entradas(t=t + 7, pose=pose, quieto_s=6.0, obstaculo=(0.0, 0.30), plan=plan,
                    puntos=puntos))
    assert m.maniobra is not None
    return m.paso(Entradas(t=t + 30, pose=pose, obstaculo=(0.0, 0.30), plan=plan,
                           puntos=puntos)), plan, puntos


def test_maniobra_sin_avance_tapa_el_hueco_y_no_reintenta_ahi_con_otro_goal():
    # maze_1 y laberinto_s4: el explorador proponía el mismo candidato desde otra pose, el
    # contador por goal volvía a cero y el robot se quedaba frente al hueco hasta el corte.
    m = mision(usar_paso_angosto=True)
    t = localizada(m)
    a, plan, puntos = maniobra_sin_avance(m, t + 0.1, (2.0, 0.0, 0.0))
    assert m.maniobra is None and a.bloquear and any('sin avance' in x for x in a.avisos)
    # Otro goal por el mismo hueco: no hay maniobra, se tapa y se descarta.
    otro = (2.0, 0.3, 0.0)
    m.paso(Entradas(t=t + 40, pose=(0.05, 0, 0), goal_exploracion=otro))
    a = m.paso(Entradas(t=t + 47, pose=(0.05, 0, 0), quieto_s=6.0, obstaculo=(0.0, 0.30),
                        plan=plan, puntos=puntos))
    assert m.maniobra is None and a.bloquear and a.descartar == otro[:2]


def test_maniobra_que_avanzo_algo_no_marca_el_lugar():
    m = mision(usar_paso_angosto=True)
    t = localizada(m)
    goal = (2.0, 0.0, 0.0)
    plan = [(0.05 * k, 0.0) for k in range(41)]
    puntos = pasillo_con_caja()
    m.paso(Entradas(t=t + 0.1, pose=(0, 0, 0), goal_exploracion=goal))
    m.paso(Entradas(t=t + 7, pose=(0, 0, 0), quieto_s=6.0, obstaculo=(0.0, 0.30), plan=plan,
                    puntos=puntos))
    m.paso(Entradas(t=t + 10, pose=(0.3, 0, 0), obstaculo=(0.0, 0.30), plan=plan,
                    puntos=puntos))
    a = m.paso(Entradas(t=t + 60, pose=(0.3, 0, 0), obstaculo=(0.0, 0.30), plan=plan,
                        puntos=puntos))
    assert m.maniobra is None and not a.bloquear and m.lugares_fallidos == []


def test_vigilante_corta_una_maniobra_quieta_antes_del_corte_del_juez():
    # Con tiempo_sin_progreso alto, la maniobra sola no se rendiría: corta el vigilante.
    m = mision(usar_paso_angosto=True, tiempo_sin_progreso_paso_s=100.0)
    t = localizada(m)
    goal = (2.0, 0.0, 0.0)
    plan = [(0.05 * k, 0.0) for k in range(41)]
    puntos = pasillo_con_caja()
    m.paso(Entradas(t=t + 0.1, pose=(0, 0, 0), goal_exploracion=goal))
    m.paso(Entradas(t=t + 7, pose=(0, 0, 0), quieto_s=6.0, obstaculo=(0.0, 0.30), plan=plan,
                    puntos=puntos))
    a = m.paso(Entradas(t=t + 20, pose=(0, 0, 0), quieto_s=20.0, obstaculo=(0.0, 0.30),
                        plan=plan, puntos=puntos))
    assert m.maniobra is not None
    a = m.paso(Entradas(t=t + 31, pose=(0, 0, 0), quieto_s=31.0, obstaculo=(0.0, 0.25),
                        salida=(-1.0, 0.0), plan=plan, puntos=puntos))
    assert m.maniobra is None and a.descartar == goal[:2] and a.bloquear
    assert any('sin moverse' in x for x in a.avisos)
    a = m.paso(Entradas(t=t + 31.1, pose=(0, 0, 0), quieto_s=31.1, obstaculo=(0.0, 0.25),
                        salida=(-1.0, 0.0)))
    assert a.cmd[0] < 0                                     # escapa hacia la salida libre
    # No vuelve a cortar enseguida: espera otro período entero.
    a = m.paso(Entradas(t=t + 40, pose=(0, 0, 0), quieto_s=40.0))
    assert not any('sin moverse' in x for x in a.avisos)


def test_vigilante_al_volver_reintenta_la_vuelta():
    m = mision()
    t = localizada(m)
    m._iniciar_regreso(Entradas(t=t + 1, pose=(1, 0, 0)), Acciones(), 'prueba')
    a = m.paso(Entradas(t=t + 2, pose=(1, 0, 0)))
    assert a.goal is not None and m.regreso_enviado
    a = m.paso(Entradas(t=t + 33, pose=(1, 0, 0), quieto_s=31.0))
    assert a.cancelar and not m.regreso_enviado


# Cruzar huecos medidos con el LiDAR (r9) ----------------------------------------------------

from equipo_jar.maquina_mision import hueco_hacia, plan_por_hueco  # noqa: E402
import numpy as np  # noqa: E402

# Puerta de D en laberinto_puertas: pared en y = 2.5, hueco de 0.40 m entre x = 1.55 y 1.95.
HUECO_D = {'a': (1.55, 2.5), 'b': (1.95, 2.5), 'ancho': 0.40}
DENTRO_D = (1.75, 3.2, -1.57)
AFUERA = (1.75, 1.2, 0.0)


def test_hueco_hacia_elige_el_que_separa_al_robot_del_goal():
    centro, normal, ancho = hueco_hacia(DENTRO_D, AFUERA, [HUECO_D], 1.2)
    assert centro == pytest.approx((1.75, 2.5)) and ancho == 0.40
    assert normal == pytest.approx((0.0, -1.0))            # de D hacia afuera
    assert hueco_hacia(DENTRO_D, (1.75, 3.8, 0.0), [HUECO_D], 1.2) is None   # mismo lado
    assert hueco_hacia((1.75, 4.2, 0.0), AFUERA, [HUECO_D], 1.2) is None     # lejos
    otro = {'a': (2.55, 2.5), 'b': (2.95, 2.5), 'ancho': 0.40}
    assert hueco_hacia(DENTRO_D, AFUERA, [otro, HUECO_D], 1.2)[0] == pytest.approx((1.75, 2.5))


def test_plan_por_hueco_pasa_por_el_centro():
    plan = plan_por_hueco(DENTRO_D, (1.75, 2.5), (0.0, -1.0), 0.4, 0.5)
    assert plan[0] == pytest.approx(DENTRO_D[:2]) and plan[-1] == pytest.approx((1.75, 2.0))
    assert min(math.hypot(x - 1.75, y - 2.5) for x, y in plan) < 1e-9
    assert max(math.hypot(x1 - x0, y1 - y0)
               for (x0, y0), (x1, y1) in zip(plan, plan[1:])) <= 0.05 + 1e-9


def trabada_en_d(m, t, huecos):
    """Explorando dentro de D con el goal afuera: Nav2 sin plan y el robot quieto 6 s."""
    m.paso(Entradas(t=t + 0.1, pose=DENTRO_D, goal_exploracion=AFUERA))
    return m.paso(Entradas(t=t + 7, pose=DENTRO_D, quieto_s=6.0, obstaculo=(0.6, 0.0),
                           puntos=np.zeros((0, 2)), huecos=huecos))


def test_trabado_sin_plan_cruza_por_el_hueco_medido():
    m = mision(usar_paso_angosto=True, cruzar_huecos=True)
    t = localizada(m)
    a = trabada_en_d(m, t, [HUECO_D])
    assert m.maniobra is not None and a.cancelar
    assert any('cruza por el hueco medido de 0.40 m' in x for x in a.avisos)
    assert m.maniobra['goal'] == AFUERA
    assert min(math.hypot(x - 1.75, y - 2.5) for x, y in m.maniobra['plan_map']) < 0.05


def test_sin_el_flag_no_cruza_por_el_hueco_medido():
    m = mision(usar_paso_angosto=True)
    t = localizada(m)
    trabada_en_d(m, t, [HUECO_D])
    assert m.maniobra is None


def con_cuello(m):
    m.buscar_cuello = lambda pose, goal: ((1.75, 2.5), (0.0, -1.0))
    return m


def falla_dentro_de_d(m, t):
    m.paso(Entradas(t=t + 0.1, pose=DENTRO_D, goal_exploracion=AFUERA))
    return m.paso(Entradas(t=t + 1, pose=DENTRO_D, nav_resultado='falla'))


def test_antes_de_descartar_va_a_medir_y_cruza_si_hay_hueco():
    m = con_cuello(mision(usar_paso_angosto=True, cruzar_huecos=True,
                          medir_antes_de_descartar=True))
    t = localizada(m)
    a = falla_dentro_de_d(m, t)
    assert a.descartar is None
    assert a.goal == pytest.approx((1.75, 3.1, -math.pi / 2))   # 0.6 m antes del cuello
    m.paso(Entradas(t=t + 8, pose=(1.75, 3.1, -1.57), nav_resultado='ok'))
    a = m.paso(Entradas(t=t + 9, pose=(1.75, 3.1, -1.57)))
    assert a.cmd == (0.0, 0.0, 0.0) and m.maniobra is None       # quieto, midiendo
    a = m.paso(Entradas(t=t + 10.5, pose=(1.75, 3.1, -1.57), puntos=np.zeros((0, 2)),
                        huecos=[HUECO_D]))
    assert m.maniobra is not None and m.maniobra['goal'] == AFUERA


def test_medido_sin_hueco_pasable_recien_ahi_descarta_y_no_vuelve_a_medir():
    m = con_cuello(mision(usar_paso_angosto=True, cruzar_huecos=True,
                          medir_antes_de_descartar=True))
    t = localizada(m)
    falla_dentro_de_d(m, t)
    m.paso(Entradas(t=t + 8, pose=(1.75, 3.1, -1.57), nav_resultado='ok'))
    a = m.paso(Entradas(t=t + 10.5, pose=(1.75, 3.1, -1.57), huecos=[]))
    assert a.descartar == AFUERA[:2]
    assert any('no hay hueco pasable' in x for x in a.avisos)
    assert m.goals_medidos == [AFUERA[:2]]


def test_sin_medir_antes_de_descartar_descarta_como_siempre():
    m = con_cuello(mision(usar_paso_angosto=True, cruzar_huecos=True))
    t = localizada(m)
    a = falla_dentro_de_d(m, t)
    assert a.descartar == AFUERA[:2]


def test_trabado_al_lado_del_goal_se_da_por_alcanzado():
    m = mision(radio_llegada_m=0.35)
    t = localizada(m)
    goal = (1.90, 2.90, 0.89)
    m.paso(Entradas(t=t + 0.1, pose=(1.75, 2.74, 0.0), goal_exploracion=goal))
    a = m.paso(Entradas(t=t + 7, pose=(1.75, 2.74, 0.0), quieto_s=6.0, obstaculo=(0.2, 0.0)))
    assert m.estado == Estado.ORIENTANDO and a.descartar is None and a.cancelar
    assert m.yaw_objetivo == goal[2] and m.goal_llegado == goal


def test_paso_sin_avance_al_lado_del_goal_no_lo_descarta():
    m = mision(usar_paso_angosto=True, descartar_paso_sin_avance=True)
    t = localizada(m)
    goal = (0.25, 0.0, 0.0)                         # a 0.25 m: ya llegó
    m.paso(Entradas(t=t + 0.1, pose=(0, 0, 0), goal_exploracion=goal))
    plan = [(0.0, 0.0), (0.05, 0.0), (0.10, 0.0)]
    m.paso(Entradas(t=t + 7, pose=(0, 0, 0), quieto_s=6.0, obstaculo=(0.0, 0.30),
                    plan=plan, puntos=pasillo_con_caja()))
    assert m.maniobra is not None                   # (con el radio apagado arranca la maniobra)
    m.config.radio_llegada_m = 0.35
    a = m.paso(Entradas(t=t + 7.1, pose=(0, 0, 0), obstaculo=(0.0, 0.30), plan=plan,
                        puntos=pasillo_con_caja()))
    assert m.estado == Estado.ORIENTANDO and a.descartar is None
    # Lejos del goal sigue descartando.
    m2 = mision(usar_paso_angosto=True, descartar_paso_sin_avance=True, radio_llegada_m=0.35)
    t = localizada(m2)
    a = maniobra_con_plan_corto(m2, t, (2.0, 0.0, 0.0))
    assert a.descartar == (2.0, 0.0)


def giro_frenado(m, t, frenado_s, quieto_s=3.0):
    return m.paso(Entradas(t=t, pose=(1.0, 1.0, 0.0), quieto_s=quieto_s, obstaculo=(0.15, 0.1),
                           salida=(-1.0, 0.0), giro_frenado_s=frenado_s))


def test_giro_frenado_escapa_sin_soltar_el_goal():
    m = mision(escapar_giro_frenado=True)
    t = localizada(m)
    goal = (3.0, 1.0, 0.0)
    m.paso(Entradas(t=t + 0.1, pose=(1.0, 1.0, 0.0), goal_exploracion=goal))
    a = giro_frenado(m, t + 1.0, frenado_s=1.0)
    assert m.escape_hasta is None                       # todavía no
    a = giro_frenado(m, t + 3.0, frenado_s=2.5)
    assert m.escape_hasta is not None and not a.cancelar and a.descartar is None
    assert m.goal_actual == goal
    a = giro_frenado(m, t + 3.1, frenado_s=2.6)
    assert a.cmd[0] < 0                                 # se aleja (salida hacia -x)


def test_giro_frenado_no_escapa_sin_el_flag_ni_dos_veces_seguidas():
    m = mision()
    t = localizada(m)
    m.paso(Entradas(t=t + 0.1, pose=(1.0, 1.0, 0.0), goal_exploracion=(3.0, 1.0, 0.0)))
    giro_frenado(m, t + 3.0, frenado_s=3.0)
    assert m.escape_hasta is None
    m = mision(escapar_giro_frenado=True)
    t = localizada(m)
    m.paso(Entradas(t=t + 0.1, pose=(1.0, 1.0, 0.0), goal_exploracion=(3.0, 1.0, 0.0)))
    giro_frenado(m, t + 3.0, frenado_s=3.0)
    m.escape_hasta = None                               # terminó el escape
    giro_frenado(m, t + 5.0, frenado_s=3.0)
    assert m.escape_hasta is None                       # espera_escape_giro_s
    giro_frenado(m, t + 9.0, frenado_s=3.0)
    assert m.escape_hasta is not None
