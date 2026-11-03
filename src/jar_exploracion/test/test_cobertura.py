import math

import numpy as np
from scipy import ndimage

from jar_exploracion.cobertura import Grilla

RES = 0.05


def sala(ancho_m=4.0, alto_m=2.0, pared_media=True):
    """Dos cuartos de 2×2 m (origen en -1, -1) separados por una pared con una puerta."""
    w, h = int(ancho_m / RES), int(alto_m / RES)
    m = np.zeros((h, w), dtype=np.int8)
    m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = 100
    if pared_media:
        m[:, w // 2] = 100
        m[h // 2 - 6:h // 2 + 6, w // 2] = 0          # puerta de 0.6 m
    return Grilla(m.ravel().tolist(), w, h, RES, -1.0, -1.0)


def test_grilla_gruesa_conserva_paredes():
    g = sala()
    assert g.res == 0.10
    assert g.forma == (20, 40)
    assert g.ocupado[0, :].all() and g.ocupado[:, 0].all()
    assert g.ocupado[2, 20] and not g.ocupado[10, 20]   # pared media y su puerta
    assert not g.ocupado[g.celda(0.0, 0.0)]


def test_marcar_visto_no_atraviesa_paredes():
    g = sala(pared_media=False)
    g.marcar_visto(0.0, 0.0, 0.0, fov=1.0, alcance=1.5)
    adelante = g.celda(0.8, 0.0)
    atras = g.celda(-0.7, 0.0)
    assert g.visto[adelante] and not g.visto[atras]
    # La pared derecha está en x = 3 m - 1 = 2.95: más allá de 1.5 m de alcance nada.
    assert not g.visto[g.celda(1.7, 0.0)]


def test_cobertura_sube_y_el_goal_apunta_a_lo_no_visto():
    g = sala()
    x, y = -0.2, 0.0                                    # en el cuarto izquierdo
    for yaw in np.linspace(0, 2 * math.pi, 12, endpoint=False):
        g.marcar_visto(x, y, yaw, fov=1.0, alcance=2.0)
    cobertura_un_cuarto = g.cobertura(x, y)
    assert 0.3 < cobertura_un_cuarto < 0.7
    goal = g.mejor_goal(x, y, fov=1.0, alcance=2.0)
    assert goal is not None
    gx, gy, gyaw, ganancia = goal
    # Va hacia la puerta y mira al cuarto derecho, que es lo que falta ver.
    assert gx > 0.5 and abs(gyaw) < math.pi / 2
    assert ganancia > 10


def test_sin_nada_por_ver_no_hay_goal():
    g = sala()
    g.visto[:] = True
    assert g.mejor_goal(-0.2, 0.0, fov=1.0, alcance=2.0) is None
    assert g.cobertura(-0.2, 0.0) == 1.0


def test_descartados_no_se_proponen():
    g = sala(pared_media=False)
    gx, gy, _, _ = g.mejor_goal(-0.5, 0.0, fov=1.0, alcance=1.0)
    otro = g.mejor_goal(-0.5, 0.0, fov=1.0, alcance=1.0, descartados=[(gx, gy)])
    assert otro is not None
    assert math.hypot(otro[0] - gx, otro[1] - gy) >= 0.3


def test_goal_respeta_el_despeje_de_las_paredes():
    g = sala()
    for _ in range(3):
        goal = g.mejor_goal(-0.2, 0.0, fov=1.0, alcance=2.0, despeje_min=0.3)
        i, j = g.celda(goal[0], goal[1])
        assert g.despeje[i, j] >= 0.3
        g.marcar_visto(goal[0], goal[1], goal[2], fov=1.0, alcance=2.0)


def sala_puerta_abajo():
    """Dos cuartos de 2×2 m (origen en -1, -1) con la puerta abajo de todo, en y ≈ -0.7."""
    w, h = int(4.0 / RES), int(2.0 / RES)
    m = np.zeros((h, w), dtype=np.int8)
    m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = 100
    m[:, w // 2] = 100
    m[2:8, w // 2] = 0                                  # puerta de 0.3 m, de y=-0.9 a y=-0.6
    return Grilla(m.ravel().tolist(), w, h, RES, -1.0, -1.0)


def test_distancia_por_camino_rodea_las_paredes():
    g = sala_puerta_abajo()
    mascara = g.despeje >= 0.10
    d = g.distancia_por_camino(mascara, 0.7, 0.7)
    # (1.3, 0.7) está a 0.6 m en línea recta, pero del otro lado de la pared.
    assert d[g.celda(1.3, 0.7)] > 2.5
    assert d[g.celda(0.3, 0.7)] < 0.5


def test_goal_prefiere_lo_cercano_por_camino_y_no_en_linea_recta():
    g = sala_puerta_abajo()
    g.visto[:] = True
    # Falta ver un parche de 0.4×0.4 m en cada cuarto.
    for x0, y0 in ((1.2, 0.3), (-0.8, -0.8)):
        i0, j0 = g.celda(x0, y0)
        g.visto[i0:i0 + 4, j0:j0 + 4] = False
    x, y = 0.7, 0.7                                     # cuarto izquierdo, pegado a la pared
    gx, gy, _, _ = g.mejor_goal(x, y, fov=1.0, alcance=0.6, despeje_min=0.10)
    # El parche del cuarto derecho está más cerca en línea recta, pero hay que dar la
    # vuelta por la puerta de abajo: conviene el del propio cuarto.
    assert gx < 1.0


def test_radio_local_termina_lo_cercano_antes_de_irse():
    g = sala(pared_media=False)
    g.visto[:] = True
    i0, j0 = g.celda(-0.9, 0.5)
    g.visto[i0:i0 + 4, j0:j0 + 5] = False               # un rincón chico al lado del robot
    i1, j1 = g.celda(1.5, -0.9)
    g.visto[i1:, j1:] = False                           # mucho sin ver al fondo
    x, y = -0.7, 0.0
    lejos = g.mejor_goal(x, y, fov=1.0, alcance=1.0)
    assert lejos[0] > 0.5
    cerca = g.mejor_goal(x, y, fov=1.0, alcance=1.0, radio_local=1.0, ganancia_local=5.0)
    assert cerca[0] < 0.0 and cerca[3] >= 5.0


def test_goal_queda_lejos_de_las_paredes_pero_pasa_por_lo_angosto():
    g = sala()                                          # puerta de 0.6 m entre los cuartos
    x, y = -0.5, 0.0
    for yaw in np.linspace(0, 2 * math.pi, 12, endpoint=False):
        g.marcar_visto(x, y, yaw, fov=1.0, alcance=2.0)
    goal = g.mejor_goal(x, y, fov=1.0, alcance=2.0, despeje_min=0.10, despeje_goal=0.35)
    assert goal is not None and goal[0] > 0.1            # llega al otro cuarto por la puerta
    assert g.despeje[g.celda(goal[0], goal[1])] >= 0.35


def costmap_con_cubo(cx, cy, lado=0.25, radio=0.15):
    """Costmap global como el de Nav2 para `sala()`: paredes y un cubo que el mapa no tiene,
    letales, e inflados con el radio del robot."""
    w, h = int(4.0 / RES), int(2.0 / RES)
    letal = np.zeros((h, w), dtype=bool)
    letal[0, :] = letal[-1, :] = letal[:, 0] = letal[:, -1] = True
    letal[:, w // 2] = True
    letal[h // 2 - 6:h // 2 + 6, w // 2] = False
    xs = -1.0 + (np.arange(w) + 0.5) * RES
    ys = -1.0 + (np.arange(h) + 0.5) * RES
    letal |= ((np.abs(xs[None, :] - cx) <= lado / 2) & (np.abs(ys[:, None] - cy) <= lado / 2))
    d = ndimage.distance_transform_edt(~letal) * RES
    c = np.where(letal, 100, np.where(d <= radio, 99, 0)).astype(np.int8)
    return c.ravel().tolist(), w, h, RES, -1.0, -1.0


def test_cubo_fuera_del_mapa_tapa_la_camara_y_corta_el_camino():
    g = sala()
    x, y = -0.5, 0.0
    for yaw in np.linspace(0, 2 * math.pi, 12, endpoint=False):
        g.marcar_visto(x, y, yaw, fov=1.0, alcance=2.0)
    g.actualizar_costmap(*costmap_con_cubo(1.0, 0.0))  # justo en la puerta
    assert g.ocupado[g.celda(1.0, 0.0)] and not g.ocupado_mapa[g.celda(1.0, 0.0)]
    goal = g.mejor_goal(x, y, fov=1.0, alcance=2.0, despeje_min=0.10)
    assert goal is None or goal[0] < 1.0               # no lo manda al otro cuarto (pared x=1)
    # Mirando a la puerta desde cerca, la cámara no ve a través del cubo.
    g = sala()
    g.actualizar_costmap(*costmap_con_cubo(1.0, 0.0))
    g.marcar_visto(0.6, 0.0, 0.0, fov=0.2, alcance=1.5)
    assert g.visto[g.celda(0.8, 0.0)] and not g.visto[g.celda(1.4, 0.0)]


def test_llegada_da_por_visto_y_no_repite_el_lugar():
    g = sala()
    goal = g.mejor_goal(-0.5, 0.0, fov=1.0, alcance=2.0)
    gx, gy, gyaw, _ = goal
    # El robot quedó corrido y mirando a otro lado: desde ahí no vio lo prometido.
    g.marcar_visto(gx - 0.15, gy, gyaw + math.pi, fov=1.0, alcance=2.0)
    g.llegada(gx, gy, gyaw, fov=1.0, alcance=2.0)
    otro = g.mejor_goal(gx, gy, fov=1.0, alcance=2.0)
    assert otro is None or math.hypot(otro[0] - gx, otro[1] - gy) >= 0.3


def test_un_paso_justo_al_costado_de_un_cubo_sigue_siendo_navegable():
    # Puerta de 0.6 m con un cubo pegado arriba: abajo quedan ~0.33 m. El robot entra (lo
    # resuelve la misión con la maniobra): el explorador no tiene que darlo por cerrado.
    g = sala()
    g.actualizar_costmap(*costmap_con_cubo(1.0, 0.18))
    x, y = -0.5, 0.0
    navegable = g._componente((g.despeje >= 0.10) & ~g.bloqueo_nav, x, y)
    assert navegable[g.celda(1.5, 0.0)]


def test_paredes_vistas_corridas_por_amcl_no_cierran_un_paso():
    # Pasillo con un cubo que deja ~0.38 m; el LiDAR ve las paredes 6 cm corridas hacia
    # adentro (error de AMCL): el paso tiene que seguir abierto para el explorador.
    w, h = int(4.0 / RES), int(2.0 / RES)
    m = np.zeros((h, w), dtype=np.int8)
    m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = 100
    g = Grilla(m.ravel().tolist(), w, h, RES, -1.0, -1.0)
    c = m.astype(np.int16).copy()
    c[3, :] = 100                                      # pared de abajo vista 0.1 m más arriba
    xs = -1.0 + (np.arange(w) + 0.5) * RES
    ys = -1.0 + (np.arange(h) + 0.5) * RES
    cubo = (np.abs(xs[None, :] - 1.0) <= 0.125) & (np.abs(ys[:, None] - 0.25) <= 0.125)
    c[cubo] = 100                                      # cubo: deja ~0.4 m arriba del piso
    g.actualizar_costmap(c.astype(np.int8).ravel().tolist(), w, h, RES, -1.0, -1.0)
    navegable = g._componente((g.despeje >= 0.10) & ~g.bloqueo_nav, -0.5, -0.3)
    assert navegable[g.celda(2.0, -0.3)]
    assert g.ocupado[g.celda(1.0, 0.25)]               # el cubo sí cuenta


def test_se_elige_desde_donde_mirar_de_frente_cada_candidato():
    g = sala(pared_media=False)
    g.visto[:] = True                                   # el piso ya está todo "visto"
    goal, ident, sin_pose = g.mejor_inspeccion(-0.5, 0.0, {7: (2.0, 0.0)})
    assert ident == 7 and not sin_pose
    gx, gy, gyaw = goal
    assert 0.7 <= math.hypot(2.0 - gx, 0.0 - gy) <= 1.3
    assert abs(math.atan2(-gy, 2.0 - gx) - gyaw) < 1e-6     # mirando al candidato


def test_rojo_cortado_se_mira_desde_mas_lejos():
    # Caja roja demasiado cerca para ubicarla (cortada por el borde de la imagen): la próxima
    # inspección pide el rango lejano, aunque el robot esté al lado.
    g = sala(pared_media=False)
    g.visto[:] = True
    goal, ident, _ = g.mejor_inspeccion(1.4, 0.0, {7: (2.0, 0.0)}, rangos={7: (1.0, 1.5)})
    assert ident == 7
    assert 1.0 <= math.hypot(2.0 - goal[0], 0.0 - goal[1]) <= 1.5


def test_la_inspeccion_fija_se_mantiene_aunque_el_robot_se_mueva():
    # Desde la otra punta de la sala la pose más cercana por camino es otra; con `fija` se
    # sigue yendo a la de antes (antes saltaba de un lado al otro de un hueco angosto).
    g = sala(pared_media=False)
    g.visto[:] = True
    goal, ident, _ = g.mejor_inspeccion(-0.5, 0.0, {7: (2.0, 0.0)})
    otra, _, _ = g.mejor_inspeccion(2.6, 0.6, {7: (2.0, 0.0)})
    assert math.hypot(otra[0] - goal[0], otra[1] - goal[1]) > 0.3
    fija, ident_fija, _ = g.mejor_inspeccion(2.6, 0.6, {7: (2.0, 0.0)}, fija=(ident, goal))
    assert ident_fija == 7 and fija[:2] == goal[:2]


def test_la_inspeccion_fija_se_suelta_si_ya_no_sirve():
    g = sala(pared_media=False)
    g.visto[:] = True
    goal, ident, _ = g.mejor_inspeccion(-0.5, 0.0, {7: (2.0, 0.0)})
    # Descartada (la misión no pudo llegar): se busca otra pose.
    nueva, _, _ = g.mejor_inspeccion(-0.5, 0.0, {7: (2.0, 0.0)}, fija=(ident, goal),
                                     descartados=[goal[:2]])
    assert math.hypot(nueva[0] - goal[0], nueva[1] - goal[1]) >= 0.30
    # El candidato ya no está pendiente: no se mantiene una inspección de otro.
    otro, ident_otro, _ = g.mejor_inspeccion(-0.5, 0.0, {9: (-0.2, 0.5)}, fija=(ident, goal))
    assert ident_otro == 9 and otro[:2] != goal[:2]


def test_candidato_sin_lugar_desde_donde_mirarlo():
    g = sala(pared_media=False)
    # Candidato metido en la pared del fondo: no hay pose a 0.7-1.3 m con vista libre... salvo
    # adentro de la sala; uno fuera del mapa no tiene ninguna.
    goal, ident, sin_pose = g.mejor_inspeccion(-0.5, 0.0, {3: (10.0, 10.0)})
    assert goal is None and sin_pose == [3]


def test_el_diagnostico_explica_la_decision():
    g = sala()
    goal = g.mejor_goal(-0.5, 0.0, fov=1.0, alcance=1.5)
    d = g.diagnostico
    assert goal is not None and d['elegido']['x'] == round(goal[0], 2)
    assert 'local' in d and isinstance(d['alternativas'], list)
    assert d['zonas_sin_ver'] and all(z['celdas'] > 0 for z in d['zonas_sin_ver'])


def test_migajas_cercanas_no_le_ganan_a_una_sala_entera_a_un_paso():
    # maze_1: con una sala de ~700 celdas a 0.9 m de camino, el explorador terminaba migajas
    # al lado (ganancia ~40-90) porque sólo competían los candidatos a menos del radio local.
    g = sala()
    g.visto[:] = True
    i, j = g.celda(-0.5, 0.6)
    g.visto[i:i + 3, j:j + 3] = False                   # migajas cerca del robot
    i0, j0 = g.celda(1.05, -0.95)
    g.visto[i0:, j0:] = False                           # todo el cuarto derecho sin ver
    x, y = -0.6, 0.0                                    # lejos de la puerta
    kw = dict(fov=1.0, alcance=1.5, radio_local=0.8, ganancia_local=5.0)
    migajas = g.mejor_goal(x, y, **kw)
    sala_grande = g.mejor_goal(x, y, fraccion_local=0.5, **kw)
    assert migajas[0] < 0.5 and migajas[3] < 50            # sin fracción: lo cercano chico
    assert sala_grande[0] > 1.0 and sala_grande[3] > 3 * migajas[3]   # con fracción: la sala


def sala_puerta_angosta():
    """Dos cuartos de 2×2 m (origen en -1, -1) y en x = 1 una pared con una puerta de 0.4 m."""
    w, h = int(4.0 / RES), int(2.0 / RES)
    m = np.zeros((h, w), dtype=np.int8)
    m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = 100
    m[:, w // 2] = 100
    m[h // 2 - 4:h // 2 + 4, w // 2] = 0
    g = Grilla(m.ravel().tolist(), w, h, RES, -1.0, -1.0)
    g.visto[:] = True
    return g


def test_con_penalidad_mira_desde_este_lado_en_vez_de_cruzar_el_hueco():
    g = sala_puerta_angosta()
    cruza, _, _ = g.mejor_inspeccion(0.5, -0.6, {7: (1.3, -0.6)})
    assert cruza[0] > 1.0                              # sin penalidad: del otro lado
    queda, ident, _ = g.mejor_inspeccion(0.5, -0.6, {7: (1.3, -0.6)}, penalidad_paso=3.0)
    assert ident == 7 and queda[0] < 1.0               # mira a través de la puerta
    # Si sólo se lo puede mirar del otro lado, cruza igual.
    lejos, ident, _ = g.mejor_inspeccion(0.5, -0.6, {8: (2.3, 0.0)}, penalidad_paso=3.0)
    assert ident == 8 and lejos[0] > 1.0
