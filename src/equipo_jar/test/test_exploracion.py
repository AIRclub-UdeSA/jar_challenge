import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from equipo_jar.exploracion import (
    angle_difference, first_pending, already_visited, yaw_from_quaternion, goal_position,
    box_zones, near_box, return_verdict, goal_blocked, start_ready)

DIST = 0.30
ANG = 0.5


def quaternion(yaw):
    """Cuaternión de un giro `yaw` sobre el eje vertical."""
    return SimpleNamespace(x=0.0, y=0.0, z=math.sin(yaw / 2), w=math.cos(yaw / 2))


def pose(x, y, yaw):
    """PoseStamped falso: mismos campos que usa first_pending."""
    return SimpleNamespace(pose=SimpleNamespace(
        position=SimpleNamespace(x=x, y=y), orientation=quaternion(yaw)))


# --- yaw_from_quaternion ---

def test_yaw_no_rotation():
    assert yaw_from_quaternion(quaternion(0.0)) == pytest.approx(0.0)


def test_yaw_quarter_turn():
    assert yaw_from_quaternion(quaternion(math.pi / 2)) == pytest.approx(math.pi / 2)


def test_yaw_negative():
    assert yaw_from_quaternion(quaternion(-1.0)) == pytest.approx(-1.0)


# --- angle_difference ---

def test_difference_simple():
    assert angle_difference(0.1, 0.3) == pytest.approx(0.2)


def test_difference_does_not_depend_on_order():
    assert angle_difference(0.3, 0.1) == pytest.approx(0.2)


def test_difference_wraps_around():
    # 179° y -179° están a 2°, no a 358°.
    assert angle_difference(math.radians(179), math.radians(-179)) == pytest.approx(math.radians(2))


def test_difference_full_turn_is_zero():
    assert angle_difference(0.0, 2 * math.pi) == pytest.approx(0.0, abs=1e-9)


# --- already_visited ---

def test_no_visited():
    assert not already_visited(1.0, 1.0, 0.0, [], DIST, ANG)


def test_same_point_far_from_origin():
    assert already_visited(3.0, 3.0, 0.0, [(3.0, 3.0, 0.0)], DIST, ANG)


def test_near_origin_not_visited():
    assert not already_visited(0.1, 0.1, 0.0, [(5.0, 5.0, 0.0)], DIST, ANG)


def test_almost_equal_counts_as_visited():
    assert already_visited(1.1, 1.0, 0.1, [(1.0, 1.0, 0.0)], DIST, ANG)


def test_same_place_facing_other_way_is_not_visited():
    assert not already_visited(1.0, 1.0, math.pi / 2, [(1.0, 1.0, 0.0)], DIST, ANG)


def test_same_heading_elsewhere_is_not_visited():
    assert not already_visited(2.0, 1.0, 0.0, [(1.0, 1.0, 0.0)], DIST, ANG)


# --- first_pending ---

def test_empty_plan():
    assert first_pending([], [], DIST, ANG, []) is None


def test_no_visited_returns_first():
    poses = [pose(1.0, 0.5, 0.0), pose(2.0, 0.5, 0.0)]
    assert first_pending(poses, [], DIST, ANG, []) is poses[0]


def test_skips_visited_and_keeps_order():
    poses = [pose(1.0, 0.5, 0.0), pose(2.0, 0.5, 0.0), pose(3.0, 0.5, 0.0)]
    assert first_pending(poses, [(1.0, 0.5, 0.0)], DIST, ANG, []) is poses[1]


def test_all_visited():
    poses = [pose(1.0, 0.5, 0.0), pose(2.0, 0.5, 0.0)]
    visited = [(1.0, 0.5, 0.0), (2.0, 0.5, 0.0)]
    assert first_pending(poses, visited, DIST, ANG, []) is None


def test_same_place_other_heading_is_still_pending():
    # El plan de la prueba del paso 8: los dos últimos puntos están en el mismo lugar.
    poses = [pose(1.0, 0.5, 0.0), pose(2.0, 0.5, 0.0), pose(2.0, 0.5, math.pi / 2)]
    visited = [(1.0, 0.5, 0.0), (2.0, 0.5, 0.0)]
    assert first_pending(poses, visited, DIST, ANG, []) is poses[2]


# --- Identidad de una vista del plan de cobertura ---
# Cobertura genera las vistas sobre una grilla fija (centros de celda y múltiplos de
# 360°/heading_count), así que la misma vista sale siempre con las mismas coordenadas.
# Misión las compara con tolerancias que solo filtran redondeo (params.yaml).

PLAN_DIST = 0.02
PLAN_ANG = 0.1
MAP_RESOLUTION = 0.05   # m por celda en los mapas de práctica


def test_coverage_views_25_cm_apart_are_different():
    # Caso de un mapa de práctica: vistas 31 y 34, misma dirección, a 0.25 m.
    yaw = 4.71238898038469
    poses = [pose(-2.675, 2.125, yaw), pose(-2.925, 2.125, yaw)]
    visited = [(-2.675, 2.125, yaw_from_quaternion(quaternion(yaw)))]
    assert first_pending(poses, visited, PLAN_DIST, PLAN_ANG, []) is poses[1]


def test_coverage_headings_45_degrees_apart_are_different():
    poses = [pose(1.025, 0.475, 0.0), pose(1.025, 0.475, math.pi / 4)]
    visited = [(1.025, 0.475, 0.0)]
    assert first_pending(poses, visited, PLAN_DIST, PLAN_ANG, []) is poses[1]


def test_same_view_in_a_new_plan_is_recognized():
    # El mismo punto, publicado otra vez en un plan nuevo y leído desde el cuaternión.
    yaw = 3 * math.pi / 4
    visited = [(1.025, 0.475, yaw_from_quaternion(quaternion(yaw)))]
    assert already_visited(1.025, 0.475, yaw_from_quaternion(quaternion(yaw)),
                           visited, PLAN_DIST, PLAN_ANG)


def test_params_respect_the_coverage_resolution():
    # Regla acordada con cobertura: las tolerancias de misión tienen que ser menores que
    # media celda y que media separación entre direcciones. Si cobertura cambia su grilla
    # o la cantidad de direcciones, este test avisa.
    yaml = pytest.importorskip('yaml')
    path = Path(__file__).resolve().parents[1] / 'params' / 'params.yaml'
    params = yaml.safe_load(path.read_text())
    mision = params['mision']['ros__parameters']
    heading_count = params['cobertura']['ros__parameters']['heading_count']
    assert mision['same_point_distance_m'] < MAP_RESOLUTION / 2
    assert mision['same_point_angle_rad'] < math.pi / heading_count


# --- goal_position: girar en el lugar ---
# Nav2 da por llegado un objetivo a menos de xy_goal_tolerance (0.10 m) y en el giro se
# puede correr un poco. Si misión volviera a pedir el x,y del plan para el giro siguiente,
# Nav2 tendría que acercarse otra vez (en un mapa de práctica quedó 23 s trabado).

TURN_RADIUS = 0.20


def test_without_tf_goes_to_the_plan_point():
    assert goal_position(1.0, 2.0, None, TURN_RADIUS) == (1.0, 2.0, False)


def test_robot_near_the_plan_point_turns_in_place():
    # Caso medido: el objetivo (-2.17, 2.43) se dio por llegado con el robot a 0.14 m.
    robot = (-2.03, 2.43, 0.5)
    assert goal_position(-2.17, 2.43, robot, TURN_RADIUS) == (-2.03, 2.43, True)


def test_robot_far_from_the_plan_point_travels():
    robot = (1.5, 2.0, 0.0)
    assert goal_position(1.0, 2.0, robot, TURN_RADIUS) == (1.0, 2.0, False)


def test_robot_exactly_at_the_radius_travels():
    # La comparación es estricta (<): justo en el borde, viaja.
    robot = (0.2, 0.0, 0.0)
    assert goal_position(0.0, 0.0, robot, TURN_RADIUS) == (0.0, 0.0, False)


def test_turn_radius_covers_nav2_arrivals():
    # turn_in_place_m tiene que ser mayor que la tolerancia de llegada de Nav2: si no, un
    # robot que Nav2 dio por llegado puede quedar afuera del radio y tener que acercarse
    # otra vez. Si navegación cambia su tolerancia, este test avisa.
    yaml = pytest.importorskip('yaml')
    params_dir = Path(__file__).resolve().parents[1] / 'params'
    params = yaml.safe_load((params_dir / 'params.yaml').read_text())
    mision = params['mision']['ros__parameters']
    nav = yaml.safe_load((params_dir / 'navegacion.yaml').read_text())
    checker = nav['controller_server']['ros__parameters']['general_goal_checker']
    assert mision['turn_in_place_m'] > checker['xy_goal_tolerance']


# --- Zonas alrededor de las cajas ---
# Las cajas no están en /map: un punto del plan puede caer pegado a una. Misión saltea
# los puntos a menos de box_clearance_m + sigma_factor * dispersión de una candidata.

CLEARANCE = 0.42


def box_cand(id_, x, y, observaciones=3, sigma=0.15):
    """Candidata falsa con los campos que usa box_zones (covarianza redonda de `sigma`)."""
    var = sigma ** 2
    return SimpleNamespace(id=id_, centro=SimpleNamespace(x=x, y=y),
                           observaciones=observaciones, covarianza_xy=[var, 0.0, 0.0, var])


def test_zone_radius_adds_the_uncertainty():
    zones = box_zones([box_cand(1, 1.0, 2.0, sigma=0.15)], 2, CLEARANCE, 1.0)
    assert len(zones) == 1
    x, y, radius = zones[0]
    assert (x, y) == (1.0, 2.0)
    assert radius == pytest.approx(CLEARANCE + 0.15)


def test_zone_factor_zero_ignores_the_uncertainty():
    radius = box_zones([box_cand(1, 0.0, 0.0, sigma=0.15)], 2, CLEARANCE, 0.0)[0][2]
    assert radius == pytest.approx(CLEARANCE)


def test_candidate_with_few_observations_makes_no_zone():
    # Una detección suelta puede ser falsa: no tiene que borrar zonas del plan.
    assert box_zones([box_cand(1, 0.0, 0.0, observaciones=1)], 2, CLEARANCE, 1.0) == []


def test_non_finite_center_makes_no_zone():
    assert box_zones([box_cand(1, float('nan'), 0.0)], 2, CLEARANCE, 1.0) == []


def test_near_box():
    zones = [(0.0, 0.0, 0.5)]
    assert near_box(0.3, 0.0, zones)
    assert not near_box(0.6, 0.0, zones)
    assert not near_box(0.5, 0.0, zones)   # justo en el borde: la comparación es estricta
    assert not near_box(0.3, 0.0, [])


def test_first_pending_skips_a_point_next_to_a_box():
    # Caso de un mapa de práctica: vista (0.144, 1.969) con la caja roja en (0.0, 2.0).
    poses = [pose(0.144, 1.969, 0.0), pose(1.0, 1.0, 0.0)]
    zones = box_zones([box_cand(1, 0.0, 2.0, sigma=0.10)], 2, CLEARANCE, 1.0)
    assert first_pending(poses, [], DIST, ANG, zones) is poses[1]


def test_a_skipped_point_is_still_pending():
    # Saltear no es visitar: si fusión corre la caja, el punto vuelve a estar disponible.
    poses = [pose(0.144, 1.969, 0.0)]
    near = box_zones([box_cand(1, 0.0, 2.0)], 2, CLEARANCE, 1.0)
    moved = box_zones([box_cand(1, 0.0, 3.0)], 2, CLEARANCE, 1.0)
    assert first_pending(poses, [], DIST, ANG, near) is None
    assert first_pending(poses, [], DIST, ANG, moved) is poses[0]


def launch_constants(path):
    """Constantes numéricas sueltas de un archivo .py, leídas sin importarlo."""
    values = {}
    for node in ast.parse(path.read_text()).body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.Constant)):
            values[node.targets[0].id] = node.value.value
    return values


def test_box_clearance_covers_the_turning_robot_and_the_box():
    # box_clearance_m = radio de la esquina más lejana del robot (al llegar a un objetivo
    # gira en el lugar y barre ese círculo) + media diagonal de la caja (cualquier
    # orientación). Si navegación cambia la forma o el margen del robot, o percepción el
    # tamaño de la caja, este test avisa.
    yaml = pytest.importorskip('yaml')
    base = Path(__file__).resolve().parents[1]
    chassis = launch_constants(base / 'launch' / 'navegacion.launch.py')
    nav = yaml.safe_load((base / 'params' / 'navegacion.yaml').read_text())
    padding = max(nav[c][c]['ros__parameters']['footprint_padding']
                  for c in ('local_costmap', 'global_costmap'))
    params = yaml.safe_load((base / 'params' / 'params.yaml').read_text())
    box_side = params['deteccion']['ros__parameters']['box_side_m']
    corner = math.hypot(max(chassis['ADELANTE'], chassis['ATRAS']) + padding,
                        chassis['MEDIO_ANCHO'] + padding)
    needed = corner + box_side * math.sqrt(2) / 2
    assert params['mision']['ros__parameters']['box_clearance_m'] >= needed


# --- Regreso a la largada ---
# Al terminar la vuelta, misión mide cuánto quedó de la pose de largada y decide:
# 'ok' (termina), 'retry' (vuelve a mandar la largada) o 'give_up' (termina igual).
# Solo mira la posición: el bonus pide las 4 ruedas dentro del círculo de 0.25 m, y las
# ruedas son simétricas alrededor del centro, así que la orientación no cambia nada.
# Argumentos: error (m), vueltas ya mandadas, tolerancia (m), máximo de vueltas.

def test_return_close_enough_is_ok():
    assert return_verdict(0.03, 1, 0.05, 3) == 'ok'


def test_return_exactly_at_the_tolerance_is_ok():
    assert return_verdict(0.05, 1, 0.05, 3) == 'ok'


def test_return_far_with_tries_left_retries():
    assert return_verdict(0.08, 1, 0.05, 3) == 'retry'


def test_return_far_on_the_last_try_gives_up():
    assert return_verdict(0.08, 3, 0.05, 3) == 'give_up'


def test_return_close_enough_on_the_last_try_is_ok():
    assert return_verdict(0.03, 3, 0.05, 3) == 'ok'


def test_return_tree_uses_the_precise_goal_checker():
    # Misión manda behavior_trees/vuelta_precisa.xml en el goal de la vuelta (go_to). Si
    # alguien renombra el archivo o el checker, la vuelta pierde la precisión sin avisar.
    tree = Path(__file__).resolve().parents[1] / 'behavior_trees' / 'vuelta_precisa.xml'
    assert 'goal_checker_id="precise_goal_checker"' in tree.read_text()


def test_return_tolerance_fits_the_precise_goal_checker_and_the_rule():
    # Más chica que la del checker: misión reintentaría vueltas que Nav2 ya da por cumplidas,
    # sin mover el robot. 0.10 m o más: con el error de AMCL encima, alguna rueda puede quedar
    # fuera del círculo (la punta de rueda más lejana está a 0.150 m del centro).
    yaml = pytest.importorskip('yaml')
    params_dir = Path(__file__).resolve().parents[1] / 'params'
    params = yaml.safe_load((params_dir / 'params.yaml').read_text())
    mision = params['mision']['ros__parameters']
    nav = yaml.safe_load((params_dir / 'navegacion.yaml').read_text())
    checker = nav['controller_server']['ros__parameters']['precise_goal_checker']
    assert checker['xy_goal_tolerance'] <= mision['return_tolerance_m'] < 0.10


def test_return_params_have_the_types_mision_declares():
    # parametros_en_vivo declara cada parámetro con el tipo de su valor por defecto: un
    # entero en el YAML para un float (o al revés) hace fallar el nodo al arrancar.
    yaml = pytest.importorskip('yaml')
    params_dir = Path(__file__).resolve().parents[1] / 'params'
    mision = yaml.safe_load((params_dir / 'params.yaml').read_text())['mision']['ros__parameters']
    for name in ('return_tolerance_m', 'return_still_s', 'return_still_m', 'return_check_max_s'):
        assert isinstance(mision[name], float), name
    assert isinstance(mision['return_max_tries'], int)
    assert mision['return_max_tries'] >= 1


def test_return_check_has_time_to_see_the_robot_still():
    # El chequeo espera return_still_s quieto (el ajuste fino de AMCL llega 0.3-2.5 s después
    # de frenar). Si el tope fuera menor, terminaría siempre por el tope, sin medir.
    # Y el tope tiene que quedar lejos de los 60 s quieto que cortan la corrida.
    yaml = pytest.importorskip('yaml')
    params_dir = Path(__file__).resolve().parents[1] / 'params'
    mision = yaml.safe_load((params_dir / 'params.yaml').read_text())['mision']['ros__parameters']
    assert mision['return_still_s'] + 2.0 <= mision['return_check_max_s'] <= 30.0


def test_mision_params_in_yaml_match_the_ones_declared_in_code():
    # Un parámetro del YAML que mision.py no declara se ignora sin avisar (un nombre mal
    # escrito, o uno que ya no existe). Compara las claves de params.yaml con el dict que
    # mision.py le pasa a parametros_en_vivo.
    yaml = pytest.importorskip('yaml')
    package = Path(__file__).resolve().parents[1]
    mision = yaml.safe_load((package / 'params' / 'params.yaml').read_text())['mision']['ros__parameters']
    declared = None
    for node in ast.walk(ast.parse((package / 'equipo_jar' / 'mision.py').read_text())):
        if isinstance(node, ast.Call) and getattr(node.func, 'id', None) == 'parametros_en_vivo':
            declared = {key.value for key in node.args[1].keys}
    assert declared is not None, 'parametros_en_vivo call not found in mision.py'
    assert set(mision) == declared


# --- Objetivos sobre obstáculos que no están en /map ---
# goal_blocked lee el costmap global de Nav2 (/global_costmap/costmap) en el punto del
# objetivo. Valores: 100 obstáculo, 99 el centro del robot ahí tocaría algo, 1-98 solo
# margen, 0 libre, -1 desconocido. Desconocido o fuera de la grilla: no bloquea (decide Nav2).
# Grilla de prueba: 3 x 2 celdas de 0.5 m, origen en (-1.0, 0.0).

def costmap(values):
    return SimpleNamespace(
        info=SimpleNamespace(width=3, height=2, resolution=0.5,
                             origin=SimpleNamespace(position=SimpleNamespace(x=-1.0, y=0.0))),
        data=values)


def test_goal_on_an_obstacle_is_blocked():
    grid = costmap([0, 100, 0,
                    0, 0, 0])
    assert goal_blocked(grid, -0.25, 0.25, 99)


def test_goal_where_the_robot_would_touch_is_blocked():
    grid = costmap([0, 99, 0,
                    0, 0, 0])
    assert goal_blocked(grid, -0.25, 0.25, 99)


def test_goal_in_the_safety_margin_is_not_blocked():
    grid = costmap([0, 98, 0,
                    0, 0, 0])
    assert not goal_blocked(grid, -0.25, 0.25, 99)


def test_free_goal_is_not_blocked():
    grid = costmap([100, 0, 100,
                    100, 100, 100])
    assert not goal_blocked(grid, -0.25, 0.25, 99)


def test_unknown_goal_is_not_blocked():
    grid = costmap([0, -1, 0,
                    0, 0, 0])
    assert not goal_blocked(grid, -0.25, 0.25, 99)


def test_goal_outside_the_costmap_is_not_blocked():
    grid = costmap([100] * 6)
    assert not goal_blocked(grid, 5.0, 5.0, 99)


def test_no_costmap_yet_is_not_blocked():
    assert not goal_blocked(None, 0.0, 0.0, 99)


def test_costmap_blocked_value_is_an_int_cost():
    # parametros_en_vivo declara el tipo con el valor por defecto (int): 99.0 en el YAML
    # haría fallar el nodo al arrancar. Fuera de 1-100 nunca bloquearía o bloquearía todo.
    yaml = pytest.importorskip('yaml')
    params_dir = Path(__file__).resolve().parents[1] / 'params'
    mision = yaml.safe_load((params_dir / 'params.yaml').read_text())['mision']['ros__parameters']
    value = mision['costmap_blocked_value']
    assert isinstance(value, int) and 1 <= value <= 100


# --- Arranque ---
# Misión arranca cuando localización publica true en /jar/localizacion_lista (AMCL ya se
# acomodó), o al llegar al tope si el flag nunca llega. Argumentos: si llegó el true,
# segundos desde /map, tope (s).

def test_start_waits_for_localization():
    assert not start_ready(False, 2.0, 10.0)


def test_start_as_soon_as_localization_is_ready():
    assert start_ready(True, 0.5, 10.0)


def test_start_anyway_at_the_max_wait():
    assert start_ready(False, 10.0, 10.0)
    assert start_ready(False, 12.0, 10.0)


def test_localization_wait_max_is_a_float_well_under_the_still_limit():
    # parametros_en_vivo declara el tipo con el valor por defecto (float). El robot no se
    # mueve hasta el arranque, y 60 s quieto termina la corrida (§5): el tope debe quedar
    # muy por debajo, y por encima de lo que tarda AMCL en acomodarse (~2-3 s en el sim).
    yaml = pytest.importorskip('yaml')
    params_dir = Path(__file__).resolve().parents[1] / 'params'
    mision = yaml.safe_load((params_dir / 'params.yaml').read_text())['mision']['ros__parameters']
    value = mision['localization_wait_max_s']
    assert isinstance(value, float) and 3.0 <= value <= 30.0
