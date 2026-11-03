"""La misión de navigator.py en una simulación cinemática del laberinto sintético de test_core, sin ROS.

El robot se mueve con el comando (mecanum: vx, vy y giro; exacto o con motores como los del bag) y el lidar se simula contra el mundo real, que
puede tener cajas que el mapa no tiene. La percepción se simula con la regla de DetectionClusterer.ready() (rayos_reporte = 100 en 3 scans): una caja
roja se reporta tras ~0.8 s en cuadro (lo que tardó en el bag), y desde el segundo scan aparece en /jar/rojo. Se mide lo que cuesta
puntos o tiempo: chocar, no ver el laberinto, no encontrar una víctima, girar de más, no volver a la largada o volver tarde.
"""
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from scipy import ndimage

from equipo_jar.core import Board, line_of_sight, nearest_drivable, normalize, visibility
from equipo_jar.navigator import Mission, detect_boxes, slow_down, tail_barrier
from test_core import RES, maze

PARAMS = Path(__file__).parents[1] / 'params' / 'params.yaml'
NAVIGATOR = yaml.safe_load(PARAMS.read_text())['navigator']['ros__parameters']
P = SimpleNamespace(**NAVIGATOR) # los parámetros del robot (params.yaml del repo), como los atributos del nodo
START = (0.0, -0.25, 0.0)
ANG = np.linspace(-math.pi, math.pi, 1080, endpoint=False) # como el lidar del robot y el de Gazebo: un rayo cada 0.33°
RAY_STEPS = np.arange(0.12, 4.0, RES / 2)


def with_box(grid, x, y, side=0.25):
    """El mismo laberinto con una caja que el mapa no tiene: en las celdas (para medir choques) y como cuadrado exacto para el lidar (boxes)."""
    g = grid.occupied.astype(np.int16) * 100
    i, j = np.mgrid[0:g.shape[0], 0:g.shape[1]]
    cx, cy = grid.to_world(i, j)
    g[(np.abs(cx - x) < side / 2) & (np.abs(cy - y) < side / 2)] = 100
    out = Board(g.ravel(), g.shape[1], g.shape[0], RES, grid.ox, grid.oy)
    out.walls, out.boxes = getattr(grid, 'walls', grid), getattr(grid, 'boxes', []) + [(x, y, side)]
    return out


def lidar(world, x, y):
    """Puntos (frame map) donde pega cada rayo hasta 4 m: las paredes del mapa por celdas y las cajas como cuadrados (por celdas, sus caras serían escalones)."""
    walls = getattr(world, 'walls', world)
    gx, gy = x + np.outer(np.cos(ANG), RAY_STEPS), y + np.outer(np.sin(ANG), RAY_STEPS)
    i, j = walls.to_cell(gx, gy)
    ok = walls.inside(i, j)
    hit = ~ok | walls.occupied[np.where(ok, i, 0), np.where(ok, j, 0)]
    r = np.where(hit.any(axis=1), RAY_STEPS[np.argmax(hit, axis=1)], np.inf)
    c, s = np.cos(ANG), np.sin(ANG)
    with np.errstate(divide='ignore', invalid='ignore'):
        for bx, by, side in getattr(world, 'boxes', []):
            t1, t2 = (bx - side / 2 - x) / c, (bx + side / 2 - x) / c
            t3, t4 = (by - side / 2 - y) / s, (by + side / 2 - y) / s
            t_in = np.maximum(np.minimum(t1, t2), np.minimum(t3, t4))
            t_out = np.minimum(np.maximum(t1, t2), np.maximum(t3, t4))
            r = np.where((t_in <= t_out) & (t_in >= RAY_STEPS[0]) & (t_in < r), t_in, r)
    ok = r < 4.0
    return np.column_stack([x + r[ok] * c[ok], y + r[ok] * s[ok]])


def real_motors(dt=0.2, tau=0.45, tau_turn=0.67, deadband=0.1):
    """
    Motores como los del bag laberinto_victimas: llegan al 63 % del avance pedido en ~0.45 s y del giro en ~0.67 s, de costado van más lento y el
    giro horario da 0.6-0.86 rad/s cuando se piden 1. La zona muerta no se pudo medir (el bag es a 0.5 m/s y 1 rad/s): el test usa 0.2 m/s, un
    supuesto pesimista.
    """
    v, taus = np.zeros(3), np.array([tau, tau, tau_turn])

    def apply(vx, vy, w):
        cmd = np.array([vx, 0.8 * vy, w * (0.7 if w < 0 else 1.0)])
        if math.hypot(cmd[0], cmd[1]) < deadband: cmd[:2] = 0
        if abs(cmd[2]) < 2 * deadband: cmd[2] = 0
        v[:] += (cmd - v) * np.minimum(1.0, dt / taus)

        return tuple(v)

    return apply


def in_camera(grid, x, y, yaw, bx, by, p):
    """La caja (bx, by) en la foto: a menos de camera_range, en la parte central (view_fraction) y con al menos 2 de 5 puntos (centro y esquinas) a la vista."""
    if math.hypot(bx - x, by - y) > p.camera_range or abs(normalize(math.atan2(by - y, bx - x) - yaw)) > 0.5 * p.view_fraction * p.camera_fov: return False
    corners = ((0, 0), (.12, .12), (.12, -.12), (-.12, .12), (-.12, -.12))
    return sum(line_of_sight(grid, (x, y), (bx + dx, by + dy)) for dx, dy in corners) >= 2


def simulate(world, grid, p=None, red=(), reports=(), victims=3, t_max=600.0, dt=0.2, motors=None):
    """
    Corre la misión hasta /done. red: cajas rojas del mundo, que la percepción reporta. motors: comando -> velocidad real (None: exacta).

    Returns:
        (misión, trayectoria (T, 3) con x, y, yaw, terminó, t final, reportes, grados girados).
    """
    m = Mission(grid, START[:2], p or P)
    x, y, yaw = START
    traj, t, done, turned, reports = [], 0.0, False, 0.0, list(reports)
    views = {box: [] for box in red}
    while t < t_max and not done:
        seen_red = [box for box, v in views.items() if len(v) * dt > 0.28 and box not in reports] # /jar/rojo: desde el segundo scan
        (vx, vy, w), done = m.step(x, y, yaw, t, lidar(world, x, y), reports, victims, seen_red)
        if motors: vx, vy, w = motors(vx, vy, w)
        c, s = math.cos(yaw), math.sin(yaw)
        x, y, yaw = x + (c * vx - s * vy) * dt, y + (s * vx + c * vy) * dt, normalize(yaw + w * dt)
        traj.append((x, y, yaw))
        turned += abs(w) * dt
        t += dt
        for box, v in views.items(): # percepción: ~0.8 s en cuadro
            if box in reports or not in_camera(grid, x, y, yaw, *box, m.p): continue
            v.append((x, y))
            if len(v) * dt > 0.8: reports.append(box)
    return m, np.array(traj), done, t, reports, math.degrees(turned)


def home(traj):
    """Distancia de la rueda más lejana a la largada al terminar (bono: las cuatro en un círculo de 0.25 m)."""
    x, y, yaw = traj[-1]
    c, s = math.cos(yaw), math.sin(yaw)
    return max(math.hypot(x + c * a - s * b - START[0], y + s * a + c * b - START[1]) for a in (-P.wheel_x, P.wheel_x) for b in (-P.wheel_y, P.wheel_y))


def come_back(x, y, yaw, motors=None, dt=0.1):
    """Vuelve desde (x, y, yaw) hasta /done. Returns: (pose al /done, velocidad real con la que venía (frame map), t, metros del camino)."""
    grid = maze()
    m = Mission(grid, START[:2], P)
    m.state = 'returning'
    t, v, path = 0.0, (0.0, 0.0), m.start_dist[nearest_drivable(grid, m.reachable, x, y)]
    while t < 30.0:
        (vx, vy, w), done = m.step(x, y, yaw, t, lidar(grid, x, y))
        if done: return (x, y, yaw), v, t, path
        if motors: vx, vy, w = motors(vx, vy, w)
        c, s = math.cos(yaw), math.sin(yaw)
        v = (c * vx - s * vy, s * vx + c * vy)
        x, y, yaw, t = x + v[0] * dt, y + v[1] * dt, normalize(yaw + w * dt), t + dt
    raise AssertionError('no volvió')


def test_covers_maze_with_lidar_and_returns():
    grid = maze()
    m, traj, done, t, _, _ = simulate(grid, grid)
    assert done and t < m.p.time_limit
    assert home(traj) < P.start_tolerance  # bono de regreso
    to_see = np.logical_or.reduce(m.vis)
    assert m.seen[to_see].mean() > 0.95
    assert grid.distance(traj[:, 0], traj[:, 1]).min() >= 0.15  # el centro nunca a menos de 15 cm de una pared


def test_does_not_turn_without_candidates():
    """Sin nada fuera del mapa no hay a qué apuntar la cámara: recorre y vuelve sin girar (el robot real gira lento). La barrera de la cola puede girar para
    apartarla: acá se apaga."""
    grid = maze()
    _, _, done, _, _, turned = simulate(grid, grid, SimpleNamespace(**{**NAVIGATOR, 'tail_start': math.inf}))
    assert done and turned < 1.0


def test_finds_red_box_among_decoy_and_returns():
    """Una víctima y un señuelo que el mapa no tiene: los dos son candidatos, la cámara los mira, reporta la roja y vuelve girando poco."""
    grid = maze()
    red, blue = (3.0, 1.5), (-1.0, -1.8)
    world = with_box(with_box(grid, *red), *blue)
    m, traj, done, t, reports, turned = simulate(world, grid, red=[red], victims=1)
    assert reports == [red] and done and home(traj) < P.start_tolerance
    assert any(math.hypot(cd['xy'][0] - red[0], cd['xy'][1] - red[1]) < 0.35 for cd in m.candidates)
    assert world.distance(traj[:, 0], traj[:, 1]).min() >= 0.15  # ni paredes ni cajas
    assert turned < 360


def test_avoids_box_not_in_map():
    """Una caja que no es víctima en el pasillo de la largada: la esquiva, la mira hasta descartarla y sigue."""
    grid = maze()
    box = (1.5, -0.3)
    world = with_box(grid, *box)
    m, traj, done, t, reports, _ = simulate(world, grid)
    assert done and not reports
    assert world.distance(traj[:, 0], traj[:, 1]).min() >= 0.15  # ni paredes ni la caja
    assert m.obstacles.any()  # la anotó como obstáculo
    assert [cd['done'] for cd in m.candidates if math.hypot(cd['xy'][0] - box[0], cd['xy'][1] - box[1]) < 0.35] == [True]


def test_returns_right_away_when_all_victims_reported():
    grid = maze()
    _, _, done, t, _, _ = simulate(grid, grid, reports=[(1.0, 1.0)] * 3, victims=3)
    assert done and t < 1.0


def test_returns_before_time_limit():
    grid = maze()
    p = SimpleNamespace(**{**NAVIGATOR, 'time_limit': 60.0})
    _, traj, done, t, _, _ = simulate(grid, grid, p)
    assert done and t <= 60.0
    assert home(traj) < P.start_tolerance


def test_returns_with_real_motors():
    """Con retardo, zona muerta y giro desparejo igual encuentra la víctima, no choca y termina con las cuatro ruedas en el círculo de 0.25 m (lo que
    frena después del /done, también)."""
    grid = maze()
    red = (3.0, 1.5)
    world = with_box(grid, *red)
    m, traj, done, t, reports, _ = simulate(world, grid, red=[red], victims=1, dt=0.1, motors=real_motors(dt=0.1, deadband=0.2))
    assert reports == [red] and done and t < m.p.time_limit
    assert home(traj) < 0.25  # a max_speed la zona muerta no lo deja trabado antes de la largada
    assert world.distance(traj[:, 0], traj[:, 1]).min() >= 0.15


@pytest.mark.parametrize('yaw', [0.0, math.pi / 4, math.pi / 2])
def test_return_finishes_when_all_wheels_are_inside(yaw):
    """Volviendo, termina en el ciclo en que la rueda más lejana entra en start_tolerance y pide frenar; un poco antes, sigue andando."""
    grid = maze()
    ds = np.linspace(0.0, 0.4, 4001)
    reach = [home(np.array([(START[0] + d, START[1], yaw)])) for d in ds]
    for offset, expected_done in ((0.001, False), (-0.001, True)):
        m = Mission(grid, START[:2], P)
        m.state = 'returning'
        x = START[0] + float(np.interp(P.start_tolerance + offset, reach, ds)) # rueda más lejana a start_tolerance + offset
        (vx, vy, w), done = m.step(x, START[1], yaw, 1.0, lidar(grid, x, START[1]))
        assert done == expected_done
        assert (math.hypot(vx, vy) == 0.0) == done


@pytest.mark.parametrize('x, y, yaw', [(1.0, -0.25, 0.0), (-1.0, -0.25, math.pi / 2), (0.7, 0.2, math.pi / 4), (-0.7, -0.7, 3.0), (-0.5, 0.25, -2.0)])
@pytest.mark.parametrize('real', [False, True])
def test_return_keeps_all_wheels_inside_when_it_brakes(x, y, yaw, real):
    """
    Llega desde distintos lados (con motores reales, de costado anda más lento y el camino se desvía), frenando el último tramo, y da /done con las cuatro
    ruedas a menos de start_tolerance. Después siguen en el círculo de 0.25 m si frena en seco, si desliza como en el bag (3-6 cm desde 0.5 m/s: ~0.12 s a la
    velocidad que traía), el doble de eso, o lo del bag con 30° de derrape.
    """
    (x, y, yaw), (vx, vy), t, path = come_back(x, y, yaw, real_motors(dt=0.1) if real else None)
    # Frenando a speed_gain por metro desde (max_speed - min_speed) / speed_gain hasta ~4 cm: lo que tarda de más contra ir a max_speed
    slow = (P.max_speed - P.min_speed) / P.speed_gain
    brake = math.log((P.speed_gain * slow + P.min_speed) / (P.speed_gain * 0.04 + P.min_speed)) / P.speed_gain - slow / P.max_speed
    assert t < path / P.max_speed + brake + (1.0 if real else 0.1) # los motores reales tardan ~0.45 s en arrancar y de costado andan al 80 %
    assert home(np.array([(x, y, yaw)])) <= P.start_tolerance
    for slide, skid in ((0.0, 0.0), (0.12, 0.0), (0.24, 0.0), (0.12, 0.5), (0.12, -0.5)):
        c, s = math.cos(skid), math.sin(skid)
        assert home(np.array([(x + slide * (c * vx - s * vy), y + slide * (s * vx + c * vy), yaw)])) < 0.25, (slide, skid)


def test_tail_barrier_slides_instead_of_backing_into_what_is_behind():
    """Marcha atrás en diagonal hacia algo a 5 cm de la cola: recorta solo lo que acerca (a (0.05 - guard_margin) / guard_horizon) y deja lo que desliza.
    Hacia adelante no toca el comando."""
    behind = np.array([(-P.body_rear - 0.05, 0.0)])
    vx, vy, w = tail_barrier(behind, (-0.5, 0.3, 0.0), P)
    assert abs(vx + (0.05 - P.guard_margin) / P.guard_horizon) < 1e-3 and abs(vy - 0.3) < 1e-3 and abs(w) < 1e-3
    assert tail_barrier(behind, (0.5, 0.0, 0.0), P) == (0.5, 0.0, 0.0)


def test_slows_down_with_something_ahead_but_never_stops():
    """Algo adelante en la franja recorta la velocidad con la distancia, sin bajar de slow_min; de costado o atrás no la toca."""
    ahead = lambda d: np.array([(d, 0.0)])
    assert slow_down(ahead(1.0), (0.5, 0.0, 0.3), P) == (0.5, 0.0, 0.3)
    vx, vy, w = slow_down(ahead(0.3), (0.5, 0.0, 0.3), P)
    assert abs(vx - (0.3 - P.body_radius - P.wall_stop) / P.slow_time) < 1e-9 and vy == 0.0 and w == 0.3
    assert slow_down(ahead(P.body_radius), (0.5, 0.0, 0.0), P)[0] == P.slow_min
    assert slow_down(np.array([(0.0, 0.3), (-0.3, 0.0)]), (0.5, 0.0, 0.0), P) == (0.5, 0.0, 0.0)


def test_return_at_full_speed_still_avoids_nearby_obstacle():
    """A max_speed el freno sigue mirando: un obstáculo detrás en el camino a la largada lo frena y se aleja."""
    grid = maze()
    m = Mission(grid, START[:2], P)
    m.state = 'returning'
    x, y = START[0] + 0.30, START[1]
    scan = np.vstack([lidar(grid, x, y), (x - P.body_radius - P.stop_margin - 0.02, y)])
    (vx, vy, _), done = m.step(x, y, 0.0, 1.0, scan)
    assert not done
    assert vx > 0 and math.hypot(vx, vy) <= max(P.retreat_speed, P.motor_min_speed) + 1e-9 # el mínimo de motor sube el retroceso


def test_node_publishes_done_before_braking():
    """El nodo publica /done en el ciclo en que entran las ruedas, y recién después el freno."""
    import rclpy
    from equipo_jar.navigator import Navigator

    rclpy.init(args=['--ros-args', '--params-file', str(PARAMS), '-p', 'victims:=1'])
    node = None
    try:
        node = Navigator()
        out = []
        node.cmd_pub = SimpleNamespace(publish=lambda msg: out.append(('cmd', math.hypot(msg.linear.x, msg.linear.y))))
        node.done_pub = SimpleNamespace(publish=lambda msg: out.append(('done', None)))
        grid = maze()
        node.board, node.start, node.t0 = grid, START[:2], node.get_clock().now()
        node.mission = Mission(grid, node.start, node)
        node.mission.state = 'returning'
        node.pose_at = node.lidar_at = node.get_clock().now()
        node.lidar = lidar(grid, START[0] + 0.3, START[1])
        node.pose = lambda: (START[0] + 0.3, START[1], math.pi / 4)
        node.control()
        assert out == [('cmd', pytest.approx(P.speed_gain * 0.3 + P.min_speed))] # todavía afuera: anda, frenando al llegar
        node.pose = lambda: (START[0] + 0.02, START[1], math.pi / 4) # la rueda más lejana a 0.136 m
        node.control()
        assert out[1:] == [('done', None), ('cmd', 0.0)] and node.finished
        node.control()
        assert len(out) == 3
    finally:
        if node is not None: node.destroy_node()
        rclpy.shutdown()


def test_returns_when_noted_obstacles_close_the_way():
    """Si los obstáculos anotados (engordados por el ruido de pose) cierran el único paso a la largada, los olvida y vuelve igual: antes se quedaba
    quieto para siempre (en Gazebo, un palo en el medio de un pasillo de 0.95 m)."""
    grid = maze()
    m = Mission(grid, START[:2], P)
    cx, cy = grid.to_world(*np.indices(grid.free.shape))
    band = grid.free & (np.abs(cx - 1.5) < 0.05) & (cy > -1.0) & (cy < 0.5) # fantasmas de pared a pared en el pasillo de la largada
    m.hits[band] = 3
    m._set_obstacles(m.hits >= 3)
    m.state = 'returning'
    x, y, yaw, t, done = 0.0, -2.0, 0.0, 0.0, False # abajo: el único camino cruza la franja
    while t < 90.0 and not done:
        (vx, vy, w), done = m.step(x, y, yaw, t, lidar(grid, x, y))
        c, s = math.cos(yaw), math.sin(yaw)
        x, y, yaw = x + (c * vx - s * vy) * 0.2, y + (s * vx + c * vy) * 0.2, normalize(yaw + w * 0.2)
        t += 0.2
    assert done and home(np.array([(x, y, yaw)])) < P.start_tolerance


def test_no_phantom_candidates_when_scan_lags_while_turning():
    """Girando, el scan llega con la pose de 50 ms antes (3° a 1 rad/s): a 4 m las paredes se corren 20 cm y parecían objetos (en Gazebo aparecieron 20
    candidatos fantasma de golpe). El detector por forma no las toma por cajas: una pared corrida es una recta larga."""
    grid = maze()
    m = Mission(grid, START[:2], P)
    x, y = 2.0, -0.25 # en el medio del pasillo largo: paredes a hasta 4 m
    for k in range(30):
        yaw = 1.0 * 0.1 * k # 1 rad/s
        pts = lidar(grid, x, y) - (x, y)
        a = -0.05 # el scan es de 50 ms antes: girado 3° para atrás
        pts = pts @ np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]]) + (x, y)
        m.step(x, y, yaw, 0.1 * k, pts)
    assert m.candidates == []


def test_detects_boxes_by_shape():
    """detect_boxes reconoce una caja por sus caras en un solo scan, suelta o apoyada en una pared (RANSAC separa la cara de la pared), con el centro a
    menos de 3 cm; sin cajas, o con la pose girada 3° (paredes corridas hasta 20 cm), no ve ninguna."""
    grid, rng = maze(), np.random.default_rng(0)
    for box, at in (((1.5, -0.3), (0.0, -0.25)), ((4.33, -1.92), (2.5, -0.25))): # suelta; apoyada en una pared
        found = detect_boxes(grid, lidar(with_box(grid, *box), *at), *at, P, rng)
        assert len(found) == 1 and math.dist(found[0], box) < 0.03, found
    a = 0.05
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    for at in ((0.0, -0.25), (2.0, -0.25), (3.5, -0.25)):
        pts = lidar(grid, *at)
        assert detect_boxes(grid, pts, *at, P, rng) == []
        assert detect_boxes(grid, (pts - at) @ rot.T + at, *at, P, rng) == []


def test_glimpse_is_not_seen():
    """Una celda cuenta como vista con lo visible alrededor hasta seen_margin: el borde de una sombra no (ahí una caja muestra pocos puntos), pero el lado
    de acá de una pared sí (lo que no se ve del otro lado no lo borra)."""
    grid = maze()
    m = Mission(grid, START[:2], P)
    x, y = START[:2]
    vis, seen = visibility(grid, x, y, P.lidar_range, 720), m._visible(x, y)
    assert not (seen & ~vis).any() and seen.sum() < vis.sum()
    cx, cy = grid.to_world(*np.indices(grid.free.shape))
    by_wall = vis & (grid.dist < 0.08) & (np.hypot(cx - x, cy - y) < 1.5)
    assert by_wall.any() and seen[by_wall].mean() > 0.8


def test_looks_at_unseen_corner_before_returning():
    """Terminados los puntos de paso y con víctimas por encontrar, antes de volver va a ver el rincón que quedó sin ver (donde entra una caja); visto, vuelve."""
    grid = maze()
    m = Mission(grid, START[:2], P)
    m.k = len(m.points)
    cx, cy = grid.to_world(*np.indices(grid.free.shape))
    hidden = m.near_route & grid.free & ~ndimage.binary_dilation(visibility(grid, *START[:2], P.lidar_range, 720), iterations=4)
    i, j = np.argwhere(hidden)[0]
    m.seen[:] = True
    m.seen[hidden & (np.hypot(cx - cx[i, j], cy - cy[i, j]) < 0.15)] = False
    m.step(*START, 0.0, lidar(grid, *START[:2]), victims=1)
    assert m.state == 'going' and math.dist(m.goal, (cx[i, j], cy[i, j])) < P.route_distance + 0.15
    m.seen[:] = True
    m.step(*START, 0.1, lidar(grid, *START[:2]), victims=1)
    assert m.state == 'done' # vuelve, y como está en la largada termina


def test_looks_at_candidate_from_where_it_is():
    """Viéndolo entero (centro y esquinas) a menos de look_range y con el lidar pegándole, no se acerca: se queda quieto y gira la cámara hacia el
    candidato. Si el lidar no ve nada ahí (un fantasma), se acerca: quieto no lo descartaría nunca (_confirm actúa más cerca)."""
    grid = maze()
    for world, still in ((with_box(grid, 2.5, -0.3), True), (grid, False)): # a 2.5 m por el pasillo
        m = Mission(grid, START[:2], P)
        m.candidates.append({'xy': (2.5, -0.3), 'time': 0.0, 'views': [], 'misses': 0, 'done': False, 'why': None})
        (vx, vy, w), _ = m.step(*START, 0.0, lidar(world, *START[:2]))
        assert ((vx, vy) == (0.0, 0.0)) == still


def test_does_not_turn_to_look_with_something_where_the_front_corners_sweep():
    """Girando en el lugar, las esquinas de adelante barren 0.155 m del centro: con el lidar viendo algo a menos de turn_clearance no gira a mirar el
    candidato (en Gazebo, la esquina rozaba la pared); sin eso, sí."""
    grid = maze()
    start = (START[0], START[1], math.pi / 2) # el candidato queda 90° a la derecha
    for world, turns in ((with_box(grid, START[0] - 0.19, START[1], side=0.05), False), (grid, True)): # a la izquierda, con la cara a 0.165 m del centro
        m = Mission(grid, START[:2], P)
        m.candidates.append({'xy': (2.5, -0.3), 'time': 0.0, 'views': [], 'misses': 0, 'done': False, 'why': None})
        (_, _, w), _ = m.step(*start, 0.0, lidar(world, *START[:2]))
        assert (abs(w) > 0) == turns


def test_passes_beside_a_box_in_the_middle_of_a_corridor():
    """Una caja en el medio de un pasillo de 0.9 m deja 0.325 m a cada lado: el círculo de body_radius no entra, el chasis (0.20 x 0.30) sí, alineado con
    el paso. Vuelve a la largada, del otro lado de la caja, arrancando cruzado (90°), sin que el chasis toque nada."""
    g = np.zeros((24, 80), np.int16)
    g[:3] = g[-3:] = g[:, :2] = g[:, -2:] = 100 # pasillo de 0.9 m de x = -0.4 a 3.4
    grid = Board(g.ravel(), 80, 24, RES, -0.5, -0.6)
    world = with_box(grid, 1.5, 0.0)
    m = Mission(grid, (3.0, 0.0), P)
    m.state = 'returning'
    x, y, yaw, t, done, touch = 0.2, 0.0, math.pi / 2, 0.0, False, False
    edge = np.array([(a, b) for a in np.linspace(-P.body_rear, P.body_front, 7) for b in (-P.body_side, P.body_side)] +
                    [(a, b) for a in (-P.body_rear, P.body_front) for b in np.linspace(-P.body_side, P.body_side, 5)])
    while t < 60.0 and not done:
        (vx, vy, w), done = m.step(x, y, yaw, t, lidar(world, x, y))
        c, s = math.cos(yaw), math.sin(yaw)
        x, y, yaw, t = x + (c * vx - s * vy) * 0.1, y + (s * vx + c * vy) * 0.1, normalize(yaw + w * 0.1), t + 0.1
        c, s = math.cos(yaw), math.sin(yaw)
        i, j = world.to_cell(x + c * edge[:, 0] - s * edge[:, 1], y + s * edge[:, 0] + c * edge[:, 1])
        touch |= bool(world.occupied[i, j].any())
    assert done and not touch


def test_little_of_the_box_in_view_is_not_in_frame():
    """Con menos de media caja a la vista (asomándose detrás de una esquina) no cuenta tiempo en cuadro: si no, la descartaba sin rojo sin que la cámara
    la viera. Con media caja, sí: un señuelo se descarta tras clear_time."""
    grid = maze()
    world = with_box(grid, 1.5, -0.3)
    m = Mission(grid, START[:2], P)
    cd = {'xy': (1.5, -0.3), 'time': 0.0, 'views': [], 'misses': 0, 'done': False, 'why': None}
    m.candidates.append(cd)
    m._in_view = lambda x, y, cx, cy: 2 # el centro y una esquina
    for k in range(15): m.step(*START, 0.1 * k, lidar(world, *START[:2]))
    assert cd['time'] == 0.0 and not cd['done']
    m._in_view = lambda x, y, cx, cy: 3
    for k in range(15, 30): m.step(*START, 0.1 * k, lidar(world, *START[:2]))
    assert cd['done'] and cd['why'] == 'sin rojo'


def test_keeps_looking_at_candidate_until_resolved():
    """Elige el candidato más cercano y sigue con él hasta resolverlo, aunque al moverse otro le quede más cerca: si saltara, dejaría uno a medio mirar."""
    grid = maze()
    m = Mission(grid, START[:2], P)
    a, b = ({'xy': xy, 'time': 0.0, 'views': [], 'misses': 0, 'done': False, 'why': None} for xy in ((2.0, -0.3), (0.6, -0.3)))
    m.candidates += [a, b]
    m.step(1.5, -0.25, 0.0, 0.0, lidar(grid, 1.5, -0.25))
    assert m.target is a
    m.step(1.0, -0.25, 0.0, 0.1, lidar(grid, 1.0, -0.25)) # ahora b está más cerca
    assert m.target is a
    a['done'] = True
    m.step(1.0, -0.25, 0.0, 0.2, lidar(grid, 1.0, -0.25))
    assert m.target is b


def test_discards_phantom_candidates_with_lidar():
    """Un error de pose de unos scans (3°: el localizer corrige y la TF tarda 0.3 s en reflejarlo) corre las paredes hasta 20 cm a 4 m: detect_boxes no
    las toma por cajas (son rectas largas), pero los vistazos sí parecen objetos a 2-4 m. El robot va hacia ellos y, con línea de vista a menos de 2.5 m y
    el lidar sin ver nada ahí, los descarta sin gastar los 3.5 s de cámara."""
    grid = maze()
    m = Mission(grid, START[:2], P)
    x, y, yaw, t = 3.5, -0.25, 0.0, 0.0
    a = 0.05
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    for _ in range(4):
        m.step(x, y, yaw, t, (lidar(grid, x, y) - (x, y)) @ rot.T + (x, y))
        t += 0.1
    phantoms = list(m.candidates)
    assert phantoms
    while t < 40.0 and not all(cd['done'] for cd in phantoms):
        (vx, vy, w), _ = m.step(x, y, yaw, t, lidar(grid, x, y))
        c, s = math.cos(yaw), math.sin(yaw)
        x, y, yaw = x + (c * vx - s * vy) * 0.1, y + (s * vx + c * vy) * 0.1, normalize(yaw + w * 0.1)
        t += 0.1
    assert all(cd['done'] and cd['why'] in ('fantasma', 'sin rojo') and cd['time'] < m.p.watch_time for cd in phantoms)


def test_occluded_box_is_not_a_phantom():
    """Una caja tapada por otra que el mapa tampoco tiene: el lidar no la ve, pero no es un fantasma (algo más cerca la tapa)."""
    grid = maze()
    world = with_box(with_box(grid, 1.5, -0.25), 2.5, -0.25)
    m = Mission(grid, START[:2], P)
    m.candidates.append({'xy': (2.5, -0.25), 'time': 0.0, 'views': [], 'misses': 0, 'done': False, 'why': None}) # la vio antes, desde otro lado
    for k in range(10): m.step(0.5, -0.25, 0.0, 0.1 * k, lidar(world, 0.5, -0.25))
    far = [cd for cd in m.candidates if cd['xy'] == (2.5, -0.25)][0]
    assert far['misses'] == 0 and not far['done']


def test_discards_decoy_in_about_a_second():
    """Un señuelo: la cámara lo tiene en cuadro y la percepción no ve rojo (/jar/rojo vacío cerca): se descarta tras clear_time (1 s), no tras
    watch_time (3.5 s) como cuando el navegador solo se enteraba por los reportes."""
    grid = maze()
    blue = (1.5, -0.3)
    m, traj, done, t, reports, _ = simulate(with_box(grid, *blue), grid)
    decoy = [cd for cd in m.candidates if math.hypot(cd['xy'][0] - blue[0], cd['xy'][1] - blue[1]) < 0.35]
    assert done and not reports and len(decoy) == 1
    assert decoy[0]['why'] == 'sin rojo' and decoy[0]['time'] < m.p.clear_time + 0.3


def test_reopens_candidate_when_red_shows_up_later():
    """Un candidato descartado sin rojo vuelve a mirarse si la percepción después ve rojo ahí (lo vio desde otro lado): una víctima no se pierde por un
    primer vistazo malo."""
    grid = maze()
    m = Mission(grid, START[:2], P)
    m.candidates.append({'xy': (1.5, -0.3), 'time': 1.2, 'views': [], 'misses': 0, 'done': True, 'why': 'sin rojo'})
    m.step(0.0, -0.25, math.pi, 0.0, lidar(grid, 0.0, -0.25), red=[(1.55, -0.35)])
    assert not m.candidates[0]['done'] and m.candidates[0]['time'] == 0.0


def test_box_seen_from_another_side_is_not_a_phantom():
    """
    En Gazebo un señuelo salió como fantasma: el robot giraba hacia él y se contaban fallos. Una caja real no se descarta así:
    - quieto, aunque el candidato quede corrido 23 cm hacia las caras que el lidar vio primero: con el primer scan que la muestra pasa al centro de la caja;
    - girando, aunque el scan no la muestre (scan y pose desfasados): girando no se cuentan fallos (solo el primer ciclo, sin giro previo).
    """
    grid = maze()
    box = (1.5, -0.3)
    for xy, spin, world in (((box[0] + 0.20, box[1] + 0.12), 0.0, with_box(grid, *box)), (box, 1.0, grid)):
        m = Mission(grid, START[:2], P)
        m.candidates.append({'xy': xy, 'time': 0.0, 'views': [], 'misses': 0, 'done': False, 'why': None})
        for k in range(10): m.step(0.0, -0.25, spin * 0.1 * k, 0.1 * k, lidar(world, 0.0, -0.25))
        assert m.candidates[0]['misses'] <= 1 and m.candidates[0]['why'] != 'fantasma'


def test_node_stops_without_pose_or_scan():
    """
    El nodo frena si deja de llegar /jar/pose o el lidar: la TF seguiría con la odometría sola y el lidar quedaría congelado. Al arrancar espera el
    primer scan; al volver, el rato sin datos no cuenta para la misión (dt = 0).
    """
    import rclpy
    from geometry_msgs.msg import PoseStamped
    from rclpy.duration import Duration

    from equipo_jar.navigator import Navigator

    rclpy.init(args=['--ros-args', '--params-file', str(PARAMS), '-p', 'victims:=3'])
    try:
        node = Navigator()
        sent = []
        node.cmd_pub = SimpleNamespace(publish=sent.append)
        node.board, node.t0, node.pose = maze(), node.get_clock().now(), lambda: START
        pose = PoseStamped()
        pose.pose.position.x, pose.pose.position.y = START[:2]
        node.on_pose(pose)
        speed = lambda: math.hypot(sent[-1].linear.x, sent[-1].linear.y)

        node.control() # arma la misión; sin scan todavía
        assert node.mission is not None and speed() == 0.0

        node.lidar, node.lidar_at = lidar(maze(), *START[:2]), node.get_clock().now()
        node.control()
        assert speed() > 0.0

        node.pose_at = node.get_clock().now() - Duration(seconds=2.0) # se cayó el localizer
        node.control()
        assert speed() == 0.0 and node.mission.t_prev is None
    finally:
        rclpy.shutdown()


def test_node_retries_scan_until_its_tf_arrives():
    """Si al llegar el scan todavía no está la TF de su instante (en el robot real, el 35 % de los scans), queda pendiente y se usa en el próximo ciclo."""
    import rclpy
    from sensor_msgs.msg import LaserScan
    from equipo_jar.navigator import Navigator

    rclpy.init(args=['--ros-args', '--params-file', str(PARAMS), '-p', 'victims:=1'])
    node = None
    try:
        node = Navigator()
        node.ready_at = 0
        arrived = [False]
        tf = SimpleNamespace(transform=SimpleNamespace(translation=SimpleNamespace(x=1.0, y=2.0), rotation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0)))
        node.tf = SimpleNamespace(lookup_transform=lambda *_: tf if arrived[0] else (_ for _ in ()).throw(LookupError('todavía no')))
        scan = LaserScan(angle_min=0.0, angle_increment=0.01, time_increment=0.0, ranges=[1.0] * 10)
        scan.header.frame_id, scan.header.stamp.sec = 'laser', 1
        node.on_scan(scan)
        assert len(node.lidar) == 0 and node.pending is scan
        arrived[0] = True
        node.use_scan()
        assert node.pending is None and len(node.lidar) == 10 and node.lidar[0] == pytest.approx((2.0, 2.0))
    finally:
        if node is not None: node.destroy_node()
        rclpy.shutdown()


def test_node_near_moves_with_the_wheels_and_the_gyro_not_with_the_localizer():
    """near: el scan corrido lo que avanzó el robot según las ruedas y lo que giró según el giróscopo, sin pasar por map (un salto del localizer no lo
    mueve). Con la IMU al revés o sin IMU, el giro de las ruedas."""
    import rclpy
    from rclpy.time import Time
    from sensor_msgs.msg import Imu, LaserScan
    from equipo_jar.navigator import Navigator

    def tf(x, y, th, sec=0.0):
        return SimpleNamespace(header=SimpleNamespace(stamp=Time(seconds=sec).to_msg()), transform=SimpleNamespace(
            translation=SimpleNamespace(x=x, y=y), rotation=SimpleNamespace(x=0.0, y=0.0, z=math.sin(th / 2), w=math.cos(th / 2))))

    rclpy.init(args=['--ros-args', '--params-file', str(PARAMS), '-p', 'victims:=1'])
    node = None
    try:
        node = Navigator()
        node.ready_at = 0
        frames = {('map', 'laser'): tf(5.0, 5.0, 1.0), ('odom', 'base_footprint'): tf(0.0, 0.0, 0.0, 1.0), ('base_footprint', 'laser'): tf(0.0, 0.0, 0.0),
                  ('base_footprint', 'imu'): tf(0.0, 0.0, 0.0)}
        node.tf = SimpleNamespace(lookup_transform=lambda target, source, t: frames[(target, source)])
        scan = LaserScan(angle_min=0.0, angle_increment=0.01, time_increment=0.0, ranges=[0.5] * 3)
        scan.header.frame_id, scan.header.stamp.sec = 'laser', 1
        node.on_scan(scan)
        frames[('odom', 'base_footprint')] = tf(0.1, 0.0, 0.2, 2.0) # en 1 s las ruedas: 0.1 m adelante y 0.2 rad
        frames[('map', 'laser')] = tf(7.0, 7.0, 2.0) # el localizer saltó: no importa
        rotated = lambda th: (0.4 * math.cos(th), -0.4 * math.sin(th)) # (0.5, 0) del robot en el scan, visto desde el robot ahora

        for wz, th in ((0.1, 0.1), (-0.1, 0.2), (None, 0.2)): # giróscopo: 0.1 rad; al revés que las ruedas; sin IMU
            node.turns.clear(); node.imu_t = None
            for k in range(21) if wz is not None else ():
                m = Imu()
                m.header.frame_id, m.header.stamp, m.angular_velocity.z = 'imu', Time(seconds=1.0 + 0.05 * k).to_msg(), wz
                node.on_imu(m)
            assert node.near()[0] == pytest.approx(rotated(th), abs=1e-6), wz
    finally:
        if node is not None: node.destroy_node()
        rclpy.shutdown()
