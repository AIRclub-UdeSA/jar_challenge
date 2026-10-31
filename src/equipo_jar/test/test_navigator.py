"""La misión de navigator.py en una simulación cinemática del laberinto sintético de test_core, sin ROS.

El robot se mueve con el comando (mecanum: vx, vy y giro; exacto o con motores como los del bag) y el lidar se simula contra el mundo real, que
puede tener cajas que el mapa no tiene. La percepción se simula con la regla de DetectionClusterer.ready() (observaciones_min = 10): una caja roja
se reporta tras ~1.4 s en cuadro (10 scans) con 0.2 m de base, o ~2.8 s quieto, y desde el segundo scan aparece en /jar/rojo. Se mide lo que cuesta
puntos o tiempo: chocar, no ver el laberinto, no encontrar una víctima, girar de más, no volver a la largada o volver tarde.
"""
import math

import numpy as np

from equipo_jar.core import Board, line_of_sight, normalize
from equipo_jar.navigator import Mission, Params
from test_core import RES, maze

START = (0.0, -0.25, 0.0)
ANG = np.linspace(-math.pi, math.pi, 360, endpoint=False)
RAY_STEPS = np.arange(0.12, 4.0, RES / 2)


def with_box(grid, x, y, side=0.25):
    """El mismo laberinto con una caja que el mapa no tiene."""
    g = grid.occupied.astype(np.int16) * 100
    i, j = np.mgrid[0:g.shape[0], 0:g.shape[1]]
    cx, cy = grid.to_world(i, j)
    g[(np.abs(cx - x) < side / 2) & (np.abs(cy - y) < side / 2)] = 100
    return Board(g.ravel(), g.shape[1], g.shape[0], RES, grid.ox, grid.oy)


def lidar(world, x, y):
    """Puntos (frame map) donde pega cada rayo hasta 4 m."""
    gx, gy = x + np.outer(np.cos(ANG), RAY_STEPS), y + np.outer(np.sin(ANG), RAY_STEPS)
    i, j = world.to_cell(gx, gy)
    ok = world.inside(i, j)
    hit = ~ok | world.occupied[np.where(ok, i, 0), np.where(ok, j, 0)]
    hits = hit.any(axis=1)
    r = RAY_STEPS[np.argmax(hit, axis=1)][hits]
    return np.column_stack([x + r * np.cos(ANG[hits]), y + r * np.sin(ANG[hits])])


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
    """La caja (bx, by) en la foto: a menos de camera_range, en el 80 % central y con al menos 2 de 5 puntos (centro y esquinas) a la vista."""
    if math.hypot(bx - x, by - y) > p.camera_range or abs(normalize(math.atan2(by - y, bx - x) - yaw)) > 0.4 * p.camera_fov: return False
    corners = ((0, 0), (.12, .12), (.12, -.12), (-.12, .12), (-.12, -.12))
    return sum(line_of_sight(grid, (x, y), (bx + dx, by + dy)) for dx, dy in corners) >= 2


def simulate(world, grid, p=None, red=(), reports=(), victims=3, t_max=600.0, dt=0.2, motors=None):
    """
    Corre la misión hasta /done. red: cajas rojas del mundo, que la percepción reporta. motors: comando -> velocidad real (None: exacta).

    Returns:
        (misión, trayectoria (T, 2), terminó, t final, reportes, grados girados).
    """
    m = Mission(grid, START[:2], p)
    x, y, yaw = START
    traj, t, done, turned, reports = [], 0.0, False, 0.0, list(reports)
    views = {box: [] for box in red}
    while t < t_max and not done:
        seen_red = [box for box, v in views.items() if len(v) * dt > 0.28 and box not in reports] # /jar/rojo: desde el segundo scan
        (vx, vy, w), done = m.step(x, y, yaw, t, lidar(world, x, y), reports, victims, seen_red)
        if motors: vx, vy, w = motors(vx, vy, w)
        c, s = math.cos(yaw), math.sin(yaw)
        x, y, yaw = x + (c * vx - s * vy) * dt, y + (s * vx + c * vy) * dt, normalize(yaw + w * dt)
        traj.append((x, y))
        turned += abs(w) * dt
        t += dt
        for box, v in views.items(): # percepción: ~1.4 s en cuadro con 0.2 m de base, o ~2.8 s
            if box in reports or not in_camera(grid, x, y, yaw, *box, m.p): continue
            v.append((x, y))
            a = np.asarray(v)
            if (len(v) * dt > 1.4 and np.ptp(a[:, 0]) + np.ptp(a[:, 1]) > 0.2) or len(v) * dt > 2.8: reports.append(box)
    return m, np.array(traj), done, t, reports, math.degrees(turned)


def home(traj):
    return math.hypot(traj[-1, 0] - START[0], traj[-1, 1] - START[1])


def test_covers_maze_with_lidar_and_returns():
    grid = maze()
    m, traj, done, t, _, _ = simulate(grid, grid)
    assert done and t < m.p.time_limit
    assert home(traj) < 0.05  # bono de regreso
    to_see = np.logical_or.reduce(m.vis)
    assert m.seen[to_see].mean() > 0.95
    assert grid.distance(traj[:, 0], traj[:, 1]).min() >= 0.15  # el centro nunca a menos de 15 cm de una pared


def test_does_not_turn_without_candidates():
    """Sin nada fuera del mapa no hay a qué apuntar la cámara: recorre y vuelve sin girar (el robot real gira lento)."""
    grid = maze()
    _, _, done, _, _, turned = simulate(grid, grid)
    assert done and turned < 1.0


def test_finds_red_box_among_decoy_and_returns():
    """Una víctima y un señuelo que el mapa no tiene: los dos son candidatos, la cámara los mira, reporta la roja y vuelve girando poco."""
    grid = maze()
    red, blue = (3.0, 1.5), (-1.0, -1.8)
    world = with_box(with_box(grid, *red), *blue)
    m, traj, done, t, reports, turned = simulate(world, grid, red=[red], victims=1)
    assert reports == [red] and done and home(traj) < 0.05
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
    p = Params.from_yaml(time_limit=60.0)
    _, traj, done, t, _, _ = simulate(grid, grid, p)
    assert done and t <= 60.0
    assert home(traj) < 0.05


def test_returns_with_real_motors():
    """Con retardo, zona muerta y giro desparejo igual encuentra la víctima, no choca y termina con el centro a menos de 10 cm de la largada: el bono
    pide las cuatro ruedas en un círculo de 0.25 m y la rueda más lejana está a 0.15 m del centro."""
    grid = maze()
    red = (3.0, 1.5)
    world = with_box(grid, *red)
    m, traj, done, t, reports, _ = simulate(world, grid, red=[red], victims=1, dt=0.1, motors=real_motors(dt=0.1, deadband=0.2))
    assert reports == [red] and done and t < m.p.time_limit
    assert home(traj) < 0.10  # antes de start_radius y la rampa, con 0.15 m/s se quedaba a 9 cm para siempre
    assert world.distance(traj[:, 0], traj[:, 1]).min() >= 0.15


def test_returns_when_noted_obstacles_close_the_way():
    """Si los obstáculos anotados (engordados por el ruido de pose) cierran el único paso a la largada, los olvida y vuelve igual: antes se quedaba
    quieto para siempre (en Gazebo, un palo en el medio de un pasillo de 0.95 m)."""
    grid = maze()
    m = Mission(grid, START[:2])
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
    assert done and math.hypot(x - START[0], y - START[1]) < 0.05


def test_no_phantom_candidates_when_scan_lags_while_turning():
    """Girando, el scan llega con la pose de 50 ms antes (3° a 1 rad/s): a 4 m las paredes se corren 20 cm y parecían objetos. Así, girando no se
    anotan candidatos; quieto el mismo desfase no existe. (En Gazebo, con ese desfase aparecieron 20 candidatos fantasma de golpe.)"""
    grid = maze()
    m = Mission(grid, START[:2])
    x, y = 2.0, -0.25 # en el medio del pasillo largo: paredes a hasta 4 m
    for k in range(30):
        yaw = 1.0 * 0.1 * k # 1 rad/s
        pts = lidar(grid, x, y) - (x, y)
        a = -0.05 # el scan es de 50 ms antes: girado 3° para atrás
        pts = pts @ np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]]) + (x, y)
        m.step(x, y, yaw, 0.1 * k, pts)
    assert m.candidates == []


def test_discards_phantom_candidates_with_lidar():
    """Un error de pose de unos scans (3°: el localizer corrige y la TF tarda 0.3 s en reflejarlo) deja paredes corridas que parecen objetos a 2-4 m.
    El robot va hacia ellos y, con línea de vista a menos de 2.5 m y el lidar sin ver nada ahí, los descarta sin gastar los 3.5 s de cámara."""
    grid = maze()
    m = Mission(grid, START[:2])
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
    m = Mission(grid, START[:2])
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
    m = Mission(grid, START[:2])
    m.candidates.append({'xy': (1.5, -0.3), 'time': 1.2, 'views': [], 'misses': 0, 'done': True, 'why': 'sin rojo'})
    m.step(0.0, -0.25, math.pi, 0.0, lidar(grid, 0.0, -0.25), red=[(1.55, -0.35)])
    assert not m.candidates[0]['done'] and m.candidates[0]['time'] == 0.0
