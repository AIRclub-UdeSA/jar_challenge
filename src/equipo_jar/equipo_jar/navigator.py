import math
from collections import deque

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped, PoseArray, PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from scipy import ndimage
from scipy.optimize import minimize
from scipy.sparse.csgraph import dijkstra
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import Empty
from tf2_ros import Buffer, TransformListener

from equipo_jar.core import Board, _graph, line_of_sight, nearest_drivable, normalize, path_dist, viewpoint_tour, visibility
from equipo_jar.localizer import MAP_QOS, Localizer, node_params, turn_source, yaw_of, z_row

# --- Caminos (planificación) ----------------------------------------------

def costs(grid, obstacles, p, closed=None):
    """
    (drivable, penalty, dist): por dónde entra el chasis, cuánto cuesta cada celda y los metros hasta la pared u obstáculo más cercano.

    drivable: a más de narrow_side de todo. Cada paso cuesta 1 en el medio de un pasillo ancho, hasta 1 + wall_penalty pegado a la pared (desde
    wall_margin), close_penalty más a menos de robot_radius y narrow_penalty más a menos de body_radius + drivable_margin (paso angosto: lo cruza alineado,
    Mission._narrow): el camino va por el medio cuando hay lugar, pero un paso angosto (una caja en el medio de un pasillo) sigue abierto si es el único.
    closed: celdas de pasos angostos que no pudo cruzar (no son drivable).
    """
    dist = ndimage.distance_transform_edt(~(grid.occupied | obstacles)) * grid.res
    drivable = grid.free & (dist >= p.narrow_side) & (True if closed is None else ~closed)
    penalty = 1 + p.wall_penalty * np.clip((p.wall_margin - dist) / p.wall_margin, 0, 1) + p.close_penalty * (dist < p.robot_radius) + \
              p.narrow_penalty * (dist < p.body_radius + p.drivable_margin)

    return drivable, penalty, dist

def cost_to(grid, drivable, penalty, goal):
    """
    Costo de llegar caminando hasta `goal` desde cada celda drivable (inf afuera).

    Dijkstra como path_dist, con cada paso multiplicado por la penalty de sus dos celdas.
    """
    idx, g = _graph(drivable, grid.res)
    penalty = penalty[drivable]
    g = g.tocoo()
    g.data *= (penalty[g.row] + penalty[g.col]) / 2
    d = dijkstra(g.tocsr(), directed=False, indices=idx[tuple(nearest_drivable(grid, drivable, *goal))])
    out = np.full(drivable.shape, np.inf)
    out[drivable] = d

    return out

def reach_drivable(grid, D, i, j, max_cells=3000):
    """
    Celdas desde (i, j) hasta la drivable más cercana (D finito), caminando por lo libre del mapa. None si no hay.

    Hace falta cuando el robot quedó fuera de lo drivable (pegado a una pared o a un obstáculo). La más cercana en línea recta puede estar del otro lado de una
    pared: por eso se busca a lo ancho por celdas libres, sin cortar en diagonal entre dos paredes.
    """
    if not grid.inside(i, j): return None

    parent, queue = {(i, j): None}, [(i, j)]
    for c in queue:
        if np.isfinite(D[c]):
            route = [c]
            while parent[route[-1]] is not None:
                route.append(parent[route[-1]])

            return route[::-1]

        if len(parent) > max_cells: return None

        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                a, b = c[0] + di, c[1] + dj
                if (a, b) in parent or not (0 <= a < D.shape[0] and 0 <= b < D.shape[1]) or not grid.free[a, b]: continue
                if di and dj and not (grid.free[c[0] + di, c[1]] and grid.free[c[0], c[1] + dj]): continue # esquina: no se pasa en diagonal entre dos paredes

                parent[(a, b)] = c
                queue.append((a, b))

    return None

def path(grid, D, x, y, max_steps=500):
    """
    Puntos (x, y) bajando por el campo D desde la celda de (x, y) hasta la meta (D = 0). None si no hay camino.

    Si (x, y) cae fuera de lo drivable (pegado a una pared o a un obstáculo), primero vuelve a lo drivable por lo libre.
    """
    i, j = grid.to_cell(x, y)
    cells = reach_drivable(grid, D, i, j)
    if cells is None: return None

    i, j = cells[-1]
    for _ in range(max_steps):
        neighbors = [(i + di, j + dj) for di in (-1, 0, 1) for dj in (-1, 0, 1) if di or dj]
        neighbors = [(a, b) for a, b in neighbors if 0 <= a < D.shape[0] and 0 <= b < D.shape[1]]
        a, b = min(neighbors, key=lambda c: D[c])
        if D[a, b] >= D[i, j]: break

        i, j = a, b
        cells.append((i, j))

    return [tuple(map(float, grid.to_world(a, b))) for a, b in cells]

def lookahead_point(route, x, y, distance):
    """Primer punto de la ruta a más de `distance` del robot (o el último): hacia ahí apunta el pure pursuit."""
    for p in route:
        if math.hypot(p[0] - x, p[1] - y) >= distance: return p

    return route[-1]

# --- Cajas en el scan -----------------------------------------------------

def longest_run(mask):
    """Índices del tramo más largo de True seguidos."""
    d = np.diff(np.r_[0, mask.astype(int), 0])
    starts, ends = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
    if not len(starts): return np.zeros(0, int)

    k = np.argmax(ends - starts)

    return np.arange(starts[k], ends[k])

def ransac_segments(q, p, rng):
    """
    Rectas de un grupo de puntos del scan (en el orden de los rayos), con RANSAC secuencial. Cada vuelta prueba detect_iterations rectas por dos puntos al
    azar, se queda con la que tiene más puntos a menos de detect_tolerance y, de esos, con el tramo de rayos seguidos más largo (dos paredes alineadas a
    cada lado de algo no se juntan). Termina cuando el tramo tiene menos de detect_points.

    Returns:
        [(medio (x, y), dirección unitaria, largo (m), índices en q)].
    """
    out, left = [], np.arange(len(q))
    while len(left) >= p.detect_points:
        i, j = rng.integers(len(left), size=(2, p.detect_iterations))
        d = q[left[j]] - q[left[i]]
        length = np.hypot(d[:, 0], d[:, 1])
        ok = length > 1e-6
        if not ok.any(): break

        n = np.column_stack([-d[ok, 1], d[ok, 0]]) / length[ok, None]
        near = np.abs(q[left] @ n.T - np.sum(q[left[i[ok]]] * n, axis=1)) < p.detect_tolerance # (puntos, rectas)
        run = longest_run(near[:, np.argmax(near.sum(axis=0))])
        if len(run) < p.detect_points: break

        seg = q[left[run]]
        mid = seg.mean(axis=0)
        u = np.linalg.svd(seg - mid)[2][0]
        along = (seg - mid) @ u
        out.append((mid + u * (along.max() + along.min()) / 2, u, float(np.ptp(along)), left[run]))
        left = np.delete(left, run)

    return out

def detect_boxes(grid, pts, x, y, p, rng):
    """
    Centros (x, y) de las cajas que muestra un scan, por su forma. pts: el scan en el frame map, en el orden de los rayos; (x, y): desde dónde se midió.

    - El scan se corta en grupos donde la distancia salta más de detect_jump entre rayos vecinos o faltan rayos. Un grupo que es solo pared del mapa (a
      menos de wall_tolerance) se saltea.
    - El resto se parte en rectas (ransac_segments). Las que caen sobre una pared del mapa son pared: una caja apoyada en una pared queda en su mismo grupo y
      RANSAC las separa. Una recta de más de face_max fuera del mapa es una pared corrida (error de pose, scan y pose desfasados): el grupo no sirve.
    - Una caja es una cara de face_min a face_max, o dos perpendiculares (en L). Si en sus dos puntas termina el grupo y del otro lado hay algo más cerca,
      es un pedazo de algo tapado (una pared corrida vista por una puerta): no es una caja. Con una punta tapada, es una caja que se asoma detrás de una
      esquina: al pasar, muchas se ven enteras en un solo scan o en ninguno.
    - El centro, el medio de la cara más larga corrido medio lado hacia atrás (asomándose, queda corrido hasta medio lado: el próximo scan lo corrige).
    """
    if len(pts) < p.detect_points: return []

    r = np.hypot(pts[:, 0] - x, pts[:, 1] - y)
    da = np.abs(np.diff(np.unwrap(np.arctan2(pts[:, 1] - y, pts[:, 0] - x))))
    gap = da > 2.5 * np.median(da) # rayos sin eco en el medio: del otro lado no hay nada cerca
    cut = gap | (np.abs(np.diff(r)) > p.detect_jump)
    wall = grid.distance(pts[:, 0], pts[:, 1]) < p.wall_tolerance
    bounds = np.r_[0, np.nonzero(cut)[0] + 1, len(pts)]
    boxes = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if (~wall[a:b]).sum() < p.detect_points: continue

        segments = [(mid, u, size, idx + a, wall[a + idx].mean() > 0.5) for mid, u, size, idx in ransac_segments(pts[a:b], p, rng)]
        if any(size > p.face_max and not on_map for *_, size, _, on_map in segments): continue # pared corrida

        faces = [(mid, u, size, idx) for mid, u, size, idx, on_map in segments if not on_map]
        if not 1 <= len(faces) <= 2 or max(f[2] for f in faces) < p.face_min: continue
        if len(faces) == 2 and abs(faces[0][1] @ faces[1][1]) > 0.3: continue # dos caras que no son una L

        lo, hi = min(f[3].min() for f in faces), max(f[3].max() for f in faces)
        if (not wall[a:lo].any() and a > 0 and not gap[a - 1] and r[a - 1] < r[a]) and \
           (not wall[hi + 1:b].any() and b < len(pts) and not gap[b - 1] and r[b] < r[b - 1]): continue # la tapa algo más cerca de los dos lados

        mid, u, size, _ = max(faces, key=lambda f: f[2])
        n = np.array([-u[1], u[0]])
        if (mid - (x, y)) @ n < 0: n = -n # hacia atrás de la cara
        boxes.append(tuple(map(float, mid + n * size / 2)))

    return boxes

# --- Guarda de la cola ----------------------------------------------------

TURN_WEIGHT = 0.2 # en la barrera, 1 rad/s cuesta como 0.2 m/s: prefiere recortar el giro antes que el avance

def tail_barrier(points, cmd, p):
    """
    (vx, vy, w) lo más parecido a cmd con el que el chasis (rectángulo body_front x body_rear x body_side) no se acerca a ningún punto más rápido que
    (distancia - guard_margin) / guard_horizon, giro incluido (QP con SLSQP). Desliza en vez de frenar: solo recorta lo que acerca; dentro del margen, aleja.

    points: el lidar en el frame del robot. Usa solo los de detrás de -tail_start: adelante y a los costados frena el círculo de body_radius (_clear_ahead),
    que no cubre la cola (0.18 m, esquinas a 0.21; el 92 % de los choques en Gazebo).
    """
    pts = np.asarray(points, float).reshape(-1, 2)
    pts = pts[pts[:, 0] < -p.tail_start]

    # Distancia al rectángulo (negativa adentro) y normal hacia afuera en su punto más cercano
    rel = pts - ((p.body_front - p.body_rear) / 2, 0.0)
    q = np.abs(rel) - ((p.body_front + p.body_rear) / 2, p.body_side)
    out = np.maximum(q, 0)
    norm = np.linalg.norm(out, axis=1)
    d = norm + np.minimum(q.max(axis=1), 0)
    normal = np.sign(rel) * out / np.maximum(norm[:, None], 1e-12)
    inside = np.nonzero(norm < 1e-10)[0]
    normal[inside] = 0
    normal[inside, q[inside].argmax(axis=1)] = np.sign(rel[inside, q[inside].argmax(axis=1)])

    keep = d < p.guard_margin + 2 * p.guard_horizon * p.max_speed # más lejos ninguna restricción se activa
    if not keep.any(): return cmd

    pts, d, normal = pts[keep], d[keep], normal[keep]
    a = np.column_stack([normal, (pts[:, 0] * normal[:, 1] - pts[:, 1] * normal[:, 0]) / TURN_WEIGHT]) # velocidad con la que el chasis se acerca a cada punto
    b = (d - p.guard_margin) / p.guard_horizon
    scale = np.array([1.0, 1.0, TURN_WEIGHT])
    wanted = np.asarray(cmd, float) * scale
    if np.max(a @ wanted - b) <= 1e-8: return cmd

    r = minimize(lambda z: 0.5 * np.sum((z - wanted) ** 2), np.zeros(3), jac=lambda z: z - wanted, method='SLSQP',
                 bounds=[(-p.max_speed, p.max_speed)] * 2 + [(-TURN_WEIGHT * p.max_turn, TURN_WEIGHT * p.max_turn)],
                 constraints={'type': 'ineq', 'fun': lambda z: b - a @ z, 'jac': lambda z: -a}, options={'maxiter': 25, 'ftol': 1e-8})
    if r.success and np.max(a @ r.x - b) < 1e-5: return tuple(map(float, r.x / scale))

    # Sin solución (ya encerrado): lo que menos se acerca, entre lo que dio el QP y quedarse quieto
    z = r.x if np.isfinite(r.x).all() else np.zeros(3)

    return tuple(map(float, z / scale)) if np.max(a @ z - b) < np.max(-b) else (0.0, 0.0, 0.0)

def slow_down(points, cmd, p):
    """
    cmd con la velocidad recortada a (distancia - body_radius - wall_stop) / slow_time por lo más cercano adelante en la franja del círculo, sin bajar de
    slow_min. El freno (_clear_ahead) ve una pared a wall_stop del chasis: a velocidad plena llega tarde. No frena, así que no tapa el paso.
    """
    v = math.hypot(cmd[0], cmd[1])
    pts = np.asarray(points, float).reshape(-1, 2)
    if v <= p.slow_min or not len(pts): return cmd

    u = np.array(cmd[:2]) / v
    a, side = pts @ u, np.abs(pts @ np.array([-u[1], u[0]]))
    ahead = (a > 0) & (side < p.body_radius) & (np.hypot(*pts.T) > p.self_radius)
    if not ahead.any(): return cmd

    cap = max(p.slow_min, (a[ahead].min() - p.body_radius - p.wall_stop) / p.slow_time)

    return (cmd[0] * cap / v, cmd[1] * cap / v, cmd[2]) if cap < v else cmd

# --- Misión ---------------------------------------------------------------

class Mission:
    """
    La misión completa sin ROS: el lidar busca y la cámara confirma.

    El lidar ve 360°: el robot pasa sin frenar ni girar por puntos que entre todos le muestran el laberinto (viewpoint_tour con lidar_range). Lo que el lidar
    reconoce como caja fuera de las paredes del mapa (detect_boxes), o lo que asoma unos pocos puntos en varios scans (_mark_glimpses), es un candidato
    (víctima, señuelo u otro objeto): el robot gira la cámara hacia el más cercano
    mientras avanza (mecanum: hacia dónde va y hacia dónde mira son independientes) y se le acerca hasta que la percepción lo reporta o lo tuvo en cuadro lo
    suficiente. Sin candidatos no gira. Vuelve cuando se reportaron las N víctimas, cuando ya vio todo (los puntos de paso y después los rincones sin ver
    donde entra una caja) o cuando se acaba el tiempo.
    test_navigator.py la prueba con una simulación cinemática.

    p: los parámetros como atributos (p.max_speed, ...). En el robot es el nodo Navigator, que los lee de params.yaml; en los tests, la sección
    navigator del yaml.

    Estados:
        - 'going': buscando (hacia un candidato o un punto de paso).
        - 'returning' a la largada.
        - 'done': hay que publicar /done y no moverse más.
    """
    def __init__(self, grid, start, p=None):
        self.grid, self.p = grid, p
        self.start = (float(start[0]), float(start[1]))
        self.points, self.vis, self.reachable = viewpoint_tour(grid, self.start, self.p.robot_radius, self.p.lidar_range, self.p.tour_step, self.p.tour_coverage)
        self.hits = np.zeros(grid.free.shape, np.int16) # veces que el lidar vio algo fuera del mapa en cada celda, hasta obstacle_range (obstáculos)
        self.found = np.zeros(grid.free.shape, np.int16) # lo mismo hasta lidar_range, sin girar (vistazos: _mark_glimpses)
        self.obstacles = np.zeros(grid.free.shape, bool)
        self.closed = np.zeros(grid.free.shape, bool) # pasos angostos que no pudo cruzar
        self.drivable, self.penalty, self.clear = costs(grid, self.obstacles, self.p)
        self.seen = np.zeros(grid.free.shape, bool) # lo que ya vio el lidar (con una caja entera a la vista: _visible)
        self.near_route = ndimage.binary_dilation(self.reachable, iterations=int(self.p.route_distance / grid.res)) # el centro de una caja está cerca de donde anda el robot
        self.candidates = [] # {'xy', 'time' (s en cuadro), 'views' [(x, y) del robot], 'done'}
        self.sightings = [] # cajas vistas en un solo scan, esperando el segundo: {'xy', 'n' (scans), 't'}
        self.scan = np.zeros((0, 2)) # el último scan que pasó por detect_boxes
        self.rng = np.random.default_rng(0) # RANSAC: siempre lo mismo con el mismo scan
        self.pocket = None # el rincón sin ver adonde va al terminar los puntos de paso: (meta, celdas, cuántas sin ver al elegirlo)
        self.target = None # el candidato que está mirando (el más cercano cuando lo eligió)
        self.align = None # cruzando un paso angosto: el giro para alinearse con el paso (_narrow)
        self.passage, self.narrow_blocked = [], 0.0 # sus celdas angostas y cuánto lleva tapado ahí
        self.given_up = np.zeros(grid.free.shape, bool) # rincones que no vio ni llegando
        self.start_dist = path_dist(grid, self.reachable, [self.start])[0]
        self.k, self.state, self.blocked_since = 0, 'going', None
        self.goal, self.D, self.replan = None, None, False
        self.blocked_time, self.t_prev, self.yaw_prev = 0.0, None, None # tiempo tapado en total yendo a la meta actual
        self.near = np.zeros((0, 2)) # lidar en el frame del robot, para frenar
        self.is_wall = np.zeros(0, bool) # qué puntos de near son paredes del mapa
        self.blocker = None # el último punto del lidar que lo frenó (frame del robot)
        self.log = [] # mensajes para el log del nodo
        self.cleared_at = -math.inf # última vez que se olvidaron los obstáculos por quedarse sin camino
        self.moved, self.guard_off = None, -math.inf # (x, y, yaw, t) de la última vez que se movió; hasta cuándo está suelta la barrera de la cola
        self.min_speed = self.p.motor_min_speed # m/s: sube si pide moverse y no se mueve (zona muerta mayor que la supuesta)

    def step(self, x, y, yaw, t, lidar=(), reports=(), victims=1, red=(), near=None):
        """
        Un ciclo de control (_step) con la barrera de la cola (tail_barrier) sobre el comando y, al final, la velocidad según lo que tiene adelante
        (slow_down). Los mínimos de motor van antes de la barrera, para que vea lo que de verdad se manda, y después, porque puede dejarlo debajo de la zona
        muerta.

        Si pide moverse y no se mueve en guard_stall s (_step no se entera: para él no está tapado), la barrera lo encierra o la zona muerta es mayor que el
        mínimo: se suelta la barrera guard_release s (frena solo el círculo; un choque cuesta menos que quedarse quieto) y el mínimo sube motor_min_step.
        """
        cmd, done = self._step(x, y, yaw, t, lidar, reports, victims, red, near)
        if done: return cmd, done

        cmd = self._motors(*cmd)
        out = cmd if t < self.guard_off else self._motors(*tail_barrier(self.near, cmd, self.p))
        if self.moved is None or math.hypot(out[0], out[1]) <= 1e-3 or math.hypot(x - self.moved[0], y - self.moved[1]) > 0.03 or \
           abs(normalize(yaw - self.moved[2])) > 0.1: self.moved = (x, y, yaw, t) # el reloj corre solo mientras pide avanzar
        if t - self.moved[3] > self.p.guard_stall and math.hypot(out[0], out[1]) > 1e-3:
            self.moved, self.guard_off = (x, y, yaw, t), t + self.p.guard_release
            self.min_speed = min(self.min_speed + self.p.motor_min_step, 0.5 * self.p.max_speed)
            self.log.append(f'pide moverse y no se mueve en ({x:.2f}, {y:.2f}): suelto la barrera de la cola {self.p.guard_release:.0f} s y el mínimo de motor '
                            f'pasa a {self.min_speed:.2f} m/s')

            return self._motors(*cmd), done

        return slow_down(self.near, out, self.p), done

    def _motors(self, vx, vy, w):
        """El comando por encima de la zona muerta de los motores: si pide moverse, al menos min_speed; si pide girar, al menos motor_min_turn."""
        v = math.hypot(vx, vy)
        if 1e-3 < v < self.min_speed: vx, vy = vx * self.min_speed / v, vy * self.min_speed / v
        w = 0.0 if abs(w) < self.p.motor_turn_tolerance else math.copysign(max(abs(w), self.p.motor_min_turn), w)

        return float(vx), float(vy), float(w)

    def _step(self, x, y, yaw, t, lidar=(), reports=(), victims=1, red=(), near=None):
        """
        Un ciclo de control. lidar: puntos (M, 2) del scan en el frame map. reports: posiciones (x, y) de lo publicado en /victimas. red: dónde ve
        rojo la percepción sin haberlo reportado todavía (/jar/rojo). near: los mismos puntos en el frame del robot (el nodo los corre con las ruedas y el
        giróscopo, sin pasar por map); None: con la pose.

        Returns:
            ((vx, vy, w) en el frame del robot, terminado).
        """
        if self.state == 'done': return (0.0, 0.0, 0.0), True

        lidar = np.asarray(lidar, float).reshape(-1, 2)
        c, s = math.cos(yaw), math.sin(yaw)
        self.near = (lidar - (x, y)) @ np.array([[c, -s], [s, c]]) if near is None else np.asarray(near, float).reshape(-1, 2)
        self.is_wall = self.grid.distance(lidar[:, 0], lidar[:, 1]) < self.p.wall_tolerance
        dt, self.t_prev = (0.0 if self.t_prev is None else t - self.t_prev), t
        turning = dt > 0 and abs(normalize(yaw - self.yaw_prev)) / dt > self.p.turning_rate
        self.yaw_prev = yaw
        self._mark_obstacles(x, y, lidar[~self.is_wall])
        # La caja se reconoce por su forma en el scan: girando también sirve (una pared corrida por el desfase entre scan y pose es una recta larga)
        self._mark_candidates(x, y, t, lidar)
        # Los vistazos no: girando, el desfase (50 ms a 1 rad/s son 3°) corre 20 cm una pared a 4 m y parece un objeto
        if not turning: self._mark_glimpses(lidar[~self.is_wall])
        # Girando tampoco se confirman: el desfase corre los puntos y en Gazebo un señuelo salió como fantasma
        if not turning: self._confirm(x, y, lidar[~self.is_wall])
        objects = lidar[~self.is_wall]
        self._watch(x, y, yaw, dt, reports, red, objects)
        self.seen |= self._visible(x, y)

        if self.state != 'returning' and (len(reports) >= victims or t + self._return_time(x, y) > self.p.time_limit):
            self.state = 'returning'

        target = goal = None
        if self.state == 'going':
            pending = [cd for cd in self.candidates if not cd['done']]
            if pending: # hacia el candidato va y hacia él mira, hasta resolverlo: si saltara al más cercano, dejaría a medio mirar uno que después queda lejos
                if not any(cd is self.target for cd in pending): self.target = min(pending, key=lambda cd: math.hypot(cd['xy'][0] - x, cd['xy'][1] - y))
                target = self.target
                goal = target['xy']
            else: # sin candidatos, al próximo punto de paso que tenga algo sin ver; después, a los rincones que quedaron
                while self.k < len(self.points) and (self.vis[self.k] & ~self.seen).sum() < self._min_pending():
                    self.k += 1
                if self.k < len(self.points):
                    goal = self.points[self.k]
                    if math.hypot(goal[0] - x, goal[1] - y) < self.p.goal_tolerance: self.k += 1
                else:
                    goal = self._pocket(x, y)
                    if goal is None: self.state = 'returning'

        if self.state == 'returning':
            # Termina apenas la rueda más lejana entra en start_tolerance: el nodo publica /done y recién después el freno
            wheels = max(math.hypot(x + c * a - s * b - self.start[0], y + s * a + c * b - self.start[1])
                         for a in (-self.p.wheel_x, self.p.wheel_x) for b in (-self.p.wheel_y, self.p.wheel_y))
            if wheels <= self.p.start_tolerance:
                self.state = 'done'
                return (0.0, 0.0, 0.0), True

            goal = self.start

        w = 0.0
        r = np.hypot(*self.near.T)
        swept = ((r > self.p.self_radius) & (r < self.p.turn_clearance)).any() # algo donde barren las esquinas de adelante al girar (a 0.155 del centro)
        if target is not None and float(self._clear_at(x, y)) >= self.p.body_radius + self.p.drivable_margin and not swept: # en un paso angosto o pegado a algo, no gira a mirar
            tx, ty = target['xy']
            w = float(np.clip(self.p.turn_gain * normalize(math.atan2(ty - y, tx - x) - yaw), -self.p.max_turn, self.p.max_turn))
            # Ya está cerca viéndola (al menos media), o la ve entera a menos de look_range y el lidar le pega: solo la mira. Asomándose detrás de una
            # esquina se acerca (desde ahí la cámara no la ve); sin lidar en ella también (si es un fantasma, cerca lo descarta _confirm)
            d, n = math.hypot(tx - x, ty - y), self._in_view(x, y, tx, ty)
            if (d < self.p.approach and n >= 3) or (d < self.p.look_range and n == 5 and self._hit(objects, tx, ty)): return (0.0, 0.0, w), False

        self.align = None
        cmd = self._go(x, y, yaw, goal, t, dt, final=self.state == 'returning')
        if self.align is not None: w = self.align # cruzando un paso angosto: el giro es para alinearse con el paso
        if cmd is None: # sin camino o tapado demasiado: el candidato se da por visto, el punto se saltea
            reason = 'tapado' if self.blocked_time > self.p.max_blocked else 'sin camino'
            if target is not None:
                target['done'], target['why'] = True, reason
                self.log.append(f'dejo el candidato en ({goal[0]:.2f}, {goal[1]:.2f}): {reason}')
            elif self.k < len(self.points):
                self.log.append(f'salteo el punto {self.k} ({goal[0]:.2f}, {goal[1]:.2f}): {reason}')
                self.k += 1
            elif self.pocket is not None:
                self.log.append(f'dejo el rincón en ({goal[0]:.2f}, {goal[1]:.2f}): {reason}')
                self.given_up |= self.pocket[1]
                self.pocket = None

            return (0.0, 0.0, w), False

        # Los motores tardan en responder y mientras tanto el robot sigue girando: se pide la velocidad para el rumbo que va a tener
        c, s = math.cos(-w * self.p.turn_lead), math.sin(-w * self.p.turn_lead)

        return (c * cmd[0] - s * cmd[1], s * cmd[0] + c * cmd[1], w), False

    # ---------- piezas ----------
    def _min_pending(self):
        """Celdas sin ver que se toleran en un punto de paso: pending_fraction de lo que le toca, y al menos pending_min."""
        return max(self.p.pending_min, self.p.pending_fraction * self.vis[self.k].sum())

    def _return_time(self, x, y):
        """Segundos para volver a la largada desde (x, y): el camino a return_pace de max_speed, más return_margin."""
        i, j = nearest_drivable(self.grid, self.reachable, x, y)

        return self.start_dist[i, j] / (self.p.return_pace * self.p.max_speed) + self.p.return_margin

    def _mark_obstacles(self, x, y, lidar):
        """
        Lo que el lidar ve lejos de toda pared del mapa es un obstáculo (cajas, señuelos). Visto en obstacle_scans scans, entra en los costos como una pared más y
        se replanea.

        Solo hasta obstacle_range: más lejos, un error chico de ángulo (scan y pose desfasados al girar) corre las paredes lo suficiente para inventar obstáculos.
        """
        new = lidar[np.hypot(lidar[:, 0] - x, lidar[:, 1] - y) < self.p.obstacle_range]
        i, j = self.grid.to_cell(new[:, 0], new[:, 1])
        ok = self.grid.inside(i, j)
        np.add.at(self.hits, (i[ok], j[ok]), 1)
        obstacles = self.hits >= self.p.obstacle_scans

        if (obstacles & ~self.obstacles).any(): self._set_obstacles(obstacles)

    def _set_obstacles(self, obstacles):
        """Cambia los obstáculos: recalcula los costos y pide replanear."""
        self.obstacles = obstacles
        self.drivable, self.penalty, self.clear = costs(self.grid, obstacles, self.p, self.closed)
        self.replan = True

    def _mark_candidates(self, x, y, t, lidar):
        """
        Candidatos: las cajas que reconoce detect_boxes, vistas en candidate_scans scans con el centro a menos de detect_match (el segundo, antes de
        detect_keep s), dentro del laberinto (_in_maze). Una a menos de candidate_merge de un candidato es ese: le actualiza la posición (al centro de la
        caja, aunque antes la viera de otro lado o saliera de un vistazo).
        """
        if len(lidar) == len(self.scan) and np.array_equal(lidar, self.scan): return # el mismo scan otra vez: el control va más rápido que el lidar

        self.scan = lidar
        self.sightings = [s for s in self.sightings if t - s['t'] < self.p.detect_keep]
        for xy in detect_boxes(self.grid, lidar, x, y, self.p, self.rng):
            if not self._in_maze(xy): continue

            same = [cd for cd in self.candidates if math.hypot(cd['xy'][0] - xy[0], cd['xy'][1] - xy[1]) < self.p.candidate_merge]
            if same:
                same[0]['xy'] = xy
                continue

            s = next((s for s in self.sightings if math.hypot(s['xy'][0] - xy[0], s['xy'][1] - xy[1]) < self.p.detect_match and s['t'] < t), None)
            if s is None:
                s = {'xy': xy, 'n': 0, 't': -math.inf}
                self.sightings.append(s)
            s.update(xy=xy, n=s['n'] + 1, t=t)
            if s['n'] >= self.p.candidate_scans:
                self.sightings.remove(s)
                self._add_candidate(xy)

    def _mark_glimpses(self, lidar):
        """
        Candidatos por vistazos: de lejos o detrás de una esquina, una caja asoma unos pocos puntos y detect_boxes no la reconoce. Lo que el lidar ve lejos de
        las paredes del mapa (hasta lidar_range), en glimpse_scans scans, en manchas de hasta glimpse_size, es un candidato. Cada celda suma uno por scan, no
        por rayo: de cerca, decenas de rayos de un mismo scan caen en la misma celda y un solo scan malo inventaba candidatos. Solo agrega: la posición la
        corrige detect_boxes (el centro de una mancha queda sobre las caras que ve el lidar, no en el de la caja).
        """
        i, j = self.grid.to_cell(lidar[:, 0], lidar[:, 1])
        ok = self.grid.inside(i, j)
        hit = np.zeros(self.found.shape, bool)
        hit[i[ok], j[ok]] = True
        self.found += hit
        labels, n = ndimage.label(ndimage.binary_dilation(self.found >= self.p.glimpse_scans), structure=np.ones((3, 3)))
        for k in range(1, n + 1):
            cx, cy = self.grid.to_world(*np.nonzero((labels == k) & (self.found >= self.p.glimpse_scans)))
            xy = (float(np.mean(cx)), float(np.mean(cy)))
            if max(np.ptp(cx), np.ptp(cy)) > self.p.glimpse_size or not self._in_maze(xy): continue

            if all(math.hypot(cd['xy'][0] - xy[0], cd['xy'][1] - xy[1]) >= self.p.candidate_merge for cd in self.candidates): self._add_candidate(xy)

    def _in_maze(self, xy):
        """False si una caja en xy está en el borde del mapa, sobre una pared o a más de route_distance de donde anda el robot: algo afuera del laberinto (visto
        por una abertura)."""
        i, j = self.grid.to_cell(*xy)
        h, w = self.grid.free.shape

        return bool(2 <= i < h - 2 and 2 <= j < w - 2 and self.near_route[i, j] and self.grid.free[i, j])

    def _add_candidate(self, xy):
        self.candidates.append({'xy': xy, 'time': 0.0, 'views': [], 'misses': 0, 'done': False, 'why': None})
        self.log.append(f'candidato en ({xy[0]:.2f}, {xy[1]:.2f})')

    def _visible(self, x, y):
        """
        Lo que el lidar ve desde (x, y) con una caja entera a la vista: las celdas visibles a más de seen_margin (por lo libre, sin cruzar paredes) de una que
        no se ve. Por una rendija o en el borde de una sombra una caja muestra pocos puntos y detect_boxes no la reconoce: eso no cuenta como visto.
        """
        vis = visibility(self.grid, x, y, self.p.lidar_range, 720) # a 4 m, rayos a menos de una celda
        hidden = ndimage.binary_dilation(self.grid.free & ~vis, iterations=round(self.p.seen_margin / self.grid.res), mask=self.grid.free)

        return vis & ~hidden

    def _pocket(self, x, y):
        """
        Terminados los puntos de paso y con víctimas por encontrar: adónde ir para ver el rincón sin ver más cercano (por camino) donde entra una caja (al menos
        pending_min celdas seguidas cerca de donde anda el robot). None si no queda ninguno. Con el robot en la meta, si vio parte del rincón elige otra
        meta para lo que queda (una zona grande se ve de a poco); si no vio nada (lo tapa un obstáculo), lo abandona.
        """
        unseen = self.near_route & self.grid.free & ~self.seen & ~self.given_up
        if self.pocket is not None:
            goal, cells, before = self.pocket
            left = (unseen & cells).sum()
            if left >= self.p.pending_min and math.hypot(goal[0] - x, goal[1] - y) >= self.p.goal_tolerance: return goal

            self.pocket = None
            if left < self.p.pending_min or left >= before: # ya lo vio, o llegó y no vio nada
                self.given_up |= cells
                unseen &= ~cells

        labels, n = ndimage.label(unseen, structure=np.ones((3, 3)))
        sizes = ndimage.sum(unseen, labels, range(1, n + 1))
        dist = path_dist(self.grid, self.drivable, [(x, y)])[0] if (sizes >= self.p.pending_min).any() else None # con los obstáculos anotados
        best = None
        for k in np.nonzero(sizes >= self.p.pending_min)[0] + 1:
            cells = labels == k
            d = np.where(ndimage.binary_dilation(cells, iterations=int(self.p.route_distance / self.grid.res) + 1), dist, np.inf)
            i, j = np.unravel_index(np.argmin(d), d.shape) # desde dónde se ve: lo transitable más cerca del rincón
            if np.isfinite(d[i, j]) and (best is None or d[i, j] < best[0]): best = (d[i, j], (i, j), cells)
        if best is None: return None

        goal = tuple(map(float, self.grid.to_world(*best[1])))
        self.pocket = (goal, best[2], int((unseen & best[2]).sum()))
        self.log.append(f'rincón sin ver: voy a ({goal[0]:.2f}, {goal[1]:.2f})')

        return goal

    def _confirm(self, x, y, lidar):
        """
        Descarta candidatos fantasma sin gastar cámara: con línea de vista a menos de confirm_range, si en confirm_misses ciclos seguidos el lidar no ve nada
        a menos de confirm_radius, era una pared corrida por un error de pose de unos scans (el localizer corrige y la TF tarda tf_lead en reflejarlo). lidar: los puntos que no son
        paredes del mapa. Si uno más cercano lo tapa (otro objeto delante, a occlusion_margin de la línea), ese ciclo no cuenta. No se llama girando (ver step).
        """
        r = np.hypot(lidar[:, 0] - x, lidar[:, 1] - y)
        bearing = np.arctan2(lidar[:, 1] - y, lidar[:, 0] - x)
        for cd in self.candidates:
            cx, cy = cd['xy']
            d = math.hypot(cx - x, cy - y)
            if cd['done'] or d > self.p.confirm_range or not line_of_sight(self.grid, (x, y), (cx, cy)): continue

            if (np.hypot(lidar[:, 0] - cx, lidar[:, 1] - cy) < self.p.confirm_radius).any():
                cd['misses'] = 0
                continue
            m = self.p.occlusion_margin
            if ((r < d - m) & (np.abs(normalize(bearing - math.atan2(cy - y, cx - x))) < math.atan2(m, d))).any(): continue

            cd['misses'] += 1
            if cd['misses'] >= self.p.confirm_misses:
                cd['done'], cd['why'] = True, 'fantasma'
                self.log.append(f'candidato en ({cx:.2f}, {cy:.2f}): fantasma, el lidar no ve nada ahí')

    def _watch(self, x, y, yaw, dt, reports, red, lidar):
        """
        Decide qué es cada candidato con lo que dice la percepción mientras lo mira: a menos de camera_range, en la fracción view_fraction central de la foto,
        con al menos media caja a la vista (_in_view: el centro y dos esquinas) y el lidar pegándole en este scan (detect_points puntos fuera de las paredes
        a menos de confirm_radius). Con solo el centro a la vista, o con el candidato corrido unos centímetros (visto de lejos, asomándose), la cámara casi
        no la ve: contarlo como en cuadro descartaba víctimas sin rojo. lidar: los puntos del scan que no son paredes del mapa.

        - Un reporte a menos de match_radius: era una víctima.
        - clear_time en cuadro sin rojo cerca (red: grupos rojos que la percepción todavía no reportó): no lo es. Si después aparece rojo cerca de uno
          descartado así o como fantasma (lo vio desde otro lado), se vuelve a abrir.
        - Con rojo cerca se lo sigue mirando hasta el reporte; sin reporte tras watch_time en cuadro con watch_baseline de base, o watch_still veces eso
          quieto, se deja.
        """
        for cd in self.candidates:
            cx, cy = cd['xy']
            if cd['why'] != 'reportado' and any(math.hypot(cx - rx, cy - ry) < self.p.match_radius for rx, ry in reports):
                cd['done'], cd['why'] = True, 'reportado'
                self.log.append(f'candidato en ({cx:.2f}, {cy:.2f}): reportado')
                continue

            red_near = any(math.hypot(cx - rx, cy - ry) < self.p.match_radius for rx, ry in red)
            if cd['done']:
                if red_near and cd['why'] in ('sin rojo', 'fantasma'):
                    cd.update(done=False, why=None, time=0.0, views=[])
                    self.log.append(f'candidato en ({cx:.2f}, {cy:.2f}): la percepción ve rojo, lo vuelvo a mirar')
                continue

            bearing = abs(normalize(math.atan2(cy - y, cx - x) - yaw))
            if math.hypot(cx - x, cy - y) > self.p.camera_range or bearing > 0.5 * self.p.view_fraction * self.p.camera_fov or self._in_view(x, y, cx, cy) < 3: continue
            if not self._hit(lidar, cx, cy): continue

            cd['time'] += dt
            cd['views'].append((x, y))
            v = np.asarray(cd['views'])
            if not red_near and cd['time'] > self.p.clear_time:
                cd['done'], cd['why'] = True, 'sin rojo'
                self.log.append(f'candidato en ({cx:.2f}, {cy:.2f}): sin rojo, no es una víctima')
            elif cd['time'] > self.p.watch_still * self.p.watch_time or (cd['time'] > self.p.watch_time and np.ptp(v[:, 0]) + np.ptp(v[:, 1]) > self.p.watch_baseline):
                cd['done'], cd['why'] = True, 'rojo sin reporte'
                self.log.append(f'candidato en ({cx:.2f}, {cy:.2f}): rojo pero sin reporte, lo dejo')

    def _hit(self, objects, cx, cy):
        """True si el lidar le pega en este scan a lo que hay en (cx, cy): detect_points puntos que no son pared a menos de confirm_radius."""
        return bool((np.hypot(objects[:, 0] - cx, objects[:, 1] - cy) < self.p.confirm_radius).sum() >= self.p.detect_points)

    def _in_view(self, x, y, cx, cy):
        """Cuántos de 5 puntos de una caja en (cx, cy) (el centro y las esquinas a look_margin) ve desde (x, y), sin una pared del mapa en el medio."""
        m = self.p.look_margin

        return sum(line_of_sight(self.grid, (x, y), (cx + a, cy + b)) for a, b in ((0, 0), (m, m), (m, -m), (-m, m), (-m, -m)))

    def _clear_at(self, x, y):
        """Metros hasta la pared u obstáculo más cercano en (x, y); 0 fuera del mapa."""
        i, j = self.grid.to_cell(x, y)
        ok = self.grid.inside(i, j)

        return np.where(ok, self.clear[np.where(ok, i, 0), np.where(ok, j, 0)], 0.0)

    def _clear_ahead(self, vx, vy):
        """
        False si el lidar ve algo en la franja que el chasis va a barrer en los próximos stop_time segundos (más stop_margin).

        Las paredes del mapa ya las evita el camino: esas solo frenan si están a wall_stop del chasis.
        """
        v = math.hypot(vx, vy)
        if v < 1e-3 or not len(self.near): return True

        u = np.array([vx, vy]) / v
        far = np.hypot(*self.near.T) > self.p.self_radius # lo más cerca es el propio robot
        a = self.near[far] @ u
        side = np.abs(self.near[far] @ np.array([-u[1], u[0]]))
        reach = np.where(self.is_wall[far], self.p.body_radius + self.p.wall_stop, self.p.body_radius + self.p.stop_margin + self.p.stop_time * v)
        blocking = (a > 0) & (a < reach) & (side < self.p.body_radius)
        if blocking.any(): self.blocker = self.near[far][blocking][np.argmin(a[blocking])] # para el log: qué lo frenó

        return not blocking.any()

    def _narrow(self, x, y, yaw, route):
        """
        (vx, vy) para cruzar un paso angosto en los próximos narrow_ahead m del camino (celdas a menos de body_radius + drivable_margin de algo: una caja en
        el medio de un pasillo), o None si no hay. El círculo de body_radius no entra, pero el chasis (0.20 x 0.30) sí, alineado y centrado:
        - gira hasta tener el lado largo en la dirección del paso (self.align, que step usa en vez de mirar candidatos). Sin alinear no entra: a
          body_front + 5 cm del paso se queda girando en el lugar (afuera hay lugar para girar);
        - lejos de la entrada sigue el camino a narrow_speed; a menos de narrow_line, corrido de la línea del paso, primero se pone en ella de costado
          (mecanum): en diagonal, el chasis barre la esquina de lo que angosta el paso;
        - en la línea, y adentro, avanza a lo largo del paso a narrow_speed, corrigiendo hacia el camino de costado; saliendo, sigue el camino sin girar.
        self.passage: sus celdas angostas, para cerrarlo si no lo puede cruzar.
        """
        tight = self.p.body_radius + self.p.drivable_margin
        pts = np.asarray(route)
        along = np.r_[0.0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
        ahead = pts[along <= self.p.narrow_ahead]
        idx = np.nonzero(self._clear_at(ahead[:, 0], ahead[:, 1]) < tight)[0]
        inside = float(self._clear_at(x, y)) < tight
        if not len(idx) and not inside: return None

        self.passage = [tuple(q) for q in ahead[idx]] + ([(x, y)] if inside else [])

        ax, ay = lookahead_point(route, x, y, self.p.lookahead_min)
        if inside and len(idx) < 3: # saliendo: sin girar (las esquinas barren el paso) y por el camino
            self.align = 0.0
            a = normalize(math.atan2(ay - y, ax - x) - yaw)
            return self.p.narrow_speed * math.cos(a), self.p.narrow_speed * math.sin(a)

        # La línea del paso (u): la recta de sus celdas angostas adelante
        seg = ahead[idx] if len(idx) >= 3 else np.vstack([(x, y), ahead[min(len(ahead) - 1, 4)]])
        u = np.linalg.svd(seg - seg.mean(axis=0))[2][0]
        u = -u if (seg[-1] - seg[0]) @ u < 0 else u
        n = np.array([-u[1], u[0]])
        err = normalize(math.atan2(u[1], u[0]) - yaw)
        err = normalize(err - math.pi) if abs(err) > math.pi / 2 else err # de frente o de cola, el que esté más cerca
        self.align = float(np.clip(self.p.turn_gain * err, -self.p.max_turn, self.p.max_turn))
        if not inside and abs(err) > self.p.narrow_align and len(idx) and along[idx[0]] < self.p.body_front + 0.05: return 0.0, 0.0

        rel = seg[0] - (x, y)
        if not inside and rel @ u > self.p.narrow_line: # lejos de la entrada: por el camino
            a = normalize(math.atan2(ay - y, ax - x) - yaw)
            return self.p.narrow_speed * math.cos(a), self.p.narrow_speed * math.sin(a)

        e = float((rel if not inside else np.array([ax - x, ay - y])) @ n) # corrido de la línea del paso
        if not inside and abs(e) > self.p.narrow_center: v = n * math.copysign(min(self.p.narrow_speed, max(0.05, 2 * abs(e))), e)
        else: v = u * self.p.narrow_speed + n * float(np.clip(2 * e, -0.05, 0.05))
        c, s = math.cos(yaw), math.sin(yaw)

        return float(v @ (c, s)), float(v @ (-s, c))

    def _clear_box(self, vx, vy):
        """
        Como _clear_ahead, con el rectángulo del chasis (body_front adelante, body_rear atrás, body_side + narrow_margin a cada lado) en vez del círculo:
        en un paso angosto lo que tiene al costado está a pocos cm y solo frena si se acerca. Avanzando d, un punto queda en pts - d u (frame del robot):
        entra al rectángulo en d_in y sale en d_out (por cada eje, entre sus dos lados).
        """
        v = math.hypot(vx, vy)
        if v < 1e-3 or not len(self.near): return True

        u = np.array([vx, vy]) / v
        far = np.hypot(*self.near.T) > self.p.self_radius # lo más cerca es el propio robot
        pts = self.near[far]
        reach = np.where(self.is_wall[far], self.p.wall_stop, self.p.stop_margin + self.p.stop_time * v)
        side = self.p.body_side + self.p.narrow_margin
        lo, hi = np.array([-self.p.body_rear, -side]), np.array([self.p.body_front, side])
        step = np.where(np.abs(u) < 1e-9, 1e-9, u)
        t1, t2 = (pts - hi) / step, (pts - lo) / step
        d_in, d_out = np.minimum(t1, t2).max(axis=1), np.maximum(t1, t2).min(axis=1)
        blocking = (pts @ u > 0) & (d_in <= d_out) & (d_out > 0) & (d_in < reach)
        if blocking.any(): self.blocker = pts[blocking][np.argmin(d_in[blocking])] # para el log: qué lo frenó

        return not blocking.any()

    def _go(self, x, y, yaw, goal, t, dt, final=False):
        """
        Velocidad (vx, vy) en el frame del robot para avanzar hacia goal por el camino más corto. None si no hay camino o estuvo tapado demasiado.

        Va a velocidad plena hacia el camino sin importar hacia dónde mira: el giro lo decide step().
        """
        goal = (float(goal[0]), float(goal[1]))
        if goal != self.goal: self.blocked_time = 0.0
        if goal != self.goal or self.replan:
            self.goal, self.replan, self.blocked_since = goal, False, None
            self.D = cost_to(self.grid, self.drivable, self.penalty, goal)

        route = path(self.grid, self.D, x, y)
        if route is None and self.obstacles.any() and t - self.cleared_at > self.p.clear_cooldown:
            # Sin camino por los obstáculos anotados: el ruido de pose los engorda y pueden cerrar un pasillo que tiene lugar (un palo en el medio).
            # Se olvidan y se replanea: el lidar vuelve a anotar en obstacle_scans scans lo que de verdad está, ya sin engordar. Si sigue sin camino, no lo hay
            self.cleared_at, self.hits[:] = t, 0
            self._set_obstacles(np.zeros_like(self.obstacles))

            return 0.0, 0.0
        if route is None: return None

        route.append(goal)
        cmd = self._narrow(x, y, yaw, route)
        if cmd is None: self.narrow_blocked = 0.0 # tapado en total en este paso angosto: frenar y alejarse alternan
        elif self._clear_box(*cmd):
            self.blocked_since = None
            return cmd
        else:
            self.narrow_blocked += dt
            if self.narrow_blocked > self.p.narrow_blocked:
                # No lo puede cruzar (más angosto de lo que dice el mapa, mal alineado): se cierra ese paso y se replanea por otro lado, como antes de
                # poder pasar por pasos angostos
                i, j = self.grid.to_cell(*np.asarray(self.passage).T)
                self.closed[i, j] = True
                self.closed = ndimage.binary_dilation(self.closed, iterations=2) & (self.clear < self.p.body_radius + self.p.drivable_margin)
                self.narrow_blocked = 0.0
                self.log.append(f'no paso por el paso angosto en ({x:.2f}, {y:.2f}): lo cierro')
                self._set_obstacles(self.obstacles)

                return 0.0, 0.0

        dist = math.hypot(goal[0] - x, goal[1] - y)
        # Frena al llegar (speed_gain por metro que falta), también volviendo: con start_tolerance 0.15 el centro tiene que caer a ~4 cm de la largada y a
        # max_speed (5 cm por ciclo) la salteaba y quedaba dando vueltas
        v = min(self.p.max_speed, self.p.speed_gain * dist + self.p.min_speed)

        # Apunta adelante sobre el camino; si la recta hasta ahí pasa pegada a una pared o a un obstáculo (una esquina, una caja), prueba más cerca:
        # el camino va lejos de las paredes, así que su próximo tramo suele estar libre.
        margin = min(self.p.robot_radius, float(self._clear_at(x, y))) - 0.01
        for distance in ((self.p.lookahead, self.p.lookahead_short, self.p.lookahead_min) if cmd is None else ()): # tapado en el paso angosto: no prueba otra cosa
            ax, ay = lookahead_point(route, x, y, distance)
            s = np.linspace(0.2, 1.0, 5)
            if distance > self.p.lookahead_min and self._clear_at(x + s * (ax - x), y + s * (ay - y)).min() < margin: continue

            err = normalize(math.atan2(ay - y, ax - x) - yaw)
            vx, vy = v * math.cos(err), v * math.sin(err)
            if self._clear_ahead(vx, vy):
                self.blocked_since = None
                return vx, vy

        # Se suma todo el tiempo tapado hacia esta meta: si se reiniciara al destaparse, frenar-alejarse-avanzar-frenar podría repetirse para siempre
        self.blocked_since = self.blocked_since if self.blocked_since is not None else t
        self.blocked_time += dt
        if not final and self.blocked_time > self.p.max_blocked: return None

        if final and self.blocked_time > self.p.return_max_blocked and self.obstacles.any():
            # Volviendo no hay punto que saltear: se olvidan los obstáculos anotados (pueden ser fantasmas) y se replanea
            self.hits[:] = 0
            self._set_obstacles(np.zeros_like(self.obstacles))
            self.blocked_time = 0.0

        # Trabado: se aleja de todo lo que tiene a menos de retreat_radius (no solo de lo más cercano, así no se mete en un rincón) a retreat_speed: menos no
        # vence la zona muerta de los motores. Sin girar, y vuelve a probar
        far = self.near[np.hypot(*self.near.T) > self.p.self_radius]
        d = np.hypot(*far.T)
        close = d < self.p.retreat_radius
        q = (far[close] / d[close, None] ** 2).sum(axis=0) if close.any() else far[np.argmin(d)]
        n = math.hypot(*q)

        return (-self.p.retreat_speed * q[0] / n, -self.p.retreat_speed * q[1] / n) if n > 1e-9 else (0.0, 0.0)

# --- Nodo de ROS ----------------------------------------------------------

class Navigator(Node):
    """
    Le pasa a Mission la pose (TF map -> base_footprint del localizer), el lidar (en map y, para frenar, en el frame del robot: near) y las posiciones de
    los reportes, y publica /cmd_vel y /done.

    No se mueve antes de /map ni antes de que el localizer publique la largada (/jar/pose). Frena mientras falte /jar/pose o un scan nuevo (sensor_timeout).
    Después de /done no vuelve a publicar.
    """
    def __init__(self):
        super().__init__('navigator', automatically_declare_parameters_from_overrides=True)  # Registra el nodo en ROS con los parámetros de params.yaml
        p = node_params(self)  # Los valores están solo en params.yaml (sección navigator), comentados. Mission los lee del nodo

        # Movimiento
        self.max_speed = p['max_speed']
        self.max_turn = p['max_turn']
        self.turn_lead = p['turn_lead']
        self.turn_gain = p['turn_gain']
        self.speed_gain = p['speed_gain']
        self.min_speed = p['min_speed']

        # Cuerpo y camino
        self.body_radius = p['body_radius']
        self.robot_radius = p['robot_radius']
        self.wheel_x = p['wheel_x']
        self.wheel_y = p['wheel_y']
        self.self_radius = p['self_radius']
        self.drivable_margin = p['drivable_margin']
        self.wall_margin = p['wall_margin']
        self.wall_penalty = p['wall_penalty']
        self.close_penalty = p['close_penalty']
        self.lookahead = p['lookahead']
        self.lookahead_short = p['lookahead_short']
        self.lookahead_min = p['lookahead_min']
        self.body_front = p['body_front']
        self.body_rear = p['body_rear']
        self.body_side = p['body_side']
        self.narrow_side = p['narrow_side']
        self.narrow_penalty = p['narrow_penalty']
        self.narrow_ahead = p['narrow_ahead']
        self.narrow_align = p['narrow_align']
        self.narrow_speed = p['narrow_speed']
        self.narrow_margin = p['narrow_margin']
        self.narrow_blocked = p['narrow_blocked']
        self.narrow_line = p['narrow_line']
        self.narrow_center = p['narrow_center']

        # Frenado, obstáculos y destrabe
        self.wall_tolerance = p['wall_tolerance']
        self.stop_time = p['stop_time']
        self.stop_margin = p['stop_margin']
        self.wall_stop = p['wall_stop']
        self.obstacle_range = p['obstacle_range']
        self.obstacle_scans = int(p['obstacle_scans'])
        self.clear_cooldown = p['clear_cooldown']
        self.max_blocked = p['max_blocked']
        self.retreat_radius = p['retreat_radius']
        self.retreat_speed = p['retreat_speed']

        # Guarda de la cola y motores
        self.tail_start = p['tail_start']
        self.guard_horizon = p['guard_horizon']
        self.guard_margin = p['guard_margin']
        self.guard_stall = p['guard_stall']
        self.guard_release = p['guard_release']
        self.slow_time = p['slow_time']
        self.slow_min = p['slow_min']
        self.turn_clearance = p['turn_clearance']
        self.motor_min_speed = p['motor_min_speed']
        self.motor_min_step = p['motor_min_step']
        self.motor_min_turn = p['motor_min_turn']
        self.motor_turn_tolerance = p['motor_turn_tolerance']

        # Recorrido
        self.lidar_range = p['lidar_range']
        self.tour_step = p['tour_step']
        self.tour_coverage = p['tour_coverage']
        self.goal_tolerance = p['goal_tolerance']
        self.pending_fraction = p['pending_fraction']
        self.pending_min = int(p['pending_min'])
        self.seen_margin = p['seen_margin']

        # Candidatos
        self.detect_jump = p['detect_jump']
        self.detect_tolerance = p['detect_tolerance']
        self.detect_points = int(p['detect_points'])
        self.detect_iterations = int(p['detect_iterations'])
        self.face_min = p['face_min']
        self.face_max = p['face_max']
        self.candidate_scans = int(p['candidate_scans'])
        self.detect_match = p['detect_match']
        self.detect_keep = p['detect_keep']
        self.glimpse_scans = int(p['glimpse_scans'])
        self.glimpse_size = p['glimpse_size']
        self.candidate_merge = p['candidate_merge']
        self.route_distance = p['route_distance']
        self.turning_rate = p['turning_rate']
        self.confirm_range = p['confirm_range']
        self.confirm_radius = p['confirm_radius']
        self.confirm_misses = int(p['confirm_misses'])
        self.occlusion_margin = p['occlusion_margin']

        # Cámara y percepción
        self.camera_range = p['camera_range']
        self.camera_fov = p['camera_fov']
        self.view_fraction = p['view_fraction']
        self.approach = p['approach']
        self.look_range = p['look_range']
        self.look_margin = p['look_margin']
        self.match_radius = p['match_radius']
        self.clear_time = p['clear_time']
        self.watch_time = p['watch_time']
        self.watch_baseline = p['watch_baseline']
        self.watch_still = p['watch_still']

        # Regreso
        self.time_limit = p['time_limit']
        self.return_pace = p['return_pace']
        self.return_margin = p['return_margin']
        self.start_tolerance = p['start_tolerance']
        self.return_max_blocked = p['return_max_blocked']

        # Nodo
        self.base_frame = p['base_frame']
        self.rate_hz = p['rate_hz']
        self.sensor_timeout = p['sensor_timeout']
        self.start_match = p['start_match']
        self.victims = int(p['victims'])  # N: llega del launch (victimas:=N)
        self.imu_check = p['imu_check']
        self.imu_gap = p['imu_gap']
        self.max_age = p['max_age']

        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.done_pub = self.create_publisher(Empty, '/done', 10)

        # Escuchamos
        self.create_subscription(OccupancyGrid, '/map', self.on_map, MAP_QOS)
        self.create_subscription(LaserScan, '/scan', self.on_scan, qos_profile_sensor_data)
        self.create_subscription(PoseStamped, '/jar/pose', self.on_pose, 10)
        self.create_subscription(PoseArray, '/jar/rojo', self.on_red, 10)
        self.create_subscription(PointStamped, '/victimas', self.on_victim, 10)
        self.create_subscription(Imu, '/imu/data', self.on_imu, 50)
        self.create_timer(1.0 / self.rate_hz, self.control)

        # Estado inicial
        self.board = self.mission = self.t0 = self.start = self.ready_at = None
        self.finished, self.reports = False, [] # posiciones de lo publicado en /victimas (por nodo_equipo)
        self.lidar = np.zeros((0, 2))
        self.scan_t = 0 # ns, hora de los sensores del último scan: la nuestra puede ser otra (el software en otra compu)
        self.pose_at = self.lidar_at = None # cuándo llegó la última /jar/pose y el último scan pasado a map (nuestra hora)
        self.red = [] # dónde ve rojo la percepción sin haberlo reportado (/jar/rojo)
        self.pending = None # scan cuya TF todavía no llegó: se reintenta en cada ciclo (en el robot real llega ~20 ms después)
        self.scan_base = None # el último scan para near: (t, pose de las ruedas en t (odom), puntos en base_footprint)
        self.turns = deque(maxlen=400) # (t, giro integrado del giróscopo)
        self.imu_yaw, self.imu_t, self.imu_row = 0.0, None, None

    # ---------- entradas ----------
    def on_map(self, msg):
        """Guarda el mapa una sola vez. Su llegada es la largada: el tiempo de la misión se cuenta desde acá."""
        if self.board is None: self.board, self.t0 = Board.from_msg(msg), self.get_clock().now()

    def on_pose(self, msg):
        """
        La primera pose del localizer es la largada (adonde vuelve). No sale de la TF: map -> odom va adelantada tf_lead (0.3 s) y en ese lapso
        la TF todavía da la pose de antes de localizarse (con el robot corrido 18 cm del origen, volvía al origen).

        El localizer publica una con cada scan: si dejan de llegar, control frena.
        """
        if self.start is None: self.start = (msg.pose.position.x, msg.pose.position.y)
        self.pose_at = self.get_clock().now()

    def on_red(self, msg):
        self.red = [(p.position.x, p.position.y) for p in msg.poses]

    def on_victim(self, msg):
        self.reports.append((msg.point.x, msg.point.y))

    def on_imu(self, msg):
        """Integra el giróscopo (t, giro) para near, como localizer.on_imu. Sin sesgo: en los ~0.1 s entre el scan y el control, 0.02 rad/s son 0.1°."""
        if self.imu_row is None:
            try: self.imu_row = z_row(self.tf.lookup_transform(self.base_frame, msg.header.frame_id, Time()).transform.rotation)
            except Exception: return

        w, t = msg.angular_velocity, Time.from_msg(msg.header.stamp).nanoseconds * 1e-9
        if self.imu_t is not None and 0.0 < t - self.imu_t < self.imu_gap: self.imu_yaw += float(self.imu_row @ [w.x, w.y, w.z]) * (t - self.imu_t)
        self.imu_t = t
        self.turns.append((t, self.imu_yaw))

    def on_scan(self, msg):
        """
        Guarda los puntos del scan en el frame map (obstáculos, candidatos) y en base_footprint (near, para frenar), con la TF de su instante (use_scan).
        Si todavía no llegó, queda pendiente y se reintenta en cada ciclo de control: en el bag, al llegar el scan falta la TF de odom en el 35 % de los
        casos y llega 18 ms después (mediana; máx. 39). Descartarlos dejaba hasta 0.4 s sin lidar nuevo. Antes de reemplazarlo, el pendiente tiene otra
        oportunidad.
        """
        self.use_scan()
        self.pending = msg
        self.use_scan()

    def use_scan(self):
        """
        Pasa el scan pendiente al frame map con la TF del instante del scan, no con la última: girando, la diferencia corre las paredes e inventa obstáculos.
        También lo guarda en base_footprint con la pose de las ruedas en ese instante (scan_base, para near). Si la TF todavía no está, sigue pendiente.
        """
        msg = self.pending
        if msg is None: return

        t = Time.from_msg(msg.header.stamp) + Duration(seconds=msg.time_increment * (len(msg.ranges) - 1) / 2)
        self.scan_t = t.nanoseconds
        if self.ready_at is None or self.scan_t <= self.ready_at: # antes, la TF todavía no tenía la largada: las paredes salían corridas y parecían objetos
            self.pending = None
            return

        try:
            tf = self.tf.lookup_transform('map', msg.header.frame_id, t).transform
            ob = self.tf.lookup_transform('odom', self.base_frame, t).transform
            mount = self.tf.lookup_transform(self.base_frame, msg.header.frame_id, Time()).transform # dónde está el lidar en el robot: fijo
        except Exception:
            return

        self.pending = None

        r = np.asarray(msg.ranges)
        a = msg.angle_min + np.arange(len(r)) * msg.angle_increment
        ok = np.isfinite(r) & (r > self.self_radius) & (r < self.lidar_range)
        th = yaw_of(tf.rotation)
        c, s = math.cos(th), math.sin(th)
        x, y = r[ok] * np.cos(a[ok]), r[ok] * np.sin(a[ok])
        self.lidar = np.column_stack([tf.translation.x + c * x - s * y, tf.translation.y + s * x + c * y])
        c, s = math.cos(yaw_of(mount.rotation)), math.sin(yaw_of(mount.rotation))
        base = np.column_stack([mount.translation.x + c * x - s * y, mount.translation.y + s * x + c * y]) # mismos puntos y orden que lidar (is_wall)
        self.scan_base = (t.nanoseconds * 1e-9, (ob.translation.x, ob.translation.y, yaw_of(ob.rotation)), base)
        self.lidar_at = self.get_clock().now()

    # ---------- control ----------
    def near(self):
        """
        El último scan en el frame del robot ahora, para frenar (_clear_ahead, tail_barrier, slow_down, giro_lidar): corrido lo que se movió el robot desde
        el scan, con la traslación de las ruedas (odom) y el giro del giróscopo, que no patina. Si la IMU no llega o gira al revés, el giro de las ruedas
        (turn_source). No pasa por map: una corrección del localizer en el medio no mueve lo que tiene cerca. En el bag, llevando cada scan al siguiente,
        calza 0.90 cm (p95; 0.62 con la escala del encoder corregida) contra 1.22 pasando por map. None si todavía no hay scan o falta la TF (Mission usa
        la pose).
        """
        if self.scan_base is None: return None

        t0, (x0, y0, th0), pts = self.scan_base
        try: tf = self.tf.lookup_transform('odom', self.base_frame, Time())
        except Exception: return None

        t1, x1, y1, th1 = Time.from_msg(tf.header.stamp).nanoseconds * 1e-9, tf.transform.translation.x, tf.transform.translation.y, yaw_of(tf.transform.rotation)
        a, b = Localizer.interpolate(self.turns, t0, self.max_age), Localizer.interpolate(self.turns, t1, self.max_age)
        d_odom, d_imu = normalize(th1 - th0), None if a is None or b is None else b[0] - a[0]
        dth = d_imu if turn_source(d_odom, d_imu, self.imu_check) == 'imu' else d_odom
        c, s = math.cos(th0), math.sin(th0)
        dx, dy = c * (x1 - x0) + s * (y1 - y0), -s * (x1 - x0) + c * (y1 - y0) # lo que avanzó, en el frame del robot en el scan
        c, s = math.cos(dth), math.sin(dth)

        return (pts - (dx, dy)) @ np.array([[c, -s], [s, c]])

    def pose(self):
        """Pose (x, y, yaw) del robot en map con la última TF. None si todavía no está."""
        try:
            tf = self.tf.lookup_transform('map', self.base_frame, Time()).transform
        except Exception:
            return None

        return tf.translation.x, tf.translation.y, yaw_of(tf.rotation)

    def control(self):
        """
        Un ciclo de control (rate_hz): pose -> Mission.step -> /cmd_vel, y /done al terminar.

        La Mission se crea con la primera pose localizada, que es la largada. Si algo falla, frena y sigue intentando: un crash es la corrida perdida.
        """
        if self.board is None or self.finished: return # antes del mapa no se mueve; después de /done, tampoco

        try:
            self.use_scan() # el scan que esperaba su TF
            pose = self.pose()
            if pose is None or self.start is None: return

            if self.mission is None:
                if math.hypot(pose[0] - self.start[0], pose[1] - self.start[1]) > self.start_match: return # la TF todavía no tiene la largada del localizer
                self.mission, self.ready_at = Mission(self.board, self.start, self), self.scan_t # los scans que siguen ya tienen la largada en la TF
                self.get_logger().info(f'recorrido: {len(self.mission.points)} puntos de paso desde ({pose[0]:.2f}, {pose[1]:.2f})')

            # Sin /jar/pose o sin scan nuevo, frena: se cayó el localizer o el lidar, y la TF sigue con la odometría sola (y el lidar, congelado)
            now = self.get_clock().now()
            lost = [name for name, at in (('/jar/pose', self.pose_at), ('scan', self.lidar_at)) if at is None or (now - at).nanoseconds * 1e-9 > self.sensor_timeout]
            if lost:
                self.cmd_pub.publish(Twist())
                self.mission.t_prev = self.mission.moved = None # al volver, ese rato no cuenta (dt = 0): ni mirando un candidato, ni tapado, ni sin moverse
                if self.lidar_at is not None: # al arrancar, hasta el primer scan después de la largada, frena sin avisar
                    self.get_logger().warn(f'freno: hace más de {self.sensor_timeout:.1f} s que no llega {" ni ".join(lost)}', throttle_duration_sec=2.0)
                return

            t = (now - self.t0).nanoseconds * 1e-9
            state = self.mission.state
            (vx, vy, w), done = self.mission.step(*pose, t, self.lidar, self.reports, self.victims, self.red, self.near())

            # Log: cambios de estado, candidatos, puntos salteados y qué lo frenó
            if self.mission.state != state:
                self.get_logger().info(f'{state} -> {self.mission.state} (punto {self.mission.k}, t={t:.0f} s)')
            while self.mission.log:
                self.get_logger().info(f'{self.mission.log.pop(0)} (t={t:.0f} s)')
            if self.mission.blocked_since is not None and self.mission.blocker is not None:
                tx, ty = self.mission.blocker
                c, s = math.cos(pose[2]), math.sin(pose[2])
                mx, my = pose[0] + c * tx - s * ty, pose[1] + s * tx + c * ty
                self.get_logger().warn(
                    f'paso tapado en ({pose[0]:.2f}, {pose[1]:.2f}, {math.degrees(pose[2]):.0f}°): lidar a ({tx:.2f}, {ty:.2f}) del robot = '
                    f'({mx:.2f}, {my:.2f}) del mapa, a {float(self.board.distance(mx, my)):.2f} m de una pared del mapa; desde t={self.mission.blocked_since:.0f} s',
                    throttle_duration_sec=1.0)

            if done: # primero /done, con las ruedas adentro; después el freno
                self.done_pub.publish(Empty())
                self.finished = True
                self.get_logger().info(f'/done en ({pose[0]:.2f}, {pose[1]:.2f}), t={t:.0f} s')

            cmd = Twist()
            cmd.linear.x, cmd.linear.y, cmd.angular.z = float(vx), float(vy), float(w)
            self.cmd_pub.publish(cmd)
        except Exception as e:
            self.cmd_pub.publish(Twist())
            self.get_logger().error(f'control: {e!r}', throttle_duration_sec=1.0)

def main(args=None):
    """
    Punto de entrada del ejecutable navigator (setup.py). Lo arranca competencia.launch.py con params.yaml.

    rclpy.spin alcanza: un solo timer y callbacks cortos. Ctrl+C y el apagado del launch terminan sin traceback.
    """
    rclpy.init(args=args)
    node = Navigator()

    try: rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException): pass
    except Exception:
        if rclpy.ok(): raise # con ROS ya apagado (Ctrl+C, el launch), spin puede tirar RCLError al armar la espera: no es un error
    finally:
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()
