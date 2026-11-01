import math

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
from scipy.sparse.csgraph import dijkstra
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Empty
from tf2_ros import Buffer, TransformListener

from equipo_jar.core import Board, _graph, line_of_sight, nearest_drivable, normalize, path_dist, viewpoint_tour, visibility
from equipo_jar.localizer import MAP_QOS, node_params, yaw_of

# --- Caminos (planificación) ----------------------------------------------

def costs(grid, obstacles, p):
    """
    (drivable, penalty, dist): por dónde entra el chasis, cuánto cuesta cada celda y los metros hasta la pared u obstáculo más cercano.

    drivable: a más de body_radius (+2 cm) de todo. Cada paso cuesta 1 en el medio de un pasillo ancho, hasta 3 a wall_margin, y 10 veces más a menos de
    robot_radius: el camino va por el medio cuando hay lugar, pero un paso angosto (una caja cerca de una pared) sigue abierto si es el único.
    """
    dist = ndimage.distance_transform_edt(~(grid.occupied | obstacles)) * grid.res
    drivable = grid.free & (dist >= p.body_radius + 0.02)
    penalty = 1 + 2 * np.clip((p.wall_margin - dist) / p.wall_margin, 0, 1) + 10 * (dist < p.robot_radius)

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

# --- Misión ---------------------------------------------------------------

class Mission:
    """
    La misión completa sin ROS: el lidar busca y la cámara confirma.

    El lidar ve 360°: el robot pasa sin frenar ni girar por puntos que entre todos le muestran el laberinto (viewpoint_tour con lidar_range). Lo que el lidar
    ve lejos de las paredes del mapa, del tamaño de una caja, es un candidato (víctima, señuelo u otro objeto): el robot gira la cámara hacia el más cercano
    mientras avanza (mecanum: hacia dónde va y hacia dónde mira son independientes) y se le acerca hasta que la percepción lo reporta o lo tuvo en cuadro lo
    suficiente. Sin candidatos no gira. Vuelve cuando se reportaron las N víctimas, cuando ya vio todo o cuando se acaba el tiempo.
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
        self.points, self.vis, self.reachable = viewpoint_tour(grid, self.start, self.p.robot_radius, self.p.lidar_range)
        self.hits = np.zeros(grid.free.shape, np.int16) # veces que el lidar vio algo fuera del mapa en cada celda, hasta 1.5 m (obstáculos)
        self.found = np.zeros(grid.free.shape, np.int16) # lo mismo hasta lidar_range (candidatos)
        self.obstacles = np.zeros(grid.free.shape, bool)
        self.drivable, self.penalty, self.clear = costs(grid, self.obstacles, self.p)
        self.seen = np.zeros(grid.free.shape, bool) # lo que ya vio el lidar
        self.near_route = ndimage.binary_dilation(self.reachable, iterations=int(0.3 / grid.res)) # el centro de una caja está a menos de 0.3 m de donde anda el robot
        self.candidates = [] # {'xy', 'time' (s en cuadro), 'views' [(x, y) del robot], 'done'}
        self.start_dist = path_dist(grid, self.reachable, [self.start])[0]
        self.k, self.state, self.blocked_since = 0, 'going', None
        self.goal, self.D, self.replan = None, None, False
        self.blocked_time, self.t_prev, self.yaw_prev = 0.0, None, None # tiempo tapado en total yendo a la meta actual
        self.near = np.zeros((0, 2)) # lidar en el frame del robot, para frenar
        self.is_wall = np.zeros(0, bool) # qué puntos de near son paredes del mapa
        self.blocker = None # el último punto del lidar que lo frenó (frame del robot)
        self.log = [] # mensajes para el log del nodo
        self.closest = (math.inf, 0.0) # volviendo: (menor distancia a la largada, cuándo se acercó 1 cm más por última vez)
        self.cleared_at = -math.inf # última vez que se olvidaron los obstáculos por quedarse sin camino

    def step(self, x, y, yaw, t, lidar=(), reports=(), victims=1, red=()):
        """
        Un ciclo de control. lidar: puntos (M, 2) del scan en el frame map. reports: posiciones (x, y) de lo publicado en /victimas. red: dónde ve
        rojo la percepción sin haberlo reportado todavía (/jar/rojo).

        Returns:
            ((vx, vy, w) en el frame del robot, terminado).
        """
        if self.state == 'done': return (0.0, 0.0, 0.0), True

        lidar = np.asarray(lidar, float).reshape(-1, 2)
        c, s = math.cos(yaw), math.sin(yaw)
        self.near = (lidar - (x, y)) @ np.array([[c, -s], [s, c]])
        self.is_wall = self.grid.distance(lidar[:, 0], lidar[:, 1]) < 0.15
        dt, self.t_prev = (0.0 if self.t_prev is None else t - self.t_prev), t
        turning = dt > 0 and abs(normalize(yaw - self.yaw_prev)) / dt > 0.3
        self.yaw_prev = yaw
        self._mark_obstacles(x, y, lidar[~self.is_wall])
        # Girando, un desfase chico entre scan y pose (50 ms a 1 rad/s son 3°) corre 20 cm una pared a 4 m: parece un objeto. Hasta 1.5 m (obstáculos) no
        if not turning: self._mark_candidates(lidar[~self.is_wall])
        # Girando tampoco se confirman: el candidato no se corrige (la línea de arriba) y el desfase lo corre; en Gazebo un señuelo salió como fantasma
        if not turning: self._confirm(x, y, lidar[~self.is_wall])
        self._watch(x, y, yaw, dt, reports, red)
        self.seen |= visibility(self.grid, x, y, self.p.lidar_range, 180)

        if self.state != 'returning' and (len(reports) >= victims or t + self._return_time(x, y) > self.p.time_limit):
            self.state = 'returning'

        target = goal = None
        if self.state == 'going':
            pending = [cd for cd in self.candidates if not cd['done']]
            if pending: # el candidato más cercano: hacia él va y hacia él mira
                target = min(pending, key=lambda cd: math.hypot(cd['xy'][0] - x, cd['xy'][1] - y))
                goal = target['xy']
            else: # sin candidatos, al próximo punto de paso que tenga algo sin ver
                while self.k < len(self.points) and (self.vis[self.k] & ~self.seen).sum() < self._min_pending():
                    self.k += 1
                if self.k == len(self.points): self.state = 'returning'
                else:
                    goal = self.points[self.k]
                    if math.hypot(goal[0] - x, goal[1] - y) < self.p.goal_tolerance: self.k += 1

        if self.state == 'returning':
            # Termina en la largada, o trabado cerca (motores con zona muerta: a velocidad baja no se mueven) y sin acercarse 1 cm en 3 s
            dist = math.hypot(self.start[0] - x, self.start[1] - y)
            if dist < self.closest[0] - 0.01: self.closest = (dist, t)
            if dist < self.p.start_tolerance or (dist < self.p.start_radius and t - self.closest[1] > 3.0):
                self.state = 'done'
                return (0.0, 0.0, 0.0), True

            goal = self.start

        w = 0.0
        if target is not None:
            tx, ty = target['xy']
            w = float(np.clip(1.5 * normalize(math.atan2(ty - y, tx - x) - yaw), -self.p.max_turn, self.p.max_turn))
            if math.hypot(tx - x, ty - y) < self.p.approach and line_of_sight(self.grid, (x, y), (tx, ty)): return (0.0, 0.0, w), False # ya está cerca: solo lo mira

        cmd = self._go(x, y, yaw, goal, t, dt, final=self.state == 'returning')
        if cmd is None: # sin camino o tapado demasiado: el candidato se da por visto, el punto se saltea
            reason = 'tapado' if self.blocked_time > self.p.max_blocked else 'sin camino'
            if target is not None:
                target['done'], target['why'] = True, reason
                self.log.append(f'dejo el candidato en ({goal[0]:.2f}, {goal[1]:.2f}): {reason}')
            else:
                self.log.append(f'salteo el punto {self.k} ({goal[0]:.2f}, {goal[1]:.2f}): {reason}')
                self.k += 1

            return (0.0, 0.0, w), False

        # Los motores tardan en responder y mientras tanto el robot sigue girando: se pide la velocidad para el rumbo que va a tener
        c, s = math.cos(-w * self.p.turn_lead), math.sin(-w * self.p.turn_lead)

        return (c * cmd[0] - s * cmd[1], s * cmd[0] + c * cmd[1], w), False

    # ---------- piezas ----------
    def _min_pending(self):
        """Celdas sin ver que se toleran en un punto de paso: el 5 % de lo que le toca, y al menos 8."""
        return max(8, 0.05 * self.vis[self.k].sum())

    def _return_time(self, x, y):
        """Segundos para volver a la largada desde (x, y): el camino a media velocidad, más 15 s de margen."""
        i, j = nearest_drivable(self.grid, self.reachable, x, y)

        return self.start_dist[i, j] / (0.5 * self.p.max_speed) + 15.0

    def _mark_obstacles(self, x, y, lidar):
        """
        Lo que el lidar ve lejos de toda pared del mapa es un obstáculo (cajas, señuelos). Visto en 3 scans, entra en los costos como una pared más y se replanea.

        Solo hasta 1.5 m: más lejos, un error chico de ángulo (scan y pose desfasados al girar) corre las paredes lo suficiente para inventar obstáculos.
        """
        new = lidar[np.hypot(lidar[:, 0] - x, lidar[:, 1] - y) < 1.5]
        i, j = self.grid.to_cell(new[:, 0], new[:, 1])
        ok = self.grid.inside(i, j)
        np.add.at(self.hits, (i[ok], j[ok]), 1)
        obstacles = self.hits >= 3

        if (obstacles & ~self.obstacles).any(): self._set_obstacles(obstacles)

    def _set_obstacles(self, obstacles):
        """Cambia los obstáculos: recalcula los costos y pide replanear."""
        self.obstacles = obstacles
        self.drivable, self.penalty, self.clear = costs(self.grid, obstacles, self.p)
        self.replan = True

    def _mark_candidates(self, lidar):
        """
        Candidatos: lo que el lidar ve lejos de las paredes del mapa (hasta lidar_range), en 3 scans, en manchas del tamaño de una caja. Cada celda suma
        uno por scan, no por rayo: de cerca, decenas de rayos de un mismo scan caen en la misma celda y un solo scan malo inventaba candidatos.

        Una mancha de más de 0.6 m es una pared corrida u otro objeto; una en el borde del mapa o a más de 0.3 m de donde anda el robot, algo afuera del
        laberinto (visto por una abertura). Una mancha a menos de 0.35 m de un candidato es el mismo: le actualiza la posición (el centro de lo que vio el
        lidar, sobre las caras que mira: no es el centro de la caja).
        """
        i, j = self.grid.to_cell(lidar[:, 0], lidar[:, 1])
        ok = self.grid.inside(i, j)
        hit = np.zeros(self.found.shape, bool)
        hit[i[ok], j[ok]] = True
        self.found += hit
        labels, n = ndimage.label(ndimage.binary_dilation(self.found >= 3), structure=np.ones((3, 3)))
        for k in range(1, n + 1):
            cx, cy = self.grid.to_world(*np.nonzero((labels == k) & (self.found >= 3)))
            if max(np.ptp(cx), np.ptp(cy)) > 0.6: continue

            xy = (float(np.mean(cx)), float(np.mean(cy)))
            ci, cj = self.grid.to_cell(*xy)
            if not (2 <= ci < self.found.shape[0] - 2 and 2 <= cj < self.found.shape[1] - 2 and self.near_route[ci, cj]): continue

            same = [cd for cd in self.candidates if math.hypot(cd['xy'][0] - xy[0], cd['xy'][1] - xy[1]) < 0.35]
            if same: same[0]['xy'] = xy
            else:
                self.candidates.append({'xy': xy, 'time': 0.0, 'views': [], 'misses': 0, 'done': False, 'why': None})
                self.log.append(f'candidato en ({xy[0]:.2f}, {xy[1]:.2f})')

    def _confirm(self, x, y, lidar):
        """
        Descarta candidatos fantasma sin gastar cámara: con línea de vista a menos de 2.5 m, si en 5 ciclos seguidos el lidar no ve nada a menos de 0.25 m,
        era una pared corrida por un error de pose de unos scans (el localizer corrige y la TF tarda tf_lead en reflejarlo). lidar: los puntos que no son
        paredes del mapa. Si uno más cercano lo tapa (otro objeto delante), ese ciclo no cuenta. No se llama girando (ver step).
        """
        r = np.hypot(lidar[:, 0] - x, lidar[:, 1] - y)
        bearing = np.arctan2(lidar[:, 1] - y, lidar[:, 0] - x)
        for cd in self.candidates:
            cx, cy = cd['xy']
            d = math.hypot(cx - x, cy - y)
            if cd['done'] or d > 2.5 or not line_of_sight(self.grid, (x, y), (cx, cy)): continue

            if (np.hypot(lidar[:, 0] - cx, lidar[:, 1] - cy) < 0.25).any():
                cd['misses'] = 0
                continue
            if ((r < d - 0.3) & (np.abs(normalize(bearing - math.atan2(cy - y, cx - x))) < math.atan2(0.3, d))).any(): continue

            cd['misses'] += 1
            if cd['misses'] >= 5:
                cd['done'], cd['why'] = True, 'fantasma'
                self.log.append(f'candidato en ({cx:.2f}, {cy:.2f}): fantasma, el lidar no ve nada ahí')

    def _watch(self, x, y, yaw, dt, reports, red):
        """
        Decide qué es cada candidato con lo que dice la percepción mientras lo mira: a menos de camera_range, en el 80 % central de la foto y con línea de vista.

        - Un reporte a menos de 0.4 m: era una víctima.
        - clear_time en cuadro sin rojo cerca (red: grupos rojos que la percepción todavía no reportó): no lo es. Si después aparece rojo cerca de uno
          descartado así o como fantasma (lo vio desde otro lado), se vuelve a abrir.
        - Con rojo cerca se lo sigue mirando hasta el reporte; sin reporte tras watch_time en cuadro con 0.2 m de base, o 1.6 veces eso quieto, se deja.
        """
        for cd in self.candidates:
            cx, cy = cd['xy']
            if cd['why'] != 'reportado' and any(math.hypot(cx - rx, cy - ry) < 0.4 for rx, ry in reports):
                cd['done'], cd['why'] = True, 'reportado'
                self.log.append(f'candidato en ({cx:.2f}, {cy:.2f}): reportado')
                continue

            red_near = any(math.hypot(cx - rx, cy - ry) < 0.4 for rx, ry in red)
            if cd['done']:
                if red_near and cd['why'] in ('sin rojo', 'fantasma'):
                    cd.update(done=False, why=None, time=0.0, views=[])
                    self.log.append(f'candidato en ({cx:.2f}, {cy:.2f}): la percepción ve rojo, lo vuelvo a mirar')
                continue

            bearing = abs(normalize(math.atan2(cy - y, cx - x) - yaw))
            if math.hypot(cx - x, cy - y) > self.p.camera_range or bearing > 0.4 * self.p.camera_fov or not line_of_sight(self.grid, (x, y), (cx, cy)): continue

            cd['time'] += dt
            cd['views'].append((x, y))
            v = np.asarray(cd['views'])
            if not red_near and cd['time'] > self.p.clear_time:
                cd['done'], cd['why'] = True, 'sin rojo'
                self.log.append(f'candidato en ({cx:.2f}, {cy:.2f}): sin rojo, no es una víctima')
            elif cd['time'] > 1.6 * self.p.watch_time or (cd['time'] > self.p.watch_time and np.ptp(v[:, 0]) + np.ptp(v[:, 1]) > 0.2):
                cd['done'], cd['why'] = True, 'rojo sin reporte'
                self.log.append(f'candidato en ({cx:.2f}, {cy:.2f}): rojo pero sin reporte, lo dejo')

    def _clear_at(self, x, y):
        """Metros hasta la pared u obstáculo más cercano en (x, y); 0 fuera del mapa."""
        i, j = self.grid.to_cell(x, y)
        ok = self.grid.inside(i, j)

        return np.where(ok, self.clear[np.where(ok, i, 0), np.where(ok, j, 0)], 0.0)

    def _clear_ahead(self, vx, vy):
        """
        False si el lidar ve algo en la franja que el chasis va a barrer en el próximo medio segundo.

        Las paredes del mapa ya las evita el camino: esas solo frenan si están por tocar el chasis.
        """
        v = math.hypot(vx, vy)
        if v < 1e-3 or not len(self.near): return True

        u = np.array([vx, vy]) / v
        far = np.hypot(*self.near.T) > 0.12 # lo más cerca es el propio robot
        a = self.near[far] @ u
        side = np.abs(self.near[far] @ np.array([-u[1], u[0]]))
        reach = np.where(self.is_wall[far], self.p.body_radius + 0.03, self.p.body_radius + 0.04 + 0.5 * v)
        blocking = (a > 0) & (a < reach) & (side < self.p.body_radius)
        if blocking.any(): self.blocker = self.near[far][blocking][np.argmin(a[blocking])] # para el log: qué lo frenó

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
        if route is None and self.obstacles.any() and t - self.cleared_at > 5.0:
            # Sin camino por los obstáculos anotados: el ruido de pose los engorda y pueden cerrar un pasillo que tiene lugar (un palo en el medio).
            # Se olvidan y se replanea: el lidar vuelve a anotar en 3 scans lo que de verdad está, ya sin engordar. Si sigue sin camino, no lo hay
            self.cleared_at, self.hits[:] = t, 0
            self._set_obstacles(np.zeros_like(self.obstacles))

            return 0.0, 0.0
        if route is None: return None

        route.append(goal)
        dist = math.hypot(goal[0] - x, goal[1] - y)
        # Volviendo, si no se acerca (zona muerta de los motores) pide cada vez más velocidad: +0.05 m/s por segundo sin acercarse 1 cm
        v = min(self.p.max_speed, 0.8 * dist + (0.03 + 0.05 * (t - self.closest[1]) if final else 0.08))

        # Apunta adelante sobre el camino; si la recta hasta ahí pasa pegada a una pared o a un obstáculo (una esquina, una caja), prueba más cerca:
        # el camino va lejos de las paredes, así que su próximo tramo suele estar libre.
        margin = min(self.p.robot_radius, float(self._clear_at(x, y))) - 0.01
        for distance in (self.p.lookahead, 0.2, 0.08):
            ax, ay = lookahead_point(route, x, y, distance)
            s = np.linspace(0.2, 1.0, 5)
            if distance > 0.08 and self._clear_at(x + s * (ax - x), y + s * (ay - y)).min() < margin: continue

            err = normalize(math.atan2(ay - y, ax - x) - yaw)
            vx, vy = v * math.cos(err), v * math.sin(err)
            if self._clear_ahead(vx, vy):
                self.blocked_since = None
                return vx, vy

        # Se suma todo el tiempo tapado hacia esta meta: si se reiniciara al destaparse, frenar-alejarse-avanzar-frenar podría repetirse para siempre
        self.blocked_since = self.blocked_since if self.blocked_since is not None else t
        self.blocked_time += dt
        if not final and self.blocked_time > self.p.max_blocked: return None

        if final and self.blocked_time > 3 * self.p.max_blocked and self.obstacles.any():
            # Volviendo no hay punto que saltear: se olvidan los obstáculos anotados (pueden ser fantasmas) y se replanea
            self.hits[:] = 0
            self._set_obstacles(np.zeros_like(self.obstacles))
            self.blocked_time = 0.0

        # Trabado: se aleja de todo lo que tiene a menos de 35 cm (no solo de lo más cercano, así no se mete en un rincón) a 0.1 m/s: menos no vence la zona
        # muerta de los motores. Sin girar, y vuelve a probar
        far = self.near[np.hypot(*self.near.T) > 0.12]
        d = np.hypot(*far.T)
        q = (far[d < 0.35] / d[d < 0.35, None] ** 2).sum(axis=0) if (d < 0.35).any() else far[np.argmin(d)]
        n = math.hypot(*q)

        return (-0.1 * q[0] / n, -0.1 * q[1] / n) if n > 1e-9 else (0.0, 0.0)

# --- Nodo de ROS ----------------------------------------------------------

class Navigator(Node):
    """
    Le pasa a Mission la pose (TF map -> base_footprint del localizer), el lidar y las posiciones de los reportes, y publica /cmd_vel y /done.

    No se mueve antes de /map ni antes de que el localizer publique la largada (/jar/pose). Después de /done no vuelve a publicar.
    """
    def __init__(self):
        super().__init__('navigator', automatically_declare_parameters_from_overrides=True)  # Registra el nodo en ROS con los parámetros de params.yaml
        p = node_params(self)  # Los valores están solo en params.yaml (sección navigator), comentados. Mission los lee del nodo

        # Movimiento
        self.max_speed = p['max_speed']
        self.max_turn = p['max_turn']
        self.turn_lead = p['turn_lead']

        # Cuerpo y camino
        self.body_radius = p['body_radius']
        self.robot_radius = p['robot_radius']
        self.wall_margin = p['wall_margin']
        self.lookahead = p['lookahead']
        self.max_blocked = p['max_blocked']

        # Recorrido y candidatos
        self.lidar_range = p['lidar_range']
        self.goal_tolerance = p['goal_tolerance']
        self.camera_range = p['camera_range']
        self.camera_fov = p['camera_fov']
        self.approach = p['approach']
        self.clear_time = p['clear_time']
        self.watch_time = p['watch_time']

        # Regreso
        self.time_limit = p['time_limit']
        self.start_tolerance = p['start_tolerance']
        self.start_radius = p['start_radius']

        # Nodo
        self.base_frame = p['base_frame']
        self.rate_hz = p['rate_hz']
        self.victims = int(p['victims'])  # N: llega del launch (victimas:=N)

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
        self.create_timer(1.0 / self.rate_hz, self.control)

        # Estado inicial
        self.board = self.mission = self.t0 = self.start = self.ready_at = None
        self.finished, self.reports = False, [] # posiciones de lo publicado en /victimas (por nodo_equipo)
        self.lidar = np.zeros((0, 2))
        self.red = [] # dónde ve rojo la percepción sin haberlo reportado (/jar/rojo)

    # ---------- entradas ----------
    def on_map(self, msg):
        """Guarda el mapa una sola vez. Su llegada es la largada: el tiempo de la misión se cuenta desde acá."""
        if self.board is None: self.board, self.t0 = Board.from_msg(msg), self.get_clock().now()

    def on_pose(self, msg):
        """
        La primera pose del localizer es la largada (adonde vuelve). No sale de la TF: map -> odom va adelantada tf_lead (0.3 s) y en ese lapso
        la TF todavía da la pose de antes de localizarse (con el robot corrido 18 cm del origen, volvía al origen).
        """
        if self.start is None: self.start = (msg.pose.position.x, msg.pose.position.y)

    def on_red(self, msg):
        self.red = [(p.position.x, p.position.y) for p in msg.poses]

    def on_victim(self, msg):
        self.reports.append((msg.point.x, msg.point.y))

    def on_scan(self, msg):
        """
        Guarda los puntos del scan en el frame map (para frenar y para anotar obstáculos).

        Con la TF del instante del scan, no con la última: girando, la diferencia corre las paredes e inventa obstáculos. Si todavía no está, se descarta.
        """
        t = Time.from_msg(msg.header.stamp) + Duration(seconds=msg.time_increment * (len(msg.ranges) - 1) / 2)
        if self.ready_at is None or t < self.ready_at: return # antes, la TF todavía no tenía la largada: las paredes salían corridas y parecían objetos

        try:
            tf = self.tf.lookup_transform('map', msg.header.frame_id, t).transform
        except Exception:
            return

        r = np.asarray(msg.ranges)
        a = msg.angle_min + np.arange(len(r)) * msg.angle_increment
        ok = np.isfinite(r) & (r > 0.12) & (r < self.lidar_range)
        th = yaw_of(tf.rotation)
        c, s = math.cos(th), math.sin(th)
        x, y = r[ok] * np.cos(a[ok]), r[ok] * np.sin(a[ok])
        self.lidar = np.column_stack([tf.translation.x + c * x - s * y, tf.translation.y + s * x + c * y])

    # ---------- control ----------
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
            pose = self.pose()
            if pose is None or self.start is None: return

            if self.mission is None:
                if math.hypot(pose[0] - self.start[0], pose[1] - self.start[1]) > 0.05: return # la TF todavía no tiene la largada del localizer
                self.mission, self.ready_at = Mission(self.board, self.start, self), self.get_clock().now()
                self.get_logger().info(f'recorrido: {len(self.mission.points)} puntos de paso desde ({pose[0]:.2f}, {pose[1]:.2f})')

            t = (self.get_clock().now() - self.t0).nanoseconds * 1e-9
            state = self.mission.state
            (vx, vy, w), done = self.mission.step(*pose, t, self.lidar, self.reports, self.victims, self.red)

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

            cmd = Twist()
            cmd.linear.x, cmd.linear.y, cmd.angular.z = float(vx), float(vy), float(w)
            self.cmd_pub.publish(cmd)

            if done:
                self.done_pub.publish(Empty())
                self.finished = True
                self.get_logger().info(f'/done en ({pose[0]:.2f}, {pose[1]:.2f}), t={t:.0f} s')
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
