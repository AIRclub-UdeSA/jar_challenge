"""
Mission driven by the world model (docs/MISSION.md): where to go comes from planner.py, what
to report from the world model's objects.

- Plans on /equipo/ocupacion (the posterior map) and /equipo/busqueda (the search belief),
  for the objects on /equipo/objetos: explore where undetected victims could be, or look at
  an undecided object. Drives there along the planner's own path (the motion node follows
  it, re-sent every replan_s on the latest map), and plans again when it arrives, when the
  target stops being worth it (the object got decided, the belief got seen on the way), or
  when it gets stuck. Every leg, the way home included, drives the same way.
- A look: turn the camera to the object, then slide along a sweep (planner.sweep), camera on
  it, slowly, until it is decided or the sweep ends. The world model counts an object's color
  once per line of sight, so the sweep crosses the ones no view has counted from, and the next
  look at the same object is planned from a different direction (planner.py,
  mirada_angulo_min).
- Reports an object once P(victim) >= reporte_p_min and its center is within 25 cm with
  probability reporte_p_radio. Before /done, the reports left go to the objects with the
  highest expected points: a report that misses costs only itself.
- Stuck (no progress for atasco_s): drops the goal and marks where, and the path just ahead,
  as a stuck spot. A gap the map shows can be too narrow for the robot (a box in a corridor);
  the planner then prefers targets with another way there.
- Goes home (casa: the map origin or the localized start pose, home_target), and
  publishes /done when every report is spent, when nothing is left worth doing, or when the
  time left only covers the way back.
- A step that raises (a bug, bad data) is logged, the robot stops and the next step plans
  again; errores_max of them within errores_ventana s send it home. Whatever fails, the run
  ends with the robot stopped and /done published (finish), never with a dead node.
- Every timer runs on the ROS clock (sim time in the simulator), so behaviour doesn't depend
  on how fast the simulator runs.
- State for the run's log (docs/LOGGING.md): the map once, a state line and every pose since
  the last one every second, every object every 5 s, a line when an object appears, goes or
  its P moves, and every tour.
"""
import math
import threading
import time
import traceback

from geometry_msgs.msg import PointStamped, PoseStamped
from nav_msgs.msg import Path
from nav_msgs.msg import OccupancyGrid
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Empty

from equipo_jar.keep_alive import spin
from equipo_jar.localizer import yaw_of
from equipo_jar.log_lines import guarded, map_lines, object_changes, objects_line, path_line, \
    tour_line
from equipo_jar.planner import DEFAULTS as PLANNER_DEFAULTS, Grid, Planner, \
    expected_missing, reported_objects, shadowed_objects
from equipo_jar.reportes import armar_reporte, vale_el_bono
from equipo_jar_msgs.msg import EstadoMovimiento, ObjetoArray

PARAMS = {
    'casa': 'origen',           # home: origen (the map origin) or inicio (the start pose); the
                                # launch passes competencia.casa (home_target)
    'mission_deadline': 570.0,  # s after /map: be home and publish /done by then
    'margen_regreso': 20.0,     # s, spare time on the way home
    'factor_regreso': 1.3,      # the way home takes this times the planned path at velocidad
    'goal_timeout': 30.0,       # s per target
    'chequeo_s': 1.0,           # s between checks that the target is still worth it
    'espera_fin': 8.0,          # s with nothing worth doing before going home
    'espera_mirada': 4.0,       # s at most facing the object, until its P(victim) moves: the
                                # depth cloud is subscribed on demand and takes a moment to come
    'fallo_expira': 60.0,       # s, a goal not reached (or an explore spot reached) is skipped
    'atasco_s': 8.0,            # s without moving atasco_dist: stuck, drop the goal
    'atasco_dist': 0.15,        # m
    'atasco_cerca': 0.1,        # m, a stuck event marks the path from this far ahead...
    'atasco_adelante': 0.45,    # m, ...to this far: the stretch the robot couldn't cross
    'replan_s': 2.0,            # s between new paths to the controller while driving
    'llegada': 0.25,            # m from a target: arrived
    'llegada_precisa': 0.05,    # m, for a look retried on a mirador after a dead one
    'llegada_casa': 0.05,       # m from home (0, 0): arrived (the rules give 0.25, and the
                                # localization error takes its share: c05 ended 0.23-0.26 m off,
                                # stopped 0.09 m short of a home 0.145 m off). Also arrived when
                                # the controller says it reached the path's end
    'llegada_casa_max': 0.07,   # m: a leg that ends stuck this close to home is home (the
                                # origin may be inside the safety margin of a wall). The bonus
                                # needs all 4 wheels within 0.25 m of the tape, and the farthest
                                # wheel point is 0.155 m from base_link: 0.095 m, less the
                                # localization error
    'escape_dist': 0.3,         # m, how far an escape moves up the clearance gradient
    'escape_s': 6.0,            # s at most for one escape move
    'bloqueo_s': 3.0,           # s the motion node reports blocked: stuck, without waiting
                                # for atasco_s
    'reporte_p_radio': 0.99,    # P(center within 25 cm) to report
    'bono_s': 300.0,            # s after /map: every victim reported by then earns the bonus
    'bono_margen': 20.0,        # s before bono_s from which the reports left may be committed
    'bono_puntos': 200.0,       # rules §8: all N found in time
    'mirar_en_marcha': True,    # turn the camera to undecided objects while driving
    'barrido': True,            # looks sweep (planner.sweep); false: wait facing the object
    'espera_barrido': 1.0,      # s at a sweep's end for its last images to count (the world
                                # model publishes at 2 Hz)
    'mirada_rango': 2.5,        # m, objects closer than this are looked at on the way
    'separacion': 0.5,          # m, an object this close to a report is that report's box
    'errores_max': 3,           # mission steps that raise within errores_ventana: stop
    'errores_ventana': 60.0,    # s  exploring, go home and publish /done
    'parada_done_s': 0.5,       # s between the robot's stop and /done: the judge counts motion
                                # commands after /done, and the wheels at /done for the bonus
}


CASAS = ('origen', 'inicio')


def home_target(casa, start_xy):
    """Where home is: the map origin (casa 'origen') or where the robot started, its localized
    start pose (casa 'inicio'). The organizers have said both: it is competencia.casa in
    params.yaml, to set on site."""
    if casa == 'origen':
        return (0.0, 0.0)
    if casa == 'inicio':
        return (float(start_xy[0]), float(start_xy[1]))
    raise ValueError(f'casa must be one of {CASAS}, not {casa!r}')


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def p_within(cov, radius):
    """P(the center is within radius of the estimate), isotropic with the largest variance."""
    var = max(np.linalg.eigvalsh(np.array([[cov[0], cov[1]], [cov[1], cov[2]]])).max(), 1e-8)
    return 1.0 - math.exp(-radius ** 2 / (2 * var))


def expected_points(obj):
    """Rules §8: 100 within 0.25 m, 50 within 0.50 m, if it is a victim."""
    near, far = p_within(obj['cov'], 0.25), p_within(obj['cov'], 0.50)
    return obj['p'] * (100 * near + 50 * (far - near))


def grid_of(msg):
    i = msg.info
    return Grid(np.array(msg.data, np.int8).reshape(i.height, i.width), i.resolution,
                (i.origin.position.x, i.origin.position.y))


class Mission(Node):

    def __init__(self):
        super().__init__('mission')
        self.declare_parameter('victimas', 1)  # N, from the launch
        for key, value in {**PARAMS, **PLANNER_DEFAULTS}.items():
            self.declare_parameter(key, value)
        g = lambda k: self.get_parameter(k).value  # noqa: E731
        self.n = int(g('victimas'))
        self.p = {k: type(v)(g(k)) for k, v in PARAMS.items()}
        self.planner_params = {k: type(v)(g(k)) for k, v in PLANNER_DEFAULTS.items()}
        home_target(self.p['casa'], (0.0, 0.0))  # a bad casa fails now, not once the map is in
        self.lock = threading.Lock()
        self.map_time = None
        self.pose = None                # (x, y, yaw), map frame
        self.objects = None             # dicts: id, center, p, cov, radius
        self.occupancy = self.belief = None
        self.reported = []              # report positions
        self.home = None
        self.home_dist = 0.0            # m, path home from the last plan
        self.seen_from = {}             # object id -> where its looks were taken
        self.looks = {}                 # object id -> [looks, looks that changed its P,
                                        #   looks in a row that didn't]
        self.dead_looks = {}            # object id -> where the looks that didn't were taken
        self.stuck = []                 # per stuck event, the points it couldn't get through
        self.planner = None             # the last plan
        self.tour = []                  # the last tour's targets
        self.tour_objects = set()       # the undecided objects that tour knew about, and
                                        #   whether each had a spot to look from
        self.tour_failed = []           # and the failed spots it avoided
        self.tour_visited = []          # and the explore stops it didn't go back to
        self.prefer = None              # where the next tour should start, if not tour[1]
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(OccupancyGrid, '/map', self.on_map, latched)
        self.create_subscription(PoseStamped, '/equipo/pose_continua', self.on_pose, 10)
        self.create_subscription(ObjetoArray, '/equipo/objetos', self.on_objects, latched)
        self.create_subscription(OccupancyGrid, '/equipo/ocupacion', self.on_occupancy, latched)
        self.create_subscription(OccupancyGrid, '/equipo/busqueda', self.on_belief, latched)
        self.report_pub = self.create_publisher(PointStamped, '/victimas', 10)
        self.done_pub = self.create_publisher(Empty, '/done', 10)
        self.goal = '-'                 # what the robot is going for, for the log
        self.belief_left = None         # search belief left, from /equipo/objetos
        self.logged_p = {}              # object id -> its P in its last line in the log
        self.path = []                  # poses since the last path line
        self.create_timer(1.0, self.log_state)
        self.create_timer(1.0, self.log_path)
        self.create_timer(5.0, self.log_objects)
        self.motion_state = None        # the last /equipo/movimiento
        self.create_subscription(EstadoMovimiento, '/equipo/movimiento',
                                 lambda m: setattr(self, 'motion_state', m), 10)
        self.route_pub = self.create_publisher(Path, '/equipo/ruta', 10)
        self.slow_pub = self.create_publisher(Path, '/equipo/ruta_lenta', 10)

    # Inputs.

    def on_map(self, msg):
        if self.map_time is None:
            self.map_time = self.get_clock().now()
            self.log_map(msg)

    def on_pose(self, msg):
        p = msg.pose
        self.pose = (p.position.x, p.position.y, yaw_of(p.orientation))
        self.note_pose(msg)

    def on_occupancy(self, msg):
        self.occupancy = grid_of(msg)

    def on_belief(self, msg):
        self.belief = grid_of(msg)

    def on_objects(self, msg):
        objs = [{'id': int(o.id), 'center': (o.centro.x, o.centro.y), 'p': float(o.p_objetivo),
                 'p_rojo': float(o.p_rojo), 'cov': tuple(o.cov_centro),
                 'radius': math.hypot(o.lado_a, o.lado_b) / 2,
                 'miradores': [(m.x, m.y) for m in o.miradores],
                 'cells': np.asarray(o.celdas, int)}
                for o in msg.objetos]
        self.objects = objs
        self.report_confirmed()
        self.report_for_bonus()
        self.log_object_changes(msg)

    # The run's log.

    @guarded
    def log_map(self, msg):
        for line in map_lines(msg):
            self.get_logger().info(line)

    @guarded
    def log_state(self):
        if self.map_time is None or self.pose is None:
            return
        m = self.motion_state
        self.get_logger().info(
            f'state: t {self.elapsed():.1f} pose {self.pose[0]:.3f} {self.pose[1]:.3f} '
            f'{self.pose[2]:.3f} goal {self.goal} '
            f'motion {m.estado if m else "-"} left {m.restante if m else 0.0:.2f} '
            f'objects {len(self.objects or ())} belief {self.belief_left} '
            f'reports {len(self.reported)}')

    @guarded
    def note_pose(self, msg):
        if self.map_time is not None:
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            self.path.append((stamp, *self.pose))

    @guarded
    def log_path(self):
        """Every pose, one line a second: a logger call per pose would be ten a second."""
        if self.path:
            path, self.path = self.path, []
            self.get_logger().info(path_line(path, self.map_time.nanoseconds * 1e-9))

    @guarded
    def log_objects(self):
        if self.map_time is not None and self.objects is not None:
            self.get_logger().info(objects_line(self.objects))

    @guarded
    def log_object_changes(self, msg):
        self.belief_left = round(float(msg.busqueda_restante), 4)
        for line in object_changes(self.objects, self.logged_p):
            self.get_logger().info(line)

    @guarded
    def log_tour(self, name, tour):
        if tour:
            self.get_logger().info(tour_line(name, tour))

    # Reports.

    def report(self, obj, why):
        """Publish one report; False if the reports are spent or it is a reported box."""
        with self.lock:
            if len(self.reported) >= self.n or any(
                    math.dist(obj['center'], r) < self.p['separacion'] for r in self.reported):
                return False
            msg = armar_reporte(obj['center'][0], obj['center'][1],
                                self.get_clock().now().to_msg())
            self.report_pub.publish(msg)
            self.reported.append(obj['center'])
        self.get_logger().info(
            f'report {len(self.reported)}/{self.n} ({why}): object {obj["id"]} at '
            f'({obj["center"][0]:.2f}, {obj["center"][1]:.2f}), P {obj["p"]:.3f}, '
            f'E[points] {expected_points(obj):.0f}, at {self.elapsed():.0f} s')
        return True

    def report_confirmed(self):
        for o in sorted(self.objects or [], key=lambda o: -o['p']):
            if o['p'] >= self.planner_params['reporte_p_min'] \
                    and p_within(o['cov'], 0.25) >= self.p['reporte_p_radio']:
                self.report(o, 'confirmed')

    def report_for_bonus(self):
        """Shortly before the fast bonus's deadline, spend the reports left on the best
        candidates if the expected bonus beats what wrong reports would cost: a wrong report
        only wastes its slot, worth 100 if its victim is found later. Commit when
        bono_puntos * P(all right) > 100 * sum(P(wrong))."""
        if self.map_time is None or not self.objects:
            return
        t = self.elapsed()
        if not self.p['bono_s'] - self.p['bono_margen'] <= t < self.p['bono_s']:
            return
        with self.lock:
            left = self.n - len(self.reported)
            done = list(self.reported)
        if left <= 0:
            return
        cands = [o for o in self.objects
                 if o['p'] > 0 and all(math.dist(o['center'], r) >= self.p['separacion']
                                       for r in done)]
        best = sorted(cands, key=lambda o: -expected_points(o))[:left]
        if len(best) < left:
            return  # not enough candidates: the bonus is out of reach anyway
        if not vale_el_bono([o['p'] * p_within(o['cov'], 0.5) for o in best],
                            self.p['bono_puntos']):
            return
        for o in best:
            self.report(o, 'bonus')

    def report_leftovers(self):
        """The reports left, to the unreported objects with the most expected points."""
        for o in sorted(self.objects or [], key=lambda o: -expected_points(o)):
            if len(self.reported) >= self.n:
                break
            if o['p'] > 0:
                self.report(o, 'leftover')

    # Time.

    def now(self):
        """s, ROS clock: sim time in the simulator."""
        return self.get_clock().now().nanoseconds * 1e-9

    def elapsed(self):
        return (self.get_clock().now() - self.map_time).nanoseconds * 1e-9

    def time_to_go_home(self):
        way = self.home_dist / self.planner_params['velocidad'] * self.p['factor_regreso']
        return self.elapsed() + way + self.p['margen_regreso'] >= self.p['mission_deadline']

    def finished(self):
        return len(self.reported) >= self.n or self.time_to_go_home()

    # Planning.

    def plan(self, failed, visited=(), choose=True):
        """A Planner on the latest data, and its best target (or None; also without choose:
        only the paths are needed)."""
        planner = Planner(self.occupancy, self.belief, self.pose, self.stuck,
                          **self.planner_params)
        self.planner = planner
        self.home_dist = min(planner.path_to(self.home), 50.0)
        if not math.isfinite(self.home_dist):
            self.home_dist = 1.5 * math.dist(self.pose[:2], self.home)
        if not choose:
            return planner, None
        prefer = self.prefer or (self.tour[1].xy if len(self.tour) > 1 else None)
        self.prefer = None
        objs = self.unreported()
        self.tour = planner.tour(objs, self.n, self.seen_from, failed, prefer, visited)
        self.log_tour('tour', self.tour)
        self.tour_failed, self.tour_visited = list(failed), list(visited)
        self.tour_objects = self.undecided(objs, planner, failed)
        return planner, self.tour[0] if self.tour else None

    def undecided(self, objs, planner, failed):
        """The undecided objects, as (ID, whether it has a spot to look from) pairs: retour
        also replans when one gets a look spot."""
        pp = self.planner_params
        return {(o['id'], planner.look(o, self.seen_from.get(o['id'], ()), failed) is not None)
                for o in objs if pp['p_descartado'] < o['p'] < pp['reporte_p_min']}

    def retour(self, planner, target):
        """While driving to target: an object appeared (or one got decided, or got a spot to
        look from) since the tour was planned. True when a fresh tour, kept on target unless
        another start is tour_histeresis better, starts elsewhere: the next plan starts
        there. A box seen just after leaving a spot is looked at from nearby, not after the
        next leg and a drive all the way back. Its first mirador can fall next to the spot
        just left, where no look goes (fallo_radio): the second one, a moment later, is the
        change that counts (laberinto_1 with the sim realism: back to the corridor box after
        a 15 s leg away)."""
        objs = self.unreported()
        now = self.undecided(objs, planner, self.tour_failed)
        if now == self.tour_objects:
            return False
        self.tour_objects = now
        fresh = planner.tour(objs, self.n, self.seen_from, self.tour_failed, target.xy,
                             self.tour_visited)
        if not fresh or math.dist(fresh[0].xy, target.xy) <= 0.3:
            return False
        self.prefer = fresh[0].xy
        self.log_tour('retour', fresh)
        return True

    def unreported(self):
        """The objects, with the reported ones as found victims: counted, not looked at. Each
        report is one object, the nearest (planner.reported_objects); the other objects near a
        report are no victim (planner.shadowed_objects)."""
        with self.lock:
            done = list(self.reported)
        taken = reported_objects(self.objects, done, self.p['separacion'])
        found = set(taken)
        shadowed = shadowed_objects(self.objects, done, self.p['separacion']) - found
        # A report whose object is gone is still a victim found.
        out = [{'id': -1 - k, 'center': r, 'p': 1.0, 'radius': 0.15, 'miradores': []}
               for k, (r, oid) in enumerate(zip(done, taken)) if oid is None]
        for o in self.objects:
            if o['id'] in found:
                out.append(dict(o, p=1.0))
                continue
            if o['id'] in shadowed:
                o = dict(o, p=0.0)
            # How likely another look changes its P, from the looks so far (Laplace): a
            # box the camera can't settle (too small, hidden) stops pulling the robot back.
            n, useful, run = self.looks.get(o['id'], (0, 0, 0))
            out.append(dict(o, eficacia=(1 + useful) / (1 + n), miradas=n,
                            muertas=self.dead_looks.get(o['id'], []), muertas_seguidas=run))
        return out

    def obsolete(self, target):
        """The target is no longer worth going to: truthy, 'retour' when a new tour starts
        elsewhere (retour)."""
        if target.kind == 'look':
            obj = next((o for o in self.objects if o['id'] == target.object_id), None)
            pp = self.planner_params
            if obj is None or not pp['p_descartado'] < obj['p'] < pp['reporte_p_min']:
                return True
        planner = Planner(self.occupancy, self.belief, self.pose[:2], self.stuck,
                          **self.planner_params)
        if self.retour(planner, target):
            return 'retour'
        if target.kind == 'look':
            return False
        objs = self.unreported()
        return not planner.still_worth(target, expected_missing(self.n, [o['p'] for o in objs]))

    def mark_stuck(self, path):
        """Evidence that the stretch of path the robot was trying to cross can't be crossed:
        points from atasco_cerca to atasco_adelante ahead of it (not the robot's own spot,
        which would make every way out costlier too). Without a path, the robot's spot."""
        here = self.pose[:2]
        pts = list(path or [])
        event = []
        if pts:
            k = min(range(len(pts)), key=lambda i: math.dist(pts[i], here))
            run = 0.0
            for a, b in zip(pts[k:], pts[k + 1:]):
                run += math.dist(a, b)
                if run > self.p['atasco_adelante']:
                    break
                if run >= self.p['atasco_cerca']:
                    event.append(b)
        self.stuck.append(event or [here])
        self.get_logger().info(f'stuck at ({here[0]:.2f}, {here[1]:.2f})')


class Driver:
    """The motion node's interface (motion_node.py): send a path, read what it is doing with
    it. Each path gets its own stamp, and a state counts only for the path it names."""

    def __init__(self, node):
        self.node = node
        self.stamp = None

    def send(self, pts, yaws=None, slow=False):
        """slow: on /equipo/ruta_lenta, followed at the motion node's slow speed."""
        msg = path_msg(pts, yaws)
        t = self.node.get_clock().now().to_msg()
        if self.stamp is not None and (t.sec, t.nanosec) <= self.stamp:  # sim clock paused
            t.sec, t.nanosec = self.stamp[0], self.stamp[1] + 1
        msg.header.stamp = t
        self.stamp = (t.sec, t.nanosec)
        (self.node.slow_pub if slow else self.node.route_pub).publish(msg)

    def stop(self):
        self.send([])

    def state(self):
        """EstadoMovimiento's estado for the last path sent, None until the node has it."""
        m = self.node.motion_state
        if m is None or self.stamp is None or (m.ruta.sec, m.ruta.nanosec) != self.stamp:
            return None
        return m.estado

    def wait(self, timeout, blocked_s):
        """Until the robot arrives ('arrived'), stays blocked blocked_s ('blocked') or timeout
        passes ('timeout'), on the ROS clock."""
        t0 = self.node.now()
        since = None
        while rclpy.ok() and self.node.now() - t0 < timeout:
            st = self.state()
            if st == EstadoMovimiento.LLEGO:
                return 'arrived'
            if st == EstadoMovimiento.BLOQUEADO:
                since = since or self.node.now()
                if self.node.now() - since >= blocked_s:
                    return 'blocked'
            else:
                since = None
            time.sleep(0.05)
        return 'timeout'


def start_navigation(node):
    """The map, the pose, the world model's outputs and the motion node in."""
    while rclpy.ok() and (node.map_time is None or node.pose is None or node.objects is None
                          or node.occupancy is None or node.belief is None
                          or node.motion_state is None):
        time.sleep(0.1)


def turn(drv, node, angle, timeout=10.0):
    """Turn in place; False if the motion node can't (a wall too close to turn)."""
    if abs(angle) < 0.05:
        return True
    x, y, yaw = node.pose
    drv.send([(x, y)], [wrap(yaw + angle)])
    result = drv.wait(timeout, 1.0)
    if result != 'arrived':
        drv.stop()
    return result == 'arrived'


def path_msg(pts, yaws=None):
    """A nav_msgs/Path in the map frame (Driver.send stamps it). Each pose faces yaws[k] (the
    motion node turns the robot to it: where the camera looks, or the heading the chassis
    must hold), or along the path where that's None."""
    msg = Path()
    msg.header.frame_id = 'map'
    for k, (x, y) in enumerate(pts):
        a, b = pts[max(k - 1, 0)], pts[min(k + 1, len(pts) - 1)]
        yaw = math.atan2(b[1] - a[1], b[0] - a[0]) if a != b else 0.0
        if yaws is not None and yaws[k] is not None:
            yaw = yaws[k]
        p = PoseStamped()
        p.header.frame_id = 'map'
        p.pose.position.x, p.pose.position.y = x, y
        p.pose.orientation.z, p.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        msg.poses.append(p)
    return msg


def gaze(node, planner, poses, face=None):
    """Where the camera should look from each pose (x, y, planned heading or None) of a path.
    Where the robot has no room to turn, the planned heading (the one it fits at). Elsewhere
    the base is holonomic and the heading is free: face the most valuable undecided object in
    camera range with a line of sight (`face`, the object a look goes to, first), so views
    come on the way and the world model counts each new line of sight; else along the path."""
    if not node.p['mirar_en_marcha']:
        return [h for _, _, h in poses]
    pp = node.planner_params
    objs = [o for o in node.unreported()
            if pp['p_descartado'] < o['p'] < pp['reporte_p_min']]
    rng = node.p['mirada_rango']
    pts = [p[:2] for p in poses]
    out = []
    for k, pt in enumerate(pts):
        if poses[k][2] is not None:
            out.append(poses[k][2])
            continue
        a, b = pts[max(k - 1, 0)], pts[min(k + 1, len(pts) - 1)]
        along = math.atan2(b[1] - a[1], b[0] - a[0]) if a != b else node.pose[2]
        best, value = None, 0.0
        if face is not None and math.dist(pt, face) <= rng and planner.sees(pt, face, 0.2):
            best, value = face, math.inf
        for o in objs:
            d = math.dist(pt, o['center'])
            v = o['p'] * o.get('eficacia', 1.0)
            if pp['mirada_min'] * 0.5 <= d <= rng and v > value \
                    and planner.sees_object(pt, o):
                best, value = o['center'], v
        out.append(along if best is None else math.atan2(best[1] - pt[1], best[0] - pt[0]))
    return out


def follow(drv, node, xy, timeout, arrive, stop=lambda: None, face=None):
    """Drive to xy along the planner's path, re-planned every replan_s on the latest map and
    stuck spots, so the route the robot drives is the one its target was chosen on. Returns
    'arrived', 'stuck' (no progress for atasco_s: the spot is marked), 'nopath', 'timeout',
    or whatever stop() returns."""
    t0 = node.now()
    anchor = (t0, node.pose[:2])
    last_plan, path, blocked = -math.inf, None, None
    while rclpy.ok():
        t = node.now()
        if math.dist(node.pose[:2], xy) <= arrive:
            drv.stop()
            return 'arrived'
        if t - last_plan >= node.p['replan_s']:
            planner, _ = node.plan([], choose=False)
            poses = planner.path_poses(xy)
            if poses is None:
                drv.stop()
                return 'nopath'
            path = [p[:2] for p in poses]
            drv.send(path, gaze(node, planner, poses, face))
            last_plan = t
        state = drv.state()
        if state == EstadoMovimiento.LLEGO:
            return 'arrived'  # at the path's end: xy
        blocked = (blocked or t) if state == EstadoMovimiento.BLOQUEADO else None
        if math.dist(node.pose[:2], anchor[1]) > node.p['atasco_dist']:
            anchor = (t, node.pose[:2])
        elif t - anchor[0] > node.p['atasco_s'] or (
                blocked is not None and t - blocked > node.p['bloqueo_s']):
            drv.stop()
            node.mark_stuck(path)
            return 'stuck'
        if t - t0 > timeout:
            drv.stop()
            return 'timeout'
        why = stop()
        if why:
            drv.stop()
            return why
        time.sleep(0.1)
    return 'timeout'


def escape(drv, node):
    """Get out of a tight spot: move up the clearance gradient of the posterior map (where
    there's most room around); if that fails, a short move in each direction, the roomiest
    first, holding the heading (the base is holonomic). The motion node refuses whatever would
    touch something. A blind back-up once put the robot on a wall the map draws but isn't
    there, where every plan was refused."""
    planner = node.planner
    start, yaw = node.pose[:2], node.pose[2]
    moves = []
    goal = planner.escape_point(start, node.p['escape_dist']) if planner else None
    if goal is not None and math.dist(goal, start) > 0.05:
        moves.append(goal)
    around = [(start[0] + 0.2 * math.cos(a), start[1] + 0.2 * math.sin(a))
              for a in np.arange(8) * math.pi / 4]
    if planner:
        around.sort(key=lambda q: -planner.clearance_at(q))
    moves += around
    for goal in moves:
        drv.send([start, goal], [yaw, yaw])
        drv.wait(node.p['escape_s'], 1.0)
        drv.stop()
        if math.dist(node.pose[:2], start) > 0.08:
            return True
    return False


def p_of(node, object_id):
    return next((o['p'] for o in node.objects if o['id'] == object_id), None)


def p_rojo_of(node, object_id):
    return next((o['p_rojo'] for o in node.objects if o['id'] == object_id), None)


def look(drv, node, target):
    """Face the object, then slide along a sweep (planner.sweep), camera on it, on the motion
    node's slow path, until it is decided or the sweep ends (and espera_barrido more). Without
    a sweep there, wait facing it until a view counts, at most espera_mirada. Returns where
    views counted (its color evidence changed, one per line of sight), and where the robot
    started and ended; None if it couldn't turn to face it."""
    oid, (cx, cy) = target.object_id, target.face
    pp = node.planner_params
    dead = node.dead_looks.get(oid, [])
    seen = [q for q in node.seen_from.get(oid, ()) if q not in dead]
    views, last = [], [p_rojo_of(node, oid)]

    def watch():
        """True once the object is decided (or gone, or the mission over)."""
        obj = next((o for o in node.objects if o['id'] == oid), None)
        if obj is None:
            return True
        # In log-odds: near 0 or 1 a whole view moves the probability by less than 0.01.
        if last[0] is not None and abs(logit(obj['p_rojo']) - logit(last[0])) > 0.1:
            b = math.atan2(node.pose[1] - cy, node.pose[0] - cx)
            if all(abs(wrap(b - math.atan2(q[1] - cy, q[0] - cx))) >= pp['barrido_vista']
                   for q in seen + views):
                views.append(node.pose[:2])
        last[0] = obj['p_rojo']
        return not pp['p_descartado'] < obj['p'] < pp['reporte_p_min'] or node.finished()

    x, y, yaw = node.pose
    if not turn(drv, node, wrap(math.atan2(cy - y, cx - x) - yaw)):
        return None
    start = node.pose[:2]
    if watch():
        return views, start, start
    obj = next((o for o in node.objects if o['id'] == oid), None)
    pts = None
    if node.p['barrido'] and obj is not None:
        planner, _ = node.plan([], choose=False)
        pts = planner.sweep(obj, start, seen)
    t0 = node.now()
    if pts is None:
        while node.now() - t0 < node.p['espera_mirada'] and not watch() and not views:
            time.sleep(0.1)
        return views, start, start
    drv.send(pts, [math.atan2(cy - q[1], cx - q[0]) for q in pts], slow=True)
    end = blocked = None
    while rclpy.ok() and node.now() - t0 < node.p['goal_timeout'] and not watch():
        time.sleep(0.1)
        state, now = drv.state(), node.now()
        if state == EstadoMovimiento.LLEGO:
            end = end or now
            if now - end >= node.p['espera_barrido']:
                break
        blocked = (blocked or now) if state == EstadoMovimiento.BLOQUEADO else None
        if blocked is not None and now - blocked >= node.p['bloqueo_s']:
            break
    drv.stop()
    return views, start, node.pose[:2]


def go_home(drv, node, log):
    """Home by the deadline, with the same driving and escapes as any target:
    a home leg that only re-sent one failing goal sat for 7 minutes beside a box."""
    while rclpy.ok() and node.elapsed() < node.p['mission_deadline']:
        left = node.p['mission_deadline'] - node.elapsed()
        result = follow(drv, node, node.home, left, node.p['llegada_casa'])
        if result == 'arrived' or (result in ('stuck', 'nopath') and math.dist(
                node.pose[:2], node.home) <= node.p['llegada_casa_max']):
            return True
        log.info(f'on the way home: {result}')
        if result in ('stuck', 'nopath'):
            escape(drv, node)
    return False


def mission_failed(drv, node, error, errors):
    """A mission step raised: stop the robot and log it. True once errores_max steps failed
    within errores_ventana s (errors: their times), when exploring is given up for home."""
    now = node.now()
    errors[:] = [t for t in errors if now - t < node.p['errores_ventana']] + [now]
    node.get_logger().error(f'mission step failed ({len(errors)} in '
                            f'{node.p["errores_ventana"]:.0f} s): {error!r}\n'
                            + traceback.format_exc())
    drv.stop()
    if len(errors) >= node.p['errores_max']:
        node.get_logger().error('the mission keeps failing: going home')
        return True
    time.sleep(1.0)
    return False


def finish(drv, node, log):
    """The reports left, the way home, and /done once the robot has stopped. Each step on its
    own: a failure in one still ends the run with /done, which stops the judge's clock."""
    try:
        node.report_leftovers()
    except Exception as e:  # never die mid-run
        log.error(f'leftover reports failed: {e!r}\n' + traceback.format_exc())
    log.info(f'returning home, {node.elapsed():.0f} s')
    node.goal = 'home'
    errors = []
    while rclpy.ok() and node.elapsed() < node.p['mission_deadline']:
        try:
            go_home(drv, node, log)
            break
        except Exception as e:  # never die mid-run
            if mission_failed(drv, node, e, errors):
                break
    # The judge counts a motion command after /done against the run: stop, wait for the
    # motion node to confirm it has (it sends no command without a path), and let the base
    # brake, so the wheels at /done are where they stay.
    drv.stop()
    t0 = node.now()
    while rclpy.ok() and drv.state() != EstadoMovimiento.QUIETO and node.now() - t0 < 1.0:
        time.sleep(0.05)
    time.sleep(node.p['parada_done_s'])
    for _ in range(5):  # the judge counts the first; one lost message would hang the run
        node.done_pub.publish(Empty())
        time.sleep(0.2)
    log.info(f'done after {node.elapsed():.0f} s, {len(node.reported)} reports')


def main():
    rclpy.init()
    node = Mission()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=spin, args=(node, executor), daemon=True).start()
    drv = Driver(node)
    start_navigation(node)
    if not rclpy.ok():
        return
    # The robot hasn't moved yet: its pose is the start pose, localized.
    node.home = home_target(node.p['casa'], node.pose[:2])
    failed = []     # (xy, time): goals the robot did not reach
    visited = []    # (xy, time): explore stops reached, not explored again for a while
    idle = None     # when the planner first had nothing worth doing
    errors = []     # when mission steps failed
    log = node.get_logger()
    log.info(f'mission: N = {node.n}, home ({node.p["casa"]}) at ({node.home[0]:.2f}, '
             f'{node.home[1]:.2f}), robot at ({node.pose[0]:.2f}, {node.pose[1]:.2f})')

    while rclpy.ok() and not node.finished():
        try:
            now = node.now()
            failed = [(f, t) for f, t in failed if now - t < node.p['fallo_expira']]
            visited = [(f, t) for f, t in visited if now - t < node.p['fallo_expira']]
            planner, target = node.plan([f for f, _ in failed], [f for f, _ in visited])
            if target is None:
                if planner.wedged:  # where the map shows no room to move: get out first
                    log.info('no room around the robot on the map: escaping')
                    escape(drv, node)
                    continue
                idle = idle or now
                if now - idle > node.p['espera_fin']:
                    log.info('nothing left worth doing')
                    break
                time.sleep(0.5)
                continue
            idle = None
            looks = len(node.seen_from.get(target.object_id, ())) if target.kind == 'look' else 0
            log.info(f'target: {target}{" (on its mirador)" if target.precise else ""} '
                     f'(score {target.score:.3f}/s, {node.elapsed():.0f} s'
                     + (f', {looks} looks before' if target.kind == 'look' else '')
                     + (f', tour of {len(node.tour)})' if node.tour else ')'))
            node.goal = (f'{target.kind} {target.object_id} '
                         f'{target.xy[0]:.2f} {target.xy[1]:.2f}')
            arrive = node.p['llegada_precisa'] if target.precise else node.p['llegada']
            if math.dist(node.pose[:2], target.xy) > arrive:
                last = [node.now()]

                def stop():
                    if node.finished():
                        return 'finished'
                    if node.now() - last[0] > node.p['chequeo_s']:
                        last[0] = node.now()
                        why = node.obsolete(target)
                        if why:
                            return 'retour' if why == 'retour' else 'obsolete'
                    return None

                result = follow(drv, node, target.xy, node.p['goal_timeout'], arrive,
                                stop, target.face if target.kind == 'look' else None)
                if result in ('finished', 'obsolete', 'retour'):
                    log.info(f'target dropped: {result}')
                    continue
                if result != 'arrived':
                    if result == 'stuck' and not escape(drv, node):
                        log.info('could not back out')
                    log.info(f'target not reached: {result}')
                    failed.append((target.xy, node.now()))
                    continue
            if target.kind == 'look':
                oid = target.object_id
                before = p_of(node, oid)
                seen = look(drv, node, target)
                if seen is not None:
                    views, start, end = seen
                    after = p_of(node, oid)
                    stats = node.looks.setdefault(oid, [0, 0, 0])
                    stats[0] += 1
                    stats[1] += int(bool(views))
                    stats[2] = 0 if views else stats[2] + 1
                    # Without a view, the lines the sweep crossed stay open (a dead look's).
                    spots = views or ([start] if math.dist(start, end) < 0.05 else [start, end])
                    node.seen_from.setdefault(oid, []).extend(spots)
                    if not views:
                        node.dead_looks.setdefault(oid, []).extend(spots)
                    log.info(f'looked at object {oid} from ({start[0]:.2f}, {start[1]:.2f}) to '
                             f'({end[0]:.2f}, {end[1]:.2f}): P {before} -> {after}, '
                             f'{len(views)} views')
                else:
                    log.info('could not turn to the object here')
                    failed.append((target.xy, node.now()))
            else:  # seen from here: going straight back gives no new view
                visited.append((target.xy, node.now()))
        except Exception as e:  # never die mid-run: stop, log, and plan again
            if mission_failed(drv, node, e, errors):
                break

    finish(drv, node, log)
    while rclpy.ok():  # the node keeps spinning in its thread
        time.sleep(1.0)


if __name__ == '__main__':
    main()
