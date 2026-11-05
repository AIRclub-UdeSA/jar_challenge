"""
Following a path on the holonomic base (docs/MISSION.md, "Driving"). Plain Python, no ROS:
the `motion` node (motion_node.py) calls it, and the tests run it on synthetic scans.

The path comes from the mission planner, planned over position and heading for the chassis'
real shape (planner.py), so following it is all that's left to do here:

- translate toward a point `mirar_adelante` ahead on the path, at most `v_max`, slowing so
  the robot can stop at the end (`a_max`);
- turn toward the heading of the path's pose nearest the robot (the one the robot must hold
  in a tight spot, or where the camera should look), at most `w_max`; with a large heading
  error, translate slower until it's fixed;
- before every command, check the chassis grown by `margen_seguridad` against the latest
  scan along the way the robot would go if sent the command now and stopped `reaccion_s` later
  (the scan and network latency): from the velocity it has (measured, not the last command:
  the base takes time to follow), toward the command and then to rest at the base's own
  acceleration (`a_max`, `alfa_max`). A command that would touch something is replaced by
  the safe one that makes the most progress the way the path wants: slower, turned up to 60
  degrees away (along a wall), or without its turn (the heading waits for room); or dropped. So the speed follows the room around the robot,
  v ~ sqrt(2 a clearance), without a rule for it. A robot already inside the margin may still
  move if no point gets deeper into it than it is, or than the deepest is less the scan's
  noise, except along an edge at a depth it already has.

The base's acceleration is what limits the stop, not the command: the simulated base turns up
or down at about 1 rad/s^2 and translates at about 1 m/s^2. Stopping a 1.1 rad/s turn took it
0.6 rad; assumed to take 0.1 rad (6 rad/s^2, and from the last command instead of the turn
the base still had), the chassis' tail swept into walls on every tight turn.
"""
import math

import numpy as np

DEFAULTS = {
    'v_max': 0.35,              # m/s
    'a_max': 0.6,               # m/s^2, the base's acceleration and braking (sim: ~1.0)
    'alfa_max': 0.8,            # rad/s^2, the same for turning (sim: ~1.0)
    'w_max': 1.0,               # rad/s: the robot clamps yaw to 1.0
    'k_rumbo': 2.5,             # 1/s, heading error to turn rate
    'mirar_adelante': 0.25,     # m, lookahead on the path
    'llegada_xy': 0.04,         # m, at the last pose
    'llegada_rumbo': 0.08,      # rad
    'llegada_giro': 0.15,       # rad/s: arrived only turning slower than this (else it coasts on)
    'rumbo_lento': 0.7,         # rad of heading error from which translation slows down
    'margen_seguridad': 0.015,  # m around the chassis the scan must stay out of
    'ruido_scan': 0.01,         # m, how much deeper than it is the deepest scan point can look:
                                # the deepest of many ranges with noise (sim: sigma 4 mm)
    'v_min': 0.02,              # m/s; a safe command slower than this, when the path wants
                                # more, means the way is blocked
    'reaccion_s': 0.3,          # s from a scan to its command acting (scan period, latency)
    'paso_prediccion': 0.04,    # s, time step of the predicted stop
    'desvios': (0.0, 0.35, 0.7, 1.05),  # rad, turns of the translation tried when the command
                                        # isn't safe (either way)
    'huella_x': (-0.181, 0.119),
    'huella_y': 0.0997,
}


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def edges(points, poses, xs, half_y, margin):
    """How far each point (N, 2, robot frame now) is inside each edge of the chassis grown by
    margin (rear, front, side), with the robot at each of poses (K, 3: dx, dy, dyaw relative to
    now): (K, N, 3), m; negative outside that edge."""
    dx, dy, dth = (poses[:, k:k + 1] for k in range(3))
    c, s = np.cos(dth), np.sin(dth)
    px, py = points[None, :, 0] - dx, points[None, :, 1] - dy
    u, v = c * px + s * py, -s * px + c * py
    return np.stack([u - (xs[0] - margin), (xs[1] + margin) - u, (half_y + margin) - np.abs(v)],
                    axis=-1)


def depth(points, poses, xs, half_y, margin):
    """How deep each point is inside the chassis grown by margin (edges): (K, N), m; negative
    outside."""
    return edges(points, poses, xs, half_y, margin).min(axis=-1)


def inside(points, pose, xs, half_y, margin):
    """How many points fall inside the chassis grown by margin at pose (see depth)."""
    if not len(points):
        return 0
    return int((depth(points, np.array([pose], float), xs, half_y, margin) >= 0).sum())


def stopping_poses(v0, u, p):
    """The poses (K, 3: dx, dy, dyaw relative to now) the robot goes through if sent u = (vx,
    vy, w) now and stopped reaccion_s later: its velocity, v0 now (robot frame), goes toward u
    and then to rest at the base's acceleration (a_max for the translation, alfa_max for the
    turn)."""
    dt, a, alfa = p['paso_prediccion'], p['a_max'] * p['paso_prediccion'], \
        p['alfa_max'] * p['paso_prediccion']
    vx, vy, w = (float(c) for c in v0)
    x = y = th = t = 0.0
    out = [(0.0, 0.0, 0.0)]
    while True:
        tx, ty, tw = u if t < p['reaccion_s'] else (0.0, 0.0, 0.0)
        ex, ey = tx - vx, ty - vy
        e = math.hypot(ex, ey)
        k = 1.0 if e <= a else a / e
        vx, vy = vx + k * ex, vy + k * ey
        w += max(-alfa, min(alfa, tw - w))
        mid = th + 0.5 * w * dt
        x += (math.cos(mid) * vx - math.sin(mid) * vy) * dt
        y += (math.sin(mid) * vx + math.cos(mid) * vy) * dt
        th += w * dt
        t += dt
        out.append((x, y, th))
        if t >= p['reaccion_s'] and vx == vy == w == 0.0:
            return np.array(out)


def safe(points, v0, u, p):
    """Whether sending u now keeps the chassis (grown by margen_seguridad) off the scan points
    until the robot could stop (stopping_poses). Nothing inside when nothing is. Inside, no
    point gets deeper than it is (each compared with itself: the change is the motion's, not
    the noise's) or than the deepest one less ruido_scan, unless it slides in along an edge:
    one of its distances to the edges doesn't grow and is no deeper than the deepest point
    (driving along a wall inside the margin). Allowed up to the deepest point as measured, a
    robot wedged inside its margin crept toward one side and then the other on the noise of
    each scan, 1.3 cm in 7 s without moving along its path."""
    if not len(points):
        return True
    poses = stopping_poses(v0, u, p)
    # Only what the chassis can reach on the way matters.
    reach = np.hypot(poses[:, 0], poses[:, 1]).max() + math.hypot(
        max(-p['huella_x'][0], p['huella_x'][1]), p['huella_y']) + p['margen_seguridad']
    near = points[np.hypot(points[:, 0], points[:, 1]) <= reach]
    if not len(near):
        return True
    e = edges(near, poses, p['huella_x'], p['huella_y'], p['margen_seguridad'])
    d = e.min(axis=-1)
    deepest = max(d[0].max(), 0.0)
    ok = d[1:] <= np.maximum(d[0], max(deepest - p['ruido_scan'], 0.0)) + 1e-9
    along = ((e[1:] <= e[0] + 1e-9) & (e[0] <= deepest + 1e-9)).any(axis=-1)
    return bool((ok | along).all())


class Tracker:
    """Follows one path: (x, y, yaw) poses in the map frame."""

    def __init__(self, path, **params):
        self.p = dict(DEFAULTS, **params)
        self.path = [tuple(q) for q in path]
        xy = np.array([q[:2] for q in self.path], float).reshape(-1, 2)
        self.xy = xy
        seg = np.hypot(*np.diff(xy, axis=0).T) if len(xy) > 1 else np.zeros(0)
        self.s = np.concatenate(([0.0], np.cumsum(seg)))  # arc length at each pose
        self.k = 0             # progress: index of the pose nearest the robot
        self.v = np.zeros(3)   # last command (vx, vy, w), robot frame

    def candidates(self, cmd, want):
        """The commands to try, best first: cmd, slower, with its translation turned away from
        the way it points (desvios), and without its turn, ordered by the progress they make
        the way the path wants (want: the translation toward the lookahead point), the turn
        kept on ties. Next to a wall, the way to the path can be blocked by a millimetre while
        driving along the wall is free, or turning to the path's heading can swing a corner
        into it while driving on can't. None that makes less progress than v_min, when there
        is translation to make: creeping is not moving."""
        p = self.p
        wanted = math.hypot(cmd[0], cmd[1])
        if wanted <= p['v_min']:
            return [tuple(cmd * k) for k in (1.0, 0.5, 0.25, 0.12)]
        n = math.hypot(want[0], want[1])
        ux, uy = (want[0] / n, want[1] / n) if n > 1e-6 else (cmd[0] / wanted, cmd[1] / wanted)
        out = []
        for off in p['desvios']:
            for sign in ((1, -1) if off else (1,)):
                c, s = math.cos(sign * off), math.sin(sign * off)
                tx, ty = c * cmd[0] - s * cmd[1], s * cmd[0] + c * cmd[1]
                for k in (1.0, 0.5, 0.25):
                    progress = k * (tx * ux + ty * uy)
                    if progress >= p['v_min']:
                        for turn in (1.0, 0.0):
                            out.append((-progress, -turn, (k * tx, k * ty, k * turn * cmd[2])))
        out.sort(key=lambda c: c[:2])
        return [c[2] for c in out]

    def remaining(self):
        return float(self.s[-1] - self.s[self.k]) if len(self.s) else 0.0

    def step(self, pose, points, dt, velocity=None):
        """The command (vx, vy, w) in the robot frame and the state: 'llego', 'siguiendo' or
        'bloqueado'. pose: (x, y, yaw) in the map frame; points: the latest scan, (N, 2) in
        the robot frame; dt: s since the last step; velocity: the robot's (vx, vy, w) in its
        frame, measured (None: assume it follows the commands)."""
        p = self.p
        v0 = self.v.copy() if velocity is None else np.asarray(velocity, float)
        x, y, yaw = pose
        if not len(self.path):
            return (0.0, 0.0, 0.0), 'llego'
        # Progress along the path: the nearest pose from the last one on, a short way ahead.
        ahead = np.flatnonzero(self.s <= self.s[self.k] + 1.0)
        ahead = ahead[ahead >= self.k]
        d = np.hypot(self.xy[ahead, 0] - x, self.xy[ahead, 1] - y)
        self.k = int(ahead[np.argmin(d)])
        gx, gy, gyaw = self.path[-1]
        to_end = math.hypot(gx - x, gy - y)
        yaw_ref = self.path[min(self.k + 1, len(self.path) - 1)][2]
        if yaw_ref is None or not math.isfinite(yaw_ref):
            yaw_ref = yaw
        err = wrap(yaw_ref - yaw)
        last = self.k >= len(self.path) - 1 or self.remaining() < p['llegada_xy']
        # Arrived also means not still turning: at 'llego' the commands stop, and the base
        # coasts on at its own deceleration (from 1.3 rad/s, 0.8 rad past the heading).
        if last and to_end <= p['llegada_xy'] and abs(err) <= p['llegada_rumbo'] \
                and abs(v0[2]) <= p['llegada_giro']:
            self.v[:] = 0.0
            return (0.0, 0.0, 0.0), 'llego'
        # Where to translate: a point mirar_adelante further along the path (or the end).
        target_s = self.s[self.k] + p['mirar_adelante']
        j = int(np.searchsorted(self.s, target_s))
        tx, ty = (gx, gy) if j >= len(self.path) else self.xy[j]
        dist = math.hypot(tx - x, ty - y)
        speed = min(p['v_max'], math.sqrt(2 * p['a_max'] * to_end))
        speed *= max(0.0, 1.0 - max(0.0, abs(err) - p['rumbo_lento']) / p['rumbo_lento'])
        if to_end <= p['llegada_xy']:
            speed = 0.0
        heading = math.atan2(ty - y, tx - x) if dist > 1e-6 else yaw
        vx = speed * math.cos(heading - yaw)
        vy = speed * math.sin(heading - yaw)
        # Turn no faster than braking at alfa_max can stop at the heading, as speed does for
        # the end of the path.
        w = math.copysign(min(p['w_max'], p['k_rumbo'] * abs(err),
                              math.sqrt(2 * p['alfa_max'] * abs(err))), err)
        # Acceleration limits. The translation's change is limited as a vector: clipped per
        # axis, a command from rest toward a point ahead and a little to the side came out
        # pointing far to the side.
        dt = max(dt, 1e-3)
        dv = np.array([vx, vy]) - self.v[:2]
        n = math.hypot(dv[0], dv[1])
        if n > p['a_max'] * dt:
            dv *= p['a_max'] * dt / n
        dw = max(-p['alfa_max'] * dt, min(p['alfa_max'] * dt, w - self.v[2]))
        cmd = self.v + np.array([dv[0], dv[1], dw])
        for v in self.candidates(cmd, (vx, vy)):
            if safe(points, v0, v, p):
                self.v = np.array(v)
                return tuple(float(c) for c in v), 'siguiendo'
        # No translation is safe: turning in place, if that is (and the turn is wanted).
        self.v[:] = 0.0
        if abs(err) > p['llegada_rumbo'] and safe(points, v0, (0.0, 0.0, cmd[2] * 0.5), p):
            self.v[2] = cmd[2] * 0.5
        return (0.0, 0.0, float(self.v[2])), 'bloqueado'
