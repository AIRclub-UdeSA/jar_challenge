"""
Scan-to-map localization on the fixed /map grid. Pure numpy and scipy, no ROS: the
`localizer` node and the offline benchmark (herramientas/bench_localization.py) run
this same code.

- Startup and recovery: correlative search, i.e. score every pose in a window
  against a likelihood field of the walls and refine the best ones.
- Tracking: Gauss-Newton on the signed distance from each scan point to the nearest
  wall surface, with a Huber loss, an outlier gate that narrows over the iterations,
  and the odometry (or gyro) prediction as a prior. Returns the covariance too.

Poses are numpy arrays (x, y, yaw). `compose(a, b)` is a then b (a ⊕ b).
"""
from collections import deque
import math

import numpy as np
from scipy.ndimage import distance_transform_edt


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def compose(a, b):
    c, s = math.cos(a[2]), math.sin(a[2])
    return np.array([a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1],
                     wrap(a[2] + b[2])])


def inverse(a):
    c, s = math.cos(a[2]), math.sin(a[2])
    return np.array([-c * a[0] - s * a[1], s * a[0] - c * a[1], -a[2]])


def between(a, b):
    """The motion from a to b, in a's frame."""
    return compose(inverse(a), b)


def transform(pose, pts):
    c, s = math.cos(pose[2]), math.sin(pose[2])
    return pts @ np.array([[c, s], [-s, c]]) + pose[:2]


class DistanceField:
    """
    Signed distance to the nearest wall surface (m, positive in free space), sampled at
    the map's cell centers and bilinearly interpolated. A free cell next to a wall is
    half a cell from its surface, so the zero crossing sits on the cell boundary, where
    the lidar actually hits: a distance to occupied cell centers would pull every point
    half a cell (2.5 cm) into the wall.
    """

    def __init__(self, occupied, resolution, origin, truncate=0.5, likelihood_sigma=0.05):
        """occupied: (rows, cols) bool in OccupancyGrid order (row 0 at origin y)."""
        free = ~occupied
        sdf = np.where(free, distance_transform_edt(free) - 0.5,
                       0.5 - distance_transform_edt(occupied)) * resolution
        self.sdf = np.clip(sdf, -truncate, truncate)
        self.res = resolution
        self.origin = np.asarray(origin[:2], float)
        self.truncate = truncate
        # For the correlative search: nearest-cell lookup of a smooth wall likelihood.
        self.likelihood = np.exp(-0.5 * (self.sdf / likelihood_sigma) ** 2)

    def lookup(self, pts, offset=0.0):
        """
        Distance and gradient at world points (N, 2). Off the map: truncate, zero.
        `offset` moves every wall surface by that much into the wall (the map draws
        walls thicker than they are by that much).
        """
        rows, cols = self.sdf.shape
        u = (pts[:, 0] - self.origin[0]) / self.res - 0.5
        v = (pts[:, 1] - self.origin[1]) / self.res - 0.5
        j0, i0 = np.floor(u).astype(int), np.floor(v).astype(int)
        inside = (i0 >= 0) & (j0 >= 0) & (i0 < rows - 1) & (j0 < cols - 1)
        i0, j0 = np.clip(i0, 0, rows - 2), np.clip(j0, 0, cols - 2)
        fu, fv = u - j0, v - i0
        d00, d01 = self.sdf[i0, j0], self.sdf[i0, j0 + 1]
        d10, d11 = self.sdf[i0 + 1, j0], self.sdf[i0 + 1, j0 + 1]
        d = (1 - fv) * ((1 - fu) * d00 + fu * d01) + fv * ((1 - fu) * d10 + fu * d11)
        grad = np.stack([(1 - fv) * (d01 - d00) + fv * (d11 - d10),
                         (1 - fu) * (d10 - d00) + fu * (d11 - d01)], axis=1) / self.res
        d[~inside] = self.truncate
        grad[~inside] = 0.0
        return d + offset, grad

    def score(self, pts):
        """Mean wall likelihood of points (..., N, 2), nearest cell. Off the map: 0."""
        rows, cols = self.likelihood.shape
        j = np.floor((pts[..., 0] - self.origin[0]) / self.res).astype(int)
        i = np.floor((pts[..., 1] - self.origin[1]) / self.res).astype(int)
        inside = (i >= 0) & (j >= 0) & (i < rows) & (j < cols)
        val = self.likelihood[np.clip(i, 0, rows - 1), np.clip(j, 0, cols - 1)]
        return np.where(inside, val, 0.0).mean(axis=-1)


def match(field, pts, pose0, prior_info=None, gate_start=0.3, gate_min=0.15, huber=0.03,
          point_info=1.0 / 0.03 ** 2 / 10, iterations=12, min_points=30, offset=0.0):
    """
    Gauss-Newton from pose0, minimizing sum huber(d(T p_i)) * point_info plus the prior
    (T - pose0)' prior_info (T - pose0). The outlier gate starts at gate_start (wide
    enough for the prediction's error) and halves each iteration down to gate_min.
    point_info is 1 / (sigma^2 * k): k discounts neighbouring points, which are not
    independent, so the covariance is not wildly overconfident.

    Returns (pose, covariance, inliers mask over pts, iterations used), or None if too
    few points matched.
    """
    x = np.array(pose0, float)
    gate = max(gate_start, gate_min)
    for it in range(iterations):
        q = transform(x, pts)
        r, g = field.lookup(q, offset)
        keep = np.abs(r) < gate
        if keep.sum() < min_points:
            return None
        r, g, q = r[keep], g[keep], q[keep]
        # d(r)/d(x, y, yaw): the gradient, and for yaw its dot with the point's offset
        # from the robot turned 90 degrees.
        J = np.column_stack([g[:, 0], g[:, 1],
                             -g[:, 0] * (q[:, 1] - x[1]) + g[:, 1] * (q[:, 0] - x[0])])
        a = np.abs(r)
        w = np.where(a <= huber, 1.0, huber / np.maximum(a, 1e-9)) * point_info
        H = (J * w[:, None]).T @ J
        b = (J * w[:, None]).T @ r
        if prior_info is not None:
            e = x - pose0
            e[2] = wrap(e[2])
            H = H + prior_info
            b = b + prior_info @ e
        step = -np.linalg.solve(H + 1e-9 * np.eye(3), b)
        x += step
        x[2] = wrap(x[2])
        gate = max(gate * 0.5, gate_min)
        if gate == gate_min and abs(step[0]) + abs(step[1]) < 1e-4 and abs(step[2]) < 1e-4:
            break
    q = transform(x, pts)
    inliers = np.abs(field.lookup(q, offset)[0]) < gate_min
    return x, np.linalg.inv(H + 1e-9 * np.eye(3)), inliers, it + 1


def correlative_search(field, pts, center, xy_window, yaw_window, xy_step=0.05,
                       yaw_step=math.radians(2.0), max_points=180, candidates=8):
    """
    Score every pose on a grid around center (x, y within +-xy_window, yaw within
    +-yaw_window) and return the best `candidates` as (pose, score), best first, at
    least 3 grid steps apart so they are distinct hypotheses.
    """
    sub = pts[::max(1, len(pts) // max_points)]
    yaws = center[2] + np.arange(-yaw_window, yaw_window + 1e-9, yaw_step)
    if yaw_window >= math.pi - 1e-6:
        yaws = yaws[:-1]  # -pi and +pi are the same heading
    offs = np.arange(-xy_window, xy_window + 1e-9, xy_step)
    gx, gy = np.meshgrid(offs, offs)
    shifts = np.column_stack([gx.ravel(), gy.ravel()]) + center[:2]
    scores = np.empty((len(yaws), len(shifts)))
    for k in range(0, len(yaws), 16):
        yw = yaws[k:k + 16]
        c, s = np.cos(yw), np.sin(yw)
        rot = np.stack([c[:, None] * sub[:, 0] - s[:, None] * sub[:, 1],
                        s[:, None] * sub[:, 0] + c[:, None] * sub[:, 1]], axis=-1)
        world = rot[:, None, :, :] + shifts[None, :, None, :]
        scores[k:k + 16] = field.score(world)
    order = np.argsort(scores, axis=None)[::-1]
    picked = []
    for flat in order:
        iy, ixy = np.unravel_index(flat, scores.shape)
        pose = np.array([shifts[ixy, 0], shifts[ixy, 1], wrap(yaws[iy])])
        if all(np.hypot(*(pose[:2] - p[:2])) > 2.5 * xy_step
               or abs(wrap(pose[2] - p[2])) > 2.5 * yaw_step for p, _ in picked):
            picked.append((pose, scores[iy, ixy]))
            if len(picked) == candidates:
                break
    return picked


DEFAULTS = {
    'max_range': 8.0,            # m, ignore returns farther than this
    'max_points': 360,           # subsample the scan to at most this many points
    'deskew': True,              # use time_increment and odometry to undistort each scan
    'sweep_reversed': True,      # ray i of n taken at t + (n - 1 - i) * time_increment (the
                                 # robot's driver, measured); false: at t + i * time_increment
    'use_gyro': True,            # rotation between scans from the gyro, not wheel odometry
    'use_prior': True,           # prediction as a prior in the matcher
    'huber': 0.03,               # m
    'gate_min': 0.15,            # m, final outlier gate: points farther from a wall are dropped
    'gate_max': 0.5,             # m, widest starting gate
    'point_sigma': 0.03,         # m, scan point vs map noise
    'point_correlation': 10.0,   # k: how many points count as one independent measurement
    'iterations': 12,
    'odom_trans_noise': 0.10,    # sigma per m travelled (mecanum slips)
    'odom_rot_noise': 0.10,      # rad per rad turned, wheel odometry
    'gyro_rot_noise': 0.03,      # rad per rad turned, gyro
    'trans_noise_floor': 0.005,  # m per scan
    'rot_noise_floor': 0.005,    # rad per scan
    'init_x': 0.0,               # startup search center: the rules put the start near (0, 0, 0)
    'init_y': 0.0,
    'init_yaw': 0.0,
    'init_xy_window': 0.5,       # m
    'init_yaw_window': math.pi,  # rad; pi searches every heading
    'min_init_score': 0.5,       # startup pose needs this fraction of points within 5 cm of a wall
    'min_inlier_ratio': 0.45,    # tracking health: fraction of points on a wall
    'bad_scans_to_relocalize': 5,
    'reloc_xy_window': 0.5,      # m
    'reloc_yaw_window': 0.5,     # rad
    'unexplained_dist': 0.10,    # m, points this far from any wall go to the object map
    # Maps can draw walls thicker or thinner than they are (the practice maps differ by up
    # to ~2.5 cm, half a cell). Estimated online from the signed residuals.
    'wall_offset': 0.0,          # m, starting value: positive = map walls are thicker
    'estimate_wall_offset': True,
    'wall_offset_rate': 0.05,    # fraction of the residual median taken per scan
    'max_wall_offset': 0.05,     # m
}


class Localizer:
    """
    Feed it odometry (odom -> base pose), gyro rates and scans in arrival order; each
    scan returns the map -> base pose at the scan stamp and the map -> odom correction.
    """

    def __init__(self, **params):
        unknown = set(params) - set(DEFAULTS)
        if unknown:
            raise ValueError(f'unknown localizer params: {sorted(unknown)}')
        self.p = {**DEFAULTS, **params}
        self.field = None
        self.mount = (np.eye(2), np.zeros(2))  # laser -> base: p_base = A p_laser + b
        self.odom = deque(maxlen=400)          # (t, x, y, yaw)
        self.gyro = deque(maxlen=400)          # (t, wz)
        self.pose = None                       # map -> base at the last scan
        self.cov = None
        self.last_odom = None                  # odom -> base at the last scan
        self.last_t = None
        self.map_to_odom = None
        self.home = None
        self.bad_scans = 0
        self.wall_offset = self.p['wall_offset']

    def set_map(self, occupied, resolution, origin):
        self.field = DistanceField(occupied, resolution, origin)

    def set_laser_mount(self, A, b):
        self.mount = (np.asarray(A, float), np.asarray(b, float))

    def add_odom(self, t, pose):
        if self.odom and t <= self.odom[-1][0]:
            return
        self.odom.append((t, pose[0], pose[1], pose[2]))

    def add_gyro(self, t, wz):
        if self.gyro and t <= self.gyro[-1][0]:
            return
        self.gyro.append((t, wz))

    def odom_at(self, t, slack=0.1):
        """Odometry pose interpolated at t (or extrapolated at most `slack` s), or None."""
        if not self.odom or t < self.odom[0][0] or t > self.odom[-1][0] + slack:
            return None
        o = np.array(self.odom)
        if t >= o[-1, 0] and len(o) > 1:
            o = o[-2:]
        yaw = np.unwrap(o[:, 3])
        return np.array([_lerp(t, o[:, 0], col) for col in (o[:, 1], o[:, 2], yaw)])

    def gyro_rotation(self, t0, t1):
        """Integral of the gyro rate over [t0, t1], or None if the samples do not cover it."""
        if len(self.gyro) < 2 or self.gyro[0][0] > t0 or self.gyro[-1][0] < t1 - 0.15:
            return None
        g = np.array(self.gyro)
        ts = np.concatenate([[t0], g[(g[:, 0] > t0) & (g[:, 0] < t1), 0], [t1]])
        wz = np.interp(ts, g[:, 0], g[:, 1])
        return float(np.sum(0.5 * (wz[1:] + wz[:-1]) * np.diff(ts)))

    def scan_points(self, t, ranges, angle_min, angle_increment, range_min, range_max,
                    time_increment=0.0, limit=None):
        """Valid returns as (N, 2) points in the base frame at time t (deskewed), subsampled
        to `limit` points (default max_points; 0 keeps them all)."""
        r = np.asarray(ranges, float)
        idx = np.nonzero(np.isfinite(r) & (r > range_min)
                         & (r < min(range_max, self.p['max_range'])))[0]
        limit = self.p['max_points'] if limit is None else limit
        if limit and len(idx) > limit:
            idx = idx[np.linspace(0, len(idx) - 1, limit).astype(int)]
        a = angle_min + idx * angle_increment
        A, b = self.mount
        pts = np.column_stack([r[idx] * np.cos(a), r[idx] * np.sin(a)]) @ A.T + b
        if self.p['deskew'] and time_increment and len(self.odom) > 1:
            # Each ray's time over the sweep (sweep_reversed: the robot's driver lists the rays
            # against it, measured on a real bag spinning in place). Express each in the base
            # frame at t through the odometry motion over the sweep.
            ref = self.odom_at(t)
            if ref is not None:
                o = np.array(self.odom)
                k = len(r) - 1 - idx if self.p['sweep_reversed'] else idx
                ti = t + k * time_increment
                ti = np.clip(ti, o[0, 0], o[-1, 0])
                yaw = np.unwrap(o[:, 3])
                ox, oy, oyaw = (np.interp(ti, o[:, 0], col) for col in (o[:, 1], o[:, 2], yaw))
                # delta = inverse(ref) composed with odom(ti), per ray
                c, s = math.cos(ref[2]), math.sin(ref[2])
                dx, dy = ox - ref[0], oy - ref[1]
                ddx, ddy, dyaw = c * dx + s * dy, -s * dx + c * dy, oyaw - ref[2]
                cd, sd = np.cos(dyaw), np.sin(dyaw)
                pts = np.column_stack([ddx + cd * pts[:, 0] - sd * pts[:, 1],
                                       ddy + sd * pts[:, 0] + cd * pts[:, 1]])
        return pts

    def pose_at(self, t):
        """
        map -> base at time t near the last scan: the scan pose moved by odometry, with the
        rotation from the gyro when it covers the interval (wheel yaw spikes when the wheels
        slip). None if not localized or odometry doesn't cover t.
        """
        if self.pose is None:
            return None
        o = self.odom_at(t)
        if o is None:
            return None
        delta = between(self.last_odom, o)
        if self.p['use_gyro']:
            rot = (self.gyro_rotation(self.last_t, t) if t >= self.last_t
                   else self.gyro_rotation(t, self.last_t))
            if rot is not None:
                delta[2] = rot if t >= self.last_t else -rot
        return compose(self.pose, delta)

    def initialize(self, pts, center, xy_window, yaw_window):
        """
        Correlative search around center, then refine the best candidates and keep the
        one with the lowest mean residual. Returns (match result, fraction of points
        within 5 cm of a wall), or None. The coarse score cannot rank them: it is
        quantized to the 5 cm cells, so a pose a cell off ties with the true one.
        """
        best = None
        for cand, _ in correlative_search(self.field, pts, center, xy_window, yaw_window):
            res = match(self.field, pts, cand, gate_start=0.2, gate_min=self.p['gate_min'],
                        huber=self.p['huber'], point_info=self._point_info(),
                        iterations=self.p['iterations'], offset=self.wall_offset)
            if res is None:
                continue
            d = np.abs(self.field.lookup(transform(res[0], pts), self.wall_offset)[0])
            cost = np.minimum(d, self.p['gate_min']).mean()
            if best is None or cost < best[2]:
                best = (res, float((d < 0.05).mean()), cost)
        return None if best is None else best[:2]

    def _point_info(self):
        return 1.0 / (self.p['point_sigma'] ** 2 * self.p['point_correlation'])

    def process_scan(self, t, ranges, angle_min, angle_increment, range_min, range_max,
                     time_increment=0.0):
        """Returns a dict with pose, cov, map_to_odom, health; or None if not localized."""
        if self.field is None:
            return None
        odom_t = self.odom_at(t)
        if odom_t is None:
            return None
        pts = self.scan_points(t, ranges, angle_min, angle_increment, range_min, range_max,
                               time_increment)
        if len(pts) < 30:
            return None
        p = self.p
        event = 'track'
        if self.pose is None:
            center = np.array([p['init_x'], p['init_y'], p['init_yaw']])
            best = self.initialize(pts, center, p['init_xy_window'], p['init_yaw_window'])
            if best is None or best[1] < p['min_init_score']:
                return None
            (pose, cov, inliers, iters), _ = best
            self.home = pose.copy()
            event = 'init'
        else:
            delta = between(self.last_odom, odom_t)
            dist, turn = math.hypot(delta[0], delta[1]), abs(delta[2])
            rot_noise = p['odom_rot_noise']
            if p['use_gyro']:
                rot = self.gyro_rotation(self.last_t, t)
                if rot is not None:
                    delta[2], turn, rot_noise = rot, abs(rot), p['gyro_rot_noise']
            pred = compose(self.pose, delta)
            # Covariance: previous one carried through the motion, plus motion noise.
            c, s = math.cos(self.pose[2]), math.sin(self.pose[2])
            F = np.array([[1, 0, -s * delta[0] - c * delta[1]],
                          [0, 1, c * delta[0] - s * delta[1]], [0, 0, 1]])
            st = p['odom_trans_noise'] * dist + 0.2 * p['odom_trans_noise'] * turn \
                + p['trans_noise_floor']
            sr = rot_noise * turn + 0.05 * dist + p['rot_noise_floor']
            P = F @ self.cov @ F.T + np.diag([st ** 2, st ** 2, sr ** 2])
            gate = min(p['gate_max'], max(p['gate_min'],
                                          3 * math.sqrt(max(P[0, 0], P[1, 1])) + 0.05))
            res = match(self.field, pts, pred,
                        prior_info=np.linalg.inv(P) if p['use_prior'] else None,
                        gate_start=gate, gate_min=p['gate_min'], huber=p['huber'],
                        point_info=self._point_info(), iterations=p['iterations'],
                        offset=self.wall_offset)
            if res is None:
                pose, cov, inliers, iters = pred, P, np.zeros(len(pts), bool), 0
            else:
                pose, cov, inliers, iters = res
            ratio = inliers.mean()
            if ratio < p['min_inlier_ratio']:
                # A poor fit (lost, or many unmapped objects in view) is no information:
                # keep the uncertainty growing, so the gate widens and the prior relaxes.
                cov = P
                self.bad_scans += 1
            else:
                self.bad_scans = 0
            if self.bad_scans >= p['bad_scans_to_relocalize']:
                # Search again, but only take a pose that fits as well as a startup pose
                # does: a weak candidate far away is worse than a drifting estimate.
                best = self.initialize(pts, pose, p['reloc_xy_window'], p['reloc_yaw_window'])
                if best is not None and best[1] >= p['min_init_score']:
                    (pose, cov, inliers, iters), _ = best
                    event = 'relocalize'
                self.bad_scans = 0
        self.pose, self.cov, self.last_odom, self.last_t = pose, cov, odom_t, t
        self.map_to_odom = compose(pose, inverse(odom_t))
        world = transform(pose, pts)
        d = self.field.lookup(world, self.wall_offset)[0]
        near = np.abs(d) < 0.05
        if p['estimate_wall_offset'] and near.sum() > 50:
            # Points left inside the walls (d < 0) mean the map's walls are thicker.
            self.wall_offset = float(np.clip(
                self.wall_offset - p['wall_offset_rate'] * np.median(d[near]),
                -p['max_wall_offset'], p['max_wall_offset']))
        return {'t': t, 'pose': pose, 'cov': cov, 'map_to_odom': self.map_to_odom,
                'inlier_ratio': float(inliers.mean()), 'iterations': iters,
                'points': len(pts), 'event': event, 'wall_offset': self.wall_offset,
                'unexplained': world[d > p['unexplained_dist']]}


def _lerp(t, ts, vals):
    if t >= ts[-1] and len(ts) > 1:  # extrapolate from the last two samples
        return vals[-1] + (vals[-1] - vals[-2]) * (t - ts[-1]) / (ts[-1] - ts[-2])
    return float(np.interp(t, ts, vals))


def occupancy_from_grid(data, width, height, occupied_threshold=65, unknown_is_wall=True):
    """OccupancyGrid data (row-major, row 0 at the origin) to a bool wall mask."""
    g = np.asarray(data, np.int16).reshape(height, width)
    return (g >= occupied_threshold) | ((g < 0) if unknown_is_wall else False)
