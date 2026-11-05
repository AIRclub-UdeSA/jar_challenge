"""
World model (docs/WORLD_MODEL.md): what is in the arena that the map doesn't show, which of
it is red, and where an undetected box could still be. Pure numpy and scipy, no ROS: the
`world_model` node runs it, and herramientas/bench_world_model.py runs the same code on
recorded bags.

Evidence, updated with every message:
- occupancy log-odds on the map's grid, with the map as the prior (add_scan);
- surface points: hits away from the mapped walls, each with the sensor position it came from;
- color log-odds per cell, from pixel rays walked through the posterior map (add_image), with
  the red put at its measured depth when a depth cloud comes with the image (search belief,
  seeding objects);
- color log-odds per object, one term per line of sight from the pixels that land on its fitted
  box (_object_color): what P(victim) uses;
- the search belief over the center of an undetected box, cut by every look that would have
  seen one.

Inference (update, about 2 Hz), recomputed from the evidence each time: objects from the change
map, a box fitted to each object's hits and to the empty space around it, P(red) and P(victim).

Cells are the map's, extended by `margen_mapa` on every side. Flat indices are row * width +
col, rows along y as in OccupancyGrid.
"""
from dataclasses import dataclass, field
import math

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, label, maximum_filter
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull


DEFAULTS = {
    # Occupancy: the map as prior, the lidar's inverse sensor model.
    'mapa_logodds': 1.5,        # trust in the map: prior log-odds of its walls and free space
    'golpe_logodds': 0.85,      # a scan endpoint in the cell
    'libre_logodds': 0.4,       # a ray through the cell
    'libre_margen': 0.10,       # m, rays clear space up to this short of their endpoint
    'logodds_max': 6.0,         # clamp, so the map can still change
    'ocupado_logodds': 2.0,     # posterior occupied
    'libre_umbral': -1.5,       # posterior free (a mapped wall the rays cross is removed)
    'rango_max': 8.0,           # m, lidar returns used
    'libre_cada_rayo': 2,       # clear empty space along every n-th ray (hits use them all)
    'puntos_por_scan': 64,      # hits per scan kept for the box fit
    'margen_mapa': 1.0,         # m, the grid extends past the map by this
    # Change map: new occupancy must be this far from mapped walls.
    'sigma_pose_min': 0.04,     # m, floor for the localizer's (overconfident) sigma
    'sigma_mapa': 0.03,         # m, the map's own error
    'cambio_k': 2.0,
    'fusion': 0.10,             # m, object fragments this close are one object
    # Search belief over the center of an undetected box.
    'busqueda_huella': 0.10,    # m, smallest box to find, minus pose error
    'busqueda_eps_lidar': 0.3,  # chance one scan misses a box its rays pass through
    'busqueda_eps_camara': 0.5,  # chance one camera view misses a box it shows
    'busqueda_rojo': 2.0,       # belief factor along red rays into space nobody has seen
    # Box fit.
    'caja_lado_min': 0.10,      # m, side bounds
    'caja_lado_max': 0.50,
    'caja_lado_prior': 0.20,    # m, weak prior on the sides
    'caja_lado_sigma': 0.05,
    'ajuste_puntos_max': 800,   # hits used per fit
    'ajuste_cada': 6,           # refit at least every this many updates
    'punto_correlacion': 10.0,  # neighbouring hits are not independent: inflate the covariance
    # Color.
    'rojo_h_bajo': 10.0,        # red hue bands, OpenCV scale 0-180: h <= bajo or h >= alto
    'rojo_h_alto': 160.0,
    'rojo_s_min': 110.0,        # 0-255
    'rojo_croma_min': 15.0,     # 0-255, max - min of RGB: below it hue is camera noise. Not a
                                # brightness floor: a red face in shadow (64, 10, 10) is red
    'rojo_p_si_rojo': 0.8,      # P(red pixel | red surface)
    'rojo_p_si_otro': 0.05,     # P(red pixel | any other surface)
    'vista_logodds_max': 2.5,   # evidence one view adds to a cell: two lines of sight confirm
    'vista_logodds_neg': 4.0,   # at most this much against red from one view of an object.
                                # Measured on 7,238 views of the 2026-10-20 benchmark (ground-
                                # truth association): a view under the 0.36 red share is 1.6%
                                # of victim views and 99.5% of other views, log 0.016 = -4.1.
                                # (+2.5 for red stays below the measured +5.2: a report then
                                # needs two lines of sight, and a false one costs more)
    'previa_min': 0.10,         # P(victim) before any color: N / (objects in the arena, the
                                # unseen part's extrapolated from the seen part's),
    'previa_max': 0.50,         # within these bounds (the base rate there was 0.16-0.36)
    'vista_paso': 0.10,         # m, camera motion before a new frame is processed
    'vista_giro': 0.17,         # rad, or this much turn
    'vista_angulo': 0.17,       # rad, a cell's color counts again only from a line of sight
                                # this different: turning in place doesn't decorrelate errors
    'profundidad_tol': 0.10,    # m, red this much past the first occupied cell is behind it
    'rojo_tras_pared': 0.8,     # m, without depth: red past a mapped wall is a box's this
                                # close behind it (_object_color), and a red-only object with
                                # a box this close behind it along the camera's line is that
                                # box's red (update)
    'rojo_borde': 0.25,         # m from the arena's edge: a mapped wall's cells this close never
                                # seed a red-only object. Nothing real is behind an outer wall:
                                # its red is outside, seen through a window. 0.0: they do
    'camara_rango_min': 0.3,    # m
    'camara_rango_max': 4.0,    # m
    'camara_paso_px': 2,        # use every n-th pixel row and column
    'altura_muestra': 0.18,     # m, color samples only below the lidar plane (rules: victims reach it)
    'rojo_pixeles_min': 2,      # red pixels in a column for a red ray into unseen space
    # Object color (_object_color): each view of an object's fitted box gives one bounded term.
    'color_rojo_si_caja': 0.85,  # share of a red box's informative pixels that look red
    'color_rojo_si_otro': 0.03,  # share of another object's (noise, reflections)
    'color_muestras_max': 10.0,  # neighbouring pixels are not independent: at most this many
    'color_pixeles_por_muestra': 4.0,  # informative pixels per independent sample
    'color_pixeles_min': 8,     # fewer informative pixels on an object: no evidence
    'color_margen_k': 1.5,      # shrink the box by this many sigmas of where it lands in the
                                # image (pose, fit, camera heading): red seen through a gap
                                # beside an object falls on its edges, which this removes
    'camara_sigma_yaw': 0.02,   # rad, camera heading error
    'camara_sigma_rel': 0.02,   # m, camera vs object position: both come from the same
                                # localizer, so most of the pose error is common to them
    # Decisions.
    'forma_peso': 1.0,          # log-odds weight of the box-likeness in P(victim)
    'reporte_p_min': 0.97,      # P(victim) of a confirmed box
    'objetivos_n': 0,           # N; 0 = unknown, no global constraint
}

LOGB_MIN, LOGB_MAX = -30.0, math.log(4.0)


@dataclass
class Objeto:
    id: int
    cells: np.ndarray                          # flat cell indices
    centro: np.ndarray                         # (x, y), map frame
    cov: np.ndarray                            # 2x2, m^2, includes the pose error
    lados: tuple = (0.0, 0.0)                  # m, fitted sides
    yaw: float = 0.0
    forma: float = 0.5                         # 0..1, how well a box of the allowed sizes fits
    p_rojo: float = 0.5
    p_objetivo: float = 0.5
    scans: int = 0                             # distinct scans that hit it
    miradores: list = field(default_factory=list)  # sensor (x, y) it was seen from
    x: np.ndarray = None                       # fit parameters (cx, cy, yaw, a/2, b/2)
    fit_n: int = 0                             # hits at the last fit
    fit_age: int = 0                           # updates since the last fit
    lc: float = 0.0                            # color log-odds, from the views below
    vistas: list = field(default_factory=list)  # [bearing from the object, evidence, pixels]
    degenerado: bool = False                   # the last fit's covariance hit its prior
    l_color: float = 0.0                       # color log-odds used for P(victim)
    dentro: bool = False                       # center inside the arena
    cov_fit: np.ndarray = None                 # 2x2, the fit's own center covariance

    @property
    def sigma(self):
        return math.sqrt(max(np.linalg.eigvalsh(self.cov).max(), 0.0))


class WorldModel:

    def __init__(self, data, width, height, resolution, origin, **params):
        unknown = set(params) - set(DEFAULTS)
        if unknown:
            raise ValueError(f'unknown world model params: {sorted(unknown)}')
        p = self.p = {**DEFAULTS, **params}
        g = np.asarray(data, np.int16).reshape(height, width)
        n = self.pad = int(round(p['margen_mapa'] / resolution))
        self.res = float(resolution)
        self.map_origin = np.array(origin[:2], float)
        self.origin = self.map_origin - n * self.res
        self.prior_occ = np.pad(g >= 50, n)
        self.in_map = np.pad(np.ones(g.shape, bool), n)
        known = np.pad(g >= 0, n)
        self.shape = self.prior_occ.shape
        self.L = np.where(self.prior_occ, p['mapa_logodds'],
                          np.where(known, -p['mapa_logodds'], 0.0)).ravel()
        self.Lc = np.zeros(self.L.size)            # color log-odds: red surface or not
        self.view_xy = np.full((self.L.size, 2), np.nan)  # camera at each cell's last color view
        self.seen = np.zeros(self.L.size, np.int32)   # scans that observed the cell
        self.hits = np.zeros(self.L.size, np.int32)   # scans with an endpoint in the cell
        self.logb = np.zeros(self.L.size)          # search belief, log, relative to the prior
        self.arena = arena_mask(self.prior_occ)
        self.prior = (self.arena & ~self.prior_occ).ravel()
        self.edge_dist = distance_transform_edt(self.arena) * self.res
        self.wall_dist = distance_transform_edt(~self.prior_occ) * self.res
        self.k = max(1, int(round(p['busqueda_huella'] / self.res)))
        jj, ii = np.meshgrid(np.arange(self.shape[1]), np.arange(self.shape[0]))
        self.cx = (self.origin[0] + (jj + 0.5) * self.res).ravel()
        self.cy = (self.origin[1] + (ii + 0.5) * self.res).ravel()
        self.store = {}                            # cell -> [count, (cap, 4) x y sx sy]
        self.store_cap = 12
        self.rng = np.random.default_rng(0)
        self.objects = []
        self.owner = np.zeros(self.L.size, np.int32)   # _owners, at the last update
        self.next_id = 1
        self.sigma_pose = p['sigma_pose_min']
        self.last_view = None
        self.complete = False

    # ------------------------------------------------------------------ evidence

    def index(self, x, y):
        """Flat cell index of map points, and which ones fall on the grid."""
        j = np.floor((np.asarray(x) - self.origin[0]) / self.res).astype(int)
        i = np.floor((np.asarray(y) - self.origin[1]) / self.res).astype(int)
        ok = (i >= 0) & (j >= 0) & (i < self.shape[0]) & (j < self.shape[1])
        return np.where(ok, i * self.shape[1] + j, 0), ok

    def change_radius(self):
        p = self.p
        return p['cambio_k'] * math.hypot(max(self.sigma_pose, p['sigma_pose_min']),
                                          p['sigma_mapa'])

    def add_scan(self, sensor, pts, sigma_pose=None):
        """One scan: endpoints `pts` (N, 2) seen from `sensor` (x, y), both in the map frame,
        valid returns only. `sigma_pose`: the localizer's position sigma at the scan."""
        p = self.p
        if sigma_pose is not None:
            self.sigma_pose = float(sigma_pose)
        sensor = np.asarray(sensor, float)
        pts = np.asarray(pts, float).reshape(-1, 2)
        d = pts - sensor
        rng = np.hypot(d[:, 0], d[:, 1])
        keep = (rng > 0.05) & (rng < p['rango_max'])
        pts, d, rng = pts[keep], d[keep], rng[keep]
        if not len(pts):
            return
        step = self.res / 2
        every = max(1, int(p['libre_cada_rayo']))
        reach = rng[::every] - p['libre_margen']
        ts = np.arange(step, max(step, reach.max()) + step, step)
        m = ts[None, :] < reach[:, None]
        u = d[::every] / rng[::every, None]
        fi, fok = self.index(sensor[0] + u[:, :1] * ts, sensor[1] + u[:, 1:] * ts)
        free = np.unique(fi[m & fok])
        hi, hok = self.index(pts[:, 0], pts[:, 1])
        hit = np.unique(hi[hok])
        free = np.setdiff1d(free, hit, assume_unique=True)
        self.L[free] -= p['libre_logodds']
        self.L[hit] += p['golpe_logodds']
        np.clip(self.L, -p['logodds_max'], p['logodds_max'], out=self.L)
        self.seen[free] += 1
        self.seen[hit] += 1
        self.hits[hit] += 1
        passed = np.zeros(self.L.size, bool)
        passed[free] = True
        cut = maximum_filter(passed.reshape(self.shape), size=self.k).ravel()
        self.logb[cut] += math.log(p['busqueda_eps_lidar'])
        np.clip(self.logb, LOGB_MIN, LOGB_MAX, out=self.logb)
        # Hits away from the mapped walls feed the box fit.
        off = np.flatnonzero(hok & (self.wall_dist.ravel()[hi] > self.change_radius() - self.res)
                             & self.arena.ravel()[hi])
        if len(off) > p['puntos_por_scan']:
            off = self.rng.choice(off, int(p['puntos_por_scan']), replace=False)
        for c, x, y in zip(hi[off], pts[off, 0], pts[off, 1]):
            self._store(c, (x, y, sensor[0], sensor[1]))

    def _store(self, cell, row):
        """Reservoir sample per cell, so every viewpoint stays represented."""
        entry = self.store.get(cell)
        if entry is None:
            entry = self.store[cell] = [0, np.empty((self.store_cap, 4))]
        entry[0] += 1
        if entry[0] <= self.store_cap:
            entry[1][entry[0] - 1] = row
        else:
            j = self.rng.integers(entry[0])
            if j < self.store_cap:
                entry[1][j] = row

    def count_red(self, rgb):
        """Red pixels among the ones add_image samples: whether a depth cloud would help."""
        s = int(self.p['camara_paso_px'])
        return int(red_mask(rgb[s // 2::s, s // 2::s], self.p['rojo_h_bajo'],
                            self.p['rojo_h_alto'], self.p['rojo_s_min'],
                            self.p['rojo_croma_min']).sum())

    def red_depths(self, cam, R, K, us, cloud):
        """Per sampled image column, the horizontal distance to the red cloud points in it
        (median), NaN where it has none. cloud: (points (N, 3) map frame, rgb (N, 3) uint8)."""
        p = self.p
        out = np.full(len(us), np.nan)
        pts, rgb = cloud
        if not len(pts):
            return out
        keep = red_mask(np.asarray(rgb, np.uint8)[None], p['rojo_h_bajo'], p['rojo_h_alto'],
                        p['rojo_s_min'], p['rojo_croma_min'])[0]
        keep &= (pts[:, 2] >= 0.02) & (pts[:, 2] <= p['altura_muestra'])
        q = (pts[keep] - cam) @ R                  # optical frame
        rng = np.hypot(pts[keep, 0] - cam[0], pts[keep, 1] - cam[1])
        front = (q[:, 2] > 0.05) & (rng >= p['camara_rango_min'])
        if not front.any():
            return out
        u = K[0] * q[front, 0] / q[front, 2] + K[2]
        step = us[1] - us[0] if len(us) > 1 else 1
        col = np.round((u - us[0]) / step).astype(int)
        ok = (col >= 0) & (col < len(us))
        col, rng = col[ok], rng[front][ok]
        for c in np.unique(col):
            out[c] = np.median(rng[col == c])
        return out

    def add_image(self, cam, R, K, rgb, cloud=None):
        """One camera image. `cam` (x, y, z) and `R` (3x3, optical frame -> map) at the image
        stamp; `K` = (fx, fy, cx, cy); `rgb` (H, W, 3) uint8. `cloud`: the depth cloud of the
        same stamp, (points (N, 3) map frame, rgb (N, 3)), or None. Returns False if the frame
        was too close to the last fresh one: its color still counts where it gives a cell a new
        line of sight, but it doesn't cut the search belief again (frames from one viewpoint
        are not independent)."""
        p = self.p
        cam = np.asarray(cam, float)
        fwd = R[:, 2]
        yaw = math.atan2(fwd[1], fwd[0])
        fresh = True
        if self.last_view is not None:
            lx, ly, lyaw = self.last_view
            fresh = math.hypot(cam[0] - lx, cam[1] - ly) >= p['vista_paso'] \
                or abs(_wrap(yaw - lyaw)) >= p['vista_giro']
        if fresh:
            self.last_view = (cam[0], cam[1], yaw)
        s = int(p['camara_paso_px'])
        h_img, w_img = rgb.shape[:2]
        vs, us = np.arange(s // 2, h_img, s), np.arange(s // 2, w_img, s)
        red, info = pixel_classes(rgb[vs][:, us], p['rojo_h_bajo'], p['rojo_h_alto'],
                                  p['rojo_s_min'], p['rojo_croma_min'])
        fx, fy, cx, cy = K
        U, V = np.meshgrid(us, vs)
        d = np.stack([(U - cx) / fx, (V - cy) / fy, np.ones(U.shape)], -1) @ R.T
        hn = np.maximum(np.hypot(d[..., 0], d[..., 1]), 1e-9)
        slope = d[..., 2] / hn                  # height change per m of horizontal travel
        mid = len(vs) // 2
        bx, by = d[mid, :, 0] / hn[mid], d[mid, :, 1] / hn[mid]
        # Walk each column's ray through the posterior map to the first likely-occupied cell.
        step = self.res / 2
        ts = np.arange(step, p['camara_rango_max'] + step, step)
        ci, cok = self.index(cam[0] + bx[:, None] * ts, cam[1] + by[:, None] * ts)
        occ = cok & (self.L[ci] > 0)
        has = occ.any(1)
        k_occ = np.where(has, occ.argmax(1), len(ts) - 1)
        cols = np.arange(len(us))
        d_occ = np.where(has, ts[k_occ], np.inf)
        d_grid = d_occ                          # the grid's own, before depth opens columns
        cell_occ = ci[cols, k_occ]
        z0 = cam[2]
        with np.errstate(divide='ignore', invalid='ignore'):
            d_floor = np.where(slope < 0, z0 / -slope, np.inf)
            d_band = np.where(slope > 0, (p['altura_muestra'] - z0) / slope, np.inf)
        surf = d_occ[None, :] < d_floor
        h_surf = z0 + slope * np.where(np.isfinite(d_occ), d_occ, 0)[None, :]
        sample = surf & (h_surf >= 0.02) & (h_surf <= p['altura_muestra']) \
            & (d_occ >= p['camara_rango_min'])[None, :] & np.isfinite(d_occ)[None, :]
        cells = np.broadcast_to(cell_occ[None, :], sample.shape)[sample & info]
        is_red = red[sample & info]
        # With depth, a column's red pixels go to the cell at the depth of its red points: the
        # camera sees through gaps the grid has closed (a row of pillars), and the red belongs
        # to what is behind them, not to the first occupied cell.
        d_red = np.full(len(us), np.nan)
        if cloud is not None:
            d_red = self.red_depths(cam, R, K, us, cloud)
            has = np.isfinite(d_red)
            if has.any():
                ri, rok = self.index(cam[0] + bx * np.nan_to_num(d_red),
                                     cam[1] + by * np.nan_to_num(d_red))
                h_red = z0 + slope * np.nan_to_num(d_red)[None, :]
                moved = red & has[None, :] & rok[None, :] & (h_red >= 0.02) \
                    & (h_red <= p['altura_muestra'])
                keep = sample & info & ~(red & has[None, :])
                cells = np.concatenate([np.broadcast_to(cell_occ[None, :], sample.shape)[keep],
                                        np.broadcast_to(ri[None, :], sample.shape)[moved]])
                is_red = np.concatenate([red[keep], np.ones(moved.sum(), bool)])
                sample = keep | moved
                d_occ = np.where(has & (d_red > d_occ + p['profundidad_tol']), d_red, d_occ)
        # Color evidence, one bounded update per cell per view, and per line of sight: a cell
        # seen again along almost the same line (the camera only turned) repeats the same
        # errors, so it doesn't count again.
        if len(cells):
            uc, inv = np.unique(cells, return_inverse=True)
            n_tot = np.bincount(inv)
            n_red = np.bincount(inv, weights=is_red)
            a = math.log(p['rojo_p_si_rojo'] / p['rojo_p_si_otro'])
            b = math.log((1 - p['rojo_p_si_otro']) / (1 - p['rojo_p_si_rojo']))
            ev = np.clip(n_red * a - (n_tot - n_red) * b, -p['vista_logodds_max'],
                         p['vista_logodds_max'])
            last = self.view_xy[uc]
            now = np.column_stack([cam[0] - self.cx[uc], cam[1] - self.cy[uc]])
            before = last - np.column_stack([self.cx[uc], self.cy[uc]])
            cos = (now * before).sum(1) / np.maximum(
                np.hypot(*now.T) * np.hypot(*before.T), 1e-9)
            new = np.isnan(last[:, 0]) | (np.arccos(np.clip(cos, -1, 1)) >= p['vista_angulo'])
            self.Lc[uc[new]] = np.clip(self.Lc[uc[new]] + ev[new], -8.0, 8.0)
            self.view_xy[uc[new]] = cam[:2]
        # Object color: what each known object's own pixels say, one term per line of sight.
        self._object_color(cam, yaw, bx, by, slope, red, info, d_grid, cell_occ, d_red)
        if not fresh:
            return False
        # Search belief: an informative not-red pixel proves no box along its ray while the
        # ray stays below the lidar plane, up to what it hit (a box there would have been in
        # the way). A pixel too dark to tell proves nothing.
        d_hit = np.minimum(d_floor, d_occ[None, :])
        reach = np.where(red | ~info, 0.0, np.minimum(d_hit, d_band))
        red_at = np.where(red, d_hit, np.inf).min(0)
        col_clear = np.minimum.reduce([reach.max(0), red_at - p['busqueda_huella'],
                                       d_occ - p['libre_margen'],
                                       np.full(len(us), p['camara_rango_max'])])
        clear = cok & (ts[None, :] >= p['camara_rango_min']) & (ts[None, :] <= col_clear[:, None])
        if clear.any():
            seen = np.zeros(self.L.size, bool)
            seen[ci[clear]] = True
            cut = maximum_filter(seen.reshape(self.shape), size=self.k).ravel()
            self.logb[cut] += math.log(p['busqueda_eps_camara'])
        # Red with no known surface under it: raise the belief on the unseen cells of its ray.
        red_free = (red & ~sample).sum(0) >= p['rojo_pixeles_min']
        if red_free.any():
            span = red_free[:, None] & cok & (ts[None, :] < d_occ[:, None]) \
                & (ts[None, :] >= p['camara_rango_min'])
            idx = ci[span]
            idx = idx[(self.seen[idx] == 0) & self.prior[idx]]
            self.logb[np.unique(idx)] += math.log(p['busqueda_rojo'])
        np.clip(self.logb, LOGB_MIN, LOGB_MAX, out=self.logb)
        return True

    def _object_color(self, cam, yaw, bx, by, slope, red, info, d_grid, cell_occ, d_red):
        """Each known object's color from this frame: the sampled pixels whose ray meets its
        fitted box (shrunk by where the box may really land in the image) below the lidar
        plane, and that something says are on it, give one bounded log-likelihood of the red
        share. On it: the grid's line of sight reaches the box (d_grid: the first occupied
        cell per column), or, where the grid closes the column (a gap it can't resolve), a red
        pixel whose measured depth (d_red) is the box's. Red depth doesn't open the column for
        its other pixels: they may be the occluder, and when the cloud lands a few degrees off
        the image (taken a little apart in a turn), the grey pillar pixels beside a slit made
        the red box behind it grey. A red box shows mostly red, anything else almost none; red
        bleeding in from a gap beside an object is a minority of its pixels and pushes it
        toward not-red. Views along one line of sight keep their best evidence (most pixels);
        a new line of sight adds a term."""
        p = self.p
        z0 = cam[2]
        q_r, q_o = p['color_rojo_si_caja'], p['color_rojo_si_otro']
        a_r, a_n = math.log(q_r / q_o), math.log((1 - q_r) / (1 - q_o))
        for o in self.objects:
            a, b = o.lados[0] / 2, o.lados[1] / 2
            c, s = math.cos(o.yaw), math.sin(o.yaw)
            dx, dy = cam[0] - o.centro[0], cam[1] - o.centro[1]
            dist = math.hypot(dx, dy)
            if not p['camara_rango_min'] <= dist <= p['camara_rango_max'] + max(a, b):
                continue
            ang = math.atan2(-dy, -dx)
            if abs(_wrap(ang - yaw)) > 1.2:          # behind or far outside the 61 deg view
                continue
            if o.cov_fit is None:
                continue                             # no box fitted yet
            sig = math.sqrt(max(np.linalg.eigvalsh(o.cov_fit).max(), 0.0)
                            + p['camara_sigma_rel'] ** 2 + (dist * p['camara_sigma_yaw']) ** 2)
            m = p['color_margen_k'] * sig
            if min(a, b) - m < 0.01:
                continue                             # nothing certainly on the box
            # Each column's ray in the box frame: slab intersection with the full box (for
            # occlusion) and with the shrunk one (pixels certainly on it).
            ox, oy = c * dx + s * dy, -s * dx + c * dy
            ux, uy = c * bx + s * by, -s * bx + c * by
            t_full = _slab2(ox, oy, ux, uy, a, b)
            t_in = _slab2(ox, oy, ux, uy, a - m, b - m)
            cols = np.isfinite(t_in)
            if not cols.any():
                continue
            # Clear: the grid's first occupied cell isn't well before the box, or is its own,
            # and isn't something else's: another object's, or one the map's walls explain
            # (change()). A box flush against a wall, seen through it, showed the wall's grey:
            # within the tolerance, the wall's cells passed for the box's.
            owner = self.owner[cell_occ]
            own = np.isin(cell_occ, self._neighbourhood(o.cells, 1))
            clear = ((d_grid >= t_full - 2 * self.res) | own) & ((owner == 0) | (owner == o.id))
            # Depth, where there is some: red measured on the box is the box's, even through a
            # gap the grid closes; red measured clearly behind or in front isn't the box.
            on_box = np.isfinite(d_red) & (np.abs(d_red - t_in) <= p['profundidad_tol'] + m)
            off = np.isfinite(d_red) & ~on_box
            # Without depth: red seen past a mapped wall (a slat of a cage, a gap the grid
            # closes) is the box's when its ray meets the box just behind the wall. Only its
            # red pixels: the rest are the wall. In laberinto_1 the red of the caged box painted
            # the slats in front of it, a red-only object there was reported 0.4 m off, and the
            # box itself, which the lidar saw through the gaps, got no color.
            with np.errstate(invalid='ignore'):
                through = ~np.isfinite(d_red) & ~clear & (owner == -1) \
                    & (t_in - d_grid <= p['rojo_tras_pared'])
            cols &= clear | on_box | through
            h = z0 + slope * np.where(cols, t_in, 0.0)[None, :]
            on = cols[None, :] & (h >= 0.02) & (h <= p['altura_muestra']) & info
            on &= clear[None, :] | (red & (on_box | through)[None, :])
            on &= ~(red & off[None, :])
            n_inf = int(on.sum())
            if n_inf < p['color_pixeles_min']:
                continue
            f = float((red & on).sum()) / n_inf
            n_eff = min(n_inf / p['color_pixeles_por_muestra'], p['color_muestras_max'])
            ev = float(np.clip(n_eff * (f * a_r + (1 - f) * a_n), -p['vista_logodds_neg'],
                               p['vista_logodds_max']))
            bearing = math.atan2(dy, dx)
            same = [v for v in o.vistas if abs(_wrap(v[0] - bearing)) < p['vista_angulo']]
            if not same:
                o.vistas.append([bearing, ev, n_inf])
            elif n_inf > same[0][2]:
                same[0][1], same[0][2] = ev, n_inf
            o.lc = float(np.clip(sum(v[1] for v in o.vistas), -8.0, 8.0))

    # ------------------------------------------------------------------ inference

    def change(self):
        """New: occupied where the map says free, beyond what pose and map error explain.
        Removed: mapped walls the rays keep crossing."""
        p = self.p
        L = self.L.reshape(self.shape)
        new = (L > p['ocupado_logodds']) & (self.wall_dist > self.change_radius()) & self.arena
        return new, self.prior_occ & (L < p['libre_umbral'])

    def update(self):
        """Recompute the objects from the evidence. Returns them."""
        p = self.p
        new, _ = self.change()
        L = self.L.reshape(self.shape)
        red_seed = (self.Lc.reshape(self.shape) > _logit(p['reporte_p_min'])) & (L > 0) \
            & self.arena
        if p['rojo_borde'] > 0:
            red_seed &= ~((self.edge_dist <= p['rojo_borde'])
                          & (self.wall_dist <= self.change_radius()))
        cand = new | red_seed
        comps = self._components(cand, max(1, int(round(p['fusion'] / self.res))))
        objects = self._match(comps)
        self.owner = self._owners(objects)
        for o in objects:
            self._describe(o)
        # P(victim) before color: N victims among the objects in the arena, each as likely as
        # the others (an exchangeable prior), plus shape and color. Without N, even odds. The
        # objects in the arena include the ones not seen yet: as many per seen area as the seen
        # part holds. Counting only the seen ones, the prior started at its cap, and one red view
        # of a chair leg with a victim behind it was a report 2 s into the run.
        inside = [o for o in objects if o.dentro]
        n = p['objetivos_n']
        if n > 0 and inside:
            unseen = float(np.minimum(np.exp(self.logb), 1.0)[self.prior].sum()
                           / max(self.prior.sum(), 1))
            expected = len(inside) / max(1.0 - unseen, 1e-3)
            l0 = _logit(min(max(n / expected, p['previa_min']), p['previa_max']))
        else:
            l0 = 0.0
        shadow = self._red_shadows(objects, new.ravel())
        for o in objects:
            o.p_objetivo = _sigmoid(l0 + o.l_color + p['forma_peso'] * (o.forma - 0.5)) \
                if o.dentro and o.id not in shadow else 0.0
        confirmed = [o for o in objects if o.p_objetivo >= p['reporte_p_min']]
        self.complete = p['objetivos_n'] > 0 and len(confirmed) >= p['objetivos_n']
        if self.complete:
            for o in objects:
                if o.p_objetivo < p['reporte_p_min']:
                    o.p_objetivo = 0.0
        self.objects = objects
        return objects

    def _red_shadows(self, objects, new):
        """IDs of the red-only objects (seeded by color, no new occupancy of their own) whose
        red is a box's behind them: from the camera that last colored them, the line through
        them meets another object's box at most rojo_tras_pared further on. Red seen through
        a cage's slats colors the slats, without depth; the box behind is the victim."""
        p = self.p
        boxes = [q for q in objects if new[q.cells].any()]
        out = set()
        for o in objects:
            if new[o.cells].any():
                continue
            cams = self.view_xy[o.cells]
            cams = cams[np.isfinite(cams[:, 0])]
            if not len(cams):
                continue
            cam = cams.mean(0)
            u = o.centro - cam
            d_o = math.hypot(*u)
            if d_o < 1e-6:
                continue
            u /= d_o
            for q in boxes:
                v = q.centro - cam
                along = float(v @ u)
                across = abs(float(v[0] * u[1] - v[1] * u[0]))
                if 0.0 < along - d_o <= p['rojo_tras_pared'] + max(q.lados) / 2 \
                        and across <= max(q.lados) / 2 + o.sigma:
                    out.add(o.id)
                    break
        return out

    def _owners(self, objects):
        """Per cell, what its occupancy belongs to: an object's ID, -1 for the map's walls
        (within the change radius of one: change() explains occupancy there as the wall,
        seen a little off), 0 for nothing known."""
        owner = np.where(self.wall_dist.ravel() <= self.change_radius(), -1, 0).astype(np.int32)
        for o in objects:
            owner[o.cells] = o.id
        return owner

    def _components(self, cand, grow):
        """The connected components of the candidate cells (flat indices), with cells up to
        `grow` cells apart joined: fragments of one box seen through different gaps are one
        object. Only through space nobody has seen empty: a box has no holes, so fragments with
        free space between them are different things (a victim and a beige box 11 cm from it
        were one 0.5 m object, beige). A component too big to be one box is not fragments of one
        box: it is labeled again with less joining, down to none. On v08 a red box 17 cm from a
        yellow one became one 0.5 m object, yellow."""
        unseen = self.L.reshape(self.shape) >= self.p['libre_umbral']
        lab, n = label(binary_dilation(cand, iterations=grow, mask=unseen | cand) if grow
                       else cand, np.ones((3, 3)))
        lab[~cand] = 0
        flat = lab.ravel()
        order = np.argsort(flat, kind='stable')
        bounds = np.searchsorted(flat[order], np.arange(1, n + 2))
        out = []
        for i in range(n):
            cells = order[bounds[i]:bounds[i + 1]]
            if not len(cells):
                continue
            if grow and self._too_big(cells):
                sub = np.zeros(self.L.size, bool)
                sub[cells] = True
                out.extend(self._components(sub.reshape(self.shape), grow - 1))
            else:
                out.append(cells)
        return out

    def _too_big(self, cells):
        """Whether the cells fit in no box of the largest sides, at any heading (every 7.5
        deg): two boxes side by side can span less than one's diagonal."""
        xs, ys = self.cx[cells], self.cy[cells]
        th = np.radians(np.arange(0.0, 90.0, 7.5))[:, None]
        u = np.cos(th) * xs + np.sin(th) * ys
        v = -np.sin(th) * xs + np.cos(th) * ys
        side = self.p['caja_lado_max']
        return not bool(((u.ptp(1) <= side) & (v.ptp(1) <= side)).any())

    def _match(self, comps):
        """Keep each object's ID across updates: largest cell overlap, else a center within
        0.3 m, else a new ID. Objects don't merge into something too big to be one box: a
        component holding cells of several objects and larger than the largest box is split
        among them, each cell to the object whose cells are nearest. A box against a wall and a
        stretch of that wall seen a little off (a new object along it) came to touch, and the
        merged object took the wall's box and color: the victim was lost."""
        prev = {o.id: o for o in self.objects}
        owner = {}
        for o in self.objects:
            for c in o.cells:
                owner[int(c)] = o.id
        comps = [part for cells in comps for part in self._split(cells, owner, prev)]
        taken, out = set(), []
        for cells in sorted(comps, key=len, reverse=True):
            votes = {}
            for c in cells:
                oid = owner.get(int(c))
                if oid is not None and oid not in taken:
                    votes[oid] = votes.get(oid, 0) + 1
            center = np.array([self.cx[cells].mean(), self.cy[cells].mean()])
            oid = max(votes, key=votes.get) if votes else None
            if oid is None:
                near = [(np.hypot(*(o.centro - center)), o.id) for o in self.objects
                        if o.id not in taken]
                near = [x for x in near if x[0] < 0.3]
                oid = min(near)[1] if near else None
            if oid is None:
                o = Objeto(self.next_id, cells, center, np.eye(2))
                self.next_id += 1
            else:
                o = prev[oid]
                o.cells = cells
            taken.add(o.id)
            out.append(o)
        return out

    def _split(self, cells, owner, prev):
        """The component's cells, split among the objects it holds if it is too big for one box
        (see _match); [cells] otherwise."""
        ids = sorted({owner[int(c)] for c in cells if int(c) in owner})
        if len(ids) < 2 or not self._too_big(cells):
            return [cells]
        xs, ys = self.cx[cells], self.cy[cells]
        d = np.full((len(ids), len(cells)), np.inf)
        for k, oid in enumerate(ids):
            oc = prev[oid].cells
            ox, oy = self.cx[oc], self.cy[oc]
            for a in range(0, len(oc), 256):  # bounded memory for long objects
                dd = (xs[:, None] - ox[None, a:a + 256]) ** 2 + (ys[:, None] - oy[None, a:a + 256]) ** 2
                d[k] = np.minimum(d[k], dd.min(1))
        nearest = d.argmin(0)
        return [cells[nearest == k] for k in range(len(ids)) if (nearest == k).any()]

    def _describe(self, o):
        p = self.p
        o.scans = int(self.hits[o.cells].max())
        near = self._neighbourhood(o.cells, 1)
        stored = [(c, self.store[c][1][:min(self.store[c][0], self.store_cap)])
                  for c in near if c in self.store]
        pts = np.vstack([h for _, h in stored]) if stored else np.zeros((0, 4))
        o.fit_age += 1
        if len(pts) >= 6 and (o.x is None or o.fit_age >= p['ajuste_cada']
                              or abs(len(pts) - o.fit_n) > 0.15 * max(o.fit_n, 1)):
            self._fit(o, pts)
        if o.x is None:  # too few hits for a fit: the cells' extent
            xs, ys = self.cx[o.cells], self.cy[o.cells]
            o.centro = np.array([xs.mean(), ys.mean()])
            ext = max(xs.ptp(), ys.ptp()) + self.res
            o.lados = (ext, ext)
            o.cov = np.eye(2) * ((ext / 2) ** 2 + self.sigma_pose ** 2)
        # Where it was seen from: only its hits on cells nothing else explains (_owners). The
        # fit takes them all (a box's sides run into a wall's band, and the fit weighs each hit
        # by the face its sensor could see), but a wall's hits beside a box flush against it,
        # placed a few cm off, made spots behind the wall proven lines of sight.
        mine = [h for c, h in stored if self.owner[c] in (0, o.id)]
        o.miradores = self._viewpoints(o.centro, np.vstack(mine) if mine else np.zeros((0, 4)))
        # Color: the object's own views (_object_color). Before its first one (a new object, or
        # one that got a new ID), the median evidence on its cells, worth one view at most.
        if o.vistas:
            l_color = o.lc
        else:
            lc = self.Lc[near[np.isin(self.owner[near], (0, o.id))]]
            lc = lc[np.abs(lc) > 0.05]
            l_color = float(np.clip(np.median(lc), -p['vista_logodds_max'],
                                    p['vista_logodds_max'])) if len(lc) else 0.0
        o.p_rojo = _sigmoid(l_color)
        o.l_color = l_color
        i, j = self._cell_rc(o.centro)
        o.dentro = bool(i is not None and self.arena[i, j])

    def _fit(self, o, pts):
        p = self.p
        if len(pts) > p['ajuste_puntos_max']:
            pts = pts[self.rng.choice(len(pts), int(p['ajuste_puntos_max']), replace=False)]
        hits, sens = pts[:, :2], pts[:, 2:]
        center = hits.mean(0)
        r2 = (self.cx - center[0]) ** 2 + (self.cy - center[1]) ** 2
        carve_idx = np.flatnonzero((r2 < 0.45 ** 2) & (self.L < p['libre_umbral']))
        carve = np.column_stack([self.cx[carve_idx], self.cy[carve_idx]])
        lo, hi = p['caja_lado_min'] / 2, p['caja_lado_max'] / 2
        if o.x is not None and not o.degenerado:
            starts = [o.x]
        else:
            away = np.mean(hits - sens, 0)
            away /= np.linalg.norm(away) + 1e-9
            c0 = center + away * p['caja_lado_prior'] / 2
            starts = [np.array([c0[0], c0[1], th, p['caja_lado_prior'] / 2,
                                p['caja_lado_prior'] / 2]) for th in np.radians([0, 22.5, 45, 67.5])]
        best = None
        for x0 in starts:
            x0 = np.array(x0, float)
            x0[3:] = np.clip(x0[3:], lo + 1e-6, hi - 1e-6)
            try:
                r = least_squares(box_residuals, x0, args=(hits, sens, carve, p, center),
                                  loss='huber',
                                  f_scale=0.02, bounds=([-np.inf] * 3 + [lo] * 2,
                                                        [np.inf] * 3 + [hi] * 2), max_nfev=60)
            except ValueError:
                continue
            if best is None or r.cost < best.cost:
                best = r
        if best is None:
            return
        x = best.x
        n = len(hits)
        res = best.fun[:2 * n]
        s2 = max(float(np.mean(res ** 2)) * 2, 0.005 ** 2)
        # Information form: the data's (the Jacobian of the hit and carve rows) plus the priors
        # (the center within half the largest box of the hits, the sides around the prior), so
        # a far object seen from one side can't have an unbounded covariance.
        J = best.jac[:-4]
        s_c = p['caja_lado_max'] / 2
        prior = np.diag([s_c ** -2, s_c ** -2, math.pi ** -2, p['caja_lado_sigma'] ** -2,
                         p['caja_lado_sigma'] ** -2])
        cov = np.linalg.inv(J.T @ J / (s2 * p['punto_correlacion']) + prior)
        o.degenerado = math.sqrt(np.linalg.eigvalsh(cov[:2, :2]).max()) > 0.8 * s_c
        o.x, o.fit_n, o.fit_age = x, len(pts), 0
        o.centro = x[:2].copy()
        o.yaw = float(_wrap(x[2]))
        o.lados = (2 * x[3], 2 * x[4])
        o.cov_fit = cov[:2, :2]
        o.cov = cov[:2, :2] + np.eye(2) * max(self.sigma_pose, p['sigma_pose_min']) ** 2
        rms = math.sqrt(float(np.mean(res ** 2)) * 2)
        xs, ys = self.cx[o.cells], self.cy[o.cells]
        extent = math.hypot(xs.ptp(), ys.ptp())
        too_big = extent > math.hypot(p['caja_lado_max'], p['caja_lado_max']) + 0.1
        o.forma = math.exp(-(rms / 0.03) ** 2) * (0.1 if too_big else 1.0)

    def _viewpoints(self, center, pts, bins=8, ideal=1.2):
        """Up to `bins` sensor positions the object was seen from, one per direction."""
        if not len(pts):
            return []
        s = np.unique(np.round(pts[:, 2:], 2), axis=0)
        v = s - center
        dist = np.hypot(v[:, 0], v[:, 1])
        b = ((np.arctan2(v[:, 1], v[:, 0]) + math.pi) / (2 * math.pi) * bins).astype(int) % bins
        out = []
        for k in range(bins):
            m = np.flatnonzero(b == k)
            if len(m):
                out.append(tuple(s[m[np.argmin(np.abs(dist[m] - ideal))]]))
        return out

    def _neighbourhood(self, cells, r):
        mask = np.zeros(self.L.size, bool)
        mask[cells] = True
        if r:
            mask = binary_dilation(mask.reshape(self.shape), iterations=r).ravel()
        return np.flatnonzero(mask)

    def _cell_rc(self, xy):
        j = int(math.floor((xy[0] - self.origin[0]) / self.res))
        i = int(math.floor((xy[1] - self.origin[1]) / self.res))
        if 0 <= i < self.shape[0] and 0 <= j < self.shape[1]:
            return i, j
        return None, None

    # ------------------------------------------------------------------ outputs

    def belief(self):
        """P(an undetected box is centered in each cell), relative to the prior, on the grid.
        Known objects explain their own area; 0 everywhere once N boxes are confirmed."""
        b = np.exp(np.minimum(self.logb, LOGB_MAX)) * self.prior
        if self.complete:
            return np.zeros(self.shape)
        if self.objects:
            cells = np.concatenate([o.cells for o in self.objects])
            b[self._neighbourhood(cells, self.k + 1)] = 0.0
        return b.reshape(self.shape)

    def remaining(self, b=None):
        """Fraction of the prior's mass left: how incomplete the object list is."""
        b = self.belief() if b is None else b
        return float(np.minimum(b, 1.0).sum() / max(self.prior.sum(), 1))

    def occupancy_values(self):
        """Posterior P(occupied) as OccupancyGrid data: 0-100, -1 off the map and never seen."""
        v = np.round(100 / (1 + np.exp(-self.L))).astype(np.int8)
        v[~self.in_map.ravel() & (self.seen == 0)] = -1
        return v

    def belief_values(self, b=None):
        """The belief as OccupancyGrid data: 0-98 (RViz's costmap palette), -1 outside the prior."""
        b = (self.belief() if b is None else b).ravel()
        v = np.round(np.minimum(b, 1.0) * 98).astype(np.int8)
        v[~self.prior] = -1
        return v

    def color_values(self):
        """P(red surface) as OccupancyGrid data: 1-98 where there is evidence, else 0."""
        v = (1 + np.round(_sigmoid(self.Lc) * 97)).astype(np.int8)
        v[np.abs(self.Lc) < 0.05] = 0
        return v


# ---------------------------------------------------------------------- helpers

def box_residuals(x, hits, sens, carve, p, c0=None):
    """Residuals of a box (cx, cy, yaw, a, b; a and b half sides) against: hits on the face
    segments their sensor could see; empty space (carve) that can't be inside the box; weak
    priors on the sides and on the center (within half the largest box of the hits' centroid
    c0). The last 4 rows are the priors."""
    cx, cy, th, a, b = x
    c, s = math.cos(th), math.sin(th)
    R = np.array([[c, s], [-s, c]])
    P = (hits - (cx, cy)) @ R.T
    O = (sens - (cx, cy)) @ R.T
    d = np.stack([P[:, 0] - a, -P[:, 0] - a, P[:, 1] - b, -P[:, 1] - b], 1)
    vis = np.stack([O[:, 0] > a, -O[:, 0] > a, O[:, 1] > b, -O[:, 1] > b], 1)
    k = np.argmin(np.where(vis, np.abs(d), np.inf), 1)
    r = d[np.arange(len(hits)), k]
    along = np.where(k < 2, P[:, 1], P[:, 0])
    over = np.maximum(np.abs(along) - np.where(k < 2, b, a), 0.0)
    bad = ~vis.any(1)  # the sensor inside the box: impossible
    r[bad], over[bad] = 0.1, 0.0
    out = [r, over]
    if len(carve):
        Q = (carve - (cx, cy)) @ R.T
        out.append(np.maximum(np.minimum(a - np.abs(Q[:, 0]), b - np.abs(Q[:, 1])), 0.0) * 0.5)
    w = 0.02 / p['caja_lado_sigma']
    out.append(np.array([2 * a - p['caja_lado_prior'], 2 * b - p['caja_lado_prior']]) * w)
    c0 = (cx, cy) if c0 is None else c0
    out.append(np.array([cx - c0[0], cy - c0[1]]) * (0.02 / (p['caja_lado_max'] / 2)))
    return np.concatenate(out)


def pixel_classes(rgb, h_low, h_high, s_min, chroma_min):
    """(red, informative) masks of an (..., 3) uint8 RGB array.

    Red: hue in the red bands and saturation >= s_min (OpenCV scales: hue 0-180, saturation
    0-255), with an absolute chroma (max - min of RGB) above the camera's noise. Hue and
    saturation don't change with the light's intensity, so a red face in shadow is red.
    Informative: a pixel that would show a red surface's chroma if the surface were red, so
    not-red there is evidence. Too dark (a red surface of the least allowed saturation would
    stay under the chroma floor) or clipped (a channel at the top with the hue washed out)
    are unknown: no evidence either way."""
    c = rgb.astype(np.float32)
    mx, mn = c.max(-1), c.min(-1)
    delta = mx - mn
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    safe = np.where(delta > 0, delta, 1.0)
    h = np.where(mx == r, ((g - b) / safe) % 6,
                 np.where(mx == g, (b - r) / safe + 2, (r - g) / safe + 4)) * 30.0
    h = np.where(delta > 0, h, 0.0)
    sat = np.where(mx > 0, delta / np.where(mx > 0, mx, 1.0), 0.0) * 255
    red = ((h <= h_low) | (h >= h_high)) & (sat >= s_min) & (delta >= chroma_min)
    dark = mx < chroma_min * 255.0 / max(s_min, 1.0)
    clipped = (mx >= 250) & (sat < s_min)
    return red, red | ~(dark | clipped)


def red_mask(rgb, h_low, h_high, s_min, chroma_min):
    """Red pixels of an (..., 3) uint8 RGB array (pixel_classes)."""
    return pixel_classes(rgb, h_low, h_high, s_min, chroma_min)[0]


def arena_mask(occupied):
    """Inside the convex hull of the mapped walls, shrunk by one cell."""
    pts = np.argwhere(occupied).astype(float)
    if len(pts) < 3:
        return np.ones(occupied.shape, bool)
    hull = pts[ConvexHull(pts).vertices]
    ii, jj = np.mgrid[0:occupied.shape[0], 0:occupied.shape[1]]
    center = hull.mean(0)
    inside = np.ones(occupied.shape, bool)
    for a, b in zip(hull, np.roll(hull, -1, axis=0)):
        side = np.sign((b[0] - a[0]) * (center[1] - a[1]) - (b[1] - a[1]) * (center[0] - a[0]))
        inside &= side * ((b[0] - a[0]) * (jj - a[1]) - (b[1] - a[1]) * (ii - a[0])) >= -1e-9
    # Pad so the array's border counts as outside too.
    return inside & (distance_transform_edt(np.pad(inside, 1))[1:-1, 1:-1] > 1)


def _slab2(ox, oy, ux, uy, a, b):
    """Entry distance of rays from (ox, oy) along unit (ux, uy) (arrays) into the box
    |x| <= a, |y| <= b; inf where they miss it or start inside."""
    with np.errstate(divide='ignore', invalid='ignore'):
        tx1, tx2 = (-a - ox) / ux, (a - ox) / ux
        ty1, ty2 = (-b - oy) / uy, (b - oy) / uy
    tx1, tx2 = np.nan_to_num(tx1, nan=-np.inf), np.nan_to_num(tx2, nan=np.inf)
    ty1, ty2 = np.nan_to_num(ty1, nan=-np.inf), np.nan_to_num(ty2, nan=np.inf)
    t0 = np.maximum(np.minimum(tx1, tx2), np.minimum(ty1, ty2))
    t1 = np.minimum(np.maximum(tx1, tx2), np.maximum(ty1, ty2))
    return np.where((t1 >= t0) & (t0 > 0), t0, np.inf)


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p):
    return math.log(p / (1 - p))
