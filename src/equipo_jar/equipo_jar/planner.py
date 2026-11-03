"""
Where the robot should go next, from the world model's outputs (docs/MISSION.md). Pure numpy
and scipy, no ROS: the `mission` node calls it, and the tests run it on synthetic grids.

Every candidate is scored in the same unit, expected victims resolved per second:
- explore: a reachable spot, worth the undetected victims its lidar would see,
  (N - sum of P(victim) over the known objects) x visible belief mass / total belief mass;
- look: a reachable spot at camera range with a line of sight to an undecided object (on the
  posterior map, or proven: next to a spot the lidar saw the object from, a mirador), worth
  that object's P(victim): if it is a victim, looks confirm it. Each look must come from a
  new line of sight (mirada_angulo_min from the earlier ones, seen from the object): the world
  model counts a cell's color once per line of sight. At most miradas_max looks per object,
  so one the camera can't settle doesn't hold the robot forever.
The time is the path length on the posterior map at the cruise speed, plus a fixed cost.

At a look spot the robot sweeps (sweep): it slides through the spots around it that see the
object, camera on it, across the lines of sight not yet seen. Which one shows the box depends
on gaps and occluders the map draws only roughly, so passing through them all beats stopping at
one and retrying from the next.

The plan is a tour: explore stops by greedy set cover of the belief (two stops that see the
same corner aren't both counted), plus the best look at each undecided object, ordered as the
shortest open path from the robot (tour.py); nodes whose detour is worth less than
tour_tasa_min victims per second are dropped, and the rest reordered to bring the valuable ones
early (least value x arrival time). The robot drives to the first node and plans again there;
the previous plan's next node is kept unless another order is tour_histeresis better.
"""
from dataclasses import dataclass
import math

import numpy as np
from scipy.ndimage import binary_dilation, correlate, distance_transform_edt
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from equipo_jar.tour import latency, latency_order, shortest_path_order

DEFAULTS = {
    'ocupado_min': 65,          # posterior P(occupied) %, at or above: blocks
    # The chassis in base_link (yahboom_rosmaster_description), and the margin a planned path
    # keeps around it (footprint padding + the controller's tracking error): a path exists where
    # this rectangle fits, at some heading, all along (planned over x, y and heading).
    'huella_x': (-0.181, 0.119),
    'huella_y': 0.0997,
    'huella_margen': 0.05,
    'rumbos': 16,               # heading bins
    'giro_costo': 0.05,         # m of path a turn of one heading bin costs
    'holgura_comoda': 0.35,     # m of clearance; below it a path cell costs more, up to...
    'holgura_peso': 2.0,        # ...this many extra times its length where the robot just fits
    'holgura_meta': 0.28,       # m, clearance of a goal
    'velocidad': 0.25,          # m/s, average along a path
    't_fijo': 4.0,              # s, per target: planning, accelerating, settling
    't_mirada': 4.0,            # s, turning to face the object
    'paso_candidatos': 0.30,    # m, explore candidates: one per block this size
    'rango_vista': 3.5,         # m, belief the lidar counts as seen from a spot
    'rayos': 72,                # rays per explore candidate
    'mirada_min': 0.6,          # m, look spots: distance to the object's center
    'mirada_max': 1.6,
    'mirada_pref': 1.0,         # m, preferred distance (the camera's best resolution)
    'mirada_pref_peso': 2.0,    # s per m away from mirada_pref
    'mirada_angulo_min': 0.35,  # rad, a new look's line of sight vs the earlier ones'
    'mirada_misma_cara': 0.8,   # rad, a look this close to an earlier one sees the same face...
    'mirada_otra_cara_s': 0.0,  # ...s added to looks farther from all of them (another face):
                                # 0 now the world model counts any face's color per object
    'mirador_radio': 0.15,      # m, spots this close to a mirador share its line of sight
    'mirada_max_probada': 3.0,  # m, look distance from a proven line of sight
    'miradas_max': 6,           # looks per object
    'miradas_muertas_max': 3,   # looks in a row that didn't move its P: stop looking
    'mirada_mirador_s': 100.0,  # s per m from the nearest mirador, after a dead look: the
                                # retry stands where the lidar saw the box from, on its line
    'mirada_muerta_s': 20.0,    # s added to spots on the face of a look that didn't move
                                # its P: the camera got nothing there, try another face
    'barrido_largo': 0.8,       # m, a look sweep's path at most
    'barrido_angulo': 0.6,      # rad of new lines of sight (seen from the object) it aims for
    'barrido_vista': 0.17,      # rad, the world model's vista_angulo: a view counts again only
                                # from a line of sight this far from the earlier ones
    'p_descartado': 0.03,       # P(victim) at or below: not a victim, nothing to look at
    'reporte_p_min': 0.97,      # at or above: confirmed, nothing to look at
    'explorar_min': 0.01,       # expected victims: an explore target worth less is no target
    'faltantes_min': 0.1,       # expected victims not among the objects: below, don't explore
    'cerca': 0.4,               # m, explore spots closer than this to the robot are skipped
    'fallo_radio': 0.35,        # m, candidates this close to a failed goal are skipped
    'atasco_radio': 0.25,       # m, around the points of a stretch the robot couldn't cross
    'paso_previa': 0.9,         # P(a stretch the map shows open can be crossed), before trying
    'atasco_si_pasable': 0.3,   # P(getting stuck | it can be crossed): flukes happen
    'fallo_s': 15.0,            # s one failed crossing costs (stuck wait + escape)
    'paso_ruta': 0.10,          # m between the waypoints of a path sent to the controller
    'encierro_max': 0.15,       # m: nothing reachable farther than this, the robot is wedged
    'tour_bloque': 0.25,        # m, blocks the set cover counts the belief in
    'tour_max': 12,             # explore stops in a tour at most
    'tour_tasa_min': 0.002,     # victims/s: a node whose detour pays less is dropped
    'tour_histeresis': 1.0,     # m: keep the last plan's next node unless this much shorter
}


@dataclass
class Grid:
    """An OccupancyGrid's data as (rows, cols), row 0 at origin y."""
    values: np.ndarray
    res: float
    origin: tuple

    def cell(self, xy):
        return (int(math.floor((xy[1] - self.origin[1]) / self.res)),
                int(math.floor((xy[0] - self.origin[0]) / self.res)))

    def xy(self, rows, cols):
        return (self.origin[0] + (np.asarray(cols) + 0.5) * self.res,
                self.origin[1] + (np.asarray(rows) + 0.5) * self.res)


@dataclass
class Target:
    kind: str           # 'explore' or 'look'
    xy: tuple           # where to go, map frame
    value: float        # expected victims resolved
    time: float         # s, estimated
    score: float        # value / time
    object_id: int = -1
    face: tuple = None  # look: the point to turn the camera to
    precise: bool = False  # look: drive onto the spot (llegada_precisa), not just near it
    mass: float = 0.0   # explore: visible belief mass when planned

    def __str__(self):
        what = f'object {self.object_id}' if self.kind == 'look' else 'explore'
        return (f'{what} at ({self.xy[0]:.2f}, {self.xy[1]:.2f}): {self.value:.2f} in '
                f'{self.time:.0f} s')


class Planner:
    """One planning cycle: the posterior map and belief now, from the robot's position."""

    def __init__(self, occupancy, belief, robot, stuck=(), **params):
        """occupancy, belief: Grid of the world model's /equipo/ocupacion and /equipo/busqueda
        (same geometry). robot: (x, y) or (x, y, yaw). Paths are planned over position and
        heading: the chassis is a rectangle, so in a tight bend it fits only at some headings
        and can only turn where there's room; the base is holonomic, so it can translate at any
        heading. stuck: one list of points per time the robot couldn't
        cross a stretch (a gap the map shows but the robot doesn't fit). Each is evidence the
        stretch can't be crossed: crossing it costs its length plus the expected failures,
        (1 / P(passable) - 1) x fallo_s of driving, so a route around wins once the evidence
        outweighs the detour."""
        self.p = dict(DEFAULTS, **params)
        self.grid = occupancy
        self.res = occupancy.res
        v = occupancy.values
        self.blocked = (v >= self.p['ocupado_min']) | (v < 0)
        self.belief = np.clip(belief.values, 0, None) / 98.0
        self.clearance = distance_transform_edt(~self.blocked) * self.res
        nb = int(self.p['rumbos'])
        self.headings = np.arange(nb) * 2 * math.pi / nb
        self.fits = footprint_fits(self.blocked, self.res, self.headings, self.p['huella_x'],
                                   self.p['huella_y'], self.p['huella_margen'])
        self.turn_free = self.fits.all(axis=0)   # room to turn: any heading fits
        passable = self.fits.any(axis=0)
        yaw = robot[2] if len(robot) > 2 else None
        robot = tuple(robot[:2])
        self.start = start = self._nearest(passable, occupancy.cell(robot))
        self.dist = np.full(v.shape, np.inf)
        self.pred = self.node_dist = self.cost = None
        if start is not None:
            # Tight spots cost more: the controller is slow and fragile there, and a path
            # through the middle of a corridor is the one it follows best.
            lo = self.p['huella_y'] + self.p['huella_margen']
            hi = max(self.p['holgura_comoda'], lo + 1e-3)
            tight = np.clip((hi - self.clearance) / (hi - lo), 0.0, 1.0)
            cost = 1.0 + self.p['holgura_peso'] * tight
            if len(stuck):
                cost = cost + self.failure_cost(stuck, v.shape)
            self.cost = cost
            free_bins = np.flatnonzero(self.fits[:, start[0], start[1]])
            if yaw is None:
                b0 = int(free_bins[0])
            else:
                gap = np.abs((self.headings[free_bins] - yaw + math.pi) % (2 * math.pi) - math.pi)
                b0 = int(free_bins[np.argmin(gap)])
            self.node_dist, self.pred = pose_graph_lengths(
                self.fits, (b0, start[0], start[1]), self.res, cost, self.p['giro_costo'])
            self.dist = self.node_dist.min(axis=0)
        self.goal_ok = np.isfinite(self.dist) & (self.clearance >= self.p['holgura_meta'])
        self.robot = robot

    @property
    def wedged(self):
        """No room to move on the map: no cell the chassis fits on near the robot, or no
        reachable one farther than encierro_max from it (it fits only about as it stands).
        p06: the robot stopped 16 cm from a box's cells, the planner found no path anywhere,
        and the mission gave up with half the arena unseen."""
        if self.start is None:
            return True
        rows, cols = np.nonzero(np.isfinite(self.dist))
        far = np.hypot(rows - self.start[0], cols - self.start[1]).max() * self.res
        return far < self.p['encierro_max']

    def failure_cost(self, stuck, shape):
        """Extra cost per metre on each cell from the stuck evidence: the expected failures of
        a crossing beyond the prior's, (1 / P - 1 / P0) x fallo_s at velocidad, spread over
        the stretch's width (so a path pays it about once)."""
        gx, gy = self.grid.xy(*np.indices(shape))
        k = np.zeros(shape)
        r = self.p['atasco_radio']
        for event in stuck:
            hit = np.zeros(shape, bool)
            for sx, sy in event:
                hit |= np.hypot(gx - sx, gy - sy) < r
            k += hit
        prior = self.p['paso_previa']
        odds = prior / (1 - prior) * self.p['atasco_si_pasable'] ** k
        p_pass = odds / (1 + odds)
        # Only the evidence counts: what every stretch risks a priori is the same for all paths.
        return (1 / p_pass - 1 / prior) * self.p['fallo_s'] * self.p['velocidad'] / (2 * r)

    def _nearest(self, mask, cell, radius=0.5):
        r = int(radius / self.res)
        i0, j0 = max(cell[0] - r, 0), max(cell[1] - r, 0)
        win = mask[i0:cell[0] + r + 1, j0:cell[1] + r + 1]
        if not win.any():
            return None
        ii, jj = np.nonzero(win)
        k = np.argmin(np.hypot(ii + i0 - cell[0], jj + j0 - cell[1]))
        return ii[k] + i0, jj[k] + j0

    def path_to(self, xy):
        """Path length from the robot to xy (to the nearest passable cell), inf if none."""
        c = self._nearest(np.isfinite(self.dist), self.grid.cell(xy))
        return np.inf if c is None else float(self.dist[c])

    def path(self, xy):
        """The planned path from the robot to xy as map-frame points every paso_ruta, the
        robot first and xy last; None if xy can't be reached. The controller follows this
        path, so the route the robot drives is the one its targets were chosen on."""
        poses = self.path_poses(xy)
        return None if poses is None else [p[:2] for p in poses]

    def path_poses(self, xy):
        """Like path, as (x, y, heading): heading is the planned one where the robot has no
        room to turn (it must hold that heading through there), None where it can face any
        way."""
        c = self._nearest(np.isfinite(self.dist), self.grid.cell(xy))
        if c is None or self.pred is None:
            return None
        nb, rows, cols = self.fits.shape
        k = int(np.argmin(self.node_dist[:, c[0], c[1]])) * rows * cols + c[0] * cols + c[1]
        nodes = []
        while k >= 0:
            nodes.append(k)
            k = self.pred.flat[k]
        nodes = nodes[::-1]
        b, rem = np.divmod(np.array(nodes), rows * cols)
        r, q = np.divmod(rem, cols)
        x, y = self.grid.xy(r, q)
        free = self.turn_free[r, q]
        pts = [(float(x[i]), float(y[i]), None if free[i] else float(self.headings[b[i]]))
               for i in range(len(nodes))]
        first = (self.robot[0], self.robot[1], pts[0][2])
        last = (float(xy[0]), float(xy[1]), pts[-1][2])
        out = [first]
        for p in pts[1:]:
            if math.dist(p[:2], out[-1][:2]) >= self.p['paso_ruta'] or (
                    p[2] is not None and p[2] != out[-1][2]):
                out.append(p)
        out.append(last)
        return out

    def clearance_at(self, xy):
        """Clearance (m) at xy on the posterior map."""
        r, c = self.grid.cell(xy)
        if not (0 <= r < self.clearance.shape[0] and 0 <= c < self.clearance.shape[1]):
            return 0.0
        return float(self.clearance[r, c])

    def sees(self, a, b, own=0.0):
        """Whether the segment a -> b crosses no blocked cell of the posterior map, except
        within own of b (the object's own footprint)."""
        d = math.dist(a, b)
        n = max(2, int(d / (0.5 * self.res)))
        t = np.linspace(0.0, 1.0, n)
        x = a[0] + (b[0] - a[0]) * t
        y = a[1] + (b[1] - a[1]) * t
        far = np.hypot(x - b[0], y - b[1]) >= own
        i = np.floor((y - self.grid.origin[1]) / self.res).astype(int)
        j = np.floor((x - self.grid.origin[0]) / self.res).astype(int)
        ok = (i >= 0) & (i < self.blocked.shape[0]) & (j >= 0) & (j < self.blocked.shape[1])
        if not ok.all():
            return False
        return not (self.blocked[i, j] & far).any()

    def reaches(self, x, y, obj):
        """Whether lines of sight from the points (x, y) (arrays) reach the object: walking to
        its center, the first blocked cell is one of its cells (obj['cells'], grown by one), as
        the world model counts its color. A wall a box stands flush against is in the way
        though it is within the box's radius of the center. Without cells, anything within the
        object's radius of the center counts as its own."""
        cx, cy = obj['center']
        x, y = np.atleast_1d(np.asarray(x, float)), np.atleast_1d(np.asarray(y, float))
        d = np.maximum(np.hypot(x - cx, y - cy), 1e-9)
        t = np.arange(0.5, d.max() / self.res + 1) * self.res
        f = np.minimum(t[None, :] / d[:, None], 1.0)
        sx = x[:, None] + (cx - x)[:, None] * f
        sy = y[:, None] + (cy - y)[:, None] * f
        si = np.floor((sy - self.grid.origin[1]) / self.res).astype(int)
        sj = np.floor((sx - self.grid.origin[0]) / self.res).astype(int)
        rows, cols = self.blocked.shape
        inside = (si >= 0) & (si < rows) & (sj >= 0) & (sj < cols)
        si, sj = np.clip(si, 0, rows - 1), np.clip(sj, 0, cols - 1)
        blocked = self.blocked[si, sj] | ~inside
        cells = np.asarray(obj.get('cells', ()), int)
        cells = cells[(cells >= 0) & (cells < self.blocked.size)]
        if not len(cells):
            near = np.hypot(sx - cx, sy - cy) < obj.get('radius', 0.15) + 0.5 * self.res
            return ~(blocked & ~near).any(axis=1)
        own = np.zeros(self.blocked.size, bool)
        own[cells] = True
        own = binary_dilation(own.reshape(self.blocked.shape))
        first = blocked.argmax(axis=1)
        k = np.arange(len(x))
        return ~blocked.any(axis=1) | (own[si[k, first], sj[k, first]] & inside[k, first])

    def sees_object(self, a, obj):
        """Whether the line of sight from a reaches the object (reaches)."""
        return bool(self.reaches([a[0]], [a[1]], obj)[0])

    def escape_point(self, xy, dist=0.3):
        """Where to move to get out of a tight spot: up the clearance gradient, dist away,
        to the cell with the most clearance on the posterior map within that radius."""
        r = max(1, int(round(dist / self.res)))
        i, j = self.grid.cell(xy)
        i0, j0 = max(i - r, 0), max(j - r, 0)
        win = self.clearance[i0:i + r + 1, j0:j + r + 1]
        if not win.size:
            return None
        ii, jj = np.indices(win.shape)
        inside = np.hypot(ii + i0 - i, jj + j0 - j) <= r
        k = np.argmax(np.where(inside, win, -1.0))
        bi, bj = np.unravel_index(k, win.shape)
        x, y = self.grid.xy(bi + i0, bj + j0)
        return float(x), float(y)

    # Explore.

    def explore_candidates(self):
        """One goal per paso_candidatos block: the reachable cell with the most clearance."""
        step = max(1, int(round(self.p['paso_candidatos'] / self.res)))
        ok = self.goal_ok & (self.dist >= self.p['cerca'])
        score = np.where(ok, self.clearance, -1.0)
        rows, cols = score.shape
        pr, pc = -rows % step, -cols % step
        blocks = np.pad(score, ((0, pr), (0, pc)), constant_values=-1.0)
        blocks = blocks.reshape((rows + pr) // step, step, (cols + pc) // step, step)
        blocks = blocks.transpose(0, 2, 1, 3).reshape((rows + pr) // step, (cols + pc) // step, -1)
        best = blocks.argmax(axis=2)
        keep = np.take_along_axis(blocks, best[..., None], axis=2)[..., 0] >= 0
        bi, bj = np.nonzero(keep)
        k = best[bi, bj]
        return bi * step + k // step, bj * step + k % step

    def sight(self, rows, cols):
        """The lidar's rays from each cell: the cells (si, sj) of their samples, (cells, rays,
        samples), and the area in cells each sample stands for, 0 past the first wall."""
        n = self.p['rayos']
        th = np.arange(n) * 2 * math.pi / n
        rs = np.arange(0.5, self.p['rango_vista'] / self.res) * self.res
        x, y = self.grid.xy(rows, cols)
        sx = x[:, None, None] + rs[None, None, :] * np.cos(th)[None, :, None]
        sy = y[:, None, None] + rs[None, None, :] * np.sin(th)[None, :, None]
        si = np.floor((sy - self.grid.origin[1]) / self.res).astype(int)
        sj = np.floor((sx - self.grid.origin[0]) / self.res).astype(int)
        rows_n, cols_n = self.blocked.shape
        inside = (si >= 0) & (si < rows_n) & (sj >= 0) & (sj < cols_n)
        si, sj = np.clip(si, 0, rows_n - 1), np.clip(sj, 0, cols_n - 1)
        shadow = np.logical_or.accumulate(self.blocked[si, sj] | ~inside, axis=2)
        # Each sample stands for the area r dtheta dr around it, in cells.
        w = rs * (2 * math.pi / n) / self.res
        return si, sj, np.where(shadow, 0.0, np.broadcast_to(w, shadow.shape))

    # Look.

    def look(self, obj, seen_from=(), failed=()):
        """The best spot to look at an undecided object from, or None. seen_from: where the
        earlier looks at it were taken (or their views counted); obj['miradas']: how many looks
        (a sweep is one; default len(seen_from)). Worth its P(victim) times obj['eficacia'], the
        chance a look changes it, learned from the earlier looks (default 1). obj['muertas']:
        the earlier looks that didn't move its P (the camera got no pixels on it there, say
        through a cage's bars): their lines of sight stay open, spots on their face cost
        mirada_muerta_s more (another face first, if there is one), and after
        obj['muertas_seguidas'] reaches miradas_muertas_max in a row, no more looks."""
        p = obj['p']
        value = p * obj.get('eficacia', 1.0)
        dead = obj.get('muertas', ())  # where the looks that didn't move its P were taken
        if not self.p['p_descartado'] < p < self.p['reporte_p_min'] \
                or obj.get('miradas', len(seen_from)) >= self.p['miradas_max'] \
                or obj.get('muertas_seguidas', 0) >= self.p['miradas_muertas_max']:
            return None
        cx, cy = obj['center']
        rows, cols = np.nonzero(self.goal_ok)
        x, y = self.grid.xy(rows, cols)
        d, to_mirador, proven, ring = self.ring(obj, x, y)
        for f in failed:
            ring &= np.hypot(x - f[0], y - f[1]) > self.p['fallo_radio']
        bearing = np.arctan2(y - cy, x - cx)
        closest = np.full(len(x), np.inf)
        closest_dead = np.full(len(x), np.inf)
        for sx, sy in seen_from:
            turn = np.abs((bearing - math.atan2(sy - cy, sx - cx) + math.pi) % (2 * math.pi)
                          - math.pi)
            if any(math.dist((sx, sy), q) < 1e-6 for q in dead):
                # It added no evidence, so its line of sight isn't used up: for a box in a cage
                # the proven lines may be the only ones, and the next look there can work.
                closest_dead = np.minimum(closest_dead, turn)
                continue
            ring &= turn >= self.p['mirada_angulo_min']
            closest = np.minimum(closest, turn)
        # After a dead look, retry on a proven line of sight, as close to its mirador as the
        # robot can stand, and drive onto it: within mirador_radio and the usual arrival
        # distance the robot looked from up to 0.4 m off the line, through a gap a few cm wide
        # between a cage's walls.
        precise = obj.get('muertas_seguidas', 0) >= 1 and (proven & ring).any()
        if precise:
            ring &= proven
        rows, cols, x, y, d, proven, closest, closest_dead, to_mirador = (
            a[ring] for a in (rows, cols, x, y, d, proven, closest, closest_dead, to_mirador))
        if not len(rows):
            return None
        clear = self.reaches(x, y, obj) | proven
        if not clear.any():
            return None
        # The world model counts a cell's color once per line of sight: a box confirms when
        # the same face is seen again from a new direction, so prefer one near an earlier look.
        other_face = np.isfinite(closest) & (closest > self.p['mirada_misma_cara'])
        time = (self.dist[rows, cols] / self.p['velocidad'] + self.p['t_fijo']
                + self.p['t_mirada']
                + self.p['mirada_pref_peso'] * np.abs(d - self.p['mirada_pref'])
                + np.where(other_face, self.p['mirada_otra_cara_s'], 0.0)
                + np.where(closest_dead <= self.p['mirada_misma_cara'],
                           self.p['mirada_muerta_s'], 0.0))
        if precise:
            time = time + self.p['mirada_mirador_s'] * to_mirador
        time = np.where(clear, time, np.inf)
        k = int(np.argmin(time))
        return Target('look', (float(x[k]), float(y[k])), value, float(time[k]),
                      value / float(time[k]), obj['id'], (cx, cy), precise=bool(precise))

    def ring(self, obj, x, y):
        """For the points (x, y): distance to the object's center, to its nearest mirador,
        whether the line of sight is proven, and whether it is in camera range for a look."""
        d = np.hypot(x - obj['center'][0], y - obj['center'][1])
        # Next to a mirador, the line of sight is proven: the lidar saw the object from there.
        # The map's 5 cm grid closes gaps narrower than that (a row of pillars), so for a box
        # behind one, these may be the only spots that see it.
        to_mirador = np.full(len(x), np.inf)
        for mx, my in obj.get('miradores', ()):
            to_mirador = np.minimum(to_mirador, np.hypot(x - mx, y - my))
        proven = (to_mirador <= self.p['mirador_radio']) & (d <= self.p['mirada_max_probada'])
        ring = (d >= self.p['mirada_min']) & ((d <= self.p['mirada_max']) | proven)
        return d, to_mirador, proven, ring

    def sweep(self, obj, xy, seen_from=()):
        """A look sweep from xy: points to slide through with the camera on the object, xy
        first, or None when no spot near xy sees it with room to turn. It stays on spots in
        camera range with a line of sight (look's) where the chassis can face any way, up to
        barrido_largo of path, and ends where it has crossed the most lines of sight no view
        has counted from (seen_from: where views counted; barrido_vista around each is used
        up), up to barrido_angulo; the shortest such path."""
        ok = self.turn_free & np.isfinite(self.dist)
        rows, cols = np.nonzero(ok)
        x, y = self.grid.xy(rows, cols)
        _, _, proven, ring = self.ring(obj, x, y)
        rows, cols, x, y, proven = (a[ring] for a in (rows, cols, x, y, proven))
        clear = self.reaches(x, y, obj) | proven if len(rows) else np.zeros(0, bool)
        mask = np.zeros(ok.shape, bool)
        mask[rows[clear], cols[clear]] = True
        start = self._nearest(mask, self.grid.cell(xy), radius=0.3)
        if start is None:
            return None
        dist, pred = path_lengths(mask, start, self.res, self.cost, predecessors=True)
        ends = np.flatnonzero(dist.ravel() <= self.p['barrido_largo'])
        r, c = np.divmod(ends, mask.shape[1])
        ex, ey = self.grid.xy(r, c)
        cx, cy = obj['center']
        b0 = math.atan2(xy[1] - cy, xy[0] - cx)
        # The new lines of sight between xy's and each end's, sampled every `step` rad.
        step = 0.01
        a = np.arange(-math.pi, math.pi, step)
        new = np.ones(len(a), bool)
        for sx, sy in seen_from:
            seen = math.atan2(sy - cy, sx - cx)
            new &= np.abs((b0 + a - seen + math.pi) % (2 * math.pi) - math.pi) \
                >= self.p['barrido_vista']
        zero = len(a) // 2   # a[zero] == 0
        up = np.cumsum(new[zero:]) * step                 # new span from 0 up to a[zero + k]
        down = np.cumsum(new[zero::-1]) * step            # from 0 down to a[zero - k]
        rel = (np.arctan2(ey - cy, ex - cx) - b0 + math.pi) % (2 * math.pi) - math.pi
        k = np.minimum(np.round(np.abs(rel) / step).astype(int), min(len(up), len(down)) - 1)
        gain = np.minimum(np.where(rel >= 0, up[k], down[k]), self.p['barrido_angulo'])
        if not len(gain) or gain.max() < self.p['barrido_vista']:
            return None
        best = ends[np.lexsort((dist.ravel()[ends], -np.round(gain / step)))[0]]
        cells = []
        while best >= 0:
            cells.append(best)
            best = pred.flat[best]
        r, c = np.divmod(np.array(cells[::-1]), mask.shape[1])
        px, py = self.grid.xy(r, c)
        out = [tuple(xy)]
        for p in zip(px.tolist(), py.tolist()):
            if math.dist(p, out[-1]) >= self.p['paso_ruta']:
                out.append(p)
        if out[-1] != (px[-1], py[-1]):
            out.append((float(px[-1]), float(py[-1])))
        return out

    # Tour.

    def cover(self, missing, failed=()):
        """Explore stops by greedy set cover of the belief, most useful first. Each is worth
        the belief it sees that the ones before it don't, counted in tour_bloque blocks: the
        rays sample a cell here and there at range, a block they cross they see in part."""
        total = self.belief.sum()
        if missing <= 0 or total <= 0:
            return []
        rows, cols = self.explore_candidates()
        x, y = self.grid.xy(rows, cols)
        keep = np.ones(len(rows), bool)
        for f in failed:
            keep &= np.hypot(x - f[0], y - f[1]) > self.p['fallo_radio']
        rows, cols, x, y = rows[keep], cols[keep], x[keep], y[keep]
        if not len(rows):
            return []
        frac, left, mass = self.block_view(rows, cols)
        time = self.dist[rows, cols] / self.p['velocidad'] + self.p['t_fijo']
        out = []
        while len(out) < self.p['tour_max']:
            gain = frac @ left
            j = int(np.argmax(gain))
            value = missing * min(gain[j] / total, 1.0)
            if value < self.p['explorar_min']:
                break
            out.append(Target('explore', (float(x[j]), float(y[j])), value, float(time[j]),
                              value / float(time[j]), mass=float(mass[j])))
            left *= 1.0 - frac[j]
        return out

    def block_view(self, rows, cols):
        """What the lidar sees from each cell, in tour_bloque blocks: the share of each block
        (cells, blocks), the belief in each block, and the belief mass each cell sees."""
        si, sj, w = self.sight(rows, cols)
        b = max(1, int(round(self.p['tour_bloque'] / self.res)))
        nr, nc = -(-self.blocked.shape[0] // b), -(-self.blocked.shape[1] // b)
        k = len(rows)
        flat = (np.arange(k)[:, None, None] * (nr * nc) + (si // b) * nc + sj // b).ravel()
        seen = np.bincount(flat, weights=w.ravel(), minlength=k * nr * nc).reshape(k, -1)
        r, c = np.indices(self.blocked.shape)
        left = np.bincount(((r // b) * nc + c // b).ravel(), weights=self.belief.ravel(),
                           minlength=nr * nc)
        return (np.minimum(seen / (b * b), 1.0), left,
                (self.belief[si, sj] * w).sum(axis=(1, 2)))

    def leg_lengths(self, xys):
        """Path lengths (m) between the robot (node 0) and xys (nodes 1...): from the robot on
        the pose graph, between the others on the cells where the chassis fits at some
        heading, with the same costs (the turns left out: close enough to order them)."""
        cells = [self.grid.cell(xy) for xy in xys]
        passable = self.fits.any(axis=0) & np.isfinite(self.dist)
        d = path_lengths(passable, cells, self.res, self.cost)
        n = len(cells)
        D = np.zeros((n + 1, n + 1))
        for i, a in enumerate(cells):
            D[0, i + 1] = D[i + 1, 0] = self.dist[a]
            for j, c in enumerate(cells):
                D[i + 1, j + 1] = d[i][c]
        D[1:, 1:] = np.minimum(D[1:, 1:], D[1:, 1:].T)
        return np.where(np.isfinite(D), D, 1e6)

    def tour(self, objects, n_total, seen_from=None, failed=(), prefer=None, visited=()):
        """Targets in visiting order; empty when nothing is worth going to.

        objects: dicts with id, center (x, y), p (P(victim)), radius (m). n_total: N.
        seen_from: object id -> where its earlier looks were taken. failed: goals the robot
        couldn't reach. visited: explore stops just reached, skipped as explore stops only:
        a look from beside one is a new view (on laberinto_1 a box's only look spots were
        next to the stop it was seen from, and the search ended with it undecided).
        prefer: where the last tour went next, kept first unless a tour
        starting elsewhere is tour_histeresis shorter: replanning at every node must not swap
        between two orders that cost about the same."""
        if self.cost is None:
            return []
        seen_from = seen_from or {}
        missing = expected_missing(n_total, [o['p'] for o in objects])
        nodes = self.cover(missing, list(failed) + list(visited)) \
            if missing >= self.p['faltantes_min'] else []
        for o in objects:
            t = self.look(o, seen_from.get(o['id'], ()), failed)
            if t is not None and all(math.dist(t.xy, f) > self.p['fallo_radio'] for f in failed):
                nodes.append(t)
        if not nodes:
            return []
        D = self.leg_lengths([t.xy for t in nodes])
        v = self.p['velocidad']
        value = np.array([0.0] + [t.value for t in nodes])
        fixed = np.array([0.0] + [t.time - D[0, i + 1] / v for i, t in enumerate(nodes)])
        order = shortest_path_order(D)
        while len(order) > 1:  # drop the node whose detour pays least, while below the rate
            worst, rate = None, self.p['tour_tasa_min']
            for k in range(1, len(order)):
                a, i = order[k - 1], order[k]
                saved = D[a, i]
                if k + 1 < len(order):
                    saved += D[i, order[k + 1]] - D[a, order[k + 1]]
                r = value[i] / (saved / v + fixed[i])
                if r < rate:
                    worst, rate = k, r
            if worst is None:
                break
            sub = order[:worst] + order[worst + 1:]
            order = [sub[j] for j in shortest_path_order(D[np.ix_(sub, sub)])]
        # The shortest path, reordered so the valuable nodes come early: every detour delays
        # all the nodes after it, so it is taken only when that pays.
        T = D / v
        order = latency_order(T, value, fixed, order)
        if prefer is not None and len(order) > 2:
            near = [i for i in order[1:] if math.dist(nodes[i - 1].xy, prefer) <= 0.3]
            if near and near[0] != order[1]:
                j = near[0]
                rest = [j] + [i for i in order[1:] if i != j]
                kept = [0] + [rest[k] for k in shortest_path_order(D[np.ix_(rest, rest)])]
                kept = latency_order(T, value, fixed, kept, keep=2)
                slack = value.sum() * self.p['tour_histeresis'] / v
                if latency(T, value, fixed, kept) <= latency(T, value, fixed, order) + slack:
                    order = kept
        return [nodes[i - 1] for i in order[1:]]

    def still_worth(self, target, missing):
        """Whether an explore target is still worth explorar_min, as cover() admits a stop (the
        way there may have seen most of it already): else a stop dropped here would be chosen
        again by the next tour. missing: expected victims not among the objects."""
        c = self.grid.cell(target.xy)
        total = self.belief.sum()
        frac, left, _ = self.block_view(np.array([c[0]]), np.array([c[1]]))
        return total > 0 and missing * min(float(frac[0] @ left) / total, 1.0) \
            >= self.p['explorar_min']


def footprint_fits(blocked, res, headings, xs, half_y, margin):
    """(headings, rows, cols) bool: whether the chassis rectangle, x in xs and y in +-half_y in
    the robot frame grown by margin, fits with the robot's origin at each cell's center at
    each heading. A cell outside the map is blocked."""
    x0, x1 = xs[0] - margin, xs[1] + margin
    y1 = half_y + margin
    reach = math.hypot(max(abs(x0), abs(x1)), y1)
    n = int(math.ceil(reach / res))
    di, dj = np.mgrid[-n:n + 1, -n:n + 1]
    px, py = dj * res, di * res        # cell offsets in metres: columns are x, rows are y
    out = np.zeros((len(headings),) + blocked.shape, bool)
    b = blocked.astype(np.uint8)
    for k, th in enumerate(headings):
        c, s = math.cos(th), math.sin(th)
        u, v = c * px + s * py, -s * px + c * py     # into the robot frame
        kernel = ((u >= x0) & (u <= x1) & (np.abs(v) <= y1)).astype(np.uint8)
        out[k] = correlate(b, kernel, mode='constant', cval=1) == 0
    return out


def pose_graph_lengths(fits, start, res, cost, turn_cost):
    """Shortest paths over (heading, row, col) from start: translate to any of the 8
    neighbours at the same heading where the robot fits at both, or turn one heading bin
    where it fits at both headings. Steps cost their length (turns turn_cost) times the mean
    of both cells' cost. Returns the cost to every node and each node's predecessor (flat
    index, -1 for none)."""
    nb, rows, cols = fits.shape
    idx = -np.ones(fits.shape, int)
    nodes = np.flatnonzero(fits)
    idx.flat[nodes] = np.arange(len(nodes))
    src, dst, w = [], [], []
    for di, dj in ((0, 1), (1, 0), (1, 1), (1, -1)):
        r0, r1 = max(0, -di), rows - max(0, di)
        c0, c1 = max(0, -dj), cols - max(0, dj)
        a = idx[:, r0:r1, c0:c1]
        b = idx[:, r0 + di:r1 + di, c0 + dj:c1 + dj]
        m = (a >= 0) & (b >= 0)
        ca = np.broadcast_to(cost[r0:r1, c0:c1], a.shape)[m]
        cb = np.broadcast_to(cost[r0 + di:r1 + di, c0 + dj:c1 + dj], b.shape)[m]
        src.append(a[m])
        dst.append(b[m])
        w.append(math.hypot(di, dj) * res * 0.5 * (ca + cb))
    a, b = idx, np.roll(idx, -1, axis=0)
    m = (a >= 0) & (b >= 0)
    c3 = np.broadcast_to(cost, a.shape)[m]
    src.append(a[m])
    dst.append(b[m])
    w.append(turn_cost * c3)
    n = len(nodes)
    graph = coo_matrix((np.concatenate(w), (np.concatenate(src), np.concatenate(dst))),
                       shape=(n, n)).tocsr()
    d, pred = dijkstra(graph, directed=False, indices=idx[start], return_predecessors=True)
    out = np.full(fits.shape, np.inf)
    out.flat[nodes] = d
    prev = -np.ones(fits.shape, int)
    ok = pred >= 0
    prev.flat[nodes[ok]] = nodes[pred[ok]]
    return out, prev


def reported_objects(objects, reports, radius):
    """Which object each report found, one per report: the object nearest it within radius
    (m) that no other report took, or None (its object is gone from the list; it is still a
    victim found). Each report is one victim: other objects near it are not found victims
    (shadowed_objects says what they are). Counted as found too, the legs of a table a victim
    was under made four victims of three, and the search stopped with a third of the arena
    unseen.

    objects: dicts with id and center (x, y). reports: (x, y) each."""
    taken = []
    for r in reports:
        near = [(math.dist(o['center'], r), o['id']) for o in objects
                if o['id'] not in taken and math.dist(o['center'], r) < radius]
        taken.append(min(near)[1] if near else None)
    return taken


def shadowed_objects(objects, reports, radius):
    """The IDs of the objects within radius (m) of a report: none of them is another victim.
    Victim centers are over 1 m apart (rules), so at most one is within 0.5 m of a report, and
    the report already scores it. A box split in two by pose error left its second half at
    P 0.92 next to the report: N victims looked found, and the search stopped with a third of
    the arena unseen (v01, v02, v08 on the realistic sim)."""
    return {o['id'] for o in objects
            if any(math.dist(o['center'], r) < radius for r in reports)}


def expected_missing(n_total, ps):
    """E[victims not among the objects] = E[max(N - K, 0)], K the number of victims among the
    objects (independent, P(victim) each: a Poisson-binomial). N - sum(P) is 0 as soon as
    undecided objects add up to N, though some of them are most likely not victims."""
    dist = np.zeros(n_total + 1)
    dist[0] = 1.0
    for p in ps:  # the distribution of min(K, N)
        p = min(max(p, 0.0), 1.0)
        shifted = np.concatenate(([0.0], dist[:-1]))
        shifted[-1] += dist[-1] * p
        dist = dist * (1 - p) + shifted * p
    return float(sum((n_total - k) * dist[k] for k in range(n_total + 1)))


def path_lengths(passable, start, res, cost=None, predecessors=False):
    """Shortest 8-connected path length (m) from start to every cell of passable; a step
    into a cell costs its length times cost (default 1). predecessors: also return each cell's
    previous cell on its path (flat index, -1 for none). start a list of cells: one array per
    start, in one Dijkstra call (no predecessors)."""
    idx = -np.ones(passable.shape, int)
    cells = np.flatnonzero(passable)
    idx.flat[cells] = np.arange(len(cells))
    rows, cols = passable.shape
    src, dst, w = [], [], []
    for di, dj in ((0, 1), (1, 0), (1, 1), (1, -1)):
        r0, r1 = max(0, -di), rows - max(0, di)
        c0, c1 = max(0, -dj), cols - max(0, dj)
        a, b = idx[r0:r1, c0:c1], idx[r0 + di:r1 + di, c0 + dj:c1 + dj]
        m = (a >= 0) & (b >= 0)
        src.append(a[m])
        dst.append(b[m])
        step = math.hypot(di, dj) * res
        if cost is None:
            w.append(np.full(m.sum(), step))
        else:  # the mean of both cells' costs: the same in both directions
            ca, cb = cost[r0:r1, c0:c1][m], cost[r0 + di:r1 + di, c0 + dj:c1 + dj][m]
            w.append(step * 0.5 * (ca + cb))
    n = len(cells)
    graph = coo_matrix((np.concatenate(w), (np.concatenate(src), np.concatenate(dst))),
                       shape=(n, n)).tocsr()
    if isinstance(start, list):
        d = dijkstra(graph, directed=False, indices=[idx[c] for c in start])
        out = np.full((len(start),) + passable.shape, np.inf)
        out.reshape(len(start), -1)[:, cells] = d
        return out
    out = np.full(passable.shape, np.inf)
    if not predecessors:
        out.flat[cells] = dijkstra(graph, directed=False, indices=idx[start])
        return out
    d, pred = dijkstra(graph, directed=False, indices=idx[start], return_predecessors=True)
    out.flat[cells] = d
    prev = -np.ones(passable.shape, int)
    ok = pred >= 0
    prev.flat[cells[ok]] = cells[pred[ok]]
    return out, prev
