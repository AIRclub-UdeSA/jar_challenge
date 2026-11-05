"""Start pose from one LiDAR sweep matched against /map (no ROS here).

The robot is placed by hand near the origin, with no exact position and no known heading
(§5). AMCL alone only fixes a small error: its particles start around the guess, and with
the robot still they never spread. If the guess is 20° off, AMCL stays ~5 cm / 3° off
until the robot moves, and the start pose mission saves for the return is that far off.
With a heading error over ~45° AMCL never converges.

This class searches every pose within `radio_m` of the guess and every heading, and
scores each by how many scan points land on a wall of the map. The best pose goes to
AMCL as its initial pose with a small spread. Boxes and decoys are not in /map: their
points simply score 0 at every pose.

Search: a coarse grid (`paso_m`, `paso_yaw_rad`), then a fine grid around the best
coarse poses. A pose whose heading is far from the guess wins only if it scores clearly
better (`margen_otro_rumbo`) than the best pose near the guessed heading: in a
symmetric spot the guess decides.
"""
import math

import numpy as np

OCUPADA = 65          # OccupancyGrid value from which a cell is a wall (occupied_thresh 0.65)


def campo_distancias(celdas, ancho, alto, resolucion, tope_m):
    """Distance (m) from each cell centre to the nearest wall cell, capped at tope_m.

    `celdas`: OccupancyGrid data (row 0 = lowest y). Exact within the cap: it tries every
    offset inside a disk of radius tope_m.
    """
    ocupada = np.asarray(celdas, dtype=np.int16).reshape(alto, ancho) >= OCUPADA
    r = int(math.ceil(tope_m / resolucion))
    campo = np.full((alto, ancho), tope_m, dtype=np.float32)
    grande = np.zeros((alto + 2 * r, ancho + 2 * r), dtype=bool)
    grande[r:r + alto, r:r + ancho] = ocupada
    for di in range(-r, r + 1):
        for dj in range(-r, r + 1):
            d = math.hypot(di, dj) * resolucion
            if d >= tope_m:
                continue
            vecina = grande[r + di:r + di + alto, r + dj:r + dj + ancho]
            np.minimum(campo, np.where(vecina, d, tope_m), out=campo)
    return campo


class PoseInicial:
    """Built once from /map; `buscar(puntos, guess)` returns the best pose.

    `p`: dict with radio_m, paso_m, paso_yaw_rad, sigma_m, margen_otro_rumbo,
    rumbo_cercano_rad (see localizacion's params).
    """

    def __init__(self, celdas, ancho, alto, resolucion, origen_x, origen_y, p):
        self.p = p
        self.res = resolucion
        self.ox, self.oy = origen_x, origen_y
        self.ancho, self.alto = ancho, alto
        self.tope = 4.0 * p['sigma_m']
        self.campo = campo_distancias(celdas, ancho, alto, resolucion, self.tope)

    def distancia(self, x, y):
        """Bilinear distance to the nearest wall at map points; outside the map = cap.

        Cell (i, j) is taken as centred at origin + (i, j) * resolution, as Nav2 Humble's
        AMCL reads the map (not + 0.5 cell as in costmap_2d). Otherwise this match and AMCL
        disagree by half a cell (2.5 cm in x and y) and the fine-tune fights AMCL. In the
        simulator AMCL's reading is also the closer one to the world (~1.6 vs ~3.4 cm).
        """
        u = (x - self.ox) / self.res
        v = (y - self.oy) / self.res
        i0 = np.floor(u).astype(np.int64)
        j0 = np.floor(v).astype(np.int64)
        fu, fv = u - i0, v - j0
        dentro = (i0 >= 0) & (j0 >= 0) & (i0 < self.ancho - 1) & (j0 < self.alto - 1)
        i0 = np.clip(i0, 0, self.ancho - 2)
        j0 = np.clip(j0, 0, self.alto - 2)
        c = self.campo
        d = (c[j0, i0] * (1 - fu) * (1 - fv) + c[j0, i0 + 1] * fu * (1 - fv)
             + c[j0 + 1, i0] * (1 - fu) * fv + c[j0 + 1, i0 + 1] * fu * fv)
        return np.where(dentro, d, self.tope)

    def puntajes(self, puntos, xs, ys, yaws):
        """Mean score of the points at each pose (xs, ys, yaws same length). 1 = all on walls."""
        k = 1.0 / (2.0 * self.p['sigma_m'] ** 2)
        salida = np.empty(len(xs))
        bloque = max(1, 200000 // max(1, len(puntos)))
        for a in range(0, len(xs), bloque):
            b = slice(a, a + bloque)
            c, s = np.cos(yaws[b])[:, None], np.sin(yaws[b])[:, None]
            mx = xs[b][:, None] + c * puntos[None, :, 0] - s * puntos[None, :, 1]
            my = ys[b][:, None] + s * puntos[None, :, 0] + c * puntos[None, :, 1]
            d = self.distancia(mx, my)
            salida[b] = np.exp(-k * d * d).mean(axis=1)
        return salida

    def buscar(self, puntos, x0, y0, yaw0, radio_m=None, rango_yaw_rad=math.pi):
        """Best pose for `puntos` (N x 2, in base_footprint, metres) near (x0, y0, yaw0).

        Searches positions within `radio_m` (default p['radio_m']) and headings within
        `rango_yaw_rad` of yaw0 (default: all). Returns (x, y, yaw, score, score_at_guess).
        Score: mean point score, 0..1.
        """
        p = self.p
        radio = p['radio_m'] if radio_m is None else radio_m
        puntos = np.asarray(puntos, dtype=float)
        # Coarse grid: positions in a disk, headings in the range.
        n = int(math.floor(radio / p['paso_m']))
        g = np.arange(-n, n + 1) * p['paso_m']
        gx, gy = np.meshgrid(g, g)
        en_disco = np.hypot(gx, gy) <= radio + 1e-9
        px, py = x0 + gx[en_disco], y0 + gy[en_disco]
        if rango_yaw_rad >= math.pi:
            nyaw = max(1, int(round(2 * math.pi / p['paso_yaw_rad'])))
            yaws = yaw0 + np.arange(nyaw) * (2 * math.pi / nyaw)
        else:
            m = int(math.floor(rango_yaw_rad / p['paso_yaw_rad']))
            yaws = yaw0 + np.arange(-m, m + 1) * p['paso_yaw_rad']
            nyaw = len(yaws)
        X = np.repeat(px, nyaw)
        Y = np.repeat(py, nyaw)
        T = np.tile(yaws, len(px))
        gruesos = self.puntajes(puntos, X, Y, T)
        # Refine the best coarse poses: near the guessed heading and anywhere.
        cerca = np.abs(_envolver(T - yaw0)) <= p['rumbo_cercano_rad']
        candidatos = []
        for mascara in (cerca, np.ones_like(cerca)):
            orden = np.argsort(np.where(mascara, gruesos, -1.0))[::-1][:4]
            candidatos += [(X[i], Y[i], T[i]) for i in orden if mascara[i]]
        finos = [self._refinar(puntos, *c) for c in candidatos]
        en_rumbo = [f for f in finos if abs(_envolver(f[2] - yaw0)) <= p['rumbo_cercano_rad']]
        mejor = max(finos, key=lambda f: f[3])
        if en_rumbo:
            local = max(en_rumbo, key=lambda f: f[3])
            if mejor[3] < local[3] * (1.0 + p['margen_otro_rumbo']):
                mejor = local
        en_guess = self.puntajes(puntos, np.array([x0]), np.array([y0]), np.array([yaw0]))[0]
        return mejor[0], mejor[1], _envolver(mejor[2]), mejor[3], en_guess

    def nitidez(self, puntos, x, y, yaw, paso_m=0.05, paso_yaw_rad=0.035):
        """Smallest score drop when the pose moves paso_m along x or y, or turns paso_yaw_rad.

        Near 0: the fit slides (e.g. a corridor with no wall across it in view), so the
        pose along that direction is not known from this sweep.
        """
        d, a = paso_m, paso_yaw_rad
        xs = np.array([x, x + d, x - d, x, x, x, x])
        ys = np.array([y, y, y, y + d, y - d, y, y])
        ts = np.array([yaw, yaw, yaw, yaw, yaw, yaw + a, yaw - a])
        s = self.puntajes(np.asarray(puntos, dtype=float), xs, ys, ts)
        return float(s[0] - s[1:].max())

    def _refinar(self, puntos, x, y, yaw):
        """Two finer grids around (x, y, yaw); returns (x, y, yaw, score)."""
        paso, paso_yaw = self.p['paso_m'], self.p['paso_yaw_rad']
        mejor = (x, y, yaw, -1.0)
        for _ in range(2):
            d = np.linspace(-paso, paso, 7)
            dy = np.linspace(-paso_yaw, paso_yaw, 9)
            gx, gy, gt = np.meshgrid(d, d, dy, indexing='ij')
            X, Y, T = x + gx.ravel(), y + gy.ravel(), yaw + gt.ravel()
            s = self.puntajes(puntos, X, Y, T)
            i = int(np.argmax(s))
            if s[i] > mejor[3]:
                mejor = (X[i], Y[i], T[i], s[i])
            x, y, yaw = mejor[:3]
            paso, paso_yaw = paso / 3.0, paso_yaw / 4.0
        return mejor


def puntos_del_scan(rangos, angulo_min, incremento, laser_x, laser_y, laser_yaw,
                    rango_min, rango_max, max_puntos):
    """Scan rays to points in base_footprint (laser pose given there). Skips NaN/inf/out of range."""
    r = np.asarray(rangos, dtype=float)
    a = angulo_min + incremento * np.arange(len(r))
    ok = np.isfinite(r) & (r >= rango_min) & (r <= rango_max)
    r, a = r[ok], a[ok]
    if len(r) > max_puntos:
        cada = int(math.ceil(len(r) / max_puntos))
        r, a = r[::cada], a[::cada]
    ang = laser_yaw + a
    return np.column_stack([laser_x + r * np.cos(ang), laser_y + r * np.sin(ang)])


def _envolver(a):
    return (np.asarray(a) + math.pi) % (2 * math.pi) - math.pi
