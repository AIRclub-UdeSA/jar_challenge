"""Scan matcher on a synthetic maze: rays are cast against exact rectangles and the map
is their 5 cm rasterization, so map quantization shows up as it does with /map."""
import math

import numpy as np
import pytest

from equipo_jar.scan_matcher import (DistanceField, Localizer, compose, correlative_search,
                                     match)

RES, ORIGIN, SIZE = 0.05, (-1.0, -1.0), (5.0, 4.0)
# Walls as (xmin, ymin, xmax, ymax): an outer box, two inner walls, and a box that is
# not on the map (a victim) for the robust loss to ignore.
WALLS = [(-1.0, -1.0, 4.0, -0.85), (-1.0, 2.85, 4.0, 3.0), (-1.0, -1.0, -0.85, 3.0),
         (3.85, -1.0, 4.0, 3.0), (1.0, -0.85, 1.15, 1.5), (2.2, 1.2, 3.85, 1.35)]
UNMAPPED = [(0.3, 1.8, 0.55, 2.05)]


def rasterize(rects):
    rows, cols = int(SIZE[1] / RES), int(SIZE[0] / RES)
    cy = ORIGIN[1] + (np.arange(rows) + 0.5) * RES
    cx = ORIGIN[0] + (np.arange(cols) + 0.5) * RES
    X, Y = np.meshgrid(cx, cy)
    occ = np.zeros((rows, cols), bool)
    for x0, y0, x1, y1 in rects:
        occ |= (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)
    return occ


def cast(pose, n=720, rng=None, noise=0.01):
    """Ranges from pose (laser at the base origin) to the nearest rectangle face."""
    a = np.linspace(-math.pi, math.pi, n, endpoint=False)
    d = np.column_stack([np.cos(a + pose[2]), np.sin(a + pose[2])])
    best = np.full(n, np.inf)
    for x0, y0, x1, y1 in WALLS + UNMAPPED:
        with np.errstate(divide='ignore', invalid='ignore'):
            tx = np.stack([(x0 - pose[0]) / d[:, 0], (x1 - pose[0]) / d[:, 0]])
            ty = np.stack([(y0 - pose[1]) / d[:, 1], (y1 - pose[1]) / d[:, 1]])
        tmin = np.maximum(np.nanmin(tx, 0), np.nanmin(ty, 0))
        tmax = np.minimum(np.nanmax(tx, 0), np.nanmax(ty, 0))
        hit = (tmax >= tmin) & (tmin > 0)
        best = np.where(hit & (tmin < best), tmin, best)
    if rng is not None:
        best = best + rng.normal(0, noise, n)
    return best, a[0], a[1] - a[0]


@pytest.fixture(scope='module')
def field():
    return DistanceField(rasterize(WALLS), RES, ORIGIN)


def points(pose, rng):
    r, a0, da = cast(pose, rng=rng)
    a = a0 + np.arange(len(r)) * da
    ok = np.isfinite(r)
    return np.column_stack([r[ok] * np.cos(a[ok]), r[ok] * np.sin(a[ok])])


def test_distance_field_zero_on_wall_surface(field):
    d, g = field.lookup(np.array([[1.0, 0.5], [0.9, 0.5], [1.075, 0.5]]))
    assert abs(d[0]) < 0.01          # the wall face at x = 1.0
    assert abs(d[1] - 0.10) < 0.01   # 10 cm in front of it
    assert d[2] < 0                  # inside the wall
    assert g[1, 0] < -0.9            # distance shrinks towards the wall (+x)


def test_match_recovers_offset(field):
    rng = np.random.default_rng(0)
    truth = np.array([0.4, 0.6, 0.3])
    pts = points(truth, rng)
    pose, cov, inliers, _ = match(field, pts, truth + [0.12, -0.08, 0.08], gate_start=0.4)
    assert np.hypot(*(pose[:2] - truth[:2])) < 0.015
    assert abs(pose[2] - truth[2]) < math.radians(0.5)
    assert 0.8 < inliers.mean() < 1.0  # the unmapped box is an outlier
    assert np.all(np.linalg.eigvalsh(cov) > 0)


def test_correlative_search_finds_start_any_heading(field):
    rng = np.random.default_rng(1)
    truth = np.array([0.25, -0.2, 2.5])
    pts = points(truth, rng)
    (best, _), *_ = correlative_search(field, pts, np.zeros(3), 0.5, math.pi)
    assert np.hypot(*(best[:2] - truth[:2])) < 0.08
    assert abs(math.remainder(best[2] - truth[2], 2 * math.pi)) < math.radians(3)


def test_localizer_tracks_drifting_odometry():
    rng = np.random.default_rng(2)
    loc = Localizer(init_x=0.0, init_y=0.0, deskew=False, use_gyro=False)
    loc.set_map(rasterize(WALLS), RES, ORIGIN)
    # Up the corridor left of the inner wall, then right above it, turning to face along.
    corners = np.array([[0.1, 0.1], [0.4, 2.3], [3.3, 2.3], [3.3, 1.8]])
    path = np.concatenate([np.linspace(a, b, int(np.hypot(*(b - a)) / 0.06) + 1)[:-1]
                           for a, b in zip(corners[:-1], corners[1:])])
    heading = np.unwrap(np.arctan2(*np.diff(path, axis=0).T[::-1]))
    truth = np.array([*path[0], heading[0]])
    odom, errors = np.zeros(3), []
    for k in range(1, len(path)):
        t = 0.2 * k
        target = np.array([*path[k], heading[min(k, len(heading) - 1)]])
        c, s = math.cos(truth[2]), math.sin(truth[2])
        step = np.array([*(np.array([[c, s], [-s, c]]) @ (target[:2] - truth[:2])),
                         math.remainder(target[2] - truth[2], 2 * math.pi)])
        truth = compose(truth, step)
        # Odometry overestimates travel by 8% and drifts in heading.
        odom = compose(odom, step * [1.08, 1.0, 1.0] + [0.0, 0.004, 0.01])
        loc.add_odom(t, odom)
        r, a0, da = cast(truth, rng=rng)
        res = loc.process_scan(t, r, a0, da, 0.05, 12.0)
        assert res is not None
        errors.append(np.hypot(*(res['pose'][:2] - truth[:2])))
    assert max(errors[1:]) < 0.03
    assert np.mean(errors) < 0.015
