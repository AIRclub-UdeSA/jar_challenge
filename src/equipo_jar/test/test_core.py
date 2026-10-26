"""Chequeos del núcleo sin ROS ni simulador, sobre un laberinto sintético.

ESTO LO ESTOY HACIENDO CON CLAUDE PARA IR PROBANDO COSAS
"""
import math

import numpy as np

from equipo_jar.core import (DetectionClusterer, ParticleFilter, Board, initial_search, match_fraction, line_of_sight, normalize, viewpoint_tour, segments)

RES = 0.05


def maze():
    """6 x 5 m con perímetro y tres paredes internas; origen del mapa en (-1.5, -2.5)."""
    g = np.zeros((100, 120), np.int16)
    g[[0, -1], :] = 100
    g[:, [0, -1]] = 100
    g[30:31, 0:80] = 100     # pared horizontal desde la izquierda
    g[60:61, 40:120] = 100   # pared horizontal desde la derecha
    g[60:100, 30:31] = 100   # pared vertical arriba a la izquierda
    return Board(g.ravel(), 120, 100, RES, -1.5, -2.5)


def scan(grid, pose, n=90, max_range=8.0, noise=0.005, rng=None):
    """Puntos (B, 2) en el frame del robot, rayo a rayo hasta la primera pared."""
    rng = rng or np.random.default_rng(0)
    ang = np.linspace(-np.pi, np.pi, n, endpoint=False)
    r = np.arange(0.05, max_range, RES / 4)
    gx = pose[0] + np.outer(np.cos(ang + pose[2]), r)
    gy = pose[1] + np.outer(np.sin(ang + pose[2]), r)
    i, j = grid.to_cell(gx, gy)
    hit = ~grid.inside(i, j) | grid.occupied[np.clip(i, 0, 99), np.clip(j, 0, 119)]
    ranges = r[np.argmax(hit, axis=1)] + rng.normal(0, noise, n)
    return np.column_stack([ranges * np.cos(ang), ranges * np.sin(ang)])


def test_initial_search_finds_start():
    grid = maze()
    real = np.array([0.15, -0.12, 0.25])
    est = initial_search(grid, scan(grid, real))
    assert math.hypot(*(est[:2] - real[:2])) < 0.05
    assert abs(normalize(est[2] - real[2])) < 0.05
    assert match_fraction(grid, est, scan(grid, real)) > 0.9


def test_filter_tracks_robot_with_miscaled_odometry():
    grid, rng = maze(), np.random.default_rng(1)
    real = np.array([0.0, 0.0, 0.0])
    pf = ParticleFilter(grid, 300, seed=2)
    pf.initialize(real)
    # Avanza 2 m hacia +x y gira 90° en el lugar. La odometría informa el avance
    # un 10% largo (escala sin calibrar); el giro llega de la IMU con ruido.
    steps = [(0.05, 0.0)] * 40 + [(0.0, math.radians(5))] * 18 + [(0.05, 0.0)] * 20
    for forward, turn in steps:
        real[0] += forward * math.cos(real[2])
        real[1] += forward * math.sin(real[2])
        real[2] = normalize(real[2] + turn)
        pf.predict(1.10 * forward, 0.0, turn + rng.normal(0, 0.005))
        pf.correct(scan(grid, real, rng=rng))
    est, _ = pf.estimate()
    assert math.hypot(*(est[:2] - real[:2])) < 0.06
    assert abs(normalize(est[2] - real[2])) < 0.05


def test_tour_covers_maze():
    grid = maze()
    points, vis, drivable = viewpoint_tour(grid, (0.0, 0.0))
    assert 3 <= len(points) <= 25
    for x, y in points:
        i, j = grid.to_cell(x, y)
        assert drivable[i, j]
    seen = np.logical_or.reduce(vis)
    target = grid.free & (grid.dist > 0.10)
    assert seen[target].mean() > 0.9


def test_clusterer_uses_median():
    rng = np.random.default_rng(3)
    a = DetectionClusterer()
    for k in range(30):
        x, y = 1.0 + rng.normal(0, 0.03), 2.0 + rng.normal(0, 0.03)
        if k % 10 == 0:
            x += 0.25  # observación mala (el robot giraba)
        g = a.add(x, y, 0.01 * k, 0.0)
    assert len(a.groups) == 1
    cx, cy = a.center(g)
    assert math.hypot(cx - 1.0, cy - 2.0) < 0.03
    assert a.ready(g)
    a.add(3.0, 3.0, 0.0, 0.0)
    assert len(a.groups) == 2 and not a.ready(a.groups[1])


def test_segments_split_on_jumps_and_gaps():
    r = np.array([1.0, 1.02, 1.04, 1.5, 1.52, 1.54, 1.56, np.inf, 2.0, 2.01])
    assert segments(r, 0.1, 3) == [[0, 1, 2], [3, 4, 5, 6]]


def test_line_of_sight():
    grid = maze()
    assert line_of_sight(grid, (0.0, 0.0), (3.0, 0.0))
    assert not line_of_sight(grid, (0.0, 0.0), (0.0, -2.0))  # pared en y = -1.0
