"""Start pose from a LiDAR sweep matched against the map."""
import math

import numpy as np

from equipo_jar.pose_inicial import PoseInicial, campo_distancias, puntos_del_scan

P = {'radio_m': 0.5, 'paso_m': 0.05, 'paso_yaw_rad': math.radians(2.0), 'sigma_m': 0.05,
     'rumbo_cercano_rad': 0.8, 'margen_otro_rumbo': 0.10}
RES = 0.05
ORIGEN = (-1.0, -1.0)
LASER = (0.0435, 0.0, math.pi)     # the X3's LiDAR looks backwards


def mapa():
    """4 x 3 m room with an inner wall and a pillar: no symmetry. Row 0 = lowest y."""
    ancho, alto = 80, 60
    g = np.zeros((alto, ancho), dtype=np.int8)
    g[0, :] = g[-1, :] = 100
    g[:, 0] = g[:, -1] = 100
    g[0:35, 40] = 100            # inner wall from the bottom, at x = 1.0
    g[40:46, 15:21] = 100        # pillar
    return g


def scan(g, x, y, yaw, n=1080, caja=None):
    """Rays from the LiDAR at the robot pose, marching 1 cm; a box (x, y, half side) adds a wall."""
    lx = x + math.cos(yaw) * LASER[0] - math.sin(yaw) * LASER[1]
    ly = y + math.sin(yaw) * LASER[0] + math.cos(yaw) * LASER[1]
    a = -math.pi + np.arange(n) * (2 * math.pi / n)
    pasos = np.arange(5, 800) * 0.01
    px = lx + pasos[None, :] * np.cos(yaw + LASER[2] + a)[:, None]
    py = ly + pasos[None, :] * np.sin(yaw + LASER[2] + a)[:, None]
    # Cell i is centred at ORIGEN + i * RES, as AMCL (and PoseInicial) read the map.
    i = np.floor((px - ORIGEN[0]) / RES + 0.5).astype(int)
    j = np.floor((py - ORIGEN[1]) / RES + 0.5).astype(int)
    ok = (i >= 0) & (j >= 0) & (i < g.shape[1]) & (j < g.shape[0])
    golpe = np.zeros_like(ok)
    golpe[ok] = g[j[ok], i[ok]] >= 65
    if caja is not None:
        golpe |= (np.abs(px - caja[0]) <= caja[2]) & (np.abs(py - caja[1]) <= caja[2])
    primero = np.where(golpe.any(1), golpe.argmax(1), -1)
    rangos = np.where(primero >= 0, pasos[np.maximum(primero, 0)], np.inf)
    rangos[::4] = np.nan         # the real LiDAR returns ~25 % non-finite rays
    return puntos_del_scan(rangos, -math.pi, 2 * math.pi / n, *LASER, 0.10, 10.0, 360)


def buscador():
    g = mapa()
    return g, PoseInicial(g.ravel(), g.shape[1], g.shape[0], RES, *ORIGEN, P)


def error(res, real):
    return (math.hypot(res[0] - real[0], res[1] - real[1]),
            abs(math.atan2(math.sin(res[2] - real[2]), math.cos(res[2] - real[2]))))


def test_distance_field_is_exact_near_walls_and_capped():
    g = np.zeros((10, 10), dtype=np.int8)
    g[5, 5] = 100
    campo = campo_distancias(g.ravel(), 10, 10, RES, 0.2)
    assert campo[5, 5] == 0.0
    assert math.isclose(campo[5, 7], 0.10, abs_tol=1e-6)
    assert math.isclose(campo[6, 6], math.hypot(RES, RES), abs_tol=1e-6)
    assert campo[0, 0] == np.float32(0.2)


def test_finds_a_robot_placed_off_the_guess():
    g, b = buscador()
    real = (0.18, -0.12, 0.35)
    res = b.buscar(scan(g, *real), 0.0, 0.0, 0.0)
    pos, rumbo = error(res, real)
    assert pos < 0.015 and rumbo < math.radians(2)
    assert res[3] > 0.8 and res[4] < res[3]


def test_finds_any_heading():
    g, b = buscador()
    for real in [(0.1, 0.1, math.pi), (-0.2, 0.05, -2.0), (0.0, -0.2, 1.6)]:
        pos, rumbo = error(b.buscar(scan(g, *real), 0.0, 0.0, 0.0), real)
        assert pos < 0.015 and rumbo < math.radians(2), real


def test_a_box_missing_from_the_map_does_not_fool_it():
    g, b = buscador()
    real = (0.1, -0.1, 0.2)
    pos, rumbo = error(b.buscar(scan(g, *real, caja=(0.6, -0.1, 0.125)), 0.0, 0.0, 0.0), real)
    assert pos < 0.015 and rumbo < math.radians(2)


def test_in_a_symmetric_spot_the_guessed_heading_wins():
    # A plain rectangular room: a pose and the same pose turned 180° fit equally well.
    g = np.zeros((60, 80), dtype=np.int8)
    g[0, :] = g[-1, :] = 100
    g[:, 0] = g[:, -1] = 100
    origen = (-2.0, -1.5)        # the room is centred on (0, 0)
    b = PoseInicial(g.ravel(), 80, 60, RES, *origen, P)
    global ORIGEN
    viejo, ORIGEN = ORIGEN, origen
    try:
        puntos = scan(g, 0.0, 0.0, 0.1)
    finally:
        ORIGEN = viejo
    res = b.buscar(puntos, 0.0, 0.0, 0.0)
    assert abs(res[2] - 0.1) < math.radians(3)


def test_scan_points_skip_bad_rays_and_use_the_laser_pose():
    pts = puntos_del_scan([1.0, float('nan'), float('inf'), 0.01, 2.0], 0.0, math.pi / 2,
                          0.1, 0.0, math.pi, 0.05, 10.0, 100)
    assert pts.shape == (2, 2)
    assert np.allclose(pts[0], [0.1 - 1.0, 0.0], atol=1e-9)        # ray 0 points backwards
    assert np.allclose(pts[1], [0.1 - 2.0, 0.0], atol=1e-9)        # ray 4: 2 pi, backwards again


def test_sharpness_is_high_in_a_corner_and_low_in_a_plain_corridor():
    g, b = buscador()
    real = (0.18, -0.12, 0.35)
    assert b.nitidez(scan(g, *real), *real) > 0.1
    # Corridor 0.8 m wide along x, closed far away: sliding along x barely changes the fit.
    c = np.zeros((20, 400), dtype=np.int8)
    c[2, :] = c[18, :] = 100
    pasillo = PoseInicial(c.ravel(), 400, 20, RES, -10.0, -0.5, P)
    global ORIGEN
    viejo, ORIGEN = ORIGEN, (-10.0, -0.5)
    try:
        puntos = scan(c, 0.0, 0.0, 0.0)
    finally:
        ORIGEN = viejo
    assert pasillo.nitidez(puntos, 0.0, 0.0, 0.0) < 0.03
