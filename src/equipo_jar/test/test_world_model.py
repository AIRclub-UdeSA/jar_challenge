"""World model on a synthetic room. Lidar rays and camera pixels are cast against exact boxes;
the map is the walls' 5 cm rasterization, with one wall missing and one that isn't there."""
import math

import numpy as np
import pytest

from equipo_jar.world_model import Objeto, WorldModel, arena_mask, pixel_classes, red_mask

RES = 0.05
SIZE = (4.0, 3.0)
# (xmin, ymin, xmax, ymax, height, rgb)
GREY, RED, BLUE = (180, 170, 150), (200, 30, 30), (30, 60, 200)
WALLS = [(0.0, 0.0, 4.0, 0.1, 0.5, GREY), (0.0, 2.9, 4.0, 3.0, 0.5, GREY),
         (0.0, 0.0, 0.1, 3.0, 0.5, GREY), (3.9, 0.0, 4.0, 3.0, 0.5, GREY)]
MISSING = (2.6, 0.55, 3.6, 0.6, 0.5, GREY)      # in the room, not on the map
GHOST = (1.9, 2.2, 2.0, 2.9)                    # on the map, not in the room
RED_BOX = (2.375, 1.375, 2.625, 1.625, 0.3, RED)
BLUE_BOX = (1.075, 1.775, 1.325, 2.025, 0.3, BLUE)
SOLIDS = WALLS + [MISSING, RED_BOX, BLUE_BOX]
SPOTS = [(x, y) for x in (0.5, 1.2, 1.9, 2.6, 3.3) for y in (0.35, 0.95, 1.6, 2.4)
         if all(not (b[0] - 0.3 < x < b[2] + 0.3 and b[1] - 0.3 < y < b[3] + 0.3)
                for b in (RED_BOX, BLUE_BOX))]


def map_data():
    rows, cols = int(SIZE[1] / RES), int(SIZE[0] / RES)
    X, Y = np.meshgrid((np.arange(cols) + 0.5) * RES, (np.arange(rows) + 0.5) * RES)
    occ = np.zeros((rows, cols), bool)
    for x0, y0, x1, y1, *_ in WALLS + [GHOST]:
        occ |= (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)
    return np.where(occ, 100, 0).astype(np.int8).ravel(), cols, rows


def model(**params):
    data, w, h = map_data()
    return WorldModel(data, w, h, RES, (0.0, 0.0), **params)


def slab(origin, d, box):
    """Distance along rays d (N, 3) from origin to an axis-aligned box, inf if missed."""
    lo = np.array([box[0], box[1], 0.0])
    hi = np.array([box[2], box[3], box[4]])
    with np.errstate(divide='ignore', invalid='ignore'):
        t1, t2 = (lo - origin) / d, (hi - origin) / d
    t1, t2 = np.nan_to_num(t1, nan=-np.inf), np.nan_to_num(t2, nan=np.inf)
    tmin = np.minimum(t1, t2).max(1)
    tmax = np.maximum(t1, t2).min(1)
    return np.where((tmax >= tmin) & (tmin > 0), tmin, np.inf)


def scan(sensor, rng, n=720):
    a = np.linspace(-math.pi, math.pi, n, endpoint=False)
    d = np.column_stack([np.cos(a), np.sin(a), np.zeros(n)])
    o = np.array([sensor[0], sensor[1], 0.18])
    r = np.min([slab(o, d, b) for b in SOLIDS], axis=0)
    ok = np.isfinite(r)
    r = r[ok] + rng.normal(0, 0.01, ok.sum())
    return np.column_stack([sensor[0] + r * np.cos(a[ok]), sensor[1] + r * np.sin(a[ok])])


def camera(x, y, yaw, z=0.109):
    """Optical frame (x right, y down, z forward) to map, for a level camera."""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([x, y, z]), np.column_stack([(s, -c, 0.0), (0.0, 0.0, -1.0), (c, s, 0.0)])


def render(cam, R, K, w=160, h=120):
    fx, fy, cx, cy = K
    U, V = np.meshgrid(np.arange(w), np.arange(h))
    d = (np.stack([(U - cx) / fx, (V - cy) / fy, np.ones(U.shape)], -1) @ R.T).reshape(-1, 3)
    img = np.full((h * w, 3), 110, np.uint8)  # floor
    best = np.where(d[:, 2] < 0, cam[2] / np.maximum(-d[:, 2], 1e-9), np.inf)
    for b in SOLIDS:
        t = slab(cam, d, b)
        closer = t < best
        img[closer] = b[5]
        best = np.where(closer, t, best)
    return img.reshape(h, w, 3)


@pytest.fixture(scope='module')
def scanned():
    rng = np.random.default_rng(1)
    wm = model()
    for _ in range(3):
        for sp in SPOTS:
            wm.add_scan(sp, scan(sp, rng))
    wm.update()
    return wm


def nearest(objects, xy):
    return min(objects, key=lambda o: math.hypot(*(o.centro - xy)))


def test_red_mask():
    px = np.array([[RED, BLUE, GREY, (250, 250, 250), (90, 10, 10), (200, 120, 30)]], np.uint8)
    assert red_mask(px, 10, 170, 120, 15).tolist() == [[True, False, False, False, True, False]]


def test_pixel_classes_red_in_shadow_dark_unknown():
    # A red face in shadow and in a dim room is still red; a pixel too dark to show a red
    # surface's chroma, or clipped white, is unknown; a lit grey wall is informative not-red.
    px = np.array([[(64, 10, 10), (28, 2, 3), (10, 3, 3), (255, 252, 250), GREY, BLUE]],
                  np.uint8)
    red, info = pixel_classes(px, 10, 170, 120, 15)
    assert red.tolist() == [[True, True, False, False, False, False]]
    assert info.tolist() == [[True, True, False, False, True, True]]


def test_arena_is_inside_the_walls():
    data, w, h = map_data()
    a = arena_mask(data.reshape(h, w) == 100)
    assert a[30, 40] and not a[0, 40] and not a[30, 0]


def test_boxes_become_objects_with_accurate_fits(scanned):
    for box in (RED_BOX, BLUE_BOX):
        center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
        o = nearest(scanned.objects, center)
        assert math.hypot(*(o.centro - center)) < 0.03
        assert abs(o.lados[0] - 0.25) < 0.04 and abs(o.lados[1] - 0.25) < 0.04
        assert o.forma > 0.5
        assert o.miradores


def test_map_errors_are_corrected(scanned):
    new, removed = scanned.change()
    gi, gok = scanned.index(np.array([1.95]), np.array([2.5]))
    assert removed.ravel()[gi[0]]  # the wall the map draws but the rays cross
    wall = nearest(scanned.objects, np.array([3.1, 0.575]))
    assert math.hypot(*(wall.centro - (3.1, 0.575))) < 0.4  # the wall the map is missing
    assert wall.forma < 0.5
    assert len(scanned.objects) == 3


def test_search_belief_shrinks_and_excludes_objects(scanned):
    assert scanned.remaining() < 0.1
    b = scanned.belief()
    i, j = scanned._cell_rc((2.5, 1.5))
    assert b[i, j] == 0.0  # explained by the red box


def test_color_confirms_red_and_rejects_blue(scanned):
    wm = scanned
    K = (135.9, 135.9, 80.0, 60.0)
    # Distinct lines of sight: views along nearly the same line count once per cell.
    views = [(1.2, 0.6), (1.7, 0.4), (3.3, 2.4), (2.7, 2.6), (3.3, 0.9), (0.6, 1.0),
             (0.5, 2.6), (1.0, 2.7), (2.0, 1.0), (3.5, 1.5), (1.5, 1.5)]
    for x, y in views:
        for target in (RED_BOX, BLUE_BOX):
            tx, ty = (target[0] + target[2]) / 2, (target[1] + target[3]) / 2
            cam, R = camera(x, y, math.atan2(ty - y, tx - x))
            wm.add_image(cam, R, K, render(cam, R, K))
    wm.update()
    red = nearest(wm.objects, np.array([2.5, 1.5]))
    blue = nearest(wm.objects, np.array([1.2, 1.9]))
    assert red.p_rojo > 0.97 and red.p_objetivo > 0.97
    assert blue.p_rojo < 0.05 and blue.p_objetivo < 0.05


def test_known_count_closes_the_search():
    rng = np.random.default_rng(2)
    wm = model(objetivos_n=1)
    for sp in SPOTS[:6]:
        wm.add_scan(sp, scan(sp, rng))
    wm.update()   # the objects the images' color goes to (the node updates at 2 Hz)
    K = (135.9, 135.9, 80.0, 60.0)
    for x, y in [(1.9, 0.95), (1.9, 1.6), (2.0, 2.4), (3.3, 2.4)]:
        cam, R = camera(x, y, math.atan2(1.5 - y, 2.5 - x))
        wm.add_image(cam, R, K, render(cam, R, K))
    wm.update()
    assert wm.complete
    assert wm.remaining() == 0.0
    assert sum(o.p_objetivo > 0 for o in wm.objects) == 1


def test_repeated_frames_from_one_spot_count_once(scanned):
    wm = model()
    K = (135.9, 135.9, 80.0, 60.0)
    cam, R = camera(1.9, 0.95, 0.5)
    img = render(cam, R, K)
    assert wm.add_image(cam, R, K, img)
    assert not wm.add_image(cam, R, K, img)


# A red box seen through a 6 cm slit between two pillars, which the map draws closed (as a SLAM
# map draws a row of pillars): the grid ray stops at the pillars, the camera doesn't.
PILLARS = [(1.9, 1.0, 2.0, 1.47, 0.5, GREY), (1.9, 1.53, 2.0, 2.0, 0.5, GREY)]
SLIT_SOLIDS = WALLS + PILLARS + [RED_BOX]


def slit_model(**params):
    rows, cols = int(SIZE[1] / RES), int(SIZE[0] / RES)
    X, Y = np.meshgrid((np.arange(cols) + 0.5) * RES, (np.arange(rows) + 0.5) * RES)
    occ = np.zeros((rows, cols), bool)
    for x0, y0, x1, y1, *_ in WALLS + [(1.9, 1.0, 2.0, 2.0)]:
        occ |= (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)
    return WorldModel(np.where(occ, 100, 0).astype(np.int8).ravel(), cols, rows, RES,
                      (0.0, 0.0), **params)


def render_with(cam, R, K, solids, w=160, h=120):
    """The image, and the depth cloud of the same pixels: map-frame points and their color."""
    fx, fy, cx, cy = K
    U, V = np.meshgrid(np.arange(w), np.arange(h))
    d = (np.stack([(U - cx) / fx, (V - cy) / fy, np.ones(U.shape)], -1) @ R.T).reshape(-1, 3)
    img = np.full((h * w, 3), 110, np.uint8)
    best = np.where(d[:, 2] < 0, cam[2] / np.maximum(-d[:, 2], 1e-9), np.inf)
    for b in solids:
        t = slab(cam, d, b)
        closer = t < best
        img[closer] = b[5]
        best = np.where(closer, t, best)
    ok = np.isfinite(best)
    return img.reshape(h, w, 3), (cam + d[ok] * best[ok, None], img[ok])


def color_at(wm, xy):
    i, ok = wm.index(np.array([xy[0]]), np.array([xy[1]]))
    return wm.Lc[i[0]]


def test_red_through_a_slit_goes_on_the_pillars_without_depth():
    wm = slit_model()
    K = (135.9, 135.9, 80.0, 60.0)
    cam, R = camera(1.5, 1.5, 0.0)
    img, _ = render_with(cam, R, K, SLIT_SOLIDS)
    wm.add_image(cam, R, K, img)
    assert color_at(wm, (1.925, 1.5)) > 0     # the bug: the pillar gets the box's red
    assert color_at(wm, (2.37, 1.5)) == 0


def test_depth_puts_the_red_on_the_box_behind_the_slit():
    wm = slit_model()
    K = (135.9, 135.9, 80.0, 60.0)
    cam, R = camera(1.5, 1.5, 0.0)
    img, cloud = render_with(cam, R, K, SLIT_SOLIDS)
    assert wm.count_red(img) > 0
    wm.add_image(cam, R, K, img, cloud)
    assert color_at(wm, (2.37, 1.5)) > 0      # the box's near face
    assert color_at(wm, (1.925, 1.5)) <= 0
    for y in np.arange(1.05, 1.95, RES):       # no pillar cell turned red
        assert color_at(wm, (1.925, y)) <= 0


def test_turning_in_place_counts_a_cell_once():
    rng = np.random.default_rng(3)
    wm = model()
    for sp in SPOTS:
        wm.add_scan(sp, scan(sp, rng))
    K = (135.9, 135.9, 80.0, 60.0)
    box = (2.37, 1.5)                           # the red box's near face, from (1.4, 1.5)
    for yaw in (0.0, 0.25, -0.25, 0.0):        # each frame passes the per-frame turn gate
        cam, R = camera(1.4, 1.5, yaw)
        assert wm.add_image(cam, R, K, render(cam, R, K))
    once = color_at(wm, box)
    assert 0 < once <= wm.p['vista_logodds_max']
    # A different line of sight counts again.
    cam, R = camera(1.4, 1.1, math.atan2(0.4, 1.0))
    wm.add_image(cam, R, K, render(cam, R, K))
    assert color_at(wm, box) > once


def test_a_frame_close_to_the_last_still_gives_new_cells_color():
    # The end of a turn: a frame with the red box just outside the image, then one 7 deg
    # further where it comes in. The second frame is too close to cut the search belief again,
    # but its color is the first view of the box's cells.
    rng = np.random.default_rng(4)
    wm = model()
    for sp in SPOTS:
        wm.add_scan(sp, scan(sp, rng))
    K = (135.9, 135.9, 80.0, 60.0)
    cam, R = camera(1.4, 1.5, math.radians(38))
    assert wm.add_image(cam, R, K, render(cam, R, K))
    assert color_at(wm, (2.37, 1.55)) == 0
    cam, R = camera(1.4, 1.5, math.radians(31))
    assert not wm.add_image(cam, R, K, render(cam, R, K))
    assert color_at(wm, (2.37, 1.55)) > 0   # the part of the face in view


def test_red_through_a_slit_doesnt_make_the_pillars_red():
    # laberinto_1: pillars the map doesn't show (objects) with a red box behind the slit
    # between them. Without depth, the red seen through the slit falls on the pillars' edges;
    # the object color keeps only what is certainly on each pillar, so they don't turn red.
    rng = np.random.default_rng(5)
    rows, cols = int(SIZE[1] / RES), int(SIZE[0] / RES)
    X, Y = np.meshgrid((np.arange(cols) + 0.5) * RES, (np.arange(rows) + 0.5) * RES)
    occ = np.zeros((rows, cols), bool)
    for x0, y0, x1, y1, *_ in WALLS:
        occ |= (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)
    wm = WorldModel(np.where(occ, 100, 0).astype(np.int8).ravel(), cols, rows, RES, (0.0, 0.0))
    pillars = [(1.9, 1.2, 2.05, 1.47, 0.5, GREY), (1.9, 1.53, 2.05, 1.8, 0.5, GREY)]
    solids = WALLS + pillars + [RED_BOX]
    global SOLIDS
    saved, SOLIDS = SOLIDS, solids
    try:
        for sp in [(1.0, 1.5), (1.2, 0.6), (1.2, 2.4), (3.3, 1.5), (3.0, 0.6), (3.0, 2.4)] * 3:
            wm.add_scan(sp, scan(sp, rng))
    finally:
        SOLIDS = saved
    wm.update()
    K = (135.9, 135.9, 80.0, 60.0)
    for x, y in [(1.3, 1.5), (1.2, 1.35), (1.2, 1.65), (1.0, 1.45)]:
        cam, R = camera(x, y, math.atan2(1.5 - y, 2.0 - x))
        img, _ = render_with(cam, R, K, solids)
        wm.add_image(cam, R, K, img)
        wm.update()
    # The pillars fuse into one object (6 cm apart) with the slit in its middle: the worst case.
    pillars = [o for o in wm.objects if abs(o.centro[0] - 1.975) < 0.1
               and 1.2 < o.centro[1] < 1.8]
    assert pillars
    for o in pillars:   # a fragment behind the other gets no pixels of its own: 0.5
        assert o.p_rojo <= 0.5, (o.centro, o.vistas)
    box = nearest(wm.objects, np.array([2.5, 1.5]))
    print('box', box.p_rojo, box.vistas, [(o.p_rojo, o.vistas) for o in pillars])


def test_a_cloud_placed_a_few_degrees_off_doesnt_make_an_occluded_box_grey():
    # laberinto_1: a red box behind a row of pillars the map draws closed, seen through the slit
    # while the robot turns. The depth cloud, a few degrees off the image (captured a little
    # earlier in the turn), puts the box's red depth on columns that show the pillar. Red depth
    # opens a column the grid closes only for its red pixels: the grey pillar pixels there are
    # not the box's, and must not make it grey (it got the -4 cap, and the victim was lost).
    global SOLIDS
    rng = np.random.default_rng(6)
    wm = slit_model()
    saved, SOLIDS = SOLIDS, SLIT_SOLIDS
    try:
        for sp in [(3.3, 1.5), (3.0, 0.6), (3.0, 2.4), (2.3, 0.6), (2.3, 2.4)] * 3:
            wm.add_scan(sp, scan(sp, rng))
    finally:
        SOLIDS = saved
    wm.update()
    box = nearest(wm.objects, np.array([2.5, 1.5]))
    assert math.hypot(*(box.centro - (2.5, 1.5))) < 0.05 and box.cov_fit is not None
    K = (135.9, 135.9, 80.0, 60.0)
    for off in (math.radians(7), -math.radians(7)):   # the slit is 8 deg wide from here
        cam, R = camera(1.5, 1.5, 0.0)
        img, _ = render_with(cam, R, K, SLIT_SOLIDS)
        cam_c, R_c = camera(1.5, 1.5, off)        # where the cloud was really taken
        _, (pts, rgb) = render_with(cam_c, R_c, K, SLIT_SOLIDS)
        pts = cam + ((pts - cam_c) @ R_c) @ R.T   # but placed with the image's pose
        wm.add_image(cam, R, K, img, (pts, rgb))
    assert box.lc >= 0, box.vistas


def _behind_a_slit(**params):
    """The slit's red box, scanned from the east, then seen from the west only through the
    slit, without depth. Returns (model, box object)."""
    global SOLIDS
    rng = np.random.default_rng(8)
    wm = slit_model(**params)
    saved, SOLIDS = SOLIDS, SLIT_SOLIDS
    try:
        for sp in [(3.3, 1.5), (3.0, 0.6), (3.0, 2.4), (2.3, 0.6), (2.3, 2.4)] * 3:
            wm.add_scan(sp, scan(sp, rng))
    finally:
        SOLIDS = saved
    wm.update()
    K = (135.9, 135.9, 80.0, 60.0)
    # Straight through the slit (onto the box), then from either side: the slit's cells get
    # three lines of sight, the box only the first (the others reach its edges).
    for x, y in [(1.0, 1.5), (1.5, 1.4), (1.5, 1.6)]:
        cam, R = camera(x, y, math.atan2(1.5 - y, 1.95 - x))
        img, _ = render_with(cam, R, K, SLIT_SOLIDS)
        assert wm.count_red(img) > 0
        wm.add_image(cam, R, K, img)
        wm.update()
    return wm, nearest(wm.objects, np.array([2.5, 1.5]))


def test_red_through_a_mapped_slit_is_the_box_behind_it_without_depth():
    # laberinto_1: the caged victim seen through the slats, without depth. Its red painted the
    # slats, a red-only object there was a candidate 0.4 m off, and the box itself got no
    # color. The red seen past the mapped wall is the box's, and the red-only object is its
    # shadow, not a victim. (A lower report threshold: one view of a slit cell seeds an object.)
    wm, box = _behind_a_slit(reporte_p_min=0.9)
    assert len(box.vistas) == 1 and box.lc > 2.0, box.vistas
    assert all(o.p_objetivo == 0.0 for o in wm.objects if o is not box
               and math.hypot(*(o.centro - (1.95, 1.5))) < 0.2)
    wm, box = _behind_a_slit(reporte_p_min=0.9, rojo_tras_pared=-1.0)   # without: the bug
    assert not box.vistas
    assert any(o.p_objetivo > 0.5 for o in wm.objects
               if math.hypot(*(o.centro - (1.95, 1.5))) < 0.2)


def _window_model(**params):
    """Red outside, seen through a slit below the lidar in the east wall, which the map draws
    solid. Returns the model."""
    outside = (4.3, 1.35, 4.6, 1.65, 0.3, RED)
    wall = [b for b in WALLS if b[0] != 3.9]
    east = [(3.9, 0.0, 4.0, 1.3, 0.5, GREY), (3.9, 1.7, 4.0, 3.0, 0.5, GREY)]
    # A box starts at the floor (slab): one image with the wall only below the slit, one with
    # it whole, and the pixels whose rays cross the slit taken from the first.
    low = [(3.9, 1.3, 4.0, 1.7, 0.03, GREY)]
    high = [(3.9, 1.3, 4.0, 1.7, 0.5, GREY)]
    wm = WorldModel(*_walls_map(), RES, (0.0, 0.0), **params)
    K = (135.9, 135.9, 80.0, 60.0)
    rng = np.random.default_rng(4)
    for sp in [(2.0, 0.8), (2.0, 2.2), (2.6, 1.5)]:
        wm.add_scan(sp, scan(sp, rng))
    for x, y in [(2.0, 0.8), (2.0, 1.2), (2.0, 1.8), (2.0, 2.2), (2.6, 1.5)]:
        cam, R = camera(x, y, math.atan2(1.5 - y, 3.9 - x))
        img, _ = render_with(cam, R, K, wall + east + low + [outside])
        img2, _ = render_with(cam, R, K, wall + east + high + [outside])
        wm.add_image(cam, R, K, np.where(_in_slit(cam, R, K), img, img2))
    wm.update()
    return wm


def _walls_map():
    rows, cols = int(SIZE[1] / RES), int(SIZE[0] / RES)
    X, Y = np.meshgrid((np.arange(cols) + 0.5) * RES, (np.arange(rows) + 0.5) * RES)
    occ = np.zeros((rows, cols), bool)
    for x0, y0, x1, y1, *_ in WALLS:
        occ |= (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)
    return np.where(occ, 100, 0).astype(np.int8).ravel(), cols, rows


def _in_slit(cam, R, K, w=160, h=120, z0=0.03, z1=0.16):
    """Pixels whose ray crosses the wall's plane (x = 3.9) between the slit's heights."""
    U, V = np.meshgrid(np.arange(w), np.arange(h))
    d = np.stack([(U - K[2]) / K[0], (V - K[3]) / K[1], np.ones(U.shape)], -1) @ R.T
    t = (3.9 - cam[0]) / d[..., 0]
    z = cam[2] + t * d[..., 2]
    return ((z >= z0) & (z <= z1))[..., None]


def test_red_outside_through_a_window_seeds_no_object_on_the_outer_wall():
    assert not [o for o in _window_model(reporte_p_min=0.95).objects if o.p_objetivo > 0.5]
    # Without the rule: a red-only object on the wall's inner face, reportable.
    ghosts = [o for o in _window_model(reporte_p_min=0.95, rojo_borde=0.0).objects
              if o.p_objetivo > 0.5]
    assert ghosts and all(o.centro[0] > 3.7 and o.cov_fit is None for o in ghosts)


def test_the_outer_wall_rule_leaves_inner_walls_and_boxes_against_it():
    # The cage of laberinto_1 is an inner wall: its red still seeds. And so the shadow rule
    # still has its red-only object to drop.
    wm, box = _behind_a_slit(reporte_p_min=0.9)
    assert box.lc > 2.0 and wm.edge_dist[int(1.5 / RES) + wm.pad, int(1.9 / RES) + wm.pad] > 0.25
    assert any(math.hypot(*(o.centro - (1.95, 1.5))) < 0.2 for o in wm.objects)
    # A red box flush against the outer wall comes from the lidar, not from red seeds.
    box = (3.6, 1.35, 3.9, 1.65, 0.3, RED)
    wm = WorldModel(*_walls_map(), RES, (0.0, 0.0), reporte_p_min=0.95)
    K = (135.9, 135.9, 80.0, 60.0)
    global SOLIDS
    saved, SOLIDS = SOLIDS, WALLS + [box]
    try:
        rng = np.random.default_rng(5)
        for x, y in [(2.0, 0.8), (2.4, 1.1), (2.6, 1.5), (2.4, 1.9), (2.0, 2.2)]:
            for _ in range(3):
                wm.add_scan((x, y), scan((x, y), rng))
            wm.update()
            cam, R = camera(x, y, math.atan2(1.5 - y, 3.75 - x))
            wm.add_image(cam, R, K, render_with(cam, R, K, WALLS + [box])[0])
    finally:
        SOLIDS = saved
    wm.update()
    assert nearest(wm.objects, np.array([3.75, 1.5])).p_objetivo > 0.9


def _cells(wm, x0, y0, x1, y1):
    """Flat indices of the cells whose centers fall in the rectangle."""
    m = (wm.cx >= x0) & (wm.cx <= x1) & (wm.cy >= y0) & (wm.cy <= y1)
    return np.flatnonzero(m)


def test_a_box_and_a_wall_stretch_that_come_to_touch_stay_two_objects():
    # A 0.2 m box against a wall, and a stretch of that wall seen a little off (a new object
    # along it, 1.2 m long). Cells appearing between them join them into one component: each
    # keeps its ID and its own cells.
    wm = model()
    box = _cells(wm, 1.0, 1.0, 1.2, 1.2)
    wall = _cells(wm, 1.0, 1.3, 2.2, 1.35)
    wm.objects = [Objeto(1, box, np.array([1.1, 1.1]), np.eye(2)),
                  Objeto(2, wall, np.array([1.6, 1.325]), np.eye(2))]
    wm.next_id = 3
    bridge = _cells(wm, 1.05, 1.2, 1.15, 1.3)
    out = wm._match([np.unique(np.concatenate([box, wall, bridge]))])
    ids = {o.id: o for o in out}
    assert set(ids) == {1, 2}
    assert set(box) <= set(ids[1].cells) and set(wall) <= set(ids[2].cells)


def test_fragments_of_one_box_still_merge():
    # Two sides of one box, seen as two objects, then joined: one object, as before.
    wm = model()
    a = _cells(wm, 1.0, 1.0, 1.05, 1.2)
    b = _cells(wm, 1.15, 1.0, 1.2, 1.2)
    wm.objects = [Objeto(1, a, np.array([1.025, 1.1]), np.eye(2)),
                  Objeto(2, b, np.array([1.175, 1.1]), np.eye(2))]
    wm.next_id = 3
    out = wm._match([np.unique(np.concatenate([a, b, _cells(wm, 1.0, 1.0, 1.2, 1.05)]))])
    assert len(out) == 1


def test_two_boxes_a_little_apart_are_two_objects():
    # Two 0.25 m boxes 17 cm apart, first seen together: within the joining distance, but
    # together too big for one box, so two objects.
    wm = model()
    for cells in (_cells(wm, 1.0, 1.2, 1.25, 1.45), _cells(wm, 1.42, 1.2, 1.67, 1.45)):
        wm.L[cells] = 3.0
    objs = wm.update()
    assert len(objs) == 2
    assert sorted(round(float(o.centro[0]), 1) for o in objs) == [1.1, 1.5]


def test_fragments_of_one_box_seen_through_gaps_join():
    # One 0.25 m box seen as three pieces with 5-10 cm between them: one object.
    wm = model()
    for cells in (_cells(wm, 1.0, 1.2, 1.05, 1.45), _cells(wm, 1.12, 1.2, 1.15, 1.45),
                  _cells(wm, 1.22, 1.2, 1.25, 1.45)):
        wm.L[cells] = 3.0
    assert len(wm.update()) == 1


def test_fragments_with_empty_space_seen_between_them_stay_apart():
    # o05 (seed 1): a victim and a beige box 10 cm apart, together small enough for one box,
    # joined into one beige object. Once the lidar has seen the gap empty, they are two.
    wm = model()
    boxes = [_cells(wm, 1.0, 1.2, 1.25, 1.3), _cells(wm, 1.0, 1.4, 1.25, 1.5)]
    for cells in boxes:
        wm.L[cells] = 3.0
    assert len(wm.update()) == 1                # the gap not seen: maybe one box
    around = np.setdiff1d(_cells(wm, 0.8, 1.0, 1.45, 1.7), np.concatenate(boxes))
    wm.L[around] = -3.0                         # the lidar saw the space around them empty
    assert len(wm.update()) == 2


def test_the_prior_counts_the_objects_not_seen_yet():
    # N = 2 and two objects known: P(victim) before color is N / (objects in the arena). With
    # most of the arena unseen, more objects are to come and the prior is lower; once all of
    # it is seen, the two are all there is.
    wm = model(objetivos_n=2)
    for cells in (_cells(wm, 1.0, 1.2, 1.25, 1.45), _cells(wm, 2.5, 1.2, 2.75, 1.45)):
        wm.L[cells] = 3.0
    wm.logb[:] = 0.0                                # nothing seen
    wm.logb[np.flatnonzero(wm.cx < 1.0)] = -30.0    # a quarter of the room seen
    early = [o.p_objetivo for o in wm.update()]
    wm.logb[:] = -30.0                              # all seen
    late = [o.p_objetivo for o in wm.update()]
    assert all(e < l - 0.2 for e, l in zip(early, late)), (early, late)


def test_a_box_flush_against_a_wall_is_judged_only_from_its_open_side():
    # v06: a red box 2 cm north of a mapped 2 cm panel. Seen from the south, the camera shows
    # only the wall, and the wall's hits, placed 7.5 cm north by the pose error, land beside the
    # box. Both are the wall's: no color evidence, no proven line of sight from behind the wall.
    global SOLIDS
    wall = (1.5, 1.03, 3.5, 1.05, 0.5, GREY)
    box = (2.375, 1.07, 2.625, 1.32, 0.3, RED)
    rows, cols = int(SIZE[1] / RES), int(SIZE[0] / RES)
    X, Y = np.meshgrid((np.arange(cols) + 0.5) * RES, (np.arange(rows) + 0.5) * RES)
    occ = np.zeros((rows, cols), bool)
    for x0, y0, x1, y1, *_ in WALLS + [(1.5, 1.0, 3.5, 1.05)]:
        occ |= (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)
    wm = WorldModel(np.where(occ, 100, 0).astype(np.int8).ravel(), cols, rows, RES, (0.0, 0.0))
    rng = np.random.default_rng(7)
    saved, SOLIDS = SOLIDS, WALLS + [wall, box]
    try:
        north = [(1.6, 2.0), (2.5, 2.2), (3.4, 2.0), (3.7, 1.4)]
        south = [(1.8, 0.5), (2.5, 0.4), (3.2, 0.5)]
        for _ in range(3):
            for sp in north:
                wm.add_scan(sp, scan(sp, rng), sigma_pose=0.06)
            for sp in south:
                wm.add_scan(sp, scan(sp, rng) + (0.0, 0.075), sigma_pose=0.06)
        wm.update()
        o = nearest(wm.objects, np.array([2.5, 1.195]))
        assert math.hypot(*(o.centro - (2.5, 1.195))) < 0.06
        assert o.miradores and all(y > 1.05 for _, y in o.miradores), o.miradores
        K = (135.9, 135.9, 80.0, 60.0)
        for x in (2.2, 2.5, 2.8):          # behind the wall: only grey
            cam, R = camera(x, 0.5, math.pi / 2)
            img, _ = render_with(cam, R, K, SOLIDS)
            wm.add_image(cam, R, K, img)
        assert not o.vistas
        cam, R = camera(2.5, 2.2, -math.pi / 2)   # the open side: red
        img, cloud = render_with(cam, R, K, SOLIDS)
        wm.add_image(cam, R, K, img, cloud)
        assert len(o.vistas) == 1 and o.vistas[0][1] > 2.0
    finally:
        SOLIDS = saved
