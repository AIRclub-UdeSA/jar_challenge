"""The planner on a synthetic floor: two rooms joined by a door, and a closed pocket."""
import math

import numpy as np

from equipo_jar.planner import Grid, Planner, expected_missing, path_lengths, reported_objects, \
    shadowed_objects

RES = 0.05
W, H = 6.0, 3.0


def floor():
    """Occupancy % (0 free, 100 wall): outer walls, a wall at x = 3 with a 0.8 m door at
    y 1.1-1.9, and a closed 0.6 m pocket in the left room's corner."""
    rows, cols = int(H / RES), int(W / RES)
    X, Y = np.meshgrid((np.arange(cols) + 0.5) * RES, (np.arange(rows) + 0.5) * RES)
    wall = (X < 0.1) | (X > W - 0.1) | (Y < 0.1) | (Y > H - 0.1)
    wall |= (np.abs(X - 3.0) < 0.05) & ~((Y > 1.1) & (Y < 1.9))
    wall |= ((np.abs(X - 0.8) < 0.05) & (Y < 0.8)) | ((np.abs(Y - 0.8) < 0.05) & (X < 0.8))
    return np.where(wall, 100, 0).astype(np.int8), X, Y


def planner(belief=None, robot=(1.5, 1.5), stuck=(), **params):
    occ, X, Y = floor()
    b = np.zeros(occ.shape, np.int8) if belief is None else belief
    return Planner(Grid(occ, RES, (0.0, 0.0)), Grid(b, RES, (0.0, 0.0)), robot, stuck,
                   **params), X, Y


def first(p, objects, n_total, **kw):
    """Where the tour goes first, or None when it is empty."""
    tour = p.tour(objects, n_total, **kw)
    return tour[0] if tour else None


def test_path_lengths_go_through_the_door():
    p, X, Y = planner()
    # From (1.5, 1.5) to (4.5, 1.5): straight through the door, 3 m.
    assert abs(p.path_to((4.5, 1.5)) - 3.0) < 0.15
    # (0.4, 0.4) is inside the closed pocket.
    assert math.isinf(p.dist[p.grid.cell((0.4, 0.4))])
    # Around the door: (2.5, 0.5) -> (3.5, 0.5) is longer than 1 m.
    q, _, _ = planner(robot=(2.5, 0.5))
    assert q.path_to((3.5, 0.5)) > 2.0


def test_path_lengths_8_connected():
    free = np.ones((21, 21), bool)
    d = path_lengths(free, (0, 0), 0.05)
    assert abs(d[20, 20] - 20 * math.sqrt(2) * 0.05) < 1e-9
    assert abs(d[0, 20] - 1.0) < 1e-9


def test_explore_values_the_missing_victims():
    occ, X, Y = floor()
    belief = np.where(X > 3.2, 98, 0).astype(np.int8)
    p, _, _ = planner(belief)
    # Three victims in total, the known objects account for two: one is still missing.
    objs = [{'id': 1, 'center': (2.0, 2.4), 'p': 1.0, 'radius': 0.12},
            {'id': 2, 'center': (2.0, 0.6), 'p': 1.0, 'radius': 0.12}]
    one = p.cover(expected_missing(3, [o['p'] for o in objs]))[0]
    two = p.cover(expected_missing(3, []))[0]
    assert one.kind == 'explore' and abs(two.value / one.value - 3.0) < 1e-6
    # All accounted for: nothing to explore, nothing undecided to look at.
    assert first(p, objs, n_total=2) is None


def test_look_needs_a_line_of_sight_and_camera_range():
    # An undecided object in the right room, at (4.5, 0.6).
    obj = {'id': 7, 'center': (4.5, 0.6), 'p': 0.6, 'radius': 0.12}
    p, _, _ = planner()
    t = first(p, [obj], n_total=1)
    assert t.kind == 'look' and t.object_id == 7 and t.face == (4.5, 0.6)
    d = math.dist(t.xy, obj['center'])
    assert p.p['mirada_min'] <= d <= p.p['mirada_max']
    assert t.xy[0] > 3.05   # the dividing wall blocks the view from the left room
    # Decided objects are not looked at.
    for q in (0.01, 0.99):
        assert first(p, [dict(obj, p=q)], n_total=1) is None


def test_each_look_needs_a_new_line_of_sight():
    obj = {'id': 7, 'center': (4.5, 0.6), 'p': 0.6, 'radius': 0.12}
    p, _, _ = planner()
    look = first(p, [obj], n_total=1)
    again = first(p, [obj], n_total=1, seen_from={7: [look.xy]})
    cx, cy = obj['center']
    turn = abs((math.atan2(again.xy[1] - cy, again.xy[0] - cx)
                - math.atan2(look.xy[1] - cy, look.xy[0] - cx) + math.pi) % (2 * math.pi)
               - math.pi)
    assert turn >= p.p['mirada_angulo_min']
    assert again.value == look.value
    # Enough looks: the object is left alone.
    assert first(p, [obj], n_total=1, seen_from={7: [look.xy] * p.p['miradas_max']}) is None


def test_failed_goals_are_skipped():
    obj = {'id': 7, 'center': (4.5, 0.6), 'p': 0.6, 'radius': 0.12}
    p, _, _ = planner()
    look = first(p, [obj], n_total=1)
    other = first(p, [obj], n_total=1, failed=[look.xy])
    assert math.dist(other.xy, look.xy) > p.p['fallo_radio']


def test_stuck_evidence_grows_with_each_failure_and_stays_bounded():
    p, _, _ = planner()
    base = p.path_to((4.5, 1.5))
    extra = []
    for k in (1, 2, 3):
        door = [(3.0, 1.2), (3.0, 1.5), (3.0, 1.8)]           # the whole door's width
        stuck, _, _ = planner(stuck=[door] * k)               # k failures there
        extra.append(stuck.path_to((4.5, 1.5)) - base)
    # Each failure is more evidence the door can't be crossed: the expected cost of crossing,
    # (1 / P(passable) - 1) x fallo_s of driving, grows with it, and stays finite.
    assert 0.5 < extra[0] < extra[1] < extra[2] < 50.0
    # Elsewhere the paths don't change.
    assert abs(stuck.path_to((1.5, 2.5)) - p.path_to((1.5, 2.5))) < 1e-9


def test_miradores_are_proven_lines_of_sight():
    # Inside the closed pocket: the grid shows no line of sight to it from anywhere reachable.
    obj = {'id': 9, 'center': (0.4, 0.4), 'p': 0.6, 'radius': 0.12}
    p, _, _ = planner()
    assert first(p, [obj], n_total=1) is None
    # The lidar saw it from (1.3, 0.45), through a gap the grid doesn't resolve.
    t = first(p, [dict(obj, miradores=[(1.3, 0.45)])], n_total=1)
    assert t.kind == 'look' and math.dist(t.xy, (1.3, 0.45)) <= p.p['mirador_radio']


def test_no_exploring_when_the_objects_account_for_every_victim():
    occ, X, Y = floor()
    p, _, _ = planner(np.where(X > 3.2, 98, 0).astype(np.int8))
    found = {'id': 1, 'center': (2.0, 2.4), 'p': 1.0, 'radius': 0.12}
    # 0.02 victims unaccounted for: exploring can't pay.
    assert first(p, [found, {'id': 2, 'center': (2.0, 0.6), 'p': 0.98, 'radius': 0.12}],
                 n_total=2) is None
    # 0.3 unaccounted for: it can.
    t = first(p, [found, {'id': 2, 'center': (2.0, 0.6), 'p': 0.7, 'radius': 0.12}], n_total=2,
              seen_from={2: [(1.5, 0.6)] * p.p['miradas_max']})
    assert t is not None and t.kind == 'explore'


def test_a_second_look_comes_from_a_new_line_of_sight_on_the_cheap_side():
    # Looked at from the north; the robot is west of the object. The world model counts any
    # face's color per object, so the second look is from the west, the nearest new line.
    obj = {'id': 3, 'center': (2.0, 1.5), 'p': 0.95, 'radius': 0.12}
    p, _, _ = planner(robot=(1.2, 1.5))
    t = first(p, [obj], n_total=1, seen_from={3: [(2.0, 2.5)]})
    turn = abs((math.atan2(t.xy[1] - 1.5, t.xy[0] - 2.0) - math.pi / 2 + math.pi)
               % (2 * math.pi) - math.pi)
    assert turn >= p.p['mirada_angulo_min']
    assert t.xy[0] < 2.0


def test_sees_through_free_space_but_not_walls():
    p, _, _ = planner()
    assert p.sees((1.0, 1.5), (2.0, 1.5))
    # Across the wall (x = 3) away from the door, no line of sight.
    assert not p.sees((2.5, 0.5), (3.5, 0.5))
    # The object's own footprint doesn't hide it.
    assert p.sees((2.4, 0.5), (3.0, 0.5), own=0.1)


def corridor_world(width, bend=False):
    """A 4 x 3 m room split by a wall band with a corridor of `width` through it, straight
    (along x) or with a right-angle bend."""
    res = 0.02
    w, h = int(4.0 / res), int(3.0 / res)
    Y, X = (np.indices((h, w)) + 0.5) * res
    wall = (X < 0.1) | (X > 3.9) | (Y < 0.1) | (Y > 2.9)
    band = (X > 1.5) & (X < 2.5)
    if bend:  # in along y = 1.5 from the left, out along x = 2.0 to the top
        gap = ((np.abs(Y - 1.5) < width / 2) & (X < 2.0 + width / 2)) \
            | ((np.abs(X - 2.0) < width / 2) & (Y > 1.5 - width / 2))
        wall |= band & (Y > 1.0) & ~gap
    else:
        wall |= band & ~(np.abs(Y - 1.5) < width / 2)
    occ = np.where(wall, 100, 0).astype(np.int8)
    g = Grid(occ, res, (0.0, 0.0))
    return Planner(g, Grid(np.zeros_like(occ), res, (0.0, 0.0)), (0.6, 1.5, 0.0),
                   huella_margen=0.02)


def test_a_corridor_narrower_than_the_robot_blocks_it():
    assert not np.isfinite(corridor_world(0.20).path_to((3.4, 1.5)))


def test_a_corridor_wider_than_the_robot_but_shorter_than_it_is_crossed_straight():
    p = corridor_world(0.32)
    poses = p.path_poses((3.4, 1.5))
    assert poses is not None
    inside = [h for x, y, h in poses if 1.6 < x < 2.4]
    # No room to turn in there: every pose has a planned heading, along the corridor.
    assert inside and all(h is not None for h in inside)
    assert all(min(abs(math.remainder(h, math.pi)), math.pi - abs(math.remainder(h, math.pi)))
               < 0.25 for h in inside)


def test_a_bend_is_a_path_only_if_the_robot_can_turn_in_its_corner():
    # 0.26 m lets the padded 0.34 x 0.24 m chassis through straight, but not round a right
    # angle. At 0.32 m it can: holonomic, it slides through the corner while turning, as a
    # ladder carried round a corner.
    assert np.isfinite(corridor_world(0.26).path_to((3.4, 1.5)))
    assert not np.isfinite(corridor_world(0.26, bend=True).path_to((2.0, 2.7)))
    assert np.isfinite(corridor_world(0.32, bend=True).path_to((2.0, 2.7)))
    assert np.isfinite(corridor_world(0.60, bend=True).path_to((2.0, 2.7)))


def test_each_report_finds_one_object():
    # A victim under a table, reported; two of the table's legs are within the report radius.
    # Only the victim is found: the legs keep their own P, so with N = 3 and two reports the
    # third victim is still missing (marking all three as found made four of three, and the
    # search stopped with a third of the arena unseen).
    objects = [{'id': 1, 'center': (3.63, -0.41), 'p': 0.98},
               {'id': 8, 'center': (3.26, -0.71), 'p': 0.02},
               {'id': 17, 'center': (3.26, -0.18), 'p': 0.02},
               {'id': 2, 'center': (2.06, 1.28), 'p': 0.98},
               {'id': 5, 'center': (0.0, 0.0), 'p': 0.02}]
    reports = [(3.62, -0.41), (2.06, 1.28)]
    taken = reported_objects(objects, reports, 0.5)
    assert taken == [1, 2]
    ps = [1.0 if o['id'] in taken else o['p'] for o in objects]
    assert expected_missing(3, ps) > 0.9
    every = [1.0 if any(math.dist(o['center'], r) < 0.5 for r in reports) else o['p']
             for o in objects]
    assert expected_missing(3, every) == 0.0   # the old rule


def test_a_box_split_in_two_is_one_victim():
    # v01 on the realistic sim: victim 1 seen twice, 18 cm apart (pose error), both red. Three
    # reports, N = 4: the second half at P 0.92 made the fourth victim look found, and the
    # search stopped with a third of the arena unseen.
    objects = [{'id': 1, 'center': (-2.29, 1.30), 'p': 0.90},
               {'id': 2, 'center': (-2.11, 1.45), 'p': 0.92},
               {'id': 3, 'center': (-2.31, -4.33), 'p': 0.91},
               {'id': 4, 'center': (2.29, -0.99), 'p': 0.95},
               {'id': 5, 'center': (-0.31, -0.73), 'p': 0.02}]
    reports = [(-2.25, 1.34), (-2.30, -4.34), (2.29, -0.98)]
    taken = reported_objects(objects, reports, 0.5)
    shadowed = shadowed_objects(objects, reports, 0.5) - set(taken)
    assert shadowed == {2}
    ps = [1.0 if o['id'] in taken else 0.0 if o['id'] in shadowed else o['p'] for o in objects]
    assert expected_missing(4, ps) > 0.9
    kept = [1.0 if o['id'] in taken else o['p'] for o in objects]
    assert expected_missing(4, kept) < 0.1   # the old rule: below faltantes_min, no exploring


def test_two_reports_close_together_take_two_objects_and_a_gone_one_none():
    objects = [{'id': 1, 'center': (0.0, 0.0)}, {'id': 2, 'center': (0.3, 0.0)}]
    assert reported_objects(objects, [(0.05, 0.0), (0.1, 0.0), (5.0, 5.0)], 0.5) == [1, 2, None]


def test_a_wall_a_box_stands_flush_against_hides_it():
    # A box 1 cm east of a one-cell wall: the wall is within the box's radius of its center, so
    # counted as the box's own it let looks through the wall. Walking toward the center, the
    # first blocked cell must be one of the object's cells (the world model's: more than 10 cm
    # from the mapped wall).
    occ, X, Y = floor()
    occ[(np.abs(X - 3.0) < 0.05) & (Y < 1.1)] = 0
    occ[(np.abs(X - 3.025) < 0.01) & (Y < 1.1)] = 100
    box = (np.abs(X - 3.185) < 0.125) & (np.abs(Y - 0.6) < 0.125)
    occ[box] = 100
    cells = np.flatnonzero(box & (X > 3.13))
    p = Planner(Grid(occ, RES, (0.0, 0.0)), Grid(np.zeros(occ.shape, np.int8), RES, (0.0, 0.0)),
                (1.5, 1.5))
    obj = {'id': 5, 'center': (3.185, 0.6), 'p': 0.6, 'radius': math.hypot(0.25, 0.25) / 2}
    behind, front = (2.3, 0.6), (4.0, 0.6)
    assert p.sees_object(behind, obj)               # the radius alone: through the wall
    assert not p.sees_object(behind, dict(obj, cells=cells))
    assert p.sees_object(front, dict(obj, cells=cells))
    t = first(p, [dict(obj, cells=cells)], n_total=1)   # from the door or the east, not behind
    assert t.kind == 'look' and not (t.xy[0] < 3.0 and t.xy[1] < 1.1)


# Tour.

def length(D, order):
    return float(sum(D[a, b] for a, b in zip(order, order[1:])))


def test_path_lengths_from_several_starts_match_one_at_a_time():
    occ, _, _ = floor()
    free = occ == 0
    starts = [(30, 30), (30, 90), (10, 10)]
    many = path_lengths(free, starts, RES)
    for k, s in enumerate(starts):
        assert np.array_equal(many[k], path_lengths(free, s, RES))


def test_cover_does_not_count_the_same_belief_twice():
    occ, X, Y = floor()
    belief = np.where((X > 3.2) & (X < 5.8) & (Y > 0.2) & (Y < 2.8), 98, 0).astype(np.int8)
    p, _, _ = planner(belief)
    stops = p.cover(2.0)
    assert stops and (stops[0].xy[0] > 3.05 or 1.1 < stops[0].xy[1] < 1.9)
    values = [t.value for t in stops]
    assert values == sorted(values, reverse=True)
    assert sum(values) <= 2.0 + 1e-9
    # One stop in the open room sees nearly all of it: the rest add little.
    assert values[0] > 0.8 * 2.0
    assert all(t.mass >= t.value / 2.0 * p.belief.sum() - 1e-6 for t in stops)


def test_tour_is_empty_with_nothing_to_do():
    p, _, _ = planner()
    assert p.tour([], n_total=1) == []


def test_tour_finishes_one_room_before_the_other():
    """Greedy goes for the best score: the object behind the door first, then back for the
    two in the robot's room. The tour looks at those two on the way."""
    objs = [{'id': 1, 'center': (1.3, 2.4), 'p': 0.5, 'radius': 0.12},
            {'id': 2, 'center': (4.2, 1.5), 'p': 0.6, 'radius': 0.12},
            {'id': 3, 'center': (1.3, 0.4), 'p': 0.5, 'radius': 0.12}]
    p, _, _ = planner(robot=(2.4, 1.5), mirada_max=1.2)
    tour = p.tour(objs, n_total=1)
    ids = [t.object_id for t in tour]
    assert sorted(ids) == [1, 2, 3]
    assert ids[-1] == 2 or ids[0] == 2  # never the right room between the two on the left
    D = p.leg_lengths([t.xy for t in tour])
    tour_len = length(D, list(range(len(tour) + 1)))
    zigzag = [0] + [1 + ids.index(i) for i in (1, 2, 3)]
    assert tour_len <= length(D, zigzag) + 1e-9


def test_tour_drops_a_detour_worth_too_little():
    objs = [{'id': 1, 'center': (1.3, 2.4), 'p': 0.5, 'radius': 0.12},
            {'id': 2, 'center': (4.8, 2.4), 'p': 0.05, 'radius': 0.12}]
    p, _, _ = planner(robot=(1.5, 1.5), tour_tasa_min=0.003)
    assert [t.object_id for t in p.tour(objs, n_total=1)] == [1]
    p, _, _ = planner(robot=(1.5, 1.5), tour_tasa_min=0.0)
    assert sorted(t.object_id for t in p.tour(objs, n_total=1)) == [1, 2]


def test_tour_keeps_the_last_plans_next_node_when_it_costs_about_the_same():
    # Two objects at the same distance on either side of the robot.
    objs = [{'id': 1, 'center': (1.5, 2.6), 'p': 0.5, 'radius': 0.12},
            {'id': 2, 'center': (1.5, 0.4 + 0.0), 'p': 0.5, 'radius': 0.12}]
    p, _, _ = planner(robot=(1.5, 1.5), mirada_min=0.6, mirada_max=0.9)
    first = p.tour(objs, n_total=1)
    assert len(first) == 2
    other = first[1]
    again = p.tour(objs, n_total=1, prefer=other.xy)
    assert again[0].object_id == other.object_id
    # Much shorter elsewhere: the preference gives way.
    q, _, _ = planner(robot=(1.5, 1.5), mirada_min=0.6, mirada_max=0.9, tour_histeresis=0.0)
    far = q.tour(objs, n_total=1, prefer=(5.0, 2.0))
    assert [t.object_id for t in far] == [t.object_id for t in first]


def test_tour_skips_failed_spots():
    occ, X, Y = floor()
    belief = np.where(X > 3.2, 98, 0).astype(np.int8)
    p, _, _ = planner(belief)
    first = p.tour([], n_total=1)[0]
    again = p.tour([], n_total=1, failed=[first.xy])
    assert all(math.dist(t.xy, first.xy) > p.p['fallo_radio'] for t in again)


def test_a_visited_stop_skips_explore_stops_but_not_looks():
    # laberinto_1 with the sim realism: a box's only look spot was 0.14 m from the explore
    # stop just reached. Skipped like a failed goal, the tour was empty and the search ended
    # with the box undecided.
    occ, X, Y = floor()
    belief = np.where(X > 3.2, 98, 0).astype(np.int8)
    box = {'id': 1, 'center': (1.5, 2.4), 'p': 0.6, 'radius': 0.12}
    p, _, _ = planner(belief, robot=(1.5, 1.5))
    spot = p.look(box).xy
    tour = p.tour([box], n_total=2, visited=[spot])
    assert any(t.kind == 'look' for t in tour)
    assert all(math.dist(t.xy, spot) > p.p['fallo_radio'] for t in tour if t.kind == 'explore')
    assert not any(t.kind == 'look' and math.dist(t.xy, spot) <= p.p['fallo_radio']
                   for t in p.tour([box], n_total=2, failed=[spot]))


def test_a_stop_is_dropped_only_when_cover_would_not_choose_it():
    occ, X, Y = floor()
    belief = np.where(X > 3.2, 98, 0).astype(np.int8)
    p, _, _ = planner(belief)
    stop = p.tour([], n_total=1)[0]
    assert p.still_worth(stop, missing=1.0)
    seen = np.where((X > 3.2) & (np.hypot(X - stop.xy[0], Y - stop.xy[1]) > 0.5), 98, 0)
    q, _, _ = planner(seen.astype(np.int8))
    still = q.still_worth(stop, missing=1.0)
    again = any(math.dist(t.xy, stop.xy) < 0.05 for t in q.cover(1.0))
    assert still or not again  # dropped here: the next tour doesn't bring it back
    empty, _, _ = planner(np.zeros(occ.shape, np.int8))
    assert not empty.still_worth(stop, missing=1.0)


def test_every_stop_cover_picks_passes_the_drop_check():
    """The drop loop of laberinto_1: a stop dropped as obsolete, chosen again by the next plan,
    dropped again... The drop check and cover() must agree on what a stop is worth."""
    occ, X, Y = floor()
    rng = np.random.default_rng(0)
    checked = 0
    for _ in range(40):
        # Late in a run: a few small specks of belief left, which the rays may not sample.
        b = np.zeros(occ.shape)
        for _ in range(rng.integers(1, 6)):
            i, j = rng.integers(0, occ.shape[0]), rng.integers(0, occ.shape[1])
            b[i:i + rng.integers(1, 3), j:j + rng.integers(1, 3)] = 98
        b[occ != 0] = 0
        if not b.any():
            continue
        p, _, _ = planner(b.astype(np.int8))
        for stop in p.cover(1.0):
            assert p.still_worth(stop, missing=1.0)
            checked += 1
    assert checked > 10


def test_after_a_dead_look_the_next_comes_from_another_face():
    """laberinto_1's cages: a look that doesn't move P got no pixels on the box from there.
    The nearest spot 20 deg further along the same corridor sees the same face through the
    same bars: try another face, and after miradas_muertas_max of them, stop."""
    obj = {'id': 7, 'center': (4.5, 1.5), 'p': 0.6, 'radius': 0.12}
    p, _, _ = planner(robot=(3.4, 1.5))
    first = p.look(obj)
    turn = lambda a, b: abs((math.atan2(a[1] - 1.5, a[0] - 4.5) - math.atan2(b[1] - 1.5, b[0] - 4.5)
                             + math.pi) % (2 * math.pi) - math.pi)
    alive = p.look(obj, [first.xy])
    dead = p.look(dict(obj, muertas=[first.xy], muertas_seguidas=1), [first.xy])
    assert turn(alive.xy, first.xy) <= p.p['mirada_misma_cara']   # nearest: the same face
    assert turn(dead.xy, first.xy) > p.p['mirada_misma_cara']     # after a dead one: another
    assert p.look(dict(obj, muertas=[first.xy], muertas_seguidas=3), [first.xy]) is None


def test_a_dead_look_doesnt_use_up_the_only_line_of_sight():
    """laberinto_1, V3 in its cage: seen only near its two miradores, side by side. The one
    look from there got nothing; with its line counted as used, no spot was left, the robot
    gave up on the box and spent the last report on another one. Here a new line must be more
    than pi away from the first, so none is: only a dead look leaves a spot."""
    obj = {'id': 7, 'center': (4.5, 1.5), 'p': 0.6, 'radius': 0.12}
    p, _, _ = planner(robot=(3.4, 1.5), mirada_angulo_min=4.0)
    first = p.look(obj)
    assert p.look(obj, [first.xy]) is None
    again = p.look(dict(obj, muertas=[first.xy], muertas_seguidas=1), [first.xy])
    assert again is not None
    assert p.look(dict(obj, muertas=[first.xy] * 3, muertas_seguidas=3), [first.xy] * 3) is None


def test_after_a_dead_look_the_retry_stands_on_a_mirador():
    """Through a slot between a cage's walls, 10-30 cm off the lidar's line is another line.
    The first look goes anywhere that sees the box; after it comes back dead, the retry
    drives onto a mirador, precisely."""
    mir = (3.6, 2.1)
    obj = {'id': 7, 'center': (4.5, 1.5), 'p': 0.6, 'radius': 0.12, 'miradores': [mir]}
    p, _, _ = planner(robot=(3.4, 1.0))
    first = p.look(obj)
    assert not first.precise
    retry = p.look(dict(obj, muertas=[first.xy], muertas_seguidas=1), [first.xy])
    assert retry.precise
    assert math.dist(retry.xy, mir) <= 0.05 + 1e-9
    # Without miradores there is nothing to snap to: the retry is an ordinary one.
    plain = dict(obj, miradores=[])
    assert not p.look(dict(plain, muertas=[first.xy], muertas_seguidas=1), [first.xy]).precise


def bearings(pts, center):
    return np.array([math.atan2(y - center[1], x - center[0]) for x, y in pts])


def check_sweep(p, obj, pts):
    """Every point of a sweep is in camera range, sees the object and has room to turn."""
    d = [math.dist(q, obj['center']) for q in pts[1:]]
    assert min(d) >= p.p['mirada_min'] - 0.05 and max(d) <= p.p['mirada_max'] + 0.05
    assert p.reaches([q[0] for q in pts[1:]], [q[1] for q in pts[1:]], obj).all()
    assert all(p.turn_free[p.grid.cell(q)] for q in pts[1:])
    run = sum(math.dist(a, b) for a, b in zip(pts[1:], pts[2:]))
    assert run <= p.p['barrido_largo'] + 0.1


def test_a_sweep_crosses_new_lines_of_sight_in_reach_of_the_camera():
    obj = {'id': 7, 'center': (4.5, 1.5), 'p': 0.6, 'radius': 0.12}
    p, _, _ = planner(robot=(3.5, 1.5))
    pts = p.sweep(obj, (3.5, 1.5))
    assert pts is not None and pts[0] == (3.5, 1.5)
    check_sweep(p, obj, pts)
    b = np.unwrap(bearings(pts, obj['center']))
    assert b.max() - b.min() >= p.p['barrido_angulo'] - 0.05


def test_a_sweep_turns_away_from_lines_of_sight_already_seen():
    obj = {'id': 7, 'center': (4.5, 1.5), 'p': 0.6, 'radius': 0.12}
    p, _, _ = planner(robot=(3.5, 1.5))
    # Views counted from the south-west: the sweep heads north-west, where none did.
    pts = p.sweep(obj, (3.5, 1.5), seen_from=[(3.6, 1.0)])
    check_sweep(p, obj, pts)
    assert pts[-1][1] > 1.6
    pts = p.sweep(obj, (3.5, 1.5), seen_from=[(3.6, 2.0)])
    assert pts[-1][1] < 1.4


def test_a_sweep_through_a_door_crosses_the_line_straight_through_it():
    """laberinto_1: a box seen through a gap. Looks from oblique spots got no pixels on it;
    the one straight through the gap did. Sliding from an oblique spot, camera on the box,
    crosses that line."""
    obj = {'id': 7, 'center': (3.6, 1.5), 'p': 0.6, 'radius': 0.12}
    p, _, _ = planner(robot=(2.6, 1.15))
    pts = p.sweep(obj, (2.6, 1.15))
    check_sweep(p, obj, pts)
    assert min(q[1] for q in pts) < 1.5 - 0.05 and max(q[1] for q in pts) > 1.5 + 0.05


def test_no_sweep_where_nothing_near_sees_the_object():
    obj = {'id': 9, 'center': (0.4, 0.4), 'p': 0.6, 'radius': 0.12}   # in the closed pocket
    p, _, _ = planner()
    assert p.sweep(obj, (1.5, 1.5)) is None
    # Every line of sight around it already counted: nothing new to sweep across.
    obj = {'id': 7, 'center': (4.5, 1.5), 'p': 0.6, 'radius': 0.12}
    q, _, _ = planner(robot=(3.5, 1.5))
    ring = [(4.5 + math.cos(a), 1.5 + math.sin(a)) for a in np.arange(0, 2 * math.pi, 0.1)]
    assert q.sweep(obj, (3.5, 1.5), seen_from=ring) is None


def test_wedged_where_the_chassis_fits_only_as_it_stands():
    """p06: stopped 16 cm from a box's cells, the robot fit only at its own pose, with no step
    to any other cell. The planner found nothing anywhere, and the mission went home with half
    the arena unseen instead of getting out first."""
    res = 0.05
    Y, X = (np.indices((40, 40)) + 0.5) * res - 1.0     # the robot at (0, 0), heading 0

    def pocket(slack):
        free = (X > -0.231 - slack) & (X < 0.169 + slack) & (np.abs(Y) < 0.15 + slack)
        occ = np.where(free, 0, 100).astype(np.int8)
        g = Grid(occ, res, (-1.0, -1.0))
        return Planner(g, Grid(np.zeros_like(occ), res, (-1.0, -1.0)), (0.0, 0.0, 0.0))

    for slack in (0.0, 0.02, 0.04):     # it fits, and can shuffle a cell or two at most
        tight = pocket(slack)
        assert tight.start is not None and tight.wedged
    assert not pocket(0.3).wedged
