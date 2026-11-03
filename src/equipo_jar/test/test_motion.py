"""The path tracker on a simulated holonomic robot with synthetic obstacles. The simulated base
follows the commands at its own acceleration (about the simulator's: 1 m/s^2, 1 rad/s^2), and the
tracker gets the velocity it has."""
import math

import numpy as np

from equipo_jar.motion import Tracker, inside, safe, DEFAULTS


def run(path, obstacles=np.zeros((0, 2)), pose=(0.0, 0.0, 0.0), dt=0.05, steps=400,
        base_a=1.0, base_alfa=1.0):
    """Drive a holonomic robot with the tracker; obstacles in the world frame. The base's
    velocity follows the command at base_a (m/s^2) and base_alfa (rad/s^2). Returns the poses,
    the last state and the most obstacle points ever inside the chassis grown by its safety
    margin (less 2 mm)."""
    t = Tracker(path)
    x, y, th = pose
    v = np.zeros(3)
    poses, worst, state = [], 0, None
    for _ in range(steps):
        c, s = math.cos(th), math.sin(th)
        rel = obstacles - (x, y)
        pts = np.column_stack([c * rel[:, 0] + s * rel[:, 1], -s * rel[:, 0] + c * rel[:, 1]])
        worst = max(worst, inside(pts, (0, 0, 0), DEFAULTS['huella_x'], DEFAULTS['huella_y'],
                                  DEFAULTS['margen_seguridad'] - 0.002))
        cmd, state = t.step((x, y, th), pts, dt, tuple(v))
        if state == 'llego':
            break
        lim = np.array([base_a, base_a, base_alfa]) * dt
        v += np.clip(np.asarray(cmd) - v, -lim, lim)
        x += (c * v[0] - s * v[1]) * dt
        y += (s * v[0] + c * v[1]) * dt
        th += v[2] * dt
        poses.append((x, y, th))
    return poses, state, worst


def line(x0, x1, y=0.0, yaw=0.0, step=0.1):
    n = int(round((x1 - x0) / step))
    return [(x0 + k * step, y, yaw) for k in range(n + 1)]


def test_follows_a_straight_path_to_its_end():
    poses, state, _ = run(line(0.0, 2.0))
    assert state == 'llego'
    assert math.dist(poses[-1][:2], (2.0, 0.0)) < 0.05


def test_holds_the_heading_of_the_path_while_translating_sideways():
    # Heading 90 deg along a path in +x: the holonomic base strafes.
    poses, state, _ = run(line(0.0, 1.5, yaw=math.pi / 2), pose=(0.0, 0.0, math.pi / 2))
    assert state == 'llego'
    assert all(abs(p[2] - math.pi / 2) < 0.1 for p in poses)


def test_turns_in_place_to_a_single_pose():
    poses, state, _ = run([(0.0, 0.0, math.pi / 2)])
    assert state == 'llego'
    assert abs(poses[-1][2] - math.pi / 2) < 0.1
    assert max(math.hypot(p[0], p[1]) for p in poses) < 0.02


def test_stops_short_of_a_wall_across_the_path():
    wall = np.column_stack([np.full(41, 1.0), np.linspace(-1, 1, 41)])
    poses, state, worst = run(line(0.0, 2.0), wall)
    assert state == 'bloqueado'
    assert worst == 0                      # never touched it
    front = DEFAULTS['huella_x'][1]
    assert poses[-1][0] + front < 1.0      # stopped before it


def test_drives_through_a_corridor_barely_wider_than_the_robot():
    # 0.26 m wide: the 0.20 m chassis has 3 cm a side. Side walls don't slow it down (only
    # what is ahead of the motion does), and it never touches them.
    xs = np.linspace(-0.5, 3.0, 141)
    corridor = np.vstack([np.column_stack([xs, np.full_like(xs, 0.13)]),
                          np.column_stack([xs, np.full_like(xs, -0.13)])])
    poses, state, worst = run(line(0.0, 2.0), corridor)
    assert state == 'llego' and worst == 0


def test_drives_on_along_a_wall_when_turning_to_the_path_heading_would_touch_it():
    # 3 cm from a wall on the left, a path along it held at a heading 22.5 deg off (one of
    # the planner's heading bins): turning to it would swing the rear-left corner into the
    # wall, driving on doesn't. The robot drives on and turns once there is room.
    xs = np.linspace(-0.5, 3.0, 351)
    wall = np.column_stack([xs, np.full_like(xs, DEFAULTS['huella_y'] + 0.03)])
    wall = wall[wall[:, 0] < 1.0]                # room to turn past x = 1 m
    poses, state, worst = run(line(0.0, 1.8, yaw=-math.radians(22.5)), wall, steps=600)
    assert state == 'llego' and worst == 0


def test_a_robot_touching_something_may_move_away_from_it():
    pts = np.array([[0.125, 0.0], [0.125, 0.05], [0.125, -0.05]])   # just inside the front
    p = dict(DEFAULTS)
    assert not safe(pts, (0, 0, 0), (0.1, 0.0, 0.0), p)   # further in: no
    assert safe(pts, (0, 0, 0), (-0.1, 0.0, 0.0), p)      # back out: yes


def test_a_fast_turn_stops_before_the_tail_reaches_a_post():
    # A post 0.19 m from the center, 30 deg to the left: clear of the front corners, but the
    # rear-left one reaches it after about 120 deg of turning clockwise. By then the turn is
    # fast, and the base needs about a radian to stop it.
    a = math.radians(30)
    post = np.array([[0.19 * math.cos(a) + dx, 0.19 * math.sin(a) + dy]
                     for dx in (0.0, 0.01, 0.02) for dy in (0.0, 0.01, 0.02)])
    poses, state, worst = run([(0.0, 0.0, math.pi)], post)
    assert worst == 0
    assert state == 'bloqueado'


def test_the_stop_starts_from_the_velocity_the_robot_has():
    # A wall 0.17 m to the left: turning clockwise, the rear-left corner reaches it after
    # about 20 deg. A slow turn from rest stops well before; a turn already at 1.5 rad/s takes
    # more than a radian to stop, so even stopping it now is unsafe.
    xs = np.linspace(-1.0, 1.0, 201)
    wall = np.column_stack([xs, np.full_like(xs, 0.17)])
    p = dict(DEFAULTS)
    assert safe(wall, (0, 0, 0), (0.0, 0.0, -0.2), p)
    assert not safe(wall, (0, 0, -1.5), (0.0, 0.0, 0.0), p)



def noisy_run(path, walls, margin, steps, noise=0.004, seed=0):
    """Drive like run(), each scan with range noise (m, sigma). Returns the last state, the
    steps taken and the closest the true walls came to the chassis (m)."""
    hy, xs = DEFAULTS['huella_y'], DEFAULTS['huella_x']
    rng = np.random.default_rng(seed)
    t = Tracker(path, margen_seguridad=margin)
    x = y = th = 0.0
    v = np.zeros(3)
    closest, state = 1.0, None
    for k in range(steps):
        c, s = math.cos(th), math.sin(th)
        rel = walls - (x, y)
        pts = np.column_stack([c * rel[:, 0] + s * rel[:, 1], -s * rel[:, 0] + c * rel[:, 1]])
        gap = np.maximum.reduce([xs[0] - pts[:, 0], pts[:, 0] - xs[1], np.abs(pts[:, 1]) - hy])
        closest = min(closest, gap.min())
        r = np.hypot(pts[:, 0], pts[:, 1])
        cmd, state = t.step((x, y, th), pts * ((r + rng.normal(0, noise, len(r))) / r)[:, None],
                            0.05, tuple(v))
        if state == 'llego':
            return state, k, closest
        v += np.clip(np.asarray(cmd) - v, -0.05, 0.05)
        x += (c * v[0] - s * v[1]) * 0.05
        y += (s * v[0] + c * v[1]) * 0.05
        th += v[2] * 0.05
    return state, steps, closest


def test_a_wedged_robot_doesnt_creep_deeper_on_the_noise_of_its_scans():
    # m04: 2.5 cm from a wall on the right and one ahead, inside the 3 cm margin, the path into
    # the corner between them and turning. Each scan has 4 mm of range noise. Allowed as deep
    # as the deepest point looked, the robot turned its corners into the walls, scan after
    # scan (to 1.9 cm).
    hy, x1 = DEFAULTS['huella_y'], DEFAULTS['huella_x'][1]
    s = np.linspace(-0.4, 0.4, 161)
    walls = np.vstack([np.column_stack([s, np.full_like(s, -hy - 0.025)]),
                       np.column_stack([np.full_like(s, x1 + 0.025), s + 0.2])])
    for turn in (-0.6, 0.6):
        state, _, closest = noisy_run([(0.0, 0.0, 0.0), (0.3, -0.3, turn)], walls, 0.03, 300)
        assert closest >= 0.025 - 0.002, closest


def test_drives_along_a_corridor_inside_its_margin_with_noisy_scans():
    # 2.5 cm a side, inside the 3 cm margin: every wall point coming in at the front corners
    # slides in along a side at the depth the robot already has, noise or not.
    hy = DEFAULTS['huella_y']
    s = np.linspace(-0.5, 2.5, 601)
    walls = np.vstack([np.column_stack([s, np.full_like(s, -hy - 0.025)]),
                       np.column_stack([s, np.full_like(s, hy + 0.025)])])
    path = [(0.1 * k, 0.0, 0.05 * math.sin(k)) for k in range(21)]
    state, steps, closest = noisy_run(path, walls, 0.03, 400)
    assert state == 'llego' and steps * 0.05 < 10.0
    assert closest >= 0.025 - 0.002, closest


def settle(poses, v, dt=0.05, base_alfa=1.0):
    """Where the base comes to rest after the tracker's last command: zero, which it reaches
    at its own deceleration (the tracker stops commanding at 'llego')."""
    th = poses[-1][2]
    w = v
    while abs(w) > 1e-6:
        w -= math.copysign(min(abs(w), base_alfa * dt), w)
        th += w * dt
    return th


def test_a_turn_in_place_comes_to_rest_at_the_heading():
    """A look in laberinto_1: a 2 rad turn to face a cage. Crossing the tolerance band at full
    rate and stopping there, the base coasted ~0.8 rad past it, and the camera looked at a
    wall. The turn must brake in time, and arrive slow enough to stay inside the band."""
    for target in (2.0, -2.5, 1.0):
        t = Tracker([(0.0, 0.0, target)])
        th, v, dt = 0.0, np.zeros(3), 0.05
        for _ in range(400):
            cmd, state = t.step((0.0, 0.0, th), np.zeros((0, 2)), dt, tuple(v))
            if state == 'llego':
                break
            v += np.clip(np.asarray(cmd) - v, -dt, dt)  # 1 rad/s^2
            th += v[2] * dt
        assert state == 'llego'
        rest = settle([(0.0, 0.0, th)], v[2])
        assert abs(rest - target) < 0.15, (target, th, v[2], rest)
