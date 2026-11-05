"""Supplementary observation goals from immutable snapshots; no ROS or actions.

History records follow mission's proposed pose/at_s and pose/outcome contract.
They are structural protocols so the planner can be tested before mission's
historial_vistas module exists. Actual poses estimate observation opportunities,
not processed images. Mission owns freshness, history epochs and return budgets.
"""
from dataclasses import dataclass
import math
from typing import Protocol

from equipo_jar.mapa import Grid, Pose, check_cancelled, reachable_cells
from equipo_jar.cobertura_plan import Camera, Settings, angle_difference, distances, sampled_cells


class ObservedView(Protocol):
    pose: Pose
    at_s: float


class AttemptedView(Protocol):
    pose: Pose
    outcome: str


@dataclass(frozen=True)
class ExtraView:
    pose: Pose
    gain: int


@dataclass(frozen=True)
class ExtraPlan:
    views: tuple[ExtraView, ...]
    pending_count: int


def _number(value, name, *, minimum=0.0, positive=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < minimum or (positive and value == 0)):
        raise ValueError(f'{name} must be finite and within its allowed range')


def _pose(pose):
    if not isinstance(pose, Pose):
        raise ValueError('poses must be mapa.Pose records')
    for name in ('x', 'y', 'yaw'):
        value = getattr(pose, name)
        if isinstance(value, bool) or not math.isfinite(value):
            raise ValueError('pose components must be finite numbers')


def _overlap(points, i, j):
    """Positive-area overlap of a transformed cell and a unit square (SAT).

    Edge-only contact does not rasterize an extra obstacle cell. Ray tests later
    retain their closed-contact semantics. Tolerance is in cell coordinates.
    """
    axes = ((1.0, 0.0), (0.0, 1.0),
            (points[1][1] - points[0][1], points[0][0] - points[1][0]),
            (points[3][1] - points[0][1], points[0][0] - points[3][0]))
    for ax, ay in axes:
        norm = math.hypot(ax, ay)
        ax, ay = ax / norm, ay / norm
        projected = [ax * x + ay * y for x, y in points]
        center = ax * (i + 0.5) + ay * (j + 0.5)
        half = (abs(ax) + abs(ay)) / 2
        if min(max(projected), center + half) - max(min(projected), center - half) <= 1e-12:
            return False
    return True


def _project(base, overlay, cancel):
    """Overlay physical occupancy onto whole base cells, never only their centers."""
    if (base.width == overlay.width and base.height == overlay.height
            and base.origin == overlay.origin and base.resolution == overlay.resolution):
        return tuple(100 if v == 100 else -1 if v < 0 else 0 for v in overlay.data)
    dx, dy = base.origin.x - overlay.origin.x, base.origin.y - overlay.origin.y
    ox = (overlay.cos * dx + overlay.sin * dy) / overlay.resolution
    oy = (-overlay.sin * dx + overlay.cos * dy) / overlay.resolution
    scale = base.resolution / overlay.resolution
    angle = base.origin.yaw - overlay.origin.yaw
    ux, uy = scale * math.cos(angle), scale * math.sin(angle)
    vx, vy = -uy, ux
    result = []
    for j in range(base.height):
        check_cancelled(cancel)
        for i in range(base.width):
            x, y = ox + i * ux + j * vx, oy + i * uy + j * vy
            points = ((x, y), (x + ux, y + uy),
                      (x + ux + vx, y + uy + vy), (x + vx, y + vy))
            xmin, xmax = min(p[0] for p in points), max(p[0] for p in points)
            ymin, ymax = min(p[1] for p in points), max(p[1] for p in points)
            unknown = (xmin < -1e-12 or ymin < -1e-12
                       or xmax > overlay.width + 1e-12 or ymax > overlay.height + 1e-12)
            occupied = False
            for b in range(max(0, math.floor(ymin)), min(overlay.height, math.ceil(ymax))):
                check_cancelled(cancel)
                for a in range(max(0, math.floor(xmin)), min(overlay.width, math.ceil(xmax))):
                    value = overlay.data[b * overlay.width + a]
                    if (value == 100 or value < 0) and _overlap(points, a, b):
                        occupied |= value == 100
                        unknown |= value < 0
            result.append(100 if occupied else -1 if unknown else 0)
    return tuple(result)


def visibility_grid(map_grid: Grid, live_grid: Grid, *, cancel=None) -> Grid:
    """Mapped occupancy + live lethal cells; cost inflation is not optical opacity.

    Live 100 is opaque; 1..99 adds no optical obstacle. Live unknown/outside
    makes free map cells unresolved. Static unknown/occupied cells stay blocked.
    """
    check_cancelled(cancel)
    projected = _project(map_grid, live_grid, cancel)
    values = []
    for index, (static, live) in enumerate(zip(map_grid.data, projected)):
        if index % map_grid.width == 0:
            check_cancelled(cancel)
        values.append(-1 if static < 0 else 100 if live == 100
                      else -1 if live < 0 else static)
    return Grid(map_grid.width, map_grid.height, map_grid.resolution, values, map_grid.origin)


def _visible(grid, pose, targets, camera, settings, cancel):
    """Indices visible from the continuous actual pose (no cell-center snapping)."""
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    x = pose.x + c * camera.x - s * camera.y
    y = pose.y + s * camera.x + c * camera.y
    base = grid.cell(pose.x, pose.y)
    if (base is None or not grid.free(base, settings.free_threshold)
            or not grid.visible((pose.x, pose.y), (x, y), settings.free_threshold)):
        return []
    yaw = pose.yaw + camera.yaw
    radius2, half = settings.view_range ** 2, math.radians(settings.fov_degrees / 2)
    visible = []
    for index, (tx, ty) in targets:
        check_cancelled(cancel)
        dx, dy = tx - x, ty - y
        if dx * dx + dy * dy > radius2:
            continue
        if abs(angle_difference(math.atan2(dy, dx), yaw)) > half:
            continue
        if grid.visible((x, y), (tx, ty), settings.free_threshold):
            visible.append(index)
    return visible


def pending_samples(map_grid: Grid, live_grid: Grid,
                    observed: tuple[ObservedView, ...], camera: Camera,
                    settings: Settings, *, cancel=None,
                    occlusion_grid=None) -> tuple[tuple[float, float], ...]:
    visual = visibility_grid(map_grid, live_grid if occlusion_grid is None else occlusion_grid,
                             cancel=cancel)
    free = set()
    for j in range(map_grid.height):
        check_cancelled(cancel)
        free.update((i, j) for i in range(map_grid.width)
                    if map_grid.free((i, j), settings.free_threshold))
    cells = sampled_cells(free, settings.target_spacing, map_grid.resolution, settings.max_targets)
    pending = dict(enumerate(map_grid.center(cell) for cell in cells))
    distinct = set()
    for view in observed:
        check_cancelled(cancel)
        _pose(view.pose)
        _number(view.at_s, 'observation time')
        if view.pose in distinct:
            continue
        distinct.add(view.pose)
        if pending:
            for index in _visible(visual, view.pose, pending.items(), camera, settings, cancel):
                del pending[index]
    return tuple(pending.values())


def propose_extra_views(
    map_grid: Grid, live_grid: Grid, robot: Pose, start: Pose,
    observed: tuple[ObservedView, ...], attempted: tuple[AttemptedView, ...],
    camera: Camera, settings: Settings, *, limit: int, failed_shift_m: float,
    excluded: tuple[tuple[float, float, float], ...] = (), cancel=None,
    same_point_distance_m: float = 0.02, same_point_angle_rad: float = 0.10,
    occlusion_grid=None,
) -> ExtraPlan:
    """Rank alternative goals; mission executes one, then requests a new snapshot.

    Gain is independent per alternative, not an accumulated route prediction.
    Unsafe starts raise; disconnected endpoints never get snapped across walls.
    Empty views can coexist with pending samples that have no safe visible goal.
    Mission rechecks freshness, budget, exclusions and cancellation before travel.
    """
    check_cancelled(cancel)
    _pose(robot)
    _pose(start)
    if type(limit) is not int or limit < 0:
        raise ValueError('limit must be a nonnegative integer')
    for name, value in (('failed_shift_m', failed_shift_m),
                        ('same_point_distance_m', same_point_distance_m),
                        ('same_point_angle_rad', same_point_angle_rad)):
        _number(value, name)
    for zone in excluded:
        if len(zone) != 3:
            raise ValueError('exclusions must contain x, y, radius')
        x, y, radius = zone
        if any(isinstance(v, bool) or not math.isfinite(v) for v in (x, y)):
            raise ValueError('exclusion center must be finite')
        _number(radius, 'exclusion radius')
    failed = []
    for attempt in attempted:
        check_cancelled(cancel)
        _pose(attempt.pose)
        if attempt.outcome not in ('arrived', 'failed', 'rejected', 'cancelled'):
            raise ValueError('unknown navigation outcome')
        if attempt.outcome != 'arrived':
            failed.append(attempt.pose)
    pending = pending_samples(map_grid, live_grid, observed, camera, settings, cancel=cancel,
                              occlusion_grid=occlusion_grid)
    if not pending or limit == 0:
        return ExtraPlan((), len(pending))
    visual = visibility_grid(map_grid, live_grid if occlusion_grid is None else occlusion_grid,
                             cancel=cancel)

    # Live costs already encode navigation inflation. Project only static transit
    # safety onto them, then use radius zero on that combined navigation mask.
    static_safe = map_grid.safe_cells(settings.robot_radius + settings.safety_margin,
                                      settings.free_threshold, cancel)
    static_mask = Grid(map_grid.width, map_grid.height, map_grid.resolution,
                       [0 if (i, j) in static_safe else 100
                        for j in range(map_grid.height) for i in range(map_grid.width)],
                       map_grid.origin)
    static_on_live = _project(live_grid, static_mask, cancel)
    navigable = set()
    for j in range(live_grid.height):
        check_cancelled(cancel)
        for i in range(live_grid.width):
            k = j * live_grid.width + i
            if 0 <= live_grid.data[k] <= 98 and static_on_live[k] == 0:
                navigable.add((i, j))
    robot_cell, home_cell = live_grid.cell(robot.x, robot.y), live_grid.cell(start.x, start.y)
    if robot_cell not in navigable or home_cell not in navigable:
        raise ValueError('robot and start must lie in safe known navigation cells')
    connected = reachable_cells(navigable, robot_cell, cancel)
    if home_cell not in connected:
        return ExtraPlan((), len(pending))
    route = distances(connected, robot_cell, live_grid.resolution, cancel)
    clearance = max(settings.view_clearance, settings.robot_radius + settings.safety_margin)
    safe_views = visual.safe_cells(clearance, settings.free_threshold, cancel)
    preferred = visual.safe_cells(max(clearance, settings.view_preferred_clearance),
                                  settings.free_threshold, cancel)
    eligible = set()
    for cell in safe_views:
        check_cancelled(cancel)
        x, y = visual.center(cell)
        if live_grid.cell(x, y) not in connected:
            continue
        if any(math.hypot(x - p.x, y - p.y) <= failed_shift_m for p in failed):
            continue
        if any(math.hypot(x - zx, y - zy) <= radius for zx, zy, radius in excluded):
            continue
        eligible.add(cell)
    points = sampled_cells(eligible, settings.candidate_spacing, visual.resolution,
                           settings.max_positions)
    # Keep coarse search deterministic and recover unseen targets with finer
    # positions while there is room in the same total position budget.
    points.sort(key=lambda p: (p not in preferred, p[1], p[0]))
    options, tested, possible = [], set(), set()

    def evaluate(cell):
        check_cancelled(cancel)
        tested.add(cell)
        x, y = visual.center(cell)
        live_cell = live_grid.cell(x, y)
        for heading in range(settings.heading_count):
            pose = Pose(x, y, visual.origin.yaw + heading * math.tau / settings.heading_count)
            if any(math.hypot(x - v.pose.x, y - v.pose.y) <= same_point_distance_m
                   and abs(angle_difference(pose.yaw, v.pose.yaw)) <= same_point_angle_rad
                   for v in observed):
                continue
            visible = _visible(visual, pose, enumerate(pending), camera, settings, cancel)
            if not visible:
                continue
            possible.update(visible)
            distance = route[live_cell]
            duration = (distance / settings.travel_speed
                        + abs(angle_difference(pose.yaw, robot.yaw)) / settings.turn_speed
                        + settings.observation_time)
            options.append((-(len(visible) / duration), distance, x, y, pose.yaw,
                            ExtraView(pose, len(visible))))

    for cell in points:
        evaluate(cell)
    for index, target in enumerate(pending):
        check_cancelled(cancel)
        if len(tested) >= settings.max_positions:
            break
        if index in possible:
            continue
        # A physical/unknown endpoint cannot be seen by this planar model.
        # Keep it pending, but do not spend the fine-search budget on it.
        target_cell = visual.cell(*target)
        if target_cell is None or not visual.free(target_cell, settings.free_threshold):
            continue
        nearby = sorted((cell for cell in eligible - tested
                         if math.dist(visual.center(cell), target)
                         <= settings.view_range + math.hypot(camera.x, camera.y)),
                        key=lambda cell: (cell not in preferred,
                                          math.dist(visual.center(cell), target), cell[1], cell[0]))
        for cell in nearby:
            evaluate(cell)
            if index in possible or len(tested) >= settings.max_positions:
                break
    options.sort(key=lambda item: item[:5])
    return ExtraPlan(tuple(item[-1] for item in options[:min(limit, settings.max_views)]), len(pending))
