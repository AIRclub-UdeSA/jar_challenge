"""Map-based viewpoint planning, independent of ROS.

Visibility is a planar prediction, not proof that the camera processed an image.
Only the route below is predicted; execution and failure feedback need a team
agreement before we can maintain the remaining route during a run.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import heapq
import math

from equipo_jar.mapa import Grid, Pose, check_cancelled, reachable_cells


@dataclass(frozen=True)
class Settings:
    robot_radius: float = 0.18
    safety_margin: float = 0.03
    candidate_spacing: float = 0.45
    target_spacing: float = 0.15
    view_range: float = 2.0
    fov_degrees: float = 61.0
    heading_count: int = 8
    travel_speed: float = 0.25
    turn_speed: float = 0.7
    observation_time: float = 1.0
    max_views: int = 80
    max_positions: int = 256
    max_targets: int = 2048
    free_threshold: int = 25
    order_check_interval_s: float = 60.0
    order_start_delay_s: float = 8.0
    view_clearance: float = 0.24
    view_preferred_clearance: float = 0.30
    start_max_shift_m: float = 0.20
    start_min_goal_distance_m: float = 0.25

    def __post_init__(self):
        positive = ('robot_radius', 'candidate_spacing', 'target_spacing',
                    'view_range', 'travel_speed', 'turn_speed', 'observation_time')
        for name in positive:
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if not math.isfinite(self.safety_margin) or self.safety_margin < 0:
            raise ValueError('safety_margin must be finite and nonnegative')
        if not math.isfinite(self.fov_degrees) or not 0 < self.fov_degrees < 180:
            raise ValueError('fov_degrees must be between 0 and 180')
        for name in ('heading_count', 'max_views', 'max_positions', 'max_targets'):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if type(self.free_threshold) is not int or not 0 <= self.free_threshold < 100:
            raise ValueError('free_threshold must be an integer in [0, 100)')
        if (not math.isfinite(self.order_check_interval_s)
                or self.order_check_interval_s < self.observation_time):
            raise ValueError('order_check_interval_s must be finite and at least observation_time')
        if not math.isfinite(self.order_start_delay_s) or self.order_start_delay_s < 0:
            raise ValueError('order_start_delay_s must be finite and nonnegative')
        for name in ('view_clearance', 'view_preferred_clearance',
                     'start_max_shift_m', 'start_min_goal_distance_m'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f'{name} must be finite and nonnegative')


@dataclass(frozen=True)
class Camera:
    """Camera translation and forward-axis yaw in the robot frame, from TF."""
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0

    def __post_init__(self):
        if not all(math.isfinite(v) for v in (self.x, self.y, self.yaw)):
            raise ValueError('camera transform must be finite')


@dataclass(frozen=True)
class Plan:
    poses: tuple[Pose, ...]
    target_count: int
    predicted_count: int
    reachable_count: int
    candidate_count: int
    metrics: Metrics | None = None
    diagnostics: Diagnostics | None = None
    unpolished_metrics: Metrics | None = None
    start_anchor: Pose | None = None


@dataclass(frozen=True)
class Metrics:
    distance: float
    turning: float
    duration: float
    return_distance: float
    mean_first_view_time: float
    visible_from_candidates: int
    coverage_curve: tuple[tuple[float, int], ...]


@dataclass(frozen=True)
class Diagnostics:
    targets: tuple
    covered: tuple
    outside_candidates: tuple
    deferred: tuple
    positions: tuple
    legs: tuple
    added_positions: int


def angle_difference(a, b):
    return math.atan2(math.sin(a - b), math.cos(a - b))


def distances(cells, start, resolution, cancel=None):
    """Dijkstra distances, with no diagonal cuts through corners."""
    best, pending = {start: 0.0}, [(0.0, start)]
    while pending:
        check_cancelled(cancel)
        distance, current = heapq.heappop(pending)
        if distance != best[current]:
            continue
        i, j = current
        for dj in (-1, 0, 1):
            for di in (-1, 0, 1):
                nxt = (i + di, j + dj)
                if (not (di or dj) or nxt not in cells or
                        (di and dj and ((i + di, j) not in cells or (i, j + dj) not in cells))):
                    continue
                value = distance + resolution * math.hypot(di, dj)
                if value < best.get(nxt, math.inf):
                    best[nxt] = value
                    heapq.heappush(pending, (value, nxt))
    return best


def sampled_cells(cells, spacing, resolution, limit):
    """One representative per spatial bin; increase spacing if the map is large."""
    stride = max(1, math.ceil(spacing / resolution))
    ordered = sorted(cells, key=lambda c: (c[1], c[0]))
    while True:
        bins = {}
        for i, j in ordered:
            key = i // stride, j // stride
            center_i, center_j = (key[0] + 0.5) * stride, (key[1] + 0.5) * stride
            score = (i + 0.5 - center_i) ** 2 + (j + 0.5 - center_j) ** 2
            if key not in bins or score < bins[key][0]:
                bins[key] = score, (i, j)
        if len(bins) <= limit:
            return [entry[1] for entry in bins.values()]
        stride += max(1, stride // 4)


def visibility_mask(grid, cell, yaw, centers, settings, camera):
    x, y = grid.center(cell)
    co, si = math.cos(yaw), math.sin(yaw)
    optical = (x + co * camera.x - si * camera.y, y + si * camera.x + co * camera.y)
    mask = 0
    radius_squared = settings.view_range ** 2
    half_fov = math.radians(settings.fov_degrees / 2)
    for index, (tx, ty) in enumerate(centers):
        dx, dy = tx - optical[0], ty - optical[1]
        if dx * dx + dy * dy > radius_squared:
            continue
        if abs(angle_difference(math.atan2(dy, dx), yaw + camera.yaw)) > half_fov:
            continue
        if grid.visible(optical, (tx, ty), settings.free_threshold):
            mask |= 1 << index
    return mask


def _heading_geometry(grid, settings, camera):
    """Prepare camera terms once; retain the original arithmetic order."""
    result = []
    for heading in range(settings.heading_count):
        yaw = grid.origin.yaw + heading * math.tau / settings.heading_count
        co, si = math.cos(yaw), math.sin(yaw)
        result.append((co * camera.x, si * camera.y, si * camera.x,
                       co * camera.y, yaw + camera.yaw))
    return tuple(result)


def _has_visible_heading(grid, cell, center, geometry, radius_squared, half_fov, threshold):
    """Single-target witness without rebuilding camera transforms per heading."""
    x, y = grid.center(cell)
    tx, ty = center
    for ax, bx, ay, by, looking in geometry:
        optical = x + ax - bx, y + ay + by
        dx, dy = tx - optical[0], ty - optical[1]
        if dx * dx + dy * dy > radius_squared:
            continue
        if abs(angle_difference(math.atan2(dy, dx), looking)) > half_fov:
            continue
        if grid.visible(optical, center, threshold):
            return True
    return False


class TravelCosts:
    """Keep distances only between candidate centers, not full cached grid fields."""
    def __init__(self, cells, positions, resolution, cancel=None):
        self.cells, self.positions, self.resolution = cells, set(positions), resolution
        self.cache = {}
        self.turn_cache = {}
        self.cancel = cancel

    def between(self, first, last):
        check_cancelled(self.cancel)
        if first == last:
            return 0.0
        if first not in self.cache:
            field = distances(self.cells, first, self.resolution, self.cancel)
            self.cache[first] = {cell: field[cell] for cell in self.positions}
        return self.cache[first][last]

    def turn_between(self, first, last):
        """Reuse exact angular costs across trials of the same view set."""
        key = first, last
        if key not in self.turn_cache:
            self.turn_cache[key] = abs(angle_difference(first, last))
        return self.turn_cache[key]


def route_metrics(choices, start_cell, start_yaw, costs, settings, possible):
    distance = turning = elapsed = weighted_time = 0.0
    mask, cell, yaw = 0, start_cell, start_yaw
    curve = [(0.0, 0)]
    turn_cost = getattr(costs, 'turn_between', None)
    for nxt, pose, visible in choices:
        leg = costs.between(cell, nxt)
        turn = (turn_cost(yaw, pose.yaw) if turn_cost is not None
                else abs(angle_difference(yaw, pose.yaw)))
        distance += leg
        turning += turn
        elapsed += leg / settings.travel_speed + turn / settings.turn_speed + settings.observation_time
        gain = (visible & ~mask).bit_count()
        weighted_time += elapsed * gain
        mask |= visible
        curve.append((elapsed, mask.bit_count()))
        cell, yaw = nxt, pose.yaw
    return Metrics(distance, turning, elapsed, costs.between(cell, start_cell),
                   weighted_time / mask.bit_count() if mask else 0.0,
                   possible.bit_count(), tuple(curve))


class _RouteClock:
    """Exact duration before building visibility curves for rejected routes.

    Cache directed edge increments within one fixed-view ordering call. Retain
    route_metrics' arithmetic and accumulation order, including floating-point
    behavior at the existing duration thresholds.
    """
    def __init__(self, start_cell, start_yaw, costs, settings):
        self.start_cell, self.start_yaw = start_cell, start_yaw
        self.costs, self.settings = costs, settings
        self.turn_cost = getattr(costs, 'turn_between', None)
        self.edges = {}

    def __call__(self, choices):
        elapsed = 0.0
        cell, yaw = self.start_cell, self.start_yaw
        for nxt, pose, _ in choices:
            key = cell, yaw, nxt, pose.yaw
            increment = self.edges.get(key)
            if increment is None:
                leg = self.costs.between(cell, nxt)
                turn = (self.turn_cost(yaw, pose.yaw) if self.turn_cost is not None
                        else abs(angle_difference(yaw, pose.yaw)))
                increment = (leg / self.settings.travel_speed + turn / self.settings.turn_speed
                             + self.settings.observation_time)
                self.edges[key] = increment
            elapsed += increment
            cell, yaw = nxt, pose.yaw
        return elapsed


def improve_order(choices, start_cell, start_yaw, costs, settings, possible, cancel=None):
    """Bounded 2-opt and relocation sweeps, retaining every selected view."""
    choices = list(choices)
    metrics = route_metrics(choices, start_cell, start_yaw, costs, settings, possible)
    clock = _RouteClock(start_cell, start_yaw, costs, settings)
    for first in range(len(choices) - 1):
        for last in range(first + 1, len(choices)):
            check_cancelled(cancel)
            candidate = choices[:first] + list(reversed(choices[first:last + 1])) + choices[last + 1:]
            if clock(candidate) >= metrics.duration - 1e-9:
                continue
            trial = route_metrics(candidate, start_cell, start_yaw, costs, settings, possible)
            if (trial.duration < metrics.duration - 1e-9
                    and trial.mean_first_view_time <= metrics.mean_first_view_time + 1e-9):
                choices, metrics = candidate, trial
    # Keep every view: equal sampled coverage does not prove equal visibility
    # between samples, so removing a seemingly redundant pose would be unsafe.
    choices = relocate_order(choices, start_cell, start_yaw, costs, settings, possible,
                             settings.order_check_interval_s, settings.order_start_delay_s, cancel,
                             _clock=clock)
    # Later moves can unlock improvements for views visited early in the first
    # sweep. One cleanup sweep keeps the work bounded and repeats every guard.
    return relocate_order(choices, start_cell, start_yaw, costs, settings, possible,
                          settings.order_check_interval_s, settings.order_start_delay_s, cancel,
                          _clock=clock)


def _order_checkpoints(curve, interval, start_delay):
    """Only the first clock-grid check after each coverage increase can fail.

    At most one check per view is needed, even for a very small interval.
    Times in the curve are relative to route start; the grid includes the
    estimated initial delay. This is a prediction, not mission's actual clock.
    """
    result = []
    previous = 0
    for seconds, count in curve:
        if count == previous:
            continue
        previous = count
        check = math.ceil((seconds + start_delay - 1e-9) / interval) * interval - start_delay
        if result and abs(check - result[-1][0]) <= 1e-9:
            result[-1] = check, count
        else:
            result.append((check, count))
    return result


def _meets_order_checks(curve, checks):
    index = 0
    for seconds, required in checks:
        while index + 1 < len(curve) and curve[index + 1][0] <= seconds + 1e-9:
            index += 1
        if curve[index][1] < required:
            return False
    return True


def relocate_order(choices, start_cell, start_yaw, costs, settings, possible,
                   interval, start_delay, cancel=None, *, _clock=None):
    """One bounded relocation sweep, preserving all views and sampled checks.

    Accept only lower mean first-view time (or equal mean and shorter search),
    with no increase in search duration or search-plus-return estimate. Do not
    reduce sampled visibility on the configured predicted clock grid. This
    does not guarantee dense visibility between target samples or clock checks.
    """
    if not math.isfinite(interval) or interval < settings.observation_time:
        raise ValueError('order interval must be finite and at least observation_time')
    if not math.isfinite(start_delay) or start_delay < 0:
        raise ValueError('order start delay must be finite and nonnegative')
    check_cancelled(cancel)
    current = list(choices)
    metrics = route_metrics(current, start_cell, start_yaw, costs, settings, possible)
    clock = _RouteClock(start_cell, start_yaw, costs, settings) if _clock is None else _clock
    for entry in tuple(choices):
        check_cancelled(cancel)
        first = current.index(entry)
        rest = current[:first] + current[first + 1:]
        checks = _order_checkpoints(metrics.coverage_curve, interval, start_delay)
        best, best_metrics = current, metrics
        for last in range(len(current)):
            check_cancelled(cancel)
            if last == first:
                continue
            trial = rest[:last] + [entry] + rest[last:]
            if clock(trial) > metrics.duration + 1e-9:
                continue
            candidate = route_metrics(trial, start_cell, start_yaw, costs, settings, possible)
            if candidate.duration > metrics.duration + 1e-9:
                continue
            if (candidate.duration + candidate.return_distance / settings.travel_speed
                    > metrics.duration + metrics.return_distance / settings.travel_speed + 1e-9):
                continue
            earlier = candidate.mean_first_view_time < best_metrics.mean_first_view_time - 1e-9
            equal = abs(candidate.mean_first_view_time - best_metrics.mean_first_view_time) <= 1e-9
            if not (earlier or (equal and candidate.duration < best_metrics.duration - 1e-9)):
                continue
            if not _meets_order_checks(candidate.coverage_curve, checks):
                continue
            best, best_metrics = trial, candidate
        current, metrics = best, best_metrics
    return current


def shortest_leg(cells, start, goal, resolution):
    """Reconstruct a shortest grid leg for diagnostic plots only."""
    if start == goal:
        return (start,)
    best, parents = {start: 0.0}, {}
    pending = [(resolution * math.dist(start, goal), 0.0, start)]
    while pending:
        _, distance, current = heapq.heappop(pending)
        if distance != best[current]:
            continue
        if current == goal:
            result = [goal]
            while result[-1] != start:
                result.append(parents[result[-1]])
            return tuple(reversed(result))
        i, j = current
        for dj in (-1, 0, 1):
            for di in (-1, 0, 1):
                nxt = i + di, j + dj
                if (not (di or dj) or nxt not in cells or
                        (di and dj and ((i + di, j) not in cells or (i, j + dj) not in cells))):
                    continue
                value = distance + resolution * math.hypot(di, dj)
                if value < best.get(nxt, math.inf):
                    best[nxt], parents[nxt] = value, current
                    heapq.heappush(pending, (value + resolution * math.dist(nxt, goal), value, nxt))
    raise ValueError('no grid path between observation goals')


def observation_cells(grid, reachable, settings, cancel=None, *, preferred=False):
    """Reachable centers with room for observation, separate from transit.

    The minimum envelope must cover the robot at every requested heading.
    Prefer the larger envelope where it can supply the needed visibility.
    Neither certifies map accuracy, localization or unmapped obstacles.
    Setting both observation clearances to zero restores the transit mask.
    """
    transit_clearance = settings.robot_radius + settings.safety_margin
    clearance = max(settings.view_clearance,
                    settings.view_preferred_clearance if preferred else 0.0)
    if clearance <= transit_clearance:
        return reachable
    return reachable & grid.safe_cells(clearance, settings.free_threshold, cancel)


def planning_start_cell(grid, start, safe, settings, cancel=None):
    """Choose a nearby planning anchor only from a known-free actual start.

    A bounded, clear center segment prevents snapping through walls/unknowns.
    This does not certify the swept robot footprint or command an escape:
    Nav2 must still plan and execute from the real pose. Shared map reachability
    stays strict. A zero shift limit disables this initial-plan fallback.
    """
    check_cancelled(cancel)
    actual = grid.cell(start.x, start.y)
    if actual in safe:
        return actual
    if (actual is None or not grid.free(actual, settings.free_threshold)
            or settings.start_max_shift_m == 0):
        raise ValueError('robot pose is outside the safe free space; no local free start')
    candidates = []
    for cell in safe:
        check_cancelled(cancel)
        distance = math.dist((start.x, start.y), grid.center(cell))
        if distance <= settings.start_max_shift_m:
            candidates.append((distance, cell[1], cell[0], cell))
    for _, _, _, cell in sorted(candidates):
        check_cancelled(cancel)
        if grid.visible((start.x, start.y), grid.center(cell), settings.free_threshold):
            return cell
    raise ValueError('robot pose is outside the safe free space; no nearby visible safe anchor')


def _recovery_order(choices, start, settings):
    """Keep all selected views but avoid mission's nearby-goal in-place shortcut.

    The configured minimum must exceed mission's turn_in_place_m. This is a
    dispatch compatibility check, not an orientation/arrival guarantee from Nav2.
    """
    for index, (_, pose, _) in enumerate(choices):
        if math.dist((start.x, start.y), (pose.x, pose.y)) >= settings.start_min_goal_distance_m:
            return choices[index:index + 1] + choices[:index] + choices[index + 1:]
    raise ValueError('no observation goal beyond the start recovery distance')


def _entry_metrics(metrics, distance, settings):
    """Include the local connector estimate in outbound and return accounting."""
    seconds = distance / settings.travel_speed
    return replace(metrics, distance=metrics.distance + distance,
                   duration=metrics.duration + seconds,
                   return_distance=metrics.return_distance + distance,
                   mean_first_view_time=(metrics.mean_first_view_time + seconds
                                         if len(metrics.coverage_curve) > 1 else 0.0),
                   coverage_curve=((0.0, 0),) + tuple(
                       (t + seconds, count) for t, count in metrics.coverage_curve[1:]))


def make_plan(grid, start, settings=Settings(), camera=Camera(), *,
              refine=True, improve=True, diagnostics=False, cancel=None):
    """Greedily cover sampled free cells, balancing new visibility and travel time.

    Output poses are observation goals, NOT a collision-free trajectory between
    goals. Navigation must plan and execute each leg, with dynamic obstacles.
    """
    check_cancelled(cancel)
    safe = grid.safe_cells(settings.robot_radius + settings.safety_margin,
                           settings.free_threshold, cancel)
    start_cell = planning_start_cell(grid, start, safe, settings, cancel)
    recovery = start_cell != grid.cell(start.x, start.y)
    reachable = reachable_cells(safe, start_cell)
    observation = observation_cells(grid, reachable, settings, cancel)
    preferred = observation_cells(grid, observation, settings, cancel, preferred=True)
    positions = sampled_cells(preferred, settings.candidate_spacing, grid.resolution,
                              settings.max_positions)
    if start_cell in preferred and start_cell not in positions:
        positions = [start_cell] + positions[:settings.max_positions - 1]
    free = {(i, j) for j in range(grid.height) for i in range(grid.width)
            if grid.free((i, j), settings.free_threshold)}
    targets = sampled_cells(free, settings.target_spacing, grid.resolution,
                            settings.max_targets)
    centers = [grid.center(cell) for cell in targets]
    options, possible = [], 0

    def add_position(cell):
        nonlocal possible
        check_cancelled(cancel)
        x, y = grid.center(cell)
        for heading in range(settings.heading_count):
            yaw = grid.origin.yaw + heading * math.tau / settings.heading_count
            mask = visibility_mask(grid, cell, yaw, centers, settings, camera)
            if mask:
                options.append((cell, Pose(x, y, yaw), mask))
                possible |= mask

    for cell in positions:
        add_position(cell)
    base_options = tuple(options)
    added = 0
    if refine:
        # Search fine safe cells only for targets missed by coarse sampling.
        # Exhaust preferred centers before using a minimum-clearance witness.
        # A target in another raw free-space component cannot have a wall-free
        # line of sight from this component in the conservative planar model.
        existing = set(positions)
        raw_component = reachable_cells(free, start_cell)
        radius_squared = ((settings.view_range + math.hypot(camera.x, camera.y)) / grid.resolution) ** 2
        geometry = _heading_geometry(grid, settings, camera)
        view_radius_squared = settings.view_range ** 2
        half_fov = math.radians(settings.fov_degrees / 2)
        for index, center in enumerate(centers):
            check_cancelled(cancel)
            if len(positions) >= settings.max_positions:
                break
            if possible & (1 << index) or targets[index] not in raw_component:
                continue
            ti, tj = targets[index]
            nearby = sorted((cell for cell in observation if cell not in existing
                             and (cell[0] - ti) ** 2 + (cell[1] - tj) ** 2 <= radius_squared),
                            key=lambda cell: (cell not in preferred,
                                              (cell[0] - ti) ** 2 + (cell[1] - tj) ** 2,
                                              cell[1], cell[0]))
            for cell in nearby:
                check_cancelled(cancel)
                witness = _has_visible_heading(
                    grid, cell, center, geometry, view_radius_squared,
                    half_fov, settings.free_threshold)
                if witness:
                    existing.add(cell)
                    add_position(cell)
                    positions.append(cell)
                    added += 1
                    break
    costs = TravelCosts(reachable, positions + [start_cell], grid.resolution, cancel)
    result, predicted = [], 0
    current_cell, current_yaw = start_cell, start.yaw
    def select_views(available, budget):
        nonlocal predicted, current_cell, current_yaw
        for _ in range(budget):
            chosen, best_score = None, 0.0
            for cell, pose, mask in available:
                if (recovery and not result
                        and math.dist((start.x, start.y), (pose.x, pose.y))
                        < settings.start_min_goal_distance_m):
                    continue
                gain = (mask & ~predicted).bit_count()
                if not gain:
                    continue
                duration = (costs.between(current_cell, cell) / settings.travel_speed
                            + abs(angle_difference(pose.yaw, current_yaw)) / settings.turn_speed
                            + settings.observation_time)
                score = gain / duration
                if score > best_score:
                    chosen, best_score = (cell, pose, mask), score
            if chosen is None:
                break
            cell, pose, mask = chosen
            result.append(chosen)
            predicted |= mask
            current_yaw, current_cell = pose.yaw, cell

    # Preserve the initial plan's complete view set, including visibility
    # between target samples, then fill gaps within the remaining view budget.
    select_views(base_options, settings.max_views)
    select_views(options, settings.max_views - len(result))
    if recovery:
        result = _recovery_order(result, start, settings)
    unpolished = route_metrics(result, start_cell, start.yaw, costs, settings, possible)
    if improve:
        result = improve_order(result, start_cell, start.yaw, costs, settings, possible, cancel)
    if recovery:
        result = _recovery_order(result, start, settings)
    metrics = route_metrics(result, start_cell, start.yaw, costs, settings, possible)
    anchor = None
    if recovery:
        anchor = Pose(*grid.center(start_cell), start.yaw)
        shift = math.dist((start.x, start.y), (anchor.x, anchor.y))
        metrics = _entry_metrics(metrics, shift, settings)
        unpolished = _entry_metrics(unpolished, shift, settings)
    details = None
    if diagnostics:
        covered = tuple(cell for index, cell in enumerate(targets) if predicted & (1 << index))
        outside = tuple(cell for index, cell in enumerate(targets) if not possible & (1 << index))
        deferred = tuple(cell for index, cell in enumerate(targets)
                         if possible & (1 << index) and not predicted & (1 << index))
        legs, current = [], start_cell
        for cell, _, _ in result:
            check_cancelled(cancel)
            legs.append(shortest_leg(reachable, current, cell, grid.resolution))
            current = cell
        details = Diagnostics(tuple(targets), covered, outside, deferred,
                              tuple(positions), tuple(legs), added)
    return Plan(tuple(entry[1] for entry in result), len(targets), predicted.bit_count(),
                len(reachable), len(options), metrics, details, unpolished, anchor)
