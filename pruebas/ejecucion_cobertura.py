"""Audit commanded/recorded views without treating failed trips as observations.

Pure offline geometry. A successful navigation goal is still not an image ack.
"""
from dataclasses import dataclass
import math
from types import SimpleNamespace

from equipo_jar.cobertura_plan import Pose, angle_difference, visibility_mask
from equipo_jar.exploracion import first_pending, goal_position, near_box
from metricas_cobertura import Event, Trace


@dataclass(frozen=True)
class GoalRecord:
    index: int
    elapsed: float
    pose: Pose
    status: str = 'succeeded'
    recoveries: int | None = None

    def __post_init__(self):
        if not isinstance(self.pose, Pose):
            raise ValueError('goal pose must be a finite map Pose')
        if type(self.index) is not int or self.index < 0:
            raise ValueError('goal index must be a nonnegative integer')
        if not math.isfinite(self.elapsed) or self.elapsed < 0:
            raise ValueError('elapsed must be finite and nonnegative')
        if self.status not in ('succeeded', 'failed', 'canceled'):
            raise ValueError('unknown goal status')
        if self.recoveries is not None and (type(self.recoveries) is not int or self.recoveries < 0):
            raise ValueError('recoveries must be a nonnegative integer or unknown')


@dataclass(frozen=True)
class Prediction:
    records: tuple
    box_skipped: tuple
    identity_skipped: tuple
    shifted: tuple
    startup: float
    settle: float


def records_from_payload(payload):
    """Require provenance before mixing external goal records with predictions."""
    if not isinstance(payload, dict):
        raise ValueError('record payload must be an object')
    if payload.get('frame_id') != 'map' or payload.get('time_origin') != '/map':
        raise ValueError('records must use map poses and seconds since /map')
    if payload.get('clock') not in ('simulated', 'wall'):
        raise ValueError('one explicit simulated or wall clock is required')
    if payload.get('pose_source') not in ('localized', 'commanded', 'predicted'):
        raise ValueError('pose_source must distinguish localized, commanded and predicted poses')
    if not isinstance(payload.get('source'), str) or not payload['source'].strip():
        raise ValueError('record source is required')
    try:
        return tuple(GoalRecord(row['index'], row['elapsed'], Pose(**row['pose']), row['status'],
                                 row.get('recoveries'))
                     for row in payload['goals'])
    except (KeyError, TypeError) as exc:
        raise ValueError('each goal needs index, elapsed, map pose and explicit status') from exc


def predict_goals(grid, poses, start, costs, settings, mission, *, zones=(), planning_s=0.0):
    """Use the real selector and goal_position with exact commanded arrivals.

    Known fixed zones only. No inspections, reports, live cancellations, drift,
    recoveries or in-transit camera views. Arrival gets a geometric view; settle
    delays the NEXT trip. Do not add the planner's observation_time as a dwell.
    """
    for name in ('same_point_distance_m', 'same_point_angle_rad'):
        if not math.isfinite(mission[name]) or mission[name] <= 0:
            raise ValueError('mission identity tolerances must be finite and positive')
    for name in ('start_wait_s', 'settle_s', 'turn_in_place_m'):
        value = mission.get(name, 0.0)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f'{name} must be finite and nonnegative')
    if not math.isfinite(planning_s) or planning_s < 0:
        raise ValueError('planning time must be finite and nonnegative')
    messages = tuple(SimpleNamespace(pose=SimpleNamespace(
        position=SimpleNamespace(x=p.x, y=p.y),
        orientation=SimpleNamespace(x=0.0, y=0.0,
                                    z=math.sin(p.yaw / 2), w=math.cos(p.yaw / 2))))
                     for p in poses)
    lookup = {id(message): index for index, message in enumerate(messages)}
    blocked = tuple(index for index, p in enumerate(poses) if near_box(p.x, p.y, zones))
    visited, selected, records, shifted = [], set(), [], []
    robot = start
    startup = elapsed = max(mission['start_wait_s'], planning_s)
    settle = mission.get('settle_s', 0.0)
    while True:
        point = first_pending(messages, visited, mission['same_point_distance_m'],
                              mission['same_point_angle_rad'], zones)
        if point is None:
            break
        index = lookup[id(point)]
        if index in selected:
            raise RuntimeError('mission selector repeated a finished goal')
        p = poses[index]
        x, y, _ = goal_position(p.x, p.y, (robot.x, robot.y, robot.yaw),
                                mission.get('turn_in_place_m', 0.0))
        commanded = Pose(x, y, p.yaw)
        leg = costs.between(grid.cell(robot.x, robot.y), grid.cell(x, y))
        if not math.isfinite(leg) or leg < 0:
            raise ValueError('trip has no finite nonnegative travel cost')
        elapsed += (leg / settings.travel_speed
                    + abs(angle_difference(commanded.yaw, robot.yaw)) / settings.turn_speed)
        records.append(GoalRecord(index, elapsed, commanded))
        if (x, y) != (p.x, p.y):
            shifted.append(index)
        visited.append((p.x, p.y, p.yaw))  # Mission remembers the planned point.
        selected.add(index)
        robot = commanded
        elapsed += settle
    identity = tuple(i for i in range(len(poses)) if i not in selected and i not in blocked)
    return Prediction(tuple(records), blocked, identity, tuple(shifted), startup, settle)


def prediction_trace(grid, start, prediction, cells, settings, camera, costs):
    """Adapt exact-commanded-arrival predictions to the existing budget schema.

    This is still a proxy, not an image acknowledgement or measured navigation.
    Recorded failures/cancellations belong in audit_records, never in this path.
    """
    mask = 0
    events = []
    distance = turning = 0.0
    previous = start
    start_cell = grid.cell(start.x, start.y)
    for record in prediction.records:
        if record.status != 'succeeded':
            raise ValueError('only predicted successful arrivals can build this trace')
        cell = grid.cell(record.pose.x, record.pose.y)
        distance += costs.between(grid.cell(previous.x, previous.y), cell)
        turning += abs(angle_difference(record.pose.yaw, previous.yaw))
        mask |= view_mask(grid, record.pose, cells, settings, camera)
        events.append(Event(record.index, record.elapsed, mask, costs.between(cell, start_cell)))
        previous = record.pose
    return Trace(tuple(events), distance, turning, prediction.startup)


def view_mask(grid, pose, cells, settings, camera):
    """Visibility at the quantized pose cell, matching coverage's current model."""
    cell = grid.cell(pose.x, pose.y)
    if cell is None or not grid.free(cell, settings.free_threshold):
        return 0
    return visibility_mask(grid, cell, pose.yaw, [grid.center(c) for c in cells], settings, camera)


def perturbed_mask(grid, pose, cells, settings, camera, offsets):
    """Intersection over explicit (map dx, map dy, yaw error) samples.

    This is a sampled sensitivity envelope, not a continuous-error guarantee.
    Map/world offsets are explicit; start placement is not localization error.
    """
    if not offsets:
        raise ValueError('at least one perturbation sample is required')
    mask = (1 << len(cells)) - 1
    for dx, dy, yaw in offsets:
        sample = Pose(pose.x + dx, pose.y + dy, pose.yaw + yaw)
        mask &= view_mask(grid, sample, cells, settings, camera)
    return mask


def audit_records(grid, records, cells, settings, camera, budgets, *, offsets=((0.0, 0.0, 0.0),)):
    """Keep failed/cancelled durations, but grant no view for those records.

    Input elapsed times must share one /map origin and one clock. Inputs can
    be measured or predicted; the caller must label that provenance explicitly.
    """
    records, cells, budgets, offsets = tuple(records), tuple(cells), tuple(budgets), tuple(offsets)
    if not offsets or any(len(p) != 3 or not all(math.isfinite(v) for v in p) for p in offsets):
        raise ValueError('finite dx/dy/yaw perturbation samples are required')
    if any(not math.isfinite(t) or t < 0 for t in budgets):
        raise ValueError('budgets must be finite and nonnegative')
    if any(b.elapsed < a.elapsed for a, b in zip(records, records[1:])):
        raise ValueError('goal records must be ordered by elapsed time')
    nominal = robust = 0
    events = []
    invalid = []
    for record in records:
        if record.status == 'succeeded':
            cell = grid.cell(record.pose.x, record.pose.y)
            if cell is None or not grid.free(cell, settings.free_threshold):
                invalid.append(record.index)
            nominal |= view_mask(grid, record.pose, cells, settings, camera)
            robust |= perturbed_mask(grid, record.pose, cells, settings, camera, offsets)
        events.append((record.elapsed, nominal, robust, record.status))
    snapshots = {}
    for budget in budgets:
        n = r = succeeded = attempts = 0
        for elapsed, nm, rm, status in events:
            if elapsed > budget:
                break
            n, r = nm, rm
            attempts += 1
            succeeded += status == 'succeeded'
        snapshots[str(float(budget))] = dict(nominal=n.bit_count(), sampled_robust=r.bit_count(),
                                             succeeded=succeeded, attempts=attempts)
    known_recoveries = [record.recoveries for record in records if record.recoveries is not None]
    return dict(nominal_mask=nominal, sampled_robust_mask=robust,
                nominal=nominal.bit_count(), sampled_robust=robust.bit_count(),
                duration=records[-1].elapsed if records else None,
                succeeded=sum(r.status == 'succeeded' for r in records),
                failed=sum(r.status == 'failed' for r in records),
                canceled=sum(r.status == 'canceled' for r in records),
                recovery_counts_known=len(known_recoveries),
                recoveries_sum_known=sum(known_recoveries),
                max_recoveries_known=max(known_recoveries, default=None),
                invalid_pose_indices=invalid, budgets=snapshots)
