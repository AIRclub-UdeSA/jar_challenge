"""Offline prediction metrics, with mission's real selector and no ROS runtime."""
from dataclasses import dataclass
import math
from types import SimpleNamespace

from equipo_jar.cobertura_plan import Pose, visibility_mask
from equipo_jar.exploracion import first_pending


@dataclass(frozen=True)
class Event:
    index: int
    elapsed: float
    mask: int
    return_distance: float


@dataclass(frozen=True)
class Trace:
    events: tuple
    distance: float
    turning: float
    startup_delay: float

    @property
    def duration(self):
        return self.events[-1].elapsed if self.events else self.startup_delay


def mission_indices(poses, distance, angle, zones=()):
    """Real selector with fixed box zones, assuming each sent goal succeeds.

    Fixed zones are a diagnostic scenario, not a dynamic detection replay.
    """
    if not all(math.isfinite(v) and v > 0 for v in (distance, angle)):
        raise ValueError('mission tolerances must be finite and positive')
    messages = []
    for p in poses:
        messages.append(SimpleNamespace(pose=SimpleNamespace(
            position=SimpleNamespace(x=p.x, y=p.y),
            orientation=SimpleNamespace(x=0.0, y=0.0, z=math.sin(p.yaw / 2),
                                        w=math.cos(p.yaw / 2)))))
    lookup = {id(message): index for index, message in enumerate(messages)}
    visited, selected = [], []
    while True:
        point = first_pending(messages, visited, distance, angle, zones)
        if point is None:
            return tuple(selected)
        index = lookup[id(point)]
        if index in selected:
            raise RuntimeError('mission selector repeated a finished goal')
        selected.append(index)
        p = poses[index]
        visited.append((p.x, p.y, p.yaw))


def visibility_masks(grid, poses, cells, settings, camera):
    centers = [grid.center(cell) for cell in cells]
    return tuple(visibility_mask(grid, grid.cell(p.x, p.y), p.yaw,
                                 centers, settings, camera) for p in poses)


def mask_for_cells(cells, subset):
    return sum(1 << index for index, cell in enumerate(cells) if cell in subset)


def route_trace(grid, start, poses, masks, indices, costs, settings, startup_delay=0.0):
    """Credit a view only after travel, turning and estimated observation dwell."""
    if not math.isfinite(startup_delay) or startup_delay < 0:
        raise ValueError('startup delay must be finite and nonnegative')
    if len(poses) != len(masks):
        raise ValueError('one visibility mask is required per pose')
    cell, yaw = grid.cell(start.x, start.y), start.yaw
    start_cell = cell
    elapsed = startup_delay
    distance = turning = 0.0
    mask, events = 0, []
    for index in indices:
        p = poses[index]
        nxt = grid.cell(p.x, p.y)
        leg = costs.between(cell, nxt)
        turn = abs(math.atan2(math.sin(p.yaw - yaw), math.cos(p.yaw - yaw)))
        distance += leg
        turning += turn
        elapsed += (leg / settings.travel_speed + turn / settings.turn_speed
                    + settings.observation_time)
        mask |= masks[index]
        events.append(Event(index, elapsed, mask, costs.between(nxt, start_cell)))
        cell, yaw = nxt, p.yaw
    return Trace(tuple(events), distance, turning, startup_delay)


def at_budget(trace, seconds, *, target_mask, travel_speed):
    """Return a stepwise visibility prediction and its time integral."""
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('budget must be finite and positive')
    if not math.isfinite(travel_speed) or travel_speed <= 0:
        raise ValueError('travel speed must be finite and positive')
    count = target_mask.bit_count()
    mask, last_time, integral, completed, return_distance = 0, 0.0, 0.0, 0, 0.0
    for event in trace.events:
        if event.elapsed > seconds:
            break
        integral += (event.elapsed - last_time) * (mask & target_mask).bit_count()
        mask, last_time, return_distance = event.mask, event.elapsed, event.return_distance
        completed += 1
    integral += (seconds - last_time) * (mask & target_mask).bit_count()
    visible = (mask & target_mask).bit_count()
    return dict(seconds=seconds, completed=completed, visible=visible, targets=count,
                fraction=visible / count if count else None,
                mean_fraction=integral / (seconds * count) if count else None,
                return_from_last_completed_s=return_distance / travel_speed)


def summarize_trace(trace, budgets, target_masks, settings):
    final_mask = trace.events[-1].mask if trace.events else 0
    return dict(
        goals=len(trace.events), distance=trace.distance, turning=trace.turning,
        duration=trace.duration, startup_delay=trace.startup_delay,
        final_visible={name: (final_mask & mask).bit_count()
                       for name, mask in target_masks.items()},
        budgets={str(seconds): {name: at_budget(
            trace, seconds, target_mask=mask, travel_speed=settings.travel_speed)
            for name, mask in target_masks.items()} for seconds in budgets},
        curve=[dict(index=e.index, elapsed=e.elapsed,
                    visible={name: (e.mask & mask).bit_count()
                             for name, mask in target_masks.items()}) for e in trace.events])


def classify_outside(grid, outside, raw_component, reachable, settings, camera):
    """Exhaustive witness search under the current cell-center/heading model.

    'no_witness' does not mean physically impossible: footprint, camera model,
    range and discretization still constrain this diagnostic.
    """
    groups = dict(raw_disconnected=[], candidate_sampling_gap=[], no_witness=[])
    witnesses = []
    reach = tuple(sorted(reachable, key=lambda cell: (cell[1], cell[0])))
    radius_squared = ((settings.view_range + math.hypot(camera.x, camera.y))
                      / grid.resolution) ** 2
    for target in outside:
        if target not in raw_component:
            groups['raw_disconnected'].append(target)
            continue
        ti, tj = target
        nearby = sorted(
            (cell for cell in reach if (cell[0] - ti) ** 2 + (cell[1] - tj) ** 2
             <= radius_squared),
            key=lambda cell: ((cell[0] - ti) ** 2 + (cell[1] - tj) ** 2,
                              cell[1], cell[0]))
        witness = None
        for cell in nearby:
            for heading in range(settings.heading_count):
                yaw = grid.origin.yaw + heading * math.tau / settings.heading_count
                if visibility_mask(grid, cell, yaw, [grid.center(target)], settings, camera):
                    witness = Pose(*grid.center(cell), yaw)
                    break
            if witness is not None:
                break
        name = 'candidate_sampling_gap' if witness is not None else 'no_witness'
        groups[name].append(target)
        if witness is not None:
            witnesses.append(witness)
    return groups, tuple(witnesses)
