from dataclasses import dataclass
import math
from numbers import Real
from types import MappingProxyType
from typing import Mapping

from equipo_jar.mapa import Grid, Pose, check_cancelled
from equipo_jar.vuelta import steps_to_start


@dataclass(frozen=True)
class RouteField:
    grid: Grid
    at_s: float
    outward: Mapping
    home: Mapping


@dataclass(frozen=True)
class RouteEstimate:
    at_s: float
    outward_m: float
    home_m: float


def _valid_time(value):
    if isinstance(value, bool) or not isinstance(value, Real):
        return False
    try:
        return math.isfinite(value) and value >= 0
    except OverflowError:
        return False


def build_route_field(grid: Grid, robot: Pose, start: Pose, at_s: float, *,
                      cancel=None) -> RouteField | None:
    check_cancelled(cancel)
    if (not isinstance(grid, Grid) or not isinstance(robot, Pose)
            or not isinstance(start, Pose) or not _valid_time(at_s)):
        return None
    snapshot = Grid(grid.width, grid.height, grid.resolution, grid.data, grid.origin)
    outward = steps_to_start(snapshot, robot.x, robot.y, 0.0, 98, 0.0, cancel=cancel)
    check_cancelled(cancel)
    home = steps_to_start(snapshot, start.x, start.y, 0.0, 98, 0.0, cancel=cancel)
    check_cancelled(cancel)
    if snapshot.cell(robot.x, robot.y) not in home or not outward:
        return None
    return RouteField(snapshot, at_s, MappingProxyType(outward), MappingProxyType(home))


def estimate_route(field: RouteField | None, goal: Pose, now_s: float, *,
                   max_age_s: float) -> RouteEstimate | None:
    if (not isinstance(field, RouteField) or not isinstance(goal, Pose)
            or not _valid_time(now_s) or not _valid_time(max_age_s) or max_age_s == 0):
        return None
    if not 0 <= now_s - field.at_s <= max_age_s:
        return None
    cell = field.grid.cell(goal.x, goal.y)
    if cell not in field.outward or cell not in field.home:
        return None
    resolution = field.grid.resolution
    return RouteEstimate(field.at_s, field.outward[cell] * resolution,
                         field.home[cell] * resolution)
