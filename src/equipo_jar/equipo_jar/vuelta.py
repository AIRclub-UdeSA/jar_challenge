"""Trip back to the start in legs (no ROS here).

Nav2's NavFn planner (Humble) stops building its path after 4 x the costmap's longer side
in cells, about half a cell per step, so a path longer than ~2 x the map's side always
fails ("Failed to create plan from potential"). The limit is in Nav2's code, not a
parameter, and retrying the same goal fails the same way. For a long way back, mission
finds the way on /map itself and sends it to Nav2 in legs it can plan.

The way back is a breadth-first search over the cells where the robot fits
(`Grid.safe_cells`), moving only along x and y. Its length is never shorter than the
real path, so a leg of `leg_m` is at most that long for Nav2. Mission runs it on Nav2's
global costmap when it has one (boxes and obstacles the LiDAR saw, which /map lacks;
radius 0, since the inflation already holds the robot), and on /map otherwise.
"""
from collections import deque
import math
from equipo_jar.mapa import check_cancelled

# The leg's heading looks this far ahead along the way back, so the next leg starts
# without a turn.
LOOK_AHEAD_M = 0.5


def neighbours(cell):
    i, j = cell
    return ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1))


def snap_cells(grid, snap_m):
    """snap_m in whole cells (>= 0)."""
    return max(0, math.floor(snap_m / grid.resolution))


def nearest_cell(cells, cell, max_cells):
    """The cell of `cells` nearest to `cell` in a straight line, at most max_cells away.

    `cells` is anything that supports `in` (a set or a dict). None if there is none, or
    if `cell` is None (a point outside the map).
    """
    if cell is None:
        return None
    if cell in cells:
        return cell
    i0, j0 = cell
    best = None
    best_dist = None
    for dj in range(-max_cells, max_cells + 1):
        for di in range(-max_cells, max_cells + 1):
            dist = math.hypot(di, dj)
            near = (i0 + di, j0 + dj)
            if dist <= max_cells and near in cells and (best is None or dist < best_dist):
                best = near
                best_dist = dist
    return best


def steps_to_start(grid, x, y, radius, threshold, snap_m, *, cancel=None):
    """{cell: steps of the shortest way from the start} for every cell connected to it.

    The start is the cell of (x, y), or the nearest cell where the robot fits within
    snap_m: the start pose can be closer to a wall than `radius`. {} if there is none.
    `radius` and `threshold` are those of `Grid.safe_cells`.
    """
    safe = grid.safe_cells(radius, threshold, cancel)
    start = nearest_cell(safe, grid.cell(x, y), snap_cells(grid, snap_m))
    if start is None:
        return {}
    steps = {start: 0}
    pending = deque([start])
    while pending:
        check_cancelled(cancel)
        cell = pending.popleft()
        for nxt in neighbours(cell):
            if nxt in safe and nxt not in steps:
                steps[nxt] = steps[cell] + 1
                pending.append(nxt)
    return steps


def way_back(steps, cell):
    """Cells from `cell` to the start, each one step closer. `cell` must be in `steps`."""
    path = [cell]
    while steps[cell] > 0:
        cell = min(neighbours(cell), key=lambda c: steps.get(c, math.inf))
        path.append(cell)
    return path


def way_back_m(grid, steps, x, y, snap_m):
    """Length in m of the way back from (x, y); None if (x, y) is not connected to the start."""
    here = nearest_cell(steps, grid.cell(x, y), snap_cells(grid, snap_m))
    if here is None:
        return None
    return steps[here] * grid.resolution


def return_leg(grid, steps, x, y, leg_m, snap_m, blocked=None):
    """Goal (x, y, yaw) of the next leg: leg_m along the way back from (x, y).

    None if the start is at most leg_m away (then mission goes straight to it), or if
    (x, y) is not connected to the start on /map. `blocked(x, y)`: optional; a leg that
    ends on a blocked spot (an obstacle or a box) is shortened until it does not.
    The yaw faces along the way back.
    """
    here = nearest_cell(steps, grid.cell(x, y), snap_cells(grid, snap_m))
    leg = math.floor(leg_m / grid.resolution)
    if here is None or leg < 1 or steps[here] <= leg:
        return None
    path = way_back(steps, here)
    k = leg
    while k > 0 and blocked is not None and blocked(*grid.center(path[k])):
        k -= 1
    if k == 0:
        return None
    gx, gy = grid.center(path[k])
    ahead = max(1, round(LOOK_AHEAD_M / grid.resolution))
    ax, ay = grid.center(path[min(k + ahead, len(path) - 1)])
    return gx, gy, math.atan2(ay - gy, ax - gx)
