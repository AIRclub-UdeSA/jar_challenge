"""Shared static-map geometry, independent of ROS and module state.

Legacy point/line helpers keep their existing semantics. Grid provides the
conservative cell model already used by coverage, now reusable by mission.
"""
from collections import deque
from concurrent.futures import CancelledError
from dataclasses import dataclass
import math

def cell_value(grid, x, y):
    """Read a world point using the grid's planar origin pose.

    Return None outside the grid or where data is missing. Lightweight test
    grids without an orientation keep the previous zero-rotation behavior.
    """
    origin = grid.info.origin
    q = getattr(origin, 'orientation', None)
    yaw = 0.0 if q is None else math.atan2(
        2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = x - origin.position.x, y - origin.position.y
    # Undo the origin translation and rotation before finding the cell.
    i = math.floor((c * dx + s * dy) / grid.info.resolution)
    j = math.floor((-s * dx + c * dy) / grid.info.resolution)
    index = j * grid.info.width + i
    if i < 0 or j < 0 or i >= grid.info.width or j >= grid.info.height or index >= len(grid.data):
        return None
    return grid.data[index]

def is_free(grid, x, y, clearance):
    resolution = grid.info.resolution
    n = math.ceil(clearance / resolution)
    for a in range(-n, n+1):
        for b in range(-n, n+1):
            dx = a * resolution
            dy = b * resolution
            if math.hypot(dx, dy) > clearance:
                continue
            v = cell_value(grid, x + dx, y + dy)
            if v is None or v < 0 or v >= 50:
                return False
    return True

def line_is_clear(grid, x0, y0, x1, y1):
    resolution = grid.info.resolution
    dist = math.hypot(x1 - x0, y1 - y0)
    n = max(1, math.ceil(dist / resolution))
    for k in range(n+1):
        x = x0 + (x1 - x0) * k / n
        y = y0 + (y1 - y0) * k / n
        v = cell_value(grid, x, y)
        if v is not None and v >= 50:
            return False
    return True


def check_cancelled(cancel):
    if cancel is not None and cancel.is_set():
        raise CancelledError('coverage calculation cancelled')


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    yaw: float

    def __post_init__(self):
        if not all(math.isfinite(v) for v in (self.x, self.y, self.yaw)):
            raise ValueError('pose must be finite')


def planar_yaw(q):
    """Require a normalized quaternion representing a planar rotation."""
    values = (q.x, q.y, q.z, q.w)
    if not all(math.isfinite(v) for v in values):
        raise ValueError('rotation must be finite')
    if abs(sum(v * v for v in values) - 1.0) > 1e-3:
        raise ValueError('rotation quaternion must be normalized')
    if abs(q.x) > 1e-3 or abs(q.y) > 1e-3:
        raise ValueError('map and robot rotations must be planar')
    return math.atan2(2 * q.w * q.z, 1 - 2 * q.z * q.z)


class Grid:
    """Immutable map values with conservative disk clearance and grid geometry.

    Reachability describes this static map, not Nav2's live obstacle costmap.
    """

    def __init__(self, width, height, resolution, data, origin=Pose(0.0, 0.0, 0.0)):
        if type(width) is not int or type(height) is not int or min(width, height) < 1:
            raise ValueError('map dimensions must be positive integers')
        if not math.isfinite(resolution) or resolution <= 0:
            raise ValueError('map resolution must be finite and positive')
        if len(data) != width * height:
            raise ValueError('map data length does not match dimensions')
        if any(type(v) is not int or not -1 <= v <= 100 for v in data):
            raise ValueError('map cells must be integers in [-1, 100]')
        self.width, self.height, self.resolution = width, height, resolution
        self.data, self.origin = tuple(data), origin
        self.cos, self.sin = math.cos(origin.yaw), math.sin(origin.yaw)
        self._free_cache = None

    @classmethod
    def from_occupancy_grid(cls, message):
        """Copy a map-frame OccupancyGrid without importing ROS packages."""
        if message.header.frame_id != 'map':
            raise ValueError('map frame must be map')
        origin = message.info.origin
        return cls(message.info.width, message.info.height, message.info.resolution,
                   list(message.data), Pose(origin.position.x, origin.position.y,
                                            planar_yaw(origin.orientation)))

    def reachable_from(self, x, y, radius, threshold, cancel=None):
        """Return connected safe cells from a robot position in map metres.

        Radius is the transit disk including its safety margin, in metres.
        Unknown cells are blocked. An unsafe/outside start raises ValueError;
        never snap a suspect localization pose to the other side of a wall.
        Candidate membership is ``grid.cell(vx, vy) in reached``. Compute this
        once for all candidates, then keep separate observation-clearance and
        line-of-sight checks. It does not account for boxes absent from /map.
        """
        if not math.isfinite(radius) or radius < 0:
            raise ValueError('radius must be finite and nonnegative')
        if type(threshold) is not int or not 0 <= threshold < 100:
            raise ValueError('threshold must be an integer in [0, 100)')
        safe = self.safe_cells(radius, threshold, cancel)
        return frozenset(reachable_cells(safe, self.cell(x, y), cancel))

    def cell(self, x, y):
        if not math.isfinite(x) or not math.isfinite(y):
            return None
        dx, dy = x - self.origin.x, y - self.origin.y
        i = math.floor((self.cos * dx + self.sin * dy) / self.resolution)
        j = math.floor((-self.sin * dx + self.cos * dy) / self.resolution)
        return (i, j) if 0 <= i < self.width and 0 <= j < self.height else None

    def center(self, cell):
        i, j = cell
        x, y = (i + 0.5) * self.resolution, (j + 0.5) * self.resolution
        return (self.origin.x + self.cos * x - self.sin * y,
                self.origin.y + self.sin * x + self.cos * y)

    def free(self, cell, threshold):
        i, j = cell
        return (0 <= i < self.width and 0 <= j < self.height
                and 0 <= self.data[j * self.width + i] <= threshold)

    def _free_mask(self, threshold, cancel=None):
        """One immutable occupancy lookup per map/threshold, shared by geometry.

        Keep only the latest threshold. A local snapshot and atomic tuple
        replacement let concurrent readers use different thresholds safely.
        """
        cached = self._free_cache
        if cached is not None and cached[0] == threshold:
            return cached[1]
        mask = bytearray()
        for first in range(0, len(self.data), self.width):
            check_cancelled(cancel)
            mask.extend(0 <= value <= threshold
                        for value in self.data[first:first + self.width])
        result = bytes(mask)
        self._free_cache = threshold, result
        return result

    def safe_cells(self, radius, threshold, cancel=None):
        """Conservative circular footprint against full cell squares and map edges."""
        check_cancelled(cancel)
        clear = self._free_mask(threshold, cancel)
        width = self.width
        clearance = radius / self.resolution
        span = math.ceil(clearance + 0.5)
        offsets = []
        for dj in range(-span, span + 1):
            for di in range(-span, span + 1):
                distance = math.hypot(max(abs(di) - 0.5, 0), max(abs(dj) - 0.5, 0))
                if distance <= clearance or math.isclose(distance, clearance, abs_tol=1e-12):
                    offsets.append((di, dj))
        kernel_span = max(abs(di) for di, _ in offsets)
        safe = set()
        for j in range(self.height):
            check_cancelled(cancel)
            for i in range(self.width):
                if min(i + 0.5, j + 0.5, self.width - i - 0.5,
                       self.height - j - 0.5) <= clearance + 1e-12:
                    continue
                # isclose can include a square just beyond the exact radius.
                if min(i, j, width - i - 1, self.height - j - 1) < kernel_span:
                    continue
                if all(clear[(j + dj) * width + i + di] for di, dj in offsets):
                    safe.add((i, j))
        return safe

    def visible(self, start, end, threshold):
        """Closed-segment supercover; occupied/unknown boundary touches block.

        Traverse cells entered by the open segment; check its closed endpoints
        separately. A boundary's floor cell may be touched but never entered.
        """
        if not all(math.isfinite(v) for v in (*start, *end)):
            return False
        ax, ay = start[0] - self.origin.x, start[1] - self.origin.y
        bx, by = end[0] - self.origin.x, end[1] - self.origin.y
        sx = (self.cos * ax + self.sin * ay) / self.resolution
        sy = (-self.sin * ax + self.cos * ay) / self.resolution
        ex = (self.cos * bx + self.sin * by) / self.resolution
        ey = (-self.sin * bx + self.cos * by) / self.resolution
        # Only absorb floating-point transform noise, in units of cells.
        nx, ny, mx, my = round(sx), round(sy), round(ex), round(ey)
        sx = nx if abs(sx - nx) <= 1e-12 else sx
        sy = ny if abs(sy - ny) <= 1e-12 else sy
        ex = mx if abs(ex - mx) <= 1e-12 else ex
        ey = my if abs(ey - my) <= 1e-12 else ey
        clear = self._free_mask(threshold)
        width, height = self.width, self.height
        # A closed endpoint on a grid line/corner touches two/four squares.
        for x, y in ((sx, sy), (ex, ey)):
            i, j = math.floor(x), math.floor(y)
            if not (0 <= i < width and 0 <= j < height and clear[j * width + i]):
                return False
            if x == i and (i == 0 or not clear[j * width + i - 1]):
                return False
            if y == j and (j == 0 or not clear[(j - 1) * width + i]):
                return False
            if x == i and y == j and not clear[(j - 1) * width + i - 1]:
                return False
        dx, dy = ex - sx, ey - sy
        if dx == 0 and dy == 0:
            return True
        i, j = math.floor(sx), math.floor(sy)
        if dx < 0 and sx == i:
            i -= 1
        if dy < 0 and sy == j:
            j -= 1
        end_i, end_j = math.floor(ex), math.floor(ey)
        if dx > 0 and ex == end_i:
            end_i -= 1
        if dy > 0 and ey == end_j:
            end_j -= 1
        along_x_edge = dx == 0 and sx == i
        along_y_edge = dy == 0 and sy == j
        step_i, step_j = (1 if dx > 0 else -1), (1 if dy > 0 else -1)
        ti = ((i + (dx > 0) - sx) / dx) if dx else math.inf
        tj = ((j + (dy > 0) - sy) / dy) if dy else math.inf
        delta_i = abs(1 / dx) if dx else math.inf
        delta_j = abs(1 / dy) if dy else math.inf
        while True:
            if not (0 <= i < width and 0 <= j < height and clear[j * width + i]):
                return False
            # A ray on a grid line touches both strips for its whole length.
            if along_x_edge and (i == 0 or not clear[j * width + i - 1]):
                return False
            if along_y_edge and (j == 0 or not clear[(j - 1) * width + i]):
                return False
            if i == end_i and j == end_j:
                return True  # Endpoint contacts have already been checked.
            if math.isclose(ti, tj, rel_tol=1e-12, abs_tol=1e-12):
                if ti >= 1.0 or tj >= 1.0:
                    return True  # A near-corner tie must not step past the endpoint.
                ni, nj = i + step_i, j + step_j
                if not (0 <= ni < width and 0 <= nj < height
                        and clear[j * width + ni] and clear[nj * width + i]):
                    return False
                i, j = i + step_i, j + step_j
                ti, tj = ti + delta_i, tj + delta_j
            elif ti < tj:
                i, ti = i + step_i, ti + delta_i
            else:
                j, tj = j + step_j, tj + delta_j


def reachable_cells(safe, start, cancel=None):
    """Four-neighbour component; safe cells already include the robot radius.

    Takes integer cell pairs, not metres. Diagonal corner contacts do not
    connect rooms. Existing coverage imports remain available as aliases.
    """
    check_cancelled(cancel)
    if start not in safe:
        raise ValueError('robot pose is outside the safe free space; check TF or clearance')
    reached, pending = {start}, deque([start])
    while pending:
        check_cancelled(cancel)
        i, j = pending.popleft()
        for nxt in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
            if nxt in safe and nxt not in reached:
                reached.add(nxt)
                pending.append(nxt)
    return reached
