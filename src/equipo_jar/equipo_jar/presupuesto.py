import math
from numbers import Real


def _nonnegative(value):
    if isinstance(value, bool) or not isinstance(value, Real):
        return False
    try:
        return math.isfinite(value) and value >= 0
    except OverflowError:
        return False


def _valid_limits(now_s, reserve_s, return_at_s, finish_at_s):
    return (all(_nonnegative(value) for value in
                (now_s, reserve_s, return_at_s, finish_at_s))
            and return_at_s < finish_at_s)


def route_seconds(distance_m: float | None, *, speed_mps: float,
                  factor: float, overhead_s: float) -> float | None:
    if not all(_nonnegative(value) for value in
               (distance_m, speed_mps, factor, overhead_s)):
        return None
    if speed_mps == 0 or factor < 1:
        return None
    seconds = distance_m / speed_mps * factor + overhead_s
    return seconds if _nonnegative(seconds) else None


def fits_extra(now_s: float, *, outward_s: float | None, look_s: float,
               home_s: float | None, reserve_s: float,
               return_at_s: float, finish_at_s: float) -> bool:
    if not _valid_limits(now_s, reserve_s, return_at_s, finish_at_s):
        return False
    if not all(_nonnegative(value) for value in (outward_s, look_s, home_s)):
        return False
    arrival_end = now_s + outward_s + look_s
    return (arrival_end < return_at_s
            and arrival_end + home_s + reserve_s < finish_at_s)


def return_due(now_s: float, *, home_s: float | None, reserve_s: float,
               return_at_s: float, finish_at_s: float) -> bool:
    if not _valid_limits(now_s, reserve_s, return_at_s, finish_at_s):
        return True
    return (now_s >= return_at_s
            or (_nonnegative(home_s) and now_s + home_s + reserve_s >= finish_at_s))
