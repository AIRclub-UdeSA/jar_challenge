from dataclasses import dataclass
import math
from numbers import Real

from equipo_jar.mapa import Pose


def _nonnegative(value):
    if isinstance(value, bool) or not isinstance(value, Real):
        return False
    try:
        return math.isfinite(value) and value >= 0
    except OverflowError:
        return False


@dataclass(frozen=True)
class ObservedView:
    pose: Pose
    at_s: float

    def __post_init__(self):
        if not isinstance(self.pose, Pose) or not _nonnegative(self.at_s):
            raise ValueError('observation requires a Pose and a finite nonnegative time')


@dataclass(frozen=True)
class AttemptedView:
    pose: Pose
    outcome: str

    def __post_init__(self):
        if not isinstance(self.pose, Pose):
            raise ValueError('attempt requires a Pose')
        if self.outcome not in ('arrived', 'failed', 'rejected', 'cancelled'):
            raise ValueError('invalid attempt outcome')


def append_view(history: tuple[ObservedView, ...], view: ObservedView, *,
                max_items: int, min_interval_s: float,
                min_shift_m: float, min_turn_rad: float) -> tuple[ObservedView, ...]:
    if type(max_items) is not int or max_items < 1:
        raise ValueError('max_items must be a positive integer')
    if not all(_nonnegative(value) for value in
               (min_interval_s, min_shift_m, min_turn_rad)):
        raise ValueError('sampling thresholds must be finite and nonnegative')
    if (not isinstance(history, tuple) or not isinstance(view, ObservedView)
            or not all(isinstance(item, ObservedView) for item in history)):
        raise ValueError('history and view must contain observations only')
    if any(first.at_s >= second.at_s for first, second in zip(history, history[1:])):
        raise ValueError('history must have strictly increasing timestamps')
    history = history[-max_items:]
    if not history:
        return (view,)
    last = history[-1]
    if view.at_s <= last.at_s:
        return history
    elapsed = view.at_s - last.at_s
    shift = math.hypot(view.pose.x - last.pose.x, view.pose.y - last.pose.y)
    angle = math.remainder(view.pose.yaw, math.tau) - math.remainder(last.pose.yaw, math.tau)
    turn = abs(math.remainder(angle, math.tau))
    if elapsed >= min_interval_s or shift >= min_shift_m or turn >= min_turn_rad:
        return (history + (view,))[-max_items:]
    return history
