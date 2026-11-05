"""Generated geometry and independent checks for supplementary views."""
from concurrent.futures import CancelledError
from dataclasses import dataclass, replace
import math
from threading import Event

import pytest

from equipo_jar.mapa import Grid, Pose
from equipo_jar.cobertura_plan import Camera, Settings, angle_difference
from equipo_jar.cobertura_repaso import pending_samples, propose_extra_views, visibility_grid


@dataclass(frozen=True)
class Seen:
    pose: Pose
    at_s: float = 10.0


@dataclass(frozen=True)
class Attempt:
    pose: Pose
    outcome: str


def room(*, occluder=True, closed=False, inflation=True):
    w, h, res = 120, 80, 0.05
    static, live = [], []
    for j in range(h):
        for i in range(w):
            x, y = (i + 0.5) * res, (j + 0.5) * res
            wall = (i in (0, w-1) or j in (0, h-1)
                    or (i == 60 and (closed or y < 1.0 or y > 3.0)))
            value = 100 if wall else 0
            static.append(value)
            if occluder and not wall:
                distance = math.hypot(max(3.25-x, 0, x-3.55), max(1.65-y, 0, y-2.35))
                value = 100 if distance == 0 else (99 if distance < 0.18 else
                                                  50 if distance < 0.35 else 0) if inflation else 0
            live.append(value)
    return Grid(w, h, res, static), Grid(w, h, res, live)


def outside_history():
    return tuple(Seen(Pose(x, y, k * math.tau / 8))
                 for x in (0.7, 1.5, 2.3) for y in (0.7, 1.5, 2.5, 3.3) for k in range(8))


def extra(static, live, observed=(), attempted=(), **kwargs):
    return propose_extra_views(static, live, Pose(1.0, 2.0, 0), Pose(1.0, 2.0, 0),
                               observed, attempted, Camera(), Settings(),
                               limit=kwargs.pop('limit', 6), failed_shift_m=0.4, **kwargs)


def sees(grid, pose, target, *, fov=61, range_m=2):
    dx, dy = target[0] - pose.x, target[1] - pose.y
    return (math.hypot(dx, dy) <= range_m
            and abs(angle_difference(math.atan2(dy, dx), pose.yaw)) <= math.radians(fov/2)
            and grid.visible((pose.x, pose.y), target, 25))


def test_cylinder_restores_previously_predicted_visibility_to_pending():
    static, clear = room(occluder=False)
    _, blocked = room()
    observed = (Seen(Pose(2.0, 2.0, 0)),)
    before = pending_samples(static, clear, observed, Camera(), Settings())
    after = pending_samples(static, blocked, observed, Camera(), Settings())
    shadow = lambda p: math.hypot(p[0] - 3.9, p[1] - 2.0) < 0.16
    assert not any(map(shadow, before))
    assert any(map(shadow, after))
    assert set(before) <= set(after)


def test_new_view_reveals_shadow_without_any_victim_candidate():
    static, live = room()
    observed = outside_history()
    pending = pending_samples(static, live, observed, Camera(), Settings())
    result = extra(static, live, observed)
    assert result.pending_count == len(pending) > 0
    assert 0 < len(result.views) <= 6
    visual = visibility_grid(static, live)
    safe = visual.safe_cells(0.24, 25)
    shadow = [p for p in pending if 3.7 < p[0] < 4.5 and 1.8 < p[1] < 2.2]
    assert shadow
    assert any(v.pose.x > 3.55 for v in result.views)
    for view in result.views:
        assert view.gain == sum(sees(visual, view.pose, p) for p in pending) > 0
        assert visual.cell(view.pose.x, view.pose.y) in safe
        cell = live.cell(view.pose.x, view.pose.y)
        assert 0 <= live.data[cell[1] * live.width + cell[0]] <= 98


def test_execute_one_view_then_replan_recovers_the_shadow_within_six_goals():
    static, live = room()
    visual = visibility_grid(static, live)
    observed = outside_history()
    pending = pending_samples(static, live, observed, Camera(), Settings())
    shadow = [p for p in pending if 3.7 < p[0] < 4.5 and 1.8 < p[1] < 2.2]
    robot, start = Pose(1, 2, 0), Pose(1, 2, 0)
    recovered = False
    for _ in range(6):
        result = propose_extra_views(static, live, robot, start, observed, (), Camera(),
                                     Settings(), limit=6, failed_shift_m=0.4)
        assert result.views
        robot = result.views[0].pose
        if any(sees(visual, robot, p) for p in shadow):
            recovered = True
            break
        observed += (Seen(robot),)
    assert recovered


def test_closed_door_leaves_unresolved_room_without_unreachable_proposals():
    static, live = room(closed=True)
    result = extra(static, live, outside_history())
    assert result.pending_count > 0
    assert all(v.pose.x < 3 for v in result.views)
    assert not result.views  # The reachable side is already observed.


@pytest.mark.parametrize('cost', [0, 1, 25, 50, 98, 99])
def test_inflation_does_not_cast_an_optical_shadow(cost):
    static = Grid(30, 20, 0.1, [0] * 600)
    costs = [0] * 600
    costs[10*30+15] = cost
    visual = visibility_grid(static, Grid(30, 20, 0.1, costs))
    assert visual.visible((0.55, 1.05), (2.55, 1.05), 25)


@pytest.mark.parametrize('cost', [-1, 100])
def test_unknown_and_lethal_live_cells_block_optical_credit(cost):
    static = Grid(30, 20, 0.1, [0] * 600)
    costs = [0] * 600
    costs[10*30+15] = cost
    visual = visibility_grid(static, Grid(30, 20, 0.1, costs))
    assert not visual.visible((0.55, 1.05), (2.55, 1.05), 25)


def test_static_walls_and_unknown_are_not_cleared_by_live_free_cells():
    data = [0] * 600
    data[10*30+15], data[10*30+16] = 100, -1
    visual = visibility_grid(Grid(30, 20, 0.1, data), Grid(30, 20, 0.1, [0]*600))
    assert visual.data[10*30+15:10*30+17] == (100, -1)


def test_fine_live_obstacle_between_coarse_cell_centers_is_preserved():
    static = Grid(10, 10, 0.2, [0] * 100)
    live = [0] * 400
    live[10*20+10] = 100
    visual = visibility_grid(static, Grid(20, 20, 0.1, live))
    assert visual.data[5*10+5] == 100
    assert sum(v == 100 for v in visual.data) == 1


def test_live_grid_partial_coverage_is_unresolved():
    visual = visibility_grid(Grid(20, 20, 0.1, [0]*400),
                             Grid(10, 10, 0.1, [0]*100, Pose(0.5, 0.5, 0)))
    assert visual.data[0] == -1
    assert visual.data[10*20+10] == 0


def clip_polygon(polygon, axis, edge, greater):
    result = []
    if not polygon:
        return result
    previous = polygon[-1]
    for current in polygon:
        inside = (current[axis] >= edge) if greater else (current[axis] <= edge)
        was_inside = (previous[axis] >= edge) if greater else (previous[axis] <= edge)
        if inside != was_inside:
            fraction = (edge-previous[axis]) / (current[axis]-previous[axis])
            result.append(tuple(a+fraction*(b-a) for a,b in zip(previous,current)))
        if inside:
            result.append(current)
        previous = current
    return result


@pytest.mark.parametrize('angle', [0, math.pi/2, 0.37, -0.61])
@pytest.mark.parametrize('resolution', [0.07, 0.2, 0.31])
def test_projection_matches_independent_polygon_clipping(angle, resolution):
    base = Grid(7, 6, 0.2, [0]*42, Pose(1.7, -0.3, 0.21))
    data = [0]*600
    cells = [(12, 10), (13, 9)]
    for i,j in cells:
        data[j*30+i] = 100
    ax,ay = base.center((3,2))
    ca,sa = math.cos(angle),math.sin(angle)
    origin = Pose(ax-resolution*(ca*12.5-sa*10.5),
                  ay-resolution*(sa*12.5+ca*10.5),angle)
    live = Grid(30, 20, resolution, data, origin)
    polygons = []
    for i,j in cells:
        corners = []
        for x,y in ((i,j),(i+1,j),(i+1,j+1),(i,j+1)):
            wx = live.origin.x + resolution*(live.cos*x-live.sin*y)
            wy = live.origin.y + resolution*(live.sin*x+live.cos*y)
            dx,dy = wx-base.origin.x, wy-base.origin.y
            corners.append(((base.cos*dx+base.sin*dy)/base.resolution,
                            (-base.sin*dx+base.cos*dy)/base.resolution))
        polygons.append(corners)
    expected = set()
    for j in range(base.height):
        for i in range(base.width):
            for polygon in polygons:
                cut = polygon
                for axis,edge,greater in ((0,i,True),(0,i+1,False),(1,j,True),(1,j+1,False)):
                    cut = clip_polygon(cut,axis,edge,greater)
                area = abs(sum(a[0]*b[1]-a[1]*b[0] for a,b in zip(cut,cut[1:]+cut[:1])))/2
                if area > 1e-10:
                    expected.add((i,j))
    visual = visibility_grid(base, live)
    actual = {(i,j) for j in range(base.height) for i in range(base.width)
              if visual.data[j*base.width+i] == 100}
    assert actual == expected


def test_fine_search_recovers_regions_missed_by_the_only_coarse_position():
    w,h,res = 120,80,0.05
    data = [100 if i in (0,w-1) or j in (0,h-1) else 0
            for j in range(h) for i in range(w)]
    grid = Grid(w,h,res,data)
    # A single central coarse sample has already looked in every heading.
    # Targets at the ends remain outside its two-metre range.
    from equipo_jar.cobertura_plan import sampled_cells
    centers = grid.safe_cells(0.24,25)
    center = sampled_cells(centers,10,res,256)[0]
    x,y = grid.center(center)
    seen = tuple(Seen(Pose(x,y,k*math.tau/8)) for k in range(8))
    result = propose_extra_views(grid,grid,Pose(x,y,0),Pose(x,y,0),seen,(),Camera(),
                                 Settings(candidate_spacing=10),limit=6,failed_shift_m=0.4)
    assert result.views
    assert all(math.hypot(v.pose.x-x,v.pose.y-y) > 0.02 for v in result.views)


@pytest.mark.parametrize('cost', [-1,99,100])
def test_live_barrier_prevents_entering_a_disconnected_room(cost):
    static, live = room(occluder=False)
    data = list(live.data)
    for j in range(live.height):
        data[j*live.width+60] = cost
    live = Grid(live.width,live.height,live.resolution,data)
    result = extra(static,live)
    assert all(v.pose.x < 3 for v in result.views)


def test_disconnected_start_does_not_get_a_fake_route_home():
    static,live = room(closed=True)
    result = propose_extra_views(static,live,Pose(1,2,0),Pose(5,2,0),(),(),Camera(),
                                 Settings(),limit=6,failed_shift_m=0.4)
    assert result.views == ()
    assert result.pending_count > 0


def test_requested_failed_endpoint_never_grants_observation_credit():
    static, live = room()
    attempted = (Attempt(Pose(4, 2, 0), 'failed'),)
    assert extra(static, live, attempted=attempted).pending_count == extra(static, live).pending_count


@pytest.mark.parametrize('outcome', ['failed', 'rejected', 'cancelled'])
def test_failed_neighborhood_is_excluded_across_all_headings(outcome):
    static, live = room()
    initial = extra(static, live, outside_history())
    failed = initial.views[0].pose
    result = extra(static, live, outside_history(), (Attempt(failed, outcome),))
    assert all(math.hypot(v.pose.x-failed.x, v.pose.y-failed.y) > 0.4 for v in result.views)


def test_arrived_requested_endpoint_is_not_assumed_observed():
    static, live = room()
    assert extra(static, live, attempted=(Attempt(Pose(4, 2, 0), 'arrived'),)) == extra(static, live)


def test_other_heading_at_same_actual_position_can_reveal_more():
    static = Grid(40, 40, 0.1, [0]*1600)
    observed = (Seen(Pose(2.05, 2.05, 0)),)
    result = propose_extra_views(static, static, observed[0].pose, observed[0].pose,
                                 observed, (), Camera(), Settings(candidate_spacing=4.0),
                                 limit=8, failed_shift_m=0.4)
    assert result.views
    assert any(abs(angle_difference(v.pose.yaw, 0)) > 0.1 for v in result.views)


def test_candidate_safety_zones_exclude_endpoints():
    static, live = room()
    initial = extra(static, live)
    p = initial.views[0].pose
    result = extra(static, live, excluded=((p.x, p.y, 0.6),))
    assert all(math.hypot(v.pose.x-p.x, v.pose.y-p.y) > 0.6 for v in result.views)


def test_repeated_input_is_deterministic_and_does_not_mutate_initial_grids():
    static, live = room()
    saved = static.data, live.data
    assert extra(static, live, outside_history()) == extra(static, live, outside_history())
    assert (static.data, live.data) == saved


@pytest.mark.parametrize('limit', [0, 1, 3])
def test_result_limit(limit):
    static, live = room()
    assert len(extra(static, live, limit=limit).views) <= limit


@pytest.mark.parametrize('limit', [-1, True, 1.5])
def test_invalid_limits_raise(limit):
    static, live = room()
    with pytest.raises(ValueError):
        extra(static, live, limit=limit)


@pytest.mark.parametrize('value', [-1, math.nan, math.inf, True])
def test_invalid_failure_distance_raises(value):
    static, live = room()
    with pytest.raises(ValueError):
        propose_extra_views(static, live, Pose(1,2,0), Pose(1,2,0), (), (), Camera(),
                            Settings(), limit=6, failed_shift_m=value)


def test_invalid_history_is_not_silently_used():
    static, live = room()
    with pytest.raises(ValueError):
        extra(static, live, observed=(Seen(Pose(1,2,0), math.nan),))
    with pytest.raises(ValueError):
        extra(static, live, attempted=(Attempt(Pose(1,2,0), 'unknown'),))


def test_unsafe_start_is_not_snapped_across_a_wall():
    static, live = room()
    with pytest.raises(ValueError):
        propose_extra_views(static, live, Pose(1,2,0), Pose(3.01,0.5,0), (), (),
                            Camera(), Settings(), limit=6, failed_shift_m=0.4)


def test_cancellation_is_honored_before_and_during_calculation():
    static, live = room()
    stop = Event()
    stop.set()
    with pytest.raises(CancelledError):
        extra(static, live, cancel=stop)

    class StopLater:
        calls = 0

        def is_set(self):
            self.calls += 1
            return self.calls >= 3000

    stop = StopLater()
    with pytest.raises(CancelledError):
        extra(static, live, outside_history(), cancel=stop)
    assert stop.calls == 3000


def test_continuous_observation_pose_is_not_recentered_to_a_cell():
    static = Grid(30, 20, 0.1, [0]*600)
    a = pending_samples(static, static, (Seen(Pose(1.001,1.001,0)),), Camera(),
                        Settings(view_range=0.22, target_spacing=0.1))
    b = pending_samples(static, static, (Seen(Pose(1.099,1.099,0)),), Camera(),
                        Settings(view_range=0.22, target_spacing=0.1))
    assert a != b


def test_camera_translation_and_optical_yaw_are_used():
    static = Grid(30, 20, 0.1, [0]*600)
    seen = (Seen(Pose(1.05,1.05,0)),)
    front = pending_samples(static, static, seen, Camera(), Settings())
    back = pending_samples(static, static, seen, Camera(0.1,0,math.pi), Settings())
    assert front != back


@pytest.mark.parametrize('yaw', [0, math.pi/2, 0.37])
def test_common_world_transform_preserves_geometry(yaw):
    static, live = room()
    origin = Pose(7.13, -3.47, yaw)
    c, s = math.cos(yaw), math.sin(yaw)

    def transform(p):
        return Pose(origin.x+c*p.x-s*p.y, origin.y+s*p.x+c*p.y, p.yaw+yaw)

    a = extra(static, live, outside_history())
    shifted_static = Grid(static.width, static.height, static.resolution, static.data, origin)
    shifted_live = Grid(live.width, live.height, live.resolution, live.data, origin)
    b = propose_extra_views(shifted_static, shifted_live, transform(Pose(1,2,0)),
                            transform(Pose(1,2,0)),
                            tuple(Seen(transform(v.pose)) for v in outside_history()), (),
                            Camera(), Settings(), limit=6, failed_shift_m=0.4)
    assert a.pending_count == b.pending_count
    # Ranking ties may change at floating-point precision; compare useful gain
    # and safety rather than prescribing a translated coordinate tie breaker.
    assert len(a.views) == len(b.views)
    assert max(v.gain for v in a.views) == max(v.gain for v in b.views)
