from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from time import perf_counter
import math
from threading import Event

from rclpy.time import Time
from tf2_ros import TransformException

from equipo_jar.cobertura import camera_from_transform
from equipo_jar.cobertura_plan import Settings
from equipo_jar.cobertura_repaso import propose_extra_views
from equipo_jar.exploracion import near_box, yaw_from_quaternion, angle_difference
from equipo_jar.historial_vistas import AttemptedView
from equipo_jar.mapa import Grid, Pose
from equipo_jar.presupuesto import fits_extra, return_due, route_seconds
from equipo_jar.rutas_presupuesto import build_route_field, estimate_route


def geometry(grid):
    return grid.width, grid.height, grid.resolution, grid.origin


@dataclass(frozen=True)
class SearchJob:
    future: object
    epoch: int
    kind: str
    navigation_key: tuple
    optical_key: tuple
    number: int
    submitted_s: float


@dataclass(frozen=True)
class Calculation:
    field: object
    plan: object
    route_s: float
    planner_s: float
    compute_s: float
    queue_s: float
    stage: str
    error: Exception | None


class ExtraSearch:
    def __init__(self, node):
        self.node = node
        self.executor = None
        self.provider = propose_extra_views
        self.epoch = 0
        self.cancel = Event()
        self.job = None
        self.live = None
        self.optics = None
        self.live_at = None
        self.navigation_key = None
        self.optical_key = None
        self.field = None
        self.field_key = None
        self.views = None
        self.attempts = ()
        self.requested = None
        self.active = False
        self.phase = 'primary'
        self.accepted_count = 0
        self.failed_count = 0
        self.wait_since = None
        self.dwell_until = None
        self.done_reason = None
        self.config = None
        self.enabled = False
        self.valid = False
        self.route_attempt_at = None
        self.last_kind = None
        self.job_number = 0
        self.last_wait = None
        self.last_discard = None

    def diagnostic(self, event, **values):
        if (not self.node.p['extra_search_enabled'] and not self.enabled
                and self.job is None and 'job' not in values):
            return
        fields = ' '.join(f'{key}={value}' for key, value in values.items())
        self.node.get_logger().info(f'Extra search event={event} generation={self.epoch} {fields}')

    def pending(self):
        if self.job is None:
            return dict(pending_job='none', pending_kind='none', pending_status='none')
        future = self.job.future
        status = ('cancelled' if future.cancelled() else 'done' if future.done()
                  else 'running' if future.running() else 'queued')
        return dict(pending_job=self.job.number, pending_kind=self.job.kind, pending_status=status)

    def waiting(self, kind, reason):
        if self.phase != 'extra':
            return
        key = (kind, reason, self.job.number if self.job is not None else None)
        if key != self.last_wait:
            self.diagnostic('wait', kind=kind, reason=reason, **self.pending())
            self.last_wait = key

    def invalidate(self, reason='state_change'):
        self.diagnostic('invalidate', reason=reason, **self.pending())
        self.epoch += 1
        self.cancel.set()
        if self.job is not None:
            self.job.future.cancel()
        self.field = None
        self.views = None

    def reset_map(self):
        self.invalidate('map_changed')
        self.optics = None
        self.live = None
        self.live_at = None
        self.attempts = ()
        self.node.search_views = ()
        if self.active:
            self.done_reason = 'cambió el mapa durante la búsqueda extra'

    def on_costmap(self, message, at_s=None):
        if not self.node.p['extra_search_enabled']:
            return
        try:
            live = Grid.from_occupancy_grid(message)
        except (TypeError, ValueError, OverflowError):
            self.live = None
            self.live_at = None
            return
        if self.live is not None and geometry(self.live) != geometry(live):
            self.reset_map()
        if self.optics is None:
            self.optics = live
        else:
            remembered = tuple(100 if old == 100 or new == 100 else new
                               for old, new in zip(self.optics.data, live.data))
            self.optics = Grid(live.width, live.height, live.resolution, remembered, live.origin)
        self.live = live
        self.navigation_key = (geometry(live), bytes(0 <= value <= 98 for value in live.data))
        optical_key = (geometry(self.optics), bytes(2 if value == 100 else 1 if value < 0 else 0
                                                  for value in self.optics.data))
        if optical_key != self.optical_key:
            self.views = None
        self.optical_key = optical_key
        self.live_at = at_s if at_s is not None else self.node.costmap_at

    def settings(self):
        p = self.node.p
        for key in ('extra_max_goals', 'extra_max_failed', 'extra_history_max'):
            if type(p[key]) is not int or p[key] < 1:
                raise ValueError(key)
        for key in ('extra_goal_timeout_s', 'extra_plan_timeout_s', 'extra_costmap_max_age_s',
                    'extra_pose_max_age_s'):
            if isinstance(p[key], bool) or not math.isfinite(p[key]) or p[key] <= 0:
                raise ValueError(key)
        for key in ('return_reserve_s', 'extra_failed_shift_m', 'extra_history_interval_s',
                    'extra_history_shift_m', 'extra_history_turn_rad'):
            if isinstance(p[key], bool) or not math.isfinite(p[key]) or p[key] < 0:
                raise ValueError(key)
        if self.seconds(0.0) is None:
            raise ValueError('return estimator')
        return Settings(robot_radius=p['extra_transit_radius_m'], safety_margin=0.0,
                        candidate_spacing=p['extra_spacing_m'], target_spacing=p['extra_target_spacing_m'],
                        view_range=p['extra_range_m'], fov_degrees=p['extra_fov_deg'],
                        heading_count=p['extra_heading_count'], travel_speed=p['return_estimate_speed_mps'],
                        turn_speed=p['extra_turn_speed_rads'], observation_time=p['extra_look_s'],
                        max_views=p['extra_max_views'], max_positions=p['extra_max_positions'],
                        max_targets=p['extra_max_targets'], view_clearance=p['extra_view_clearance_m'],
                        view_preferred_clearance=p['extra_preferred_clearance_m'])

    def seconds(self, distance):
        p = self.node.p
        return route_seconds(distance, speed_mps=p['return_estimate_speed_mps'],
                             factor=p['return_estimate_factor'], overhead_s=p['return_estimate_overhead_s'])

    def readiness_reason(self):
        now = self.node.seconds_since_t0()
        limit = self.node.p['extra_costmap_max_age_s']
        if (not isinstance(limit, (int, float)) or isinstance(limit, bool)
                or not math.isfinite(limit) or limit <= 0):
            return 'costmap_age_invalid'
        if now is None:
            return 'clock_unavailable'
        if self.live is None or self.live_at is None:
            return 'costmap_missing'
        if not 0 <= now - self.live_at <= limit:
            return 'costmap_stale'
        if self.node.grid is None:
            return 'map_missing'
        if not self.node.start_locked:
            return 'start_unlocked'
        if self.node.start_pose is None:
            return 'start_missing'
        if not self.node.search_localization_ready:
            return 'localization_not_ready'
        if self.node.fresh_search_pose() is None:
            return 'pose_unavailable'
        return None

    def ready(self):
        return self.readiness_reason() is None

    def submit(self, kind):
        if self.job is not None:
            self.waiting(kind, 'job_busy')
            return
        reason = self.readiness_reason()
        if reason is not None:
            self.waiting(kind, reason)
            return
        node = self.node
        if (kind == self.last_kind and self.route_attempt_at is not None
                and node.seconds_since_t0() - self.route_attempt_at < node.p['extra_costmap_max_age_s'] / 2):
            self.waiting(kind, 'retry_interval')
            return
        try:
            settings = self.settings()
            robot = node.fresh_search_pose()
            start = Pose(*node.start_pose)
            camera = None
            if kind == 'plan':
                camera = camera_from_transform(node.tf_buffer.lookup_transform(
                    'base_footprint', node.p['extra_camera_frame'], Time()).transform)
            zones = tuple(node.current_box_zones())
        except (TypeError, ValueError, OverflowError, TransformException) as error:
            self.waiting(kind, f'input_error_{type(error).__name__}')
            node.get_logger().warning(f'Búsqueda extra no disponible: {error}', throttle_duration_sec=5.0)
            return
        live, optics, raw = self.live, self.optics, node.grid
        observed, attempted = node.search_views, self.attempts
        at_s, epoch, provider = self.live_at, self.epoch, self.provider
        self.cancel = Event()
        cancel = self.cancel
        kwargs = dict(limit=settings.max_views, failed_shift_m=node.p['extra_failed_shift_m'],
                      excluded=zones, cancel=cancel, occlusion_grid=optics,
                      same_point_distance_m=node.p['same_point_distance_m'],
                      same_point_angle_rad=node.p['same_point_angle_rad'])

        submitted = perf_counter()

        def calculate():
            started = perf_counter()
            field = plan = error = None
            route_s = planner_s = 0.0
            stage = 'route'
            try:
                field = build_route_field(live, robot, start, at_s, cancel=cancel)
                route_s = perf_counter() - started
                if kind == 'plan' and field is not None:
                    stage = 'planner'
                    plan = provider(raw, live, robot, start, observed, attempted, camera, settings, **kwargs)
            except Exception as caught:
                error = caught
            finished = perf_counter()
            if stage == 'route':
                route_s = finished - started
            else:
                planner_s = finished - started - route_s
            return Calculation(field, plan, route_s, planner_s, finished - started,
                               started - submitted, stage, error)

        if self.executor is None:
            self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='mission-extra')
        self.job_number += 1
        self.job = SearchJob(self.executor.submit(calculate), epoch, kind, self.navigation_key,
                             self.optical_key, self.job_number, node.seconds_since_t0())
        self.route_attempt_at = node.seconds_since_t0()
        self.last_kind = kind
        self.last_wait = None
        wait = 0.0 if self.wait_since is None else node.seconds_since_t0() - self.wait_since
        self.diagnostic('submit', job=self.job.number, kind=kind, phase=self.phase,
                        views=len(observed), attempts=len(attempted),
                        input_age_s=f'{node.seconds_since_t0() - at_s:.3f}', wait_s=f'{wait:.3f}')

    def poll_job(self, *, accept=True):
        job = self.job
        if job is None or not job.future.done():
            return
        self.job = None
        error = None
        calculation = None
        try:
            calculation = job.future.result()
            error = calculation.error
        except Exception as caught:
            error = caught
        timing = ({name: f'{getattr(calculation, name[:-3] + "_s") * 1000:.3f}'
                   for name in ('route_ms', 'planner_ms', 'compute_ms', 'queue_ms')}
                  if calculation is not None else {})
        self.diagnostic('result', job=job.number, kind=job.kind, job_generation=job.epoch,
                        stage=calculation.stage if calculation is not None else 'unstarted',
                        outcome=type(error).__name__ if error is not None else 'ok',
                        elapsed_s=f'{self.node.seconds_since_t0() - job.submitted_s:.3f}', **timing)
        reason = None
        changed = 0
        if not accept or self.node.state != 'exploring' or not self.node.p['extra_search_enabled']:
            reason = 'inactive'
        elif job.epoch != self.epoch:
            reason = 'generation_changed'
        elif job.navigation_key != self.navigation_key:
            reason = 'navigation_changed'
            if self.navigation_key is not None and job.navigation_key[0] == self.navigation_key[0]:
                changed = sum(a != b for a, b in zip(job.navigation_key[1], self.navigation_key[1]))
        if reason is not None:
            if (reason == 'navigation_changed' and job.kind == 'plan'
                    and calculation is not None and error is None
                    and calculation.plan is not None and calculation.plan.views
                    and job.optical_key == self.optical_key):
                self.views = calculation.plan.views
                self.field = None
                self.field_key = None
                self.diagnostic('retain_views', job=job.number, kind=job.kind,
                                reason=reason, changed_cells=changed, views=len(self.views),
                                route='refresh_required')
                return
            self.last_discard = reason
            self.diagnostic('discard', job=job.number, kind=job.kind, reason=reason, changed_cells=changed)
            return
        if error is not None:
            self.node.get_logger().warning(f'Falló la búsqueda extra: {error}')
            if self.phase == 'extra':
                self.done_reason = 'falló el cálculo de vistas extra'
            return
        self.field = calculation.field
        self.field_key = job.navigation_key
        if job.kind == 'plan':
            if job.optical_key != self.optical_key:
                self.last_discard = 'optical_changed'
                self.diagnostic('discard', job=job.number, kind=job.kind, reason='optical_changed')
                return
            plan = calculation.plan
            self.views = () if plan is None else plan.views
            pending = None if plan is None else plan.pending_count
            self.node.get_logger().info(f'Plan extra: {len(self.views)} vistas, {pending} muestras pendientes')
            if not self.views:
                self.done_reason = 'sin vistas extra alcanzables'
        self.diagnostic('accept', job=job.number, kind=job.kind,
                        route='available' if self.field is not None else 'missing',
                        field_age_s=f'{self.node.seconds_since_t0() - self.field.at_s:.3f}' if self.field is not None else 'none')

    def tick(self):
        node = self.node
        enabled = bool(node.p['extra_search_enabled'])
        key = tuple((k, v) for k, v in sorted(node.p.items())
                    if k.startswith(('extra_', 'return_estimate_', 'same_point_')) or k == 'return_reserve_s')
        if key != self.config:
            self.invalidate('config_changed')
            self.config = key
        if not enabled:
            if self.enabled:
                self.invalidate('disabled')
                self.live = self.optics = None
                self.live_at = None
                node.search_views = ()
                if self.phase == 'extra':
                    self.done_reason = 'búsqueda extra desactivada'
            self.enabled = False
            return
        self.enabled = True
        if self.live is None and node.costmap is not None:
            self.on_costmap(node.costmap, node.costmap_at)
        self.poll_job()
        self.valid = False
        try:
            self.settings()
            self.valid = True
        except (TypeError, ValueError, OverflowError):
            if self.phase == 'extra':
                self.done_reason = 'parámetros extra inválidos'
            return
        now = node.seconds_since_t0()
        if self.phase == 'extra' and self.wait_since is not None:
            if now - self.wait_since >= node.p['extra_plan_timeout_s']:
                self.diagnostic('timeout', wait_s=f'{now - self.wait_since:.3f}',
                                readiness=self.readiness_reason() or 'ready',
                                last_discard=self.last_discard or 'none', **self.pending())
                self.done_reason = 'se agotó la espera del plan extra'
        if (self.phase == 'primary' or self.active or node.current_inspection is not None) and self.job is None:
            interval = node.p['extra_costmap_max_age_s'] / 2
            if self.route_attempt_at is None or now - self.route_attempt_at >= interval:
                self.submit('route')

    def enter(self):
        if not self.valid:
            return 'parámetros extra inválidos'
        if self.phase != 'extra':
            if not self.node.start_locked:
                pose = self.node.fresh_search_pose()
                if pose is not None:
                    self.node.start_pose = (pose.x, pose.y, pose.yaw)
                    self.node.start_locked = True
            self.phase = 'extra'
            self.last_wait = None
            self.last_discard = None
            self.diagnostic('enter', readiness=self.readiness_reason() or 'ready', **self.pending())
            self.node.get_logger().info(f'Inicio de búsqueda extra, generación {self.epoch}')
            self.wait_since = self.node.seconds_since_t0()
            self.views = None
            self.submit('plan')
        return self.reason()

    def estimate(self, goal):
        node = self.node
        if not self.ready() or self.field is None or self.field_key != self.navigation_key:
            return None
        now = node.seconds_since_t0()
        estimate = estimate_route(self.field, goal, now, max_age_s=node.p['extra_costmap_max_age_s'])
        current = node.fresh_search_pose()
        root = estimate_route(self.field, current, now, max_age_s=node.p['extra_costmap_max_age_s'])
        if estimate is None or root is None:
            return None
        return estimate.outward_m + root.outward_m, estimate.home_m

    def affordable(self, goal):
        estimate = self.estimate(goal)
        if estimate is None:
            return False
        node, p = self.node, self.node.p
        return fits_extra(node.seconds_since_t0(), outward_s=self.seconds(estimate[0]),
                          home_s=self.seconds(estimate[1]), look_s=p['extra_look_s'],
                          reserve_s=p['return_reserve_s'], return_at_s=p['return_time_s'],
                          finish_at_s=p['finish_time_s'])

    def reason(self):
        node, p = self.node, self.node.p
        if self.done_reason is not None:
            return self.done_reason
        if p['extra_search_enabled'] and self.valid:
            pose = node.fresh_search_pose()
            estimate = self.estimate(pose) if pose is not None else None
            if estimate is not None and return_due(
                    node.seconds_since_t0(), home_s=self.seconds(estimate[1]),
                    reserve_s=p['return_reserve_s'], return_at_s=p['return_time_s'],
                    finish_at_s=p['finish_time_s']):
                return 'reserva necesaria para volver'
        if self.phase == 'extra' and not self.active and node.current_inspection is None:
            if self.dwell_until is not None and node.seconds_since_t0() < self.dwell_until:
                return None
            if self.accepted_count >= p['extra_max_goals']:
                return 'máximo de objetivos extra'
            if self.failed_count >= p['extra_max_failed']:
                return 'máximo de fallos extra'
        return None

    def explore(self):
        node = self.node
        if self.done_reason is not None:
            return
        now = node.seconds_since_t0()
        if self.dwell_until is not None and now < self.dwell_until:
            return
        if self.wait_since is None:
            self.wait_since = now
        if self.views is None:
            self.submit('plan')
            return
        if not self.ready() or self.estimate(node.fresh_search_pose()) is None:
            self.submit('route')
            return
        for view in self.views:
            pose = view.pose
            if not math.isfinite(view.gain) or view.gain <= 0:
                continue
            repeated = any(a.outcome == 'arrived'
                           and math.hypot(a.pose.x - pose.x, a.pose.y - pose.y) <= node.p['same_point_distance_m']
                           and angle_difference(a.pose.yaw, pose.yaw) <= node.p['same_point_angle_rad']
                           for a in self.attempts)
            if repeated or near_box(pose.x, pose.y, node.current_box_zones()) or node.is_blocked(pose.x, pose.y):
                continue
            if not self.affordable(pose):
                continue
            estimate = self.estimate(pose)
            if estimate is None:
                continue
            self.active = True
            goal = node.make_pose(pose.x, pose.y, pose.yaw)
            if node.go_to(goal):
                node.current_point = goal
                node.current_in_place = False
                self.views = None
                self.wait_since = None
                node.get_logger().info(
                    f'Vista extra: ({pose.x:.2f}, {pose.y:.2f}), ganancia {view.gain}, '
                    f'ida/vuelta {self.seconds(estimate[0]):.1f}/{self.seconds(estimate[1]):.1f} s, '
                    f'reserva {node.p["return_reserve_s"]:.1f} s')
            else:
                self.active = False
            return
        self.done_reason = 'ninguna vista extra entra en el presupuesto'

    def sent(self, pose):
        if self.node.p['extra_search_enabled'] and self.node.state == 'exploring':
            if self.phase == 'extra' and not self.active:
                self.invalidate('inspection')
                self.wait_since = None
            self.requested = Pose(pose.pose.position.x, pose.pose.position.y,
                                  yaw_from_quaternion(pose.pose.orientation))

    def accepted(self):
        if self.active:
            self.accepted_count += 1

    def result(self, outcome):
        if self.requested is not None:
            if (not self.node.inspection_is_reported()
                    and type(self.node.p['extra_history_max']) is int and self.node.p['extra_history_max'] > 0):
                self.attempts = (*self.attempts, AttemptedView(self.requested, outcome))[-self.node.p['extra_history_max']:]
            self.requested = None
            if self.phase == 'extra':
                self.views = None
                self.wait_since = None
        if self.active:
            self.active = False
            self.node.get_logger().info(f'Resultado extra: {outcome}')
            self.failed_count = 0 if outcome == 'arrived' else self.failed_count + 1
            self.dwell_until = self.node.seconds_since_t0() + (self.node.p['extra_look_s'] if outcome == 'arrived' else 0.0)
            self.wait_since = None
            self.views = None

    def close(self):
        self.invalidate('shutdown')
        self.poll_job(accept=False)
        if self.executor is not None:
            self.executor.shutdown(wait=False, cancel_futures=True)
