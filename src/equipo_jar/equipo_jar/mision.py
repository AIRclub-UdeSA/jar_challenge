"""Misión: qué hace el robot ahora, y cuándo reportar y terminar.

Recibe /map, el plan de cobertura y las candidatas; manda objetivos a
navegación con la acción NavigateToPose y publica /victimas y /done
(docs/team/INTERFACES.md). Es el único que reporta y el único cliente de
navegación. No publica /cmd_vel: eso es solo de navegación.

Esqueleto: espera el mapa y no hace nada más. Hasta recibir /map no se puede
mover el robot ni reportar (§5).
"""
import os
import time
from ament_index_python.packages import get_package_share_directory

from geometry_msgs.msg import PointStamped
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import Empty, Bool
from geometry_msgs.msg import PoseStamped

from equipo_jar.comun import QOS_ESTADO, QOS_MAPA, correr, parametros_en_vivo
from equipo_jar.reportes import armar_reporte
from equipo_jar_msgs.msg import Candidatas
from equipo_jar.confirmacion import near_any, select_to_report
from equipo_jar.exploracion import yaw_from_quaternion, first_pending, angle_difference, goal_position, box_zones, near_box, return_verdict, goal_blocked, start_ready
from equipo_jar.inspeccion import (pick_doubtful, viewpoint_for, forget_failed_inspection,
                                   unhelpful_inspection, inspection_outcome)
from equipo_jar.mapa import Grid, Pose, planar_yaw
from equipo_jar.historial_vistas import ObservedView, append_view
from equipo_jar.vuelta import steps_to_start, way_back_m, return_leg
from equipo_jar.busqueda_extra import ExtraSearch

from tf2_ros import Buffer, TransformListener, TransformException
from rclpy.time import Time
import math
from action_msgs.msg import GoalStatus

import traceback


class Mision(Node):
    def __init__(self):
        super().__init__('mision')
        # Cantidad de víctimas a buscar (N), que llega como argumento del launch (victimas:=N).
        self.declare_parameter('cantidad_victimas', 5)
        self.cantidad_victimas = self.get_parameter('cantidad_victimas').value
        self.p = parametros_en_vivo(self, {'min_observations': 5, 'max_deviation_m': 0.10, 
                                            'min_report_distance_m': 0.40, 'same_point_distance_m': 0.02, 
                                            'same_point_angle_rad': 0.1, 'return_time_s': 480.0,
                                            'min_observations_final': 1, 'max_deviation_final_m': 0.30, 
                                            'finish_time_s': 570.0, 'idle_spin_s': 10.0, 'idle_spin_rad': 1.57,
                                            'visit_min_observations': 2, 'visit_distance_m': 0.8, 
                                            'visit_clearance_m': 0.25, 'visit_angles': 12, 
                                            'visit_min_shift_m': 0.4, 'visit_box_margin_m': 0.25, 
                                            'localization_wait_max_s': 10.0, 'goal_timeout_s': 90.0, 'stuck_time_s': 20.0,
                                            'goal_accept_timeout_s': 5.0, 'stop_confirm_max_age_s': 0.5,
                                            'stuck_distance_m': 0.05, 'stuck_angle_rad': 0.2, 'stop_wait_s': 1.0,
                                            'done_limit_s': 590.0, 'turn_in_place_m': 0.20, 
                                            'visit_max_attempts': 3, 'box_min_observations': 2,
                                            'box_clearance_m': 0.42, 'box_clearance_sigmas': 1.0,    
                                            'visit_transit_radius_m': 0.15, 'visit_free_threshold': 25,
                                            'settle_s': 0.0, 'max_recoveries': 10, 'return_tolerance_m': 0.05,
                                            'return_max_tries': 3, 'return_still_s': 2.0, 'return_still_m': 0.01,
                                            'return_check_max_s': 10.0,
                                            'costmap_blocked_value': 99, 'inspection_wait_s': 2.0,
                                            'final_look_s': 1.0,
                                            'return_direct_max_m': 10.0, 'return_leg_m': 5.0,
                                            'return_max_legs': 10, 'return_max_failed_legs': 2,
                                            'return_snap_m': 0.3,
                                            'extra_search_enabled': False,
                                            'extra_history_max': 1200,
                                            'extra_history_interval_s': 1.0,
                                            'extra_history_shift_m': 0.10,
                                            'extra_history_turn_rad': 0.30,
                                            'extra_pose_max_age_s': 1.0,
                                            'return_estimate_speed_mps': 0.20,
                                            'return_estimate_factor': 1.5,
                                            'return_estimate_overhead_s': 8.0,
                                            'return_reserve_s': 45.0,
                                            'extra_look_s': 2.0,
                                            'extra_max_goals': 6,
                                            'extra_max_failed': 2,
                                            'extra_goal_timeout_s': 30.0,
                                            'extra_plan_timeout_s': 5.0,
                                            'extra_costmap_max_age_s': 2.5,
                                            'extra_failed_shift_m': 0.40,
                                            'extra_range_m': 2.0,
                                            'extra_fov_deg': 61.0,
                                            'extra_heading_count': 8,
                                            'extra_spacing_m': 0.45,
                                            'extra_target_spacing_m': 0.15,
                                            'extra_max_positions': 256,
                                            'extra_max_targets': 2048,
                                            'extra_max_views': 80,
                                            'extra_transit_radius_m': 0.21,
                                            'extra_view_clearance_m': 0.24,
                                            'extra_preferred_clearance_m': 0.30,
                                            'extra_turn_speed_rads': 0.7,
                                            'extra_camera_frame': 'cam_1_color_optical_frame',
                                            # Still guard (wall clock, like the judge's 60 s rule):
                                            # still = /odom at or below these (the judge's rule).
                                            'still_linear_mps': 0.02,
                                            'still_angular_rads': 0.05,
                                            'still_rescue_s': 30.0,   # cancel the trip in progress
                                            'still_finish_s': 45.0,   # final reports round (once)
                                            'cancel_timeout_s': 10.0,  # a cancel with no result: move on
                                            'goal_accept_max_timeouts': 3})

        self.mapa = None
        self.grid = None
        self.plan = None
        self.candidatas = None

        self.costmap = None
        self.costmap_at = None
        self.search_views = ()
        self.search_map_key = None
        self.search_localization_ready = False

        self.t0 = None
        self.create_timer(0.1, self.step)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.start_pose = None

        self.reported_ids = set()
        self.reported_points = []
        self.inspected_points = []
        self.failed_inspections = {}

        self.nav_state = 'idle'
        self.state = 'waiting_for_map'

        self.visited = []
        self.current_point = None
        self.current_inspection = None  # (id, x, y, observaciones) del viaje de inspección en curso
        self.look_until = None  # al llegar a mirar una candidata: hasta cuándo espera a que se confirme
        self.inspection_spot = None  # (x, y) desde donde se mira la candidata del viaje en curso
        self.inspection_cancel_goal = None
        self.inspection_canceling = False
        self.tried_views = {}   # por id: lugares que no dieron resultado o a los que Nav2 no llegó
        self.settle_until = None
        self.current_goal = None
        self.current_in_place = False

        self.return_sent = False
        self.return_tree = os.path.join(get_package_share_directory('equipo_jar'),
                                'behavior_trees', 'vuelta_precisa.xml')
        self.return_tries = 0
        self.return_by_legs = False   # la vuelta va por tramos (larga, o Nav2 no llegó directo)
        self.return_legs = 0          # tramos mandados
        self.return_leg_active = False  # el viaje en curso es un tramo, no la largada
        self.failed_legs = 0          # tramos fallidos seguidos
        self.check_since = None
        self.check_pose = None
        self.check_started = None
        self.canceling = False

        self.idle_since = None
        self.nothing_left_since = None  # desde cuándo no quedan puntos ni dudosas

        self.start_locked = False
        self.localization_ready = False
        self.start_timeout_logged = False

        self.goal_start_time = None
        self.last_motion_pose = None
        self.last_motion_time = None
        self.stuck_canceling = False

        self.stop_canceling = False
        self.stop_since = None
        self.stop_confirmed_at = None

        self.last_skipped = None

        self.recoveries = 0
        self.extra = ExtraSearch(self)

        # Goal bookkeeping: each sent goal gets a number, so a late answer or result of an
        # abandoned goal is not taken for the current one.
        self.goal_seq = 0
        self.goal_rejected = False
        self.accept_timeouts = 0
        self.cancel_sent_at = None     # monotonic time of the pending cancel, if any
        self.goal_ever_accepted = False
        # Still guard (wall clock): since when /odom shows the robot still.
        self.still_since = None
        self.odom_at = None
        self.still_rescued = False
        self.still_reported = False
        # Still time counts only from the first accepted goal: the robot waits still for /map
        # (any time, the judge's clock starts at /map) and while it localizes.
        self.still_from = None

        self.create_subscription(OccupancyGrid, '/map', self.al_recibir_mapa, QOS_MAPA)
        self.create_subscription(Path, '/jar/plan_cobertura', self.al_recibir_plan, QOS_ESTADO)
        self.create_subscription(
            Candidatas, '/jar/candidatas', self.al_recibir_candidatas, QOS_ESTADO)
        self.navegacion = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.pub_victimas = self.create_publisher(PointStamped, '/victimas', 10)
        self.pub_done = self.create_publisher(Empty, '/done', 10)
        self.pub_stop = self.create_publisher(Bool, '/jar/parar', QOS_ESTADO)
        self.create_subscription(Bool, '/jar/parada_confirmada', self.on_stop_confirmed, QOS_ESTADO)
        self.get_logger().info(
            f'Hay que encontrar {self.cantidad_victimas} víctimas. Esperando el mapa en /map...')
        self.create_subscription(OccupancyGrid, '/global_costmap/costmap', self.on_costmap, QOS_MAPA)
        self.create_subscription(Bool, '/jar/localizacion_lista', self.on_localization_ready, QOS_ESTADO)
        self.create_subscription(Odometry, '/odom', self.on_odom, 10)

    def on_odom(self, msg):
        """Still guard: the judge ends the run after 60 s still on /odom (wall clock)."""
        try:
            tw = msg.twist.twist
            values = (tw.linear.x, tw.linear.y, tw.angular.z)
            if not all(math.isfinite(v) for v in values):
                self.still_since = None
                return
            now = self.wall_now()
            self.odom_at = now
            # Same test as the practice judge: moving above 0.02 m/s (x-y speed) or 0.05 rad/s.
            still = (math.hypot(tw.linear.x, tw.linear.y) <= self.p['still_linear_mps']
                     and abs(tw.angular.z) <= self.p['still_angular_rads'])
            if not still:
                self.still_since = None
                self.still_rescued = False
                self.still_reported = False
            elif self.still_since is None:
                self.still_since = now
        except Exception:
            self.get_logger().error(f'Error en on_odom:\n{traceback.format_exc()}',
                                    throttle_duration_sec=5.0)

    def wall_now(self):
        """Wall clock for the still guard and cancel timeouts (tests replace it)."""
        return time.monotonic()

    def still_for(self):
        """Wall seconds the robot has been still on /odom; 0 if moving or /odom is stale."""
        now = self.wall_now()
        if (self.still_since is None or self.still_from is None or self.odom_at is None
                or now - self.odom_at > 1.0):
            return 0.0
        return now - max(self.still_since, self.still_from)

    def cancel_current(self, reason):
        """Cancel the goal in progress (stuck path). Returns False if it can't be sent."""
        try:
            self.current_goal.cancel_goal_async()
        except Exception as e:
            self.get_logger().warning(f'Navigation cancellation unavailable: {e}')
            return False
        self.stuck_canceling = True
        self.cancel_sent_at = self.wall_now()
        self.get_logger().warning(f'Cancelo el viaje: {reason}')
        return True

    def check_pending_cancel(self):
        """A cancel whose result never comes (seen when planner_server died): give the goal
        up as failed after cancel_timeout_s, so mission sends the next one (Nav2 preempts)."""
        if self.nav_state != 'moving' or self.cancel_sent_at is None:
            return
        if self.wall_now() - self.cancel_sent_at < self.p['cancel_timeout_s']:
            return
        self.get_logger().warning(f'No result {self.p["cancel_timeout_s"]:.0f} s after the cancel: '
                                  'I take the trip as failed')
        self.goal_seq += 1          # a late result of that goal is ignored
        self.current_goal = None
        self.cancel_sent_at = None
        self.stuck_canceling = False
        self.inspection_canceling = False
        self.canceling = False
        self.nav_state = 'failed'

    def check_still(self):
        """Wall-clock guard against the 60 s still rule, across goals and states."""
        if self.state not in ('exploring', 'returning', 'checking') or not self.goal_ever_accepted:
            return
        still = self.still_for()
        if still >= self.p['still_finish_s'] and not self.still_reported:
            # The 60 s rule may end the run soon: send what the end of the run would send.
            # No /done: if the robot moves again, the run goes on.
            self.still_reported = True
            self.get_logger().warning(f'Robot still for {still:.0f} s (wall clock): final '
                                      'reports round before the 60 s rule')
            self.check_reports(self.p['min_observations_final'], self.p['max_deviation_final_m'])
            return
        if (still >= self.p['still_rescue_s'] and not self.still_rescued
                and self.nav_state == 'moving' and self.cancel_sent_at is None):
            self.still_rescued = True
            self.cancel_current(f'robot still for {still:.0f} s (wall clock)')

    def al_recibir_mapa(self, mapa):
        if self.mapa is None:
            self.get_logger().info(
                f'Mapa recibido: {mapa.info.width}x{mapa.info.height} celdas, '
                f'resolución {mapa.info.resolution:.3f} m')
            self.t0 = self.get_clock().now()
        self.mapa = mapa
        try:
            self.grid = Grid.from_occupancy_grid(mapa)
            key = (self.grid.width, self.grid.height, self.grid.resolution,
                   self.grid.origin, self.grid.data)
            if key != self.search_map_key:
                self.extra.reset_map()
                self.search_views = ()
                self.search_map_key = key
        except ValueError as e:
            self.grid = None
            self.extra.reset_map()
            self.search_views = ()
            self.search_map_key = None
            self.get_logger().error(f'No pude armar la grilla del mapa: {e}. Sin inspecciones.')

    def on_costmap(self, msg):
        self.costmap = msg
        self.costmap_at = self.seconds_since_t0()
        self.extra.on_costmap(msg, self.costmap_at)

    def is_blocked(self, x, y):
        return goal_blocked(self.costmap, x, y, self.p['costmap_blocked_value'])

    def on_localization_ready(self, msg):
        self.search_localization_ready = bool(msg.data)
        if msg.data:
            self.localization_ready = True

    def fresh_search_pose(self):
        if self.grid is None or self.t0 is None or not self.search_localization_ready:
            return None
        try:
            limit = self.p['extra_pose_max_age_s']
            if isinstance(limit, bool) or not math.isfinite(limit) or limit <= 0:
                return None
            transform = self.tf_buffer.lookup_transform('map', 'base_footprint', Time())
            if transform.header.frame_id != 'map':
                return None
            stamp = Time.from_msg(transform.header.stamp)
            age = (self.get_clock().now() - stamp).nanoseconds / 1e9
            if stamp.nanoseconds <= 0 or age < 0 or age > limit:
                return None
            position = transform.transform.translation
            return Pose(position.x, position.y, planar_yaw(transform.transform.rotation))
        except (TransformException, TypeError, ValueError, OverflowError):
            return None

    def update_search_history(self):
        if not self.p['extra_search_enabled'] or self.state != 'exploring':
            return
        pose = self.fresh_search_pose()
        if pose is None:
            return
        try:
            self.search_views = append_view(
                self.search_views, ObservedView(pose, self.seconds_since_t0()),
                max_items=self.p['extra_history_max'],
                min_interval_s=self.p['extra_history_interval_s'],
                min_shift_m=self.p['extra_history_shift_m'],
                min_turn_rad=self.p['extra_history_turn_rad'])
        except (TypeError, ValueError, OverflowError) as error:
            self.get_logger().warning(f'View history unavailable: {error}', throttle_duration_sec=5.0)

    def seconds_since_t0(self):
        if self.t0 is None:
            return None
        now = self.get_clock().now()
        elapsed = now - self.t0
        seconds = elapsed.nanoseconds / 1e9
        return float(seconds)

    def current_pose(self):
        try:
            pose = self.tf_buffer.lookup_transform('map', 'base_footprint', Time())
            x = pose.transform.translation.x
            y = pose.transform.translation.y
            q = pose.transform.rotation
            yaw = yaw_from_quaternion(q)
            return (x, y, yaw)
        except TransformException:
            return None

    def odom_pose(self):
        try:
            pose = self.tf_buffer.lookup_transform('odom', 'base_footprint', Time())
            x = pose.transform.translation.x
            y = pose.transform.translation.y
            q = pose.transform.rotation
            yaw = yaw_from_quaternion(q)
            return (x, y, yaw)
        except TransformException:
            return None

    def check_reports(self, min_obs, max_deviation):
        if self.candidatas is None:
            return
        selected = select_to_report(self.candidatas, self.reported_ids, self.reported_points,
                                        self.cantidad_victimas, min_obs, 
                                        max_deviation, self.p['min_report_distance_m'])
        for c in selected:
            self.reportar_victima(c.centro.x, c.centro.y)
            self.reported_ids.add(c.id)
            self.reported_points.append((c.centro.x, c.centro.y))
            i = len(self.reported_points)
            self.get_logger().info(f'Reporte {i}. id={c.id} x={c.centro.x :.2f} y={c.centro.y:.2f}')

    def go_to(self, pose, tree=''):
        if self.state in ('stopping', 'finishing', 'finished'):
            return False
        if not self.navegacion.server_is_ready():
            self.get_logger().warn('!!! El servidor de navegacion no esta listo !!!', throttle_duration_sec=5.0)
            return False
        goal = NavigateToPose.Goal()
        goal.pose = pose
        goal.behavior_tree = tree
        self.goal_seq += 1
        seq = self.goal_seq
        self.goal_rejected = False
        self.cancel_sent_at = None
        self.current_goal = None
        self.canceling = False
        self.inspection_cancel_goal = None
        self.inspection_canceling = False
        self.nav_state = "sending"
        self.recoveries = 0
        self.extra.sent(pose)
        now = self.seconds_since_t0()
        self.goal_start_time = now
        self.last_motion_time = now
        self.last_motion_pose = self.odom_pose()
        self.stuck_canceling = False
        try:
            future = self.navegacion.send_goal_async(goal, feedback_callback=self.on_feedback)
            future.add_done_callback(lambda f, s=seq: self.on_goal_response(f, s))
        except Exception as e:
            self.get_logger().error(f'Navigation goal send failed: {e}; stopping')
            self.change_state('stopping')
            return False
        if not self.start_locked:
            self.start_locked = True
            if self.start_pose is not None:
                x, y, yaw = self.start_pose
                self.get_logger().info(f'Pose de inicio fija: x={x:.2f} y={y:.2f} yaw={yaw:.2f}')
            else:
                self.get_logger().warning('Primer viaje sin pose de inicio: no se va a poder volver')
        return True

    def inspection_is_reported(self):
        if self.current_inspection is None:
            return False
        cand_id, x, y, _ = self.current_inspection
        return (cand_id in self.reported_ids
                or near_any(x, y, self.reported_points, self.p['min_report_distance_m']))

    def cancel_reported_inspection(self):
        if (self.state != 'exploring' or self.nav_state != 'moving'
                or self.current_goal is None or self.stuck_canceling
                or not self.inspection_is_reported()
                or self.inspection_cancel_goal is self.current_goal):
            return
        goal = self.current_goal
        self.inspection_cancel_goal = goal
        self.inspection_canceling = True
        self.cancel_sent_at = self.wall_now()
        try:
            future = goal.cancel_goal_async()
            future.add_done_callback(lambda result: self.on_inspection_cancel_response(result, goal))
            self.get_logger().info('Cancelling inspection: candidate already reported')
        except Exception as e:
            self.inspection_canceling = False
            self.cancel_sent_at = None
            self.get_logger().warning(f'Inspection cancellation unavailable: {e}')

    def on_inspection_cancel_response(self, future, goal):
        if (self.current_goal is not goal or self.inspection_cancel_goal is not goal
                or self.nav_state != 'moving' or self.state in ('finishing', 'finished')):
            return
        try:
            if not future.result().goals_canceling:
                self.inspection_canceling = False
                self.cancel_sent_at = None
                self.get_logger().warning('Inspection cancellation rejected; waiting for navigation')
        except Exception as e:
            self.inspection_canceling = False
            self.cancel_sent_at = None
            self.get_logger().warning(f'Inspection cancellation response unavailable: {e}')

    def check_stuck(self):
        if self.nav_state != 'moving' or self.stuck_canceling:
            return
        now = self.seconds_since_t0()
        if self.inspection_canceling and now - self.goal_start_time <= self.p['goal_timeout_s']:
            return
        pose = self.odom_pose()

        if pose is None or self.last_motion_pose is None:
            self.last_motion_pose = pose
            self.last_motion_time = now
        else:
            x1, y1, yaw1 = pose
            x2, y2, yaw2 = self.last_motion_pose
            dist = math.hypot(x1 - x2, y1 - y2)
            spin = angle_difference(yaw1, yaw2)
            if dist > self.p['stuck_distance_m'] or spin > self.p['stuck_angle_rad']:
                self.last_motion_pose = pose
                self.last_motion_time = now

        reason = None
        timeout = self.p['goal_timeout_s']
        if self.extra.active:
            timeout = min(timeout, self.p['extra_goal_timeout_s'])
        if now - self.goal_start_time > timeout:
            reason = 'viaje demasiado largo'
        elif now - self.last_motion_time > self.p['stuck_time_s']:
            reason = 'robot quieto'
        elif self.state != 'returning' and self.recoveries >= self.p['max_recoveries']:
            reason = f'Nav2 lleva {self.recoveries} recuperaciones'
        if reason is not None:
            self.cancel_current(reason)

    def check_goal_near_box(self):
        if self.state != 'exploring':
            return
        elif self.nav_state != 'moving':
            return
        elif self.current_point is None:
            return
        elif self.stuck_canceling:
            return
        x = self.current_point.pose.position.x
        y = self.current_point.pose.position.y
        if near_box(x, y, self.current_box_zones()):
            self.current_goal.cancel_goal_async()
            self.get_logger().info(f'Cancelo el viaje: el punto del plan ({x:.2f},{y:.2f}) '
                                    f'quedó cerca de una caja')
            self.current_point = None

    def check_goal_blocked(self):
        if self.state != 'exploring':
            return
        elif self.nav_state != 'moving':
            return
        elif self.current_point is None or self.current_in_place:
            return
        elif self.stuck_canceling:
            return
        x = self.current_point.pose.position.x
        y = self.current_point.pose.position.y
        if self.is_blocked(x, y):
            self.current_goal.cancel_goal_async()
            self.get_logger().info(f'Cancelo el viaje: el punto del plan ({x:.2f},{y:.2f}) '
                                f'quedó ocupado en el costmap')
            self.current_point = None

    def handle_on_goal_response(self, future, seq=None):
        goal_handle = future.result()
        if self.state in ('stopping', 'finishing', 'finished') or (seq is not None and seq != self.goal_seq):
            # Late answer to a goal mission already gave up (or the run is ending): cancel it.
            if goal_handle.accepted:
                goal_handle.cancel_goal_async()
            return
        if not goal_handle.accepted:
            self.extra.result('rejected')
            self.goal_rejected = True
            self.nav_state = 'failed'
            self.get_logger().warn('Fallo el objetivo: Nav2 lo rechazó')
            return
        self.extra.accepted()
        self.accept_timeouts = 0
        if not self.goal_ever_accepted:
            self.still_from = self.wall_now()
        self.goal_ever_accepted = True
        self.current_goal = goal_handle
        self.nav_state = 'moving'
        goal_handle.get_result_async().add_done_callback(lambda f, s=seq: self.on_goal_result(f, s))

    def on_goal_response(self, future, seq=None):
        try:
            self.handle_on_goal_response(future, seq)
        except Exception:
            self.nav_state = 'failed'
            if self.state not in ('stopping', 'finishing', 'finished'):
                self.change_state('stopping')
            self.get_logger().error(f'Error en on_goal_response:\n{traceback.format_exc()}', throttle_duration_sec=5.0)

    def handle_on_goal_result(self, future, seq=None):
        if seq is not None and seq != self.goal_seq:
            return      # result of a goal mission already gave up
        self.inspection_canceling = False
        self.cancel_sent_at = None
        status = future.result().status
        outcome = ('arrived' if status == GoalStatus.STATUS_SUCCEEDED else
                   'cancelled' if status == GoalStatus.STATUS_CANCELED else 'failed')
        self.extra.result(outcome)
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.nav_state = 'arrived'
            self.get_logger().info(f'Status succeded (recuperaciones: {self.recoveries})')
        else:
            self.nav_state = 'failed'
            self.get_logger().info(f'Status failed (recuperaciones: {self.recoveries})')

    def on_goal_result(self, future, seq=None):
        try:
            self.handle_on_goal_result(future, seq)
        except Exception:
            if seq is None or seq == self.goal_seq:
                self.nav_state = 'failed'
            self.get_logger().error(f'Error en on_goal_result:\n{traceback.format_exc()}', throttle_duration_sec=5.0)

    def handle_on_feedback(self, msg):
        if self.nav_state != 'moving':
            return
        self.recoveries = msg.feedback.number_of_recoveries

    def on_feedback(self, msg):
        try:
            self.handle_on_feedback(msg)
        except Exception:
            self.get_logger().error(f'Error en on_feedback:\n{traceback.format_exc()}',
                                    throttle_duration_sec=5.0)

    def change_state(self, new_state):
        self.get_logger().info(f'Estado: <{self.state}> -> <{new_state}>')
        self.state = new_state
        if new_state == 'stopping':
            self.stop_since = self.seconds_since_t0()
            self.stop_confirmed_at = None
            self.pub_stop.publish(Bool(data=True))
        if new_state != 'exploring':
            self.extra.invalidate()

    def step_logic(self):
        if self.state != 'exploring' or not self.p['extra_search_enabled']:
            self.extra.poll_job(accept=False)
        if self.state == 'waiting_for_map':
            if self.t0 is not None:
                self.change_state('exploring')
            return
        if self.state == 'finished':
            return
        if self.state == 'finishing':
            if not self.stop_is_confirmed():
                self.change_state('stopping')
                return
            self.pub_done.publish(Empty())
            self.change_state('finished')
            return
        if self.state == 'stopping':
            self.wait_for_stop()
            return
        if self.nav_state == 'sending':
            limit = self.p['goal_accept_timeout_s']
            valid = (isinstance(limit, (int, float)) and not isinstance(limit, bool)
                     and math.isfinite(limit) and 0 < limit < 60)
            if (not valid or self.goal_start_time is None
                    or self.seconds_since_t0() - self.goal_start_time >= limit):
                self.accept_timeouts += 1
                if not valid or self.accept_timeouts >= self.p['goal_accept_max_timeouts']:
                    self.get_logger().error('Navigation acceptance timed out or has invalid timing; stopping')
                    self.change_state('stopping')
                    return
                # One slow answer does not end the run: give this goal up (a late acceptance
                # is cancelled in handle_on_goal_response) and go on.
                self.get_logger().warning(f'Navigation acceptance timed out '
                                          f'({self.accept_timeouts} in a row): trip failed')
                self.goal_seq += 1
                self.goal_rejected = True   # Nav2 never took it: the plan point stays pending
                self.nav_state = 'failed'
        if not self.start_locked:
            pose = self.current_pose()
            if pose is not None:
                if self.start_pose is None:
                    x, y, yaw = pose
                    self.get_logger().info(f'Pose de inicio (provisoria): x={x:.2f} y={y:.2f} yaw={yaw:.2f}')
                self.start_pose = pose
        if self.seconds_since_t0() >= self.p['finish_time_s']:
            self.get_logger().info('Se termina el tiempo de la corrida')
            self.change_state('stopping')
            return
        if not start_ready(self.localization_ready, self.seconds_since_t0(), self.p['localization_wait_max_s']):
            return
        if not self.localization_ready and not self.start_timeout_logged:
            self.get_logger().warning('Localización no avisó que está lista: arranco igual')
            self.start_timeout_logged = True
        if self.state == 'exploring':
            self.extra.tick()
        if self.p['extra_search_enabled']:
            self.update_search_history()
        # Después del arranque: sin localización lista, una candidata podría quedar mal ubicada.
        self.check_reports(self.p['min_observations'], self.p['max_deviation_m'])
        self.check_pending_cancel()
        self.check_still()
        self.cancel_reported_inspection()
        self.check_stuck()
        self.check_goal_near_box()
        self.check_goal_blocked()
        if self.state == 'exploring':
            reason = self.reason_to_return()
            if reason is not None:
                self.get_logger().info(f'Vuelvo: {reason}')
                self.change_state('returning')
            else:
                self.explore()
        if self.state == 'returning':
            self.return_to_start()
        if self.state == 'checking':
            self.check_return()

    def step(self):
        try:
            self.step_logic()
        except Exception:
            self.get_logger().error(f'Error en step:\n{traceback.format_exc()}', throttle_duration_sec=5.0)

    def current_box_zones(self):
        if self.candidatas is None:
            return []
        return box_zones(self.candidatas, self.p['box_min_observations'],
                        self.p['box_clearance_m'], self.p['box_clearance_sigmas'])

    def explore(self):
        if self.nav_state == 'sending' or self.nav_state == 'moving':
            return
        if self.nav_state == 'arrived' or self.nav_state == 'failed':
            if self.goal_rejected and self.current_point is not None:
                # Nav2 refused the goal or never answered (e.g. bt_navigator coming back up):
                # not a visit.
                self.get_logger().info('Goal not taken by Nav2: the plan point stays pending')
                self.current_point = None
            if self.current_point is not None:
                x = self.current_point.pose.position.x
                y = self.current_point.pose.position.y
                yaw = yaw_from_quaternion(self.current_point.pose.orientation)
                self.visited.append((x, y, yaw))
            if self.current_inspection is not None:
                if self.inspection_is_reported():
                    self.current_inspection = None
                    self.look_until = None
                    self.inspection_spot = None
                elif self.nav_state == 'failed':
                    # Observaciones de ahora, no las del envío: reintenta solo si llegan más después.
                    cand_id, cand_x, cand_y, observations = self.current_inspection
                    for c in self.candidatas or []:
                        if c.id == cand_id:
                            observations = c.observaciones
                    attempts = forget_failed_inspection(self.inspected_points, self.failed_inspections,
                                                        cand_id, cand_x, cand_y, observations)
                    self.tried_views.setdefault(cand_id, []).append(self.inspection_spot)
                    self.get_logger().info(f'No llegué a mirar la candidata id={cand_id}: intento fallido '
                                           f'{attempts} de {self.p["visit_max_attempts"]}')
                    self.current_inspection = None
                else:
                    # Llegó: no alcanza con estar ahí. Se queda quieto hasta que fusión la confirme
                    # o pase inspection_wait_s (look_at_inspection).
                    self.look_until = self.seconds_since_t0() + self.p['inspection_wait_s']
                    self.get_logger().info(f'Llegué a mirar la candidata id={self.current_inspection[0]}: '
                                           f'espero hasta {self.p["inspection_wait_s"]:.1f} s a que se confirme')
            self.current_point = None
            self.nav_state = 'idle'
            self.settle_until = self.seconds_since_t0() + self.p['settle_s']
            if self.goal_rejected:
                # Wait a bit before the next goal instead of retrying at 10 Hz.
                self.settle_until = max(self.settle_until, self.seconds_since_t0() + 1.0)
                self.goal_rejected = False
        if self.look_until is not None and self.look_at_inspection():
            return
        if self.settle_until is not None and self.seconds_since_t0() < self.settle_until:
            return
        if self.inspect_doubtful():
            return
        if self.extra.phase == 'extra':
            self.extra.explore()
            return
        if self.plan is None:
            self.spin_if_idle()
            return
        zones = self.current_box_zones()
        point = first_pending(self.plan.poses, self.visited, self.p['same_point_distance_m'], 
                                self.p['same_point_angle_rad'], zones)
        first_free = first_pending(self.plan.poses, self.visited, self.p['same_point_distance_m'],
                                    self.p['same_point_angle_rad'], [])
        if first_free is not None and first_free is not point and first_free is not self.last_skipped:
            x = first_free.pose.position.x
            y = first_free.pose.position.y
            self.get_logger().info(f'Salteo el punto del plan ({x:.2f},{y:.2f}): cerca de una caja')
            self.last_skipped = first_free
        if point is None:
            self.spin_if_idle()
            return
        plan_x = point.pose.position.x
        plan_y = point.pose.position.y
        q = point.pose.orientation
        yaw = yaw_from_quaternion(q)
        x, y, in_place = goal_position(plan_x, plan_y, self.current_pose(), self.p['turn_in_place_m'])
        if not in_place and self.is_blocked(x, y):
            self.visited.append((plan_x, plan_y, yaw))
            self.get_logger().info(f'Salteo el punto del plan ({plan_x:.2f},{plan_y:.2f}): ocupado en el costmap')
            return
        goal = self.make_pose(x, y, yaw)
        sent = self.go_to(goal)
        if sent:
            self.current_point = point
            self.current_in_place = in_place
            if in_place:
                self.get_logger().info(f'Giro en el lugar: yaw={yaw:.2f} desde ({x:.2f},{y:.2f}), '
                                        f'punto del plan ({plan_x:.2f},{plan_y:.2f})')
            else:
                self.get_logger().info(f'Viaje al punto del plan: ({plan_x:.2f}, {plan_y:.2f}, {yaw})')
            self.idle_since = None

    def spin_if_idle(self):
        if self.idle_since is None:
            time = self.seconds_since_t0()
            self.idle_since = time
            return
        if self.seconds_since_t0() - self.idle_since < self.p['idle_spin_s']:
            return
        pose = self.current_pose()
        if pose is None:
            return
        x, y, yaw = pose
        if self.go_to(self.make_pose(x, y, yaw + self.p['idle_spin_rad'])):
            self.get_logger().info('Sin plan: giro en el lugar')
            self.idle_since = None


    def reason_to_return(self):
        if len(self.reported_points) >= self.cantidad_victimas:
            return 'todas reportadas'
        elif self.seconds_since_t0() >= self.p['return_time_s']:
            return 'se termina el tiempo'
        extra_reason = self.extra.reason()
        if extra_reason is not None:
            return extra_reason
        if self.extra.phase == 'extra':
            return None
        if (self.plan is not None and self.nav_state != 'sending' and self.nav_state != 'moving'
                # Una inspección en curso (viaje o espera) todavía puede confirmar una caja,
                # también si su llegada todavía no se procesó en explore().
                and self.current_inspection is None
                and first_pending(self.plan.poses, self.visited, self.p['same_point_distance_m'], 
                self.p['same_point_angle_rad'], self.current_box_zones()) is None and self.next_doubtful() is None):
            # Desde el último punto del plan también se puede ver una caja: antes de volver,
            # le da final_look_s a percepción y fusión.
            now = self.seconds_since_t0()
            if self.nothing_left_since is None:
                self.nothing_left_since = now
            if now - self.nothing_left_since >= self.p['final_look_s']:
                if self.p['extra_search_enabled']:
                    return self.extra.enter()
                return 'no quedan puntos ni dudosas'
            return None
        self.nothing_left_since = None
        return None

    def make_pose(self, x, y, yaw):
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.orientation.z = math.sin(yaw / 2)
        pose.pose.orientation.w = math.cos(yaw / 2)
        return pose

    def return_to_start(self):
        if self.nav_state == 'sending':
            return
        if self.nav_state == 'moving' and not self.return_sent:
            if not self.canceling and not self.inspection_canceling and not self.stuck_canceling:
                self.current_goal.cancel_goal_async()
                self.canceling = True
                self.cancel_sent_at = self.wall_now()
                self.get_logger().info("Cancelando objetivo...")
            return
        if not self.return_sent:
            if self.start_pose is None:
                self.get_logger().warning('No hay pose de inicio, termino sin volver')
                self.change_state('stopping')
                return
            else:
                (x, y, yaw), tree, is_leg = self.next_return_goal()
                sent = self.go_to(self.make_pose(x, y, yaw), tree)
                if sent:
                    self.return_sent = True
                    self.return_leg_active = is_leg
                    if is_leg:
                        self.return_legs += 1
                        self.get_logger().info(f'Vuelta por tramos: tramo {self.return_legs} hasta '
                                               f'({x:.2f},{y:.2f})')
                    else:
                        self.return_tries += 1
                    return
        if self.return_sent and self.return_leg_active and (self.nav_state == 'arrived' or self.nav_state == 'failed'):
            # Un tramo no se mide: se manda el siguiente (o la largada) en el próximo ciclo.
            self.get_logger().info(f'Tramo {self.return_legs} terminado: {self.nav_state}')
            self.failed_legs = self.failed_legs + 1 if self.nav_state == 'failed' else 0
            if self.failed_legs >= self.p['return_max_failed_legs']:
                # El próximo tramo saldría del mismo lugar al mismo punto: vuelve a ir directo.
                self.get_logger().warning(f'{self.failed_legs} tramos fallidos seguidos: voy directo a la largada')
            self.return_sent = False
            self.return_leg_active = False
            return
        if self.return_sent and (self.nav_state == 'arrived' or self.nav_state == 'failed'):
            if self.nav_state == 'failed' and not self.return_by_legs:
                # Si Nav2 no pudo, mandar lo mismo falla igual: el próximo intento va por tramos.
                self.return_by_legs = True
                self.get_logger().info('Nav2 no llegó a la largada: el próximo intento va por tramos')
            self.get_logger().info(f'Vuelta terminada: {self.nav_state}')
            self.check_since = None
            self.check_started = self.seconds_since_t0()
            self.change_state('checking')
            return

    def next_return_goal(self):
        """((x, y, yaw), árbol, es_tramo): la largada, o un tramo hacia ella.

        Va por tramos si el camino de vuelta mide más de return_direct_max_m (0 = nunca de
        entrada), o si Nav2 ya falló yendo directo. El camino se busca de nuevo antes de cada
        tramo, en el costmap global si llegó (route_grid): así un tramo fallido por un
        obstáculo que vio el LiDAR hace que el siguiente lo rodee. Cada tramo mide hasta
        return_leg_m, como mucho return_max_legs, y deja de usarlos tras
        return_max_failed_legs fallidos seguidos. El último trecho, y cualquier problema, van
        directo a la largada con el árbol preciso, como antes (vuelta.py explica por qué).
        """
        start = (self.start_pose, self.return_tree, False)
        pose = self.current_pose()
        if (pose is None or self.grid is None or self.return_legs >= self.p['return_max_legs']
                or self.failed_legs >= self.p['return_max_failed_legs']):
            return start
        try:
            grid, radius, threshold, source = self.route_grid()
            x0, y0, _ = self.start_pose
            steps = steps_to_start(grid, x0, y0, radius, threshold, self.p['return_snap_m'])
            if not self.return_by_legs and self.p['return_direct_max_m'] > 0:
                length = way_back_m(grid, steps, pose[0], pose[1], self.p['return_snap_m'])
                if length is not None and length > self.p['return_direct_max_m']:
                    self.return_by_legs = True
                    self.get_logger().info(f'La vuelta mide {length:.1f} m en {source}: voy por tramos')
            if not self.return_by_legs:
                return start
            zones = self.current_box_zones()
            leg = return_leg(grid, steps, pose[0], pose[1], self.p['return_leg_m'],
                             self.p['return_snap_m'], lambda x, y: self.is_blocked(x, y) or near_box(x, y, zones))
        except ValueError as e:
            self.get_logger().warning(f'No pude armar la vuelta por tramos: {e}. Voy directo')
            return start
        if leg is None:
            return start
        return (leg, '', True)

    def route_grid(self):
        """(grilla, radio, umbral, nombre) donde buscar la vuelta: el costmap global, o /map.

        El costmap global de Nav2 tiene lo que /map no: cajas y obstáculos que vio el LiDAR. Su
        inflación ya incluye el radio del robot, así que va con radio 0, y libre es lo que el
        planner usa: por debajo de costmap_blocked_value (99 = inscripto, 100 = ocupado).
        Sin costmap todavía, /map con el radio de las inspecciones, como antes.
        """
        if self.costmap is not None:
            try:
                grid = Grid.from_occupancy_grid(self.costmap)
                return grid, 0.0, self.p['costmap_blocked_value'] - 1, 'el costmap'
            except ValueError as e:
                self.get_logger().warning(f'No uso el costmap para la vuelta: {e}', throttle_duration_sec=5.0)
        return self.grid, self.p['visit_transit_radius_m'], self.p['visit_free_threshold'], '/map'

    def check_return(self):
        now = self.seconds_since_t0()
        if now - self.check_started > self.p['return_check_max_s']:
            # Sin pose o sin quedarse quieto a tiempo: mejor /done que quedarse 60 s quieto.
            self.get_logger().warning('No pude medir la vuelta a tiempo: termino')
            self.change_state('stopping')
            return
        odom = self.odom_pose()
        if odom is None:
            return
        if (self.check_since is None or 
            math.hypot(odom[0] - self.check_pose[0], odom[1] - self.check_pose[1]) > self.p['return_still_m']):
            self.check_since = now
            self.check_pose = odom
            return
        if now - self.check_since < self.p['return_still_s']:
            return
        pose = self.current_pose()
        if pose is None:
            return
        x, y, _ = pose
        x0, y0, _ = self.start_pose
        error = math.hypot(x - x0, y - y0)
        self.get_logger().info(f'Regreso {self.return_tries}: a {error:.3f} m de la largada')
        verdict = return_verdict(error, self.return_tries, self.p['return_tolerance_m'],
                                self.p['return_max_tries'])
        if verdict == 'ok':
            self.change_state('stopping')
        elif verdict == 'retry':
            self.get_logger().info('Lejos de la largada: vuelvo a intentar')
            self.return_sent = False
            self.change_state('returning')
        else:
            self.get_logger().warning('Termino lejos de la largada')
            self.change_state('stopping')

    def next_doubtful(self):
        if self.candidatas is None:
            return None
        pose = self.current_pose()
        if pose is None:
            return None
        return pick_doubtful(self.candidatas, self.reported_ids, self.reported_points, self.inspected_points, 
                            self.p['visit_min_observations'], self.p['min_report_distance_m'], pose[0], pose[1], 
                            self.failed_inspections, self.p['visit_max_attempts'])

    def inspect_doubtful(self):
        c = self.next_doubtful()
        if c is None:
            return False
        x_robot, y_robot, _ = self.current_pose()
        if self.grid is None:
            return False
        try:
            reachable = self.grid.reachable_from(x_robot, y_robot, self.p['visit_transit_radius_m'],
                                                self.p['visit_free_threshold'])
        except ValueError as e:
            # Counted as a failed try, like "no place to look from": otherwise this candidate
            # stays pending and blocks the return until return_time_s.
            _, attempts = self.failed_inspections.get(c.id, (0, 0))
            self.failed_inspections[c.id] = (c.observaciones, attempts + 1)
            self.get_logger().info(f'No calculo zona alcanzable: {e} (intento {attempts + 1} de '
                                    f'{self.p["visit_max_attempts"]})', throttle_duration_sec=5.0)
            return False
        view = viewpoint_for(self.mapa, c.centro.x, c.centro.y, x_robot, y_robot, self.p['visit_distance_m'], 
                            self.p['visit_clearance_m'], self.p['visit_angles'], self.p['visit_min_shift_m'],
                            self.p['visit_box_margin_m'], self.grid, reachable, self.is_blocked,
                            self.tried_views.get(c.id, ()))
        if view is None:
            _, attempts = self.failed_inspections.get(c.id, (0, 0))
            self.failed_inspections[c.id] = (c.observaciones, attempts + 1)
            self.get_logger().info(f'Candidata id: {c.id} x={c.centro.x:.2f} y={c.centro.y:.2f} '
                                    f'obs={c.observaciones}. Sin lugar para mirarla (intento {attempts + 1} '
                                    f'de {self.p["visit_max_attempts"]})')
            return False
        else:
            x, y, yaw = view
            if self.extra.phase == 'extra' and not self.extra.affordable(Pose(x, y, yaw)):
                return False
            if self.go_to(self.make_pose(x, y, yaw)):
                self.inspected_points.append((c.centro.x, c.centro.y))
                self.current_inspection = (c.id, c.centro.x, c.centro.y, c.observaciones)
                self.inspection_spot = (x, y)
                self.get_logger().info(f'Voy a mirar la candidata id={c.id} desde ({x:.2f},{y:.2f})')
                self.idle_since = None
                return True
        return False

    def look_at_inspection(self):
        cand_id, cand_x, cand_y, _ = self.current_inspection
        outcome = inspection_outcome(self.candidatas, cand_id, self.reported_ids, self.reported_points,
                                    self.p['min_report_distance_m'], self.seconds_since_t0(),
                                    self.look_until)
        if outcome == 'looking':
            return True
        if outcome == 'reported':
            self.get_logger().info(f'La inspección confirmó la candidata id={cand_id}')
        else:
            attempts = unhelpful_inspection(self.inspected_points, self.failed_inspections,
                                            cand_id, cand_x, cand_y)
            if outcome == 'unhelpful':
                # La próxima vista no puede ser esta ni otra cerca (A -> B -> A gastaba un intento).
                self.tried_views.setdefault(cand_id, []).append(self.inspection_spot)
            if outcome == 'gone':
                self.get_logger().info(f'La candidata id={cand_id} desapareció mientras la miraba')
            elif attempts < self.p['visit_max_attempts']:
                self.get_logger().info(f'La inspección no confirmó la candidata id={cand_id} (intento '
                                        f'{attempts} de {self.p["visit_max_attempts"]}): la miro desde otro lugar')
            else:
                self.get_logger().info(f'La inspección no confirmó la candidata id={cand_id} (intento '
                                        f'{attempts} de {self.p["visit_max_attempts"]}): no la miro más')
        self.current_inspection = None
        self.look_until = None
        return False

    def on_stop_confirmed(self, msg):
        self.stop_confirmed_at = (self.seconds_since_t0()
                                  if msg.data and self.state in ('stopping', 'finishing') else None)

    def stop_is_confirmed(self):
        return (self.stop_confirmed_at is not None
                and 0 <= self.seconds_since_t0() - self.stop_confirmed_at
                <= self.p['stop_confirm_max_age_s'])

    def wait_for_stop(self):
        now = self.seconds_since_t0()
        self.pub_stop.publish(Bool(data=True))
        if self.nav_state == 'moving' and not self.stop_canceling:
            self.stop_canceling = True
            try:
                self.current_goal.cancel_goal_async()
                self.get_logger().info('Paro: cancelo el viaje')
            except Exception as e:
                self.get_logger().warning(f'Navigation cancellation unavailable: {e}')
        if not self.stop_is_confirmed():
            if now >= self.p['done_limit_s']:
                self.get_logger().warning('Waiting for confirmed stop; /done withheld', throttle_duration_sec=5.0)
            return
        if self.stop_since is None:
            self.stop_since = now
        if (now - self.stop_since) < self.p['stop_wait_s']:
            return
        self.terminar()

    def al_recibir_plan(self, plan):
        self.plan = plan

    def al_recibir_candidatas(self, msg):
        self.candidatas = msg.candidatas

    def reportar_victima(self, x, y):
        """Publica un reporte. Cada llamada gasta un reporte (§7.2)."""
        self.pub_victimas.publish(armar_reporte(x, y, self.get_clock().now().to_msg()))

    def terminar(self):
        self.check_reports(self.p['min_observations_final'], self.p['max_deviation_final_m'])
        self.change_state('finishing')


    def destroy_node(self):
        self.extra.close()
        return super().destroy_node()


def main(args=None):
    correr(Mision, args)
