"""
The `motion` node: follows the paths the mission sends (motion.py), the only publisher of
/cmd_vel when the mission is `world_model` (docs/MISSION.md, "Driving").

- /equipo/ruta (nav_msgs/Path, map frame): the path to follow, each pose's orientation the
  heading to hold there. A new path replaces the one being followed; an empty one stops.
- /equipo/ruta_lenta: the same, followed at most at v_lento and w_lento: a look sweep, camera
  on a box. Slower, the wheels slip less, so the pose each image is placed with is better, and
  each line of sight stays in view for more images.
- /equipo/movimiento (EstadoMovimiento): what it is doing with the last path.
- /equipo/pose_continua: the robot's pose; /scan: what it must not touch; /odom: the velocity
  it has (the base takes time to follow a command, and the stop starts from what it does).

Without a fresh pose or scan it stops: it can't know where it is or what is around it. A path
it fails to follow (an exception) is dropped: it stops and reports blocked, and the mission
plans again.

Returns inside the body (cuerpo_x, not the URDF's huella_x) are dropped: they can only be the
robot. The physical lidar sees a cable about 0.13 m from the head, which its driver removes from
/scan, but a ray at the edge of that mask can slip through. The real rear is at most 0.156 m
behind base_link (a wall that close in the robot's bag), the URDF's 0.181: dropped up to the
URDF's rear, a wall right behind the start pose was the robot, and backing up looked safe with
1-2 cm of real room. Kept, it is inside the safety chassis, and safe() lets no move take it
deeper.

A scan arrives late (a scan period and more after its stamp): each beam is placed in the map
frame with the pose at the time it was measured, stamp + (n - 1 - k) * time_increment (the
driver lists the rays against the sweep; sweep_reversed false: stamp + k * time_increment), and
brought into
the robot's frame at every step. Used in the frame of the moment it arrives, a scan taken while
turning at 1.5 rad/s put what is behind the robot up to 8 cm off, and the chassis' tail swept
into walls. The simulator's scan is a snapshot (time_increment 0); placing it with the pose half
a scan period later instead put walls 2 cm away from the turning tail, and it touched them.
"""
import collections
import math

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
import tf2_ros

from equipo_jar.keep_alive import spin
from equipo_jar.localizer import yaw_of
from equipo_jar.log_lines import guarded
from equipo_jar.motion import DEFAULTS, Tracker
from equipo_jar_msgs.msg import EstadoMovimiento

PARAMS = {
    'frecuencia': 20.0,         # Hz, control loop
    'pose_vieja_s': 0.5,        # s without a pose: stop
    'scan_viejo_s': 0.6,        # s without a scan: stop
    'odom_viejo_s': 0.3,        # s without odometry: assume the base follows the commands
    'base_frame': 'base_link',
    'sweep_reversed': True,     # ray k of n measured at stamp + (n - 1 - k) * time_increment
                                # (the robot's driver); false: stamp + k * time_increment
    'v_lento': 0.12,            # m/s, on /equipo/ruta_lenta
    'w_lento': 0.6,             # rad/s
    'cuerpo_x': (-0.156, 0.119),  # m, the body in base_link: returns inside it are the robot.
                                  # Rear from the robot's bag (base_link to a wall behind the
                                  # start pose), front the URDF's. huella_x stays the URDF's
}
ESTADOS = {'llego': EstadoMovimiento.LLEGO, 'siguiendo': EstadoMovimiento.SIGUIENDO,
           'bloqueado': EstadoMovimiento.BLOQUEADO}


def outside_body(bx, by, xs, half_y):
    """Which returns (base_link) are not the robot: outside the body xs (rear, front) x half_y."""
    return (bx < xs[0]) | (bx > xs[1]) | (np.abs(by) > half_y)


class Motion(Node):

    def __init__(self):
        super().__init__('motion')
        for key, value in {**PARAMS, **DEFAULTS}.items():
            self.declare_parameter(key, list(value) if isinstance(value, tuple) else value)
        g = lambda k: self.get_parameter(k).value  # noqa: E731
        self.p = {k: g(k) for k in PARAMS}
        self.tp = {k: tuple(g(k)) if isinstance(v, tuple) else type(v)(g(k))
                   for k, v in DEFAULTS.items()}
        self.tf = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf, self)
        self.mount = None               # laser (x, y, yaw) in base_link
        self.pose = self.pose_t = None
        self.poses = collections.deque(maxlen=200)   # (stamp s, x, y, yaw), map frame
        self.points = np.zeros((0, 2))               # the last scan, map frame
        self.scan_t = None
        self.vel = self.vel_t = None                 # (vx, vy, w), base frame
        self.tracker = None
        self.path_stamp = None
        self.state = EstadoMovimiento.QUIETO
        self.last_t = None
        self.last_cmd = (0.0, 0.0, 0.0)              # for the run's log
        self.closest = math.inf                      # m, nearest return since the last log line
        self.logged_t = None
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.state_pub = self.create_publisher(EstadoMovimiento, '/equipo/movimiento', 10)
        self.create_subscription(PoseStamped, '/equipo/pose_continua', self.on_pose, 20)
        self.create_subscription(LaserScan, '/scan', self.on_scan, qos_profile_sensor_data)
        self.create_subscription(Path, '/equipo/ruta', self.on_path, 10)
        self.create_subscription(Path, '/equipo/ruta_lenta',
                                 lambda m: self.on_path(m, slow=True), 10)
        self.create_subscription(Odometry, '/odom', self.on_odom, 10)
        self.create_timer(1.0 / self.p['frecuencia'], self.tick)

    def on_pose(self, msg):
        p = msg.pose
        self.pose = (p.position.x, p.position.y, yaw_of(p.orientation))
        self.pose_t = self.get_clock().now()
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if not self.poses or t > self.poses[-1][0]:
            self.poses.append((t,) + self.pose)

    def on_odom(self, msg):
        tw = msg.twist.twist
        self.vel = (tw.linear.x, tw.linear.y, tw.angular.z)
        self.vel_t = self.get_clock().now()

    def poses_at(self, t):
        """The poses (x, y, yaw arrays) at stamps t (s, array), interpolated; None if t isn't
        covered."""
        if len(self.poses) < 2 or not self.poses[0][0] <= t.min() <= t.max() <= \
                self.poses[-1][0] + 0.05:
            return None
        q = np.array(self.poses)
        return tuple(np.interp(t, q[:, 0], v) for v in (q[:, 1], q[:, 2], np.unwrap(q[:, 3])))

    def on_scan(self, msg):
        if self.mount is None:
            try:
                t = self.tf.lookup_transform(self.p['base_frame'], msg.header.frame_id, Time())
            except tf2_ros.TransformException:
                return
            q = t.transform.rotation
            self.mount = (t.transform.translation.x, t.transform.translation.y, yaw_of(q))
        r = np.asarray(msg.ranges, float)
        a = msg.angle_min + msg.angle_increment * np.arange(len(r))
        ok = np.flatnonzero(np.isfinite(r) & (r > msg.range_min) & (r < min(msg.range_max, 3.0)))
        lx, ly, lyaw = self.mount
        bx, by = lx + r[ok] * np.cos(lyaw + a[ok]), ly + r[ok] * np.sin(lyaw + a[ok])
        # A return inside the chassis, where it was when measured, is the robot itself: nothing
        # else can be there. Kept, it reads as an obstacle the robot is already on.
        outside = outside_body(bx, by, self.p['cuerpo_x'], self.tp['huella_y'])
        ok, bx, by = ok[outside], bx[outside], by[outside]
        self.note_closest(r[ok])
        # Into the map frame, each beam with the pose at the time it was measured. The robot's
        # driver lists the rays against the sweep: the last ray is measured first.
        k = len(r) - 1 - ok if self.p['sweep_reversed'] else ok
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9 + msg.time_increment * k
        pose = self.poses_at(t) if len(t) else (0.0, 0.0, 0.0)
        if pose is None:
            return
        x, y, th = pose
        c, s = np.cos(th), np.sin(th)
        self.points = np.column_stack([x + c * bx - s * by, y + s * bx + c * by])
        self.scan_t = self.get_clock().now()

    def points_now(self):
        """The last scan's points in the robot's frame now."""
        x, y, th = self.pose
        c, s = np.cos(th), np.sin(th)
        dx, dy = self.points[:, 0] - x, self.points[:, 1] - y
        return np.column_stack([c * dx + s * dy, -s * dx + c * dy])

    def on_path(self, msg, slow=False):
        self.path_stamp = msg.header.stamp
        if not msg.poses:
            self.tracker = None
            self.stop(EstadoMovimiento.QUIETO)
            return
        path = [(q.pose.position.x, q.pose.position.y, yaw_of(q.pose.orientation))
                for q in msg.poses]
        tp = self.tp
        if slow:
            tp = dict(tp, v_max=min(tp['v_max'], self.p['v_lento']),
                      w_max=min(tp['w_max'], self.p['w_lento']))
        self.tracker = Tracker(path, **tp)

    def stop(self, state):
        self.cmd_pub.publish(Twist())
        self.last_cmd = (0.0, 0.0, 0.0)
        self.state = state

    @guarded
    def note_closest(self, ranges):
        if len(ranges):
            self.closest = min(self.closest, float(ranges.min()))

    @guarded
    def log_motion(self, now):
        """A line a second for the run's log (docs/LOGGING.md), once there is a pose."""
        if self.pose is None or (self.logged_t is not None
                                 and (now - self.logged_t).nanoseconds < 1e9):
            return
        self.logged_t = now
        vx, vy, w = self.last_cmd
        self.get_logger().info(f'motion: cmd {vx:.2f} {vy:.2f} {w:.2f} state {self.state} '
                               f'closest {self.closest:.3f}')
        self.closest = math.inf

    def fresh(self, t, now, limit):
        return t is not None and (now - t).nanoseconds * 1e-9 <= limit

    def tick(self):
        now = self.get_clock().now()
        dt = 1.0 / self.p['frecuencia'] if self.last_t is None else \
            max((now - self.last_t).nanoseconds * 1e-9, 1e-3)
        self.last_t = now
        if self.tracker is not None:
            if not (self.fresh(self.pose_t, now, self.p['pose_vieja_s'])
                    and self.fresh(self.scan_t, now, self.p['scan_viejo_s'])):
                self.stop(EstadoMovimiento.BLOQUEADO)  # blind: don't move
            else:
                vel = self.vel if self.fresh(self.vel_t, now, self.p['odom_viejo_s']) else None
                try:
                    (vx, vy, w), state = self.tracker.step(self.pose, self.points_now(), dt,
                                                           vel)
                except Exception as e:  # never die mid-run: stop, and the mission plans again
                    self.get_logger().error(f'tracking failed, path dropped: {e!r}')
                    self.tracker = None
                    self.stop(EstadoMovimiento.BLOQUEADO)
                else:
                    cmd = Twist()
                    cmd.linear.x, cmd.linear.y, cmd.angular.z = vx, vy, w
                    self.cmd_pub.publish(cmd)
                    self.last_cmd = (vx, vy, w)
                    self.state = ESTADOS[state]
                    if state == 'llego':
                        self.tracker = None
        msg = EstadoMovimiento()
        msg.header.stamp = now.to_msg()
        if self.path_stamp is not None:
            msg.ruta = self.path_stamp
        msg.estado = self.state
        msg.restante = float(self.tracker.remaining()) if self.tracker else 0.0
        self.state_pub.publish(msg)
        self.log_motion(now)


def main():
    rclpy.init()
    node = Motion()
    try:
        spin(node)
    finally:
        if rclpy.ok():
            node.cmd_pub.publish(Twist())


if __name__ == '__main__':
    main()
