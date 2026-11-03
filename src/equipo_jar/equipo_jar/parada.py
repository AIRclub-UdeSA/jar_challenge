"""Navigation's sole velocity output; terminal stop cannot be undone by Nav2."""
import math
import time

from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from std_msgs.msg import Bool, Empty

from equipo_jar.comun import QOS_ESTADO, QOS_MAPA, correr, parametros_en_vivo


class Parada(Node):
    def __init__(self):
        super().__init__('parada')
        self.p = parametros_en_vivo(self, {
            'command_max_age_s': 0.3, 'odom_max_age_s': 0.5,
            'still_time_s': 0.5, 'still_linear_mps': 0.01,
            'still_angular_rads': 0.02, 'confirm_timeout_s': 3.0,
            'confirm_max_s': 10.0,
        })
        self.has_map = False
        self.blocked = False
        self.blocked_at = None
        self.command = None
        self.command_at = None
        self.odom_at = None
        self.odom_stamp = None
        self.still_since = None
        self.moving = False
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_confirmed = self.create_publisher(Bool, '/jar/parada_confirmada', QOS_ESTADO)
        self.create_subscription(Twist, '/jar/cmd_vel_smoothed', self.on_command, 10)
        self.create_subscription(OccupancyGrid, '/map', self.on_map, QOS_MAPA)
        self.create_subscription(Bool, '/jar/parar', self.on_stop, QOS_ESTADO)
        self.create_subscription(Empty, '/done', self.on_done, 10)
        self.create_subscription(Odometry, '/odom', self.on_odom, 10)
        # Wall time also expires commands while the simulated clock is paused.
        self.create_timer(0.05, self.tick, clock=Clock(clock_type=ClockType.STEADY_TIME))

    def now(self):
        return time.monotonic()

    def valid_config(self):
        return all(isinstance(v, (int, float)) and not isinstance(v, bool)
                   and math.isfinite(v) and v > 0 for v in self.p.values())

    def on_map(self, _msg):
        if not self.has_map:
            self.command = None
            self.command_at = None
        self.has_map = True

    def on_command(self, msg):
        values = (msg.linear.x, msg.linear.y, msg.linear.z,
                  msg.angular.x, msg.angular.y, msg.angular.z)
        if self.has_map and not self.blocked and all(map(math.isfinite, values)):
            self.command, self.command_at = msg, self.now()
        else:
            self.command, self.command_at = None, None

    def on_stop(self, msg):
        if msg.data and not self.blocked:
            self.blocked = True
            self.blocked_at = self.now()
            self.command = None
            self.still_since = None
            self.odom_at = None
            self.moving = False
            self.pub_cmd.publish(Twist())
            self.pub_confirmed.publish(Bool(data=False))

    def on_done(self, _msg):
        self.on_stop(Bool(data=True))

    def on_odom(self, msg):
        if not self.blocked or not self.valid_config():
            return
        # Freshness is the arrival time, not the stamp: on the real robot /odom is stamped by
        # the robot's computer, whose clock may differ from this one. Stamps must only advance.
        now = self.now()
        stamp = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        velocity = msg.twist.twist
        values = (velocity.linear.x, velocity.linear.y, velocity.linear.z,
                  velocity.angular.x, velocity.angular.y, velocity.angular.z)
        if (stamp <= 0
                or (self.odom_stamp is not None and stamp <= self.odom_stamp)
                or not all(map(math.isfinite, values))):
            self.still_since = None
            self.moving = False
            return
        if self.odom_at is None or now - self.odom_at > self.p['odom_max_age_s']:
            self.still_since = None
        self.odom_at, self.odom_stamp = now, stamp
        self.moving = (math.hypot(*values[:2]) > self.p['still_linear_mps']
                       or abs(values[2]) > self.p['still_linear_mps']
                       or any(abs(v) > self.p['still_angular_rads'] for v in values[3:]))
        if self.moving:
            self.still_since = None
        elif self.still_since is None:
            self.still_since = now

    def tick(self):
        now = self.now()
        valid = self.valid_config()
        fresh = (valid and self.command_at is not None
                 and 0 <= now - self.command_at <= self.p['command_max_age_s'])
        self.pub_cmd.publish(self.command if self.has_map and not self.blocked and fresh else Twist())
        if (not valid or self.odom_at is None
                or not 0 <= now - self.odom_at <= self.p['odom_max_age_s']):
            self.still_since = None
            self.moving = False
        still = (self.still_since is not None
                 and now - self.still_since >= self.p['still_time_s'])
        # Fallback: zero has been published this long, so the robot has stopped (it stops by
        # itself after 0.5 s without /cmd_vel) even if /odom is missing, late or invalid.
        # Fresh odometry that still shows motion vetoes it, up to confirm_max_s.
        blocked_for = now - self.blocked_at if self.blocked_at is not None else None
        timed_out = (valid and blocked_for is not None
                     and blocked_for >= self.p['confirm_timeout_s']
                     and (not self.moving or blocked_for >= self.p['confirm_max_s']))
        confirmed = self.blocked and (still or timed_out)
        self.pub_confirmed.publish(Bool(data=confirmed))


def main(args=None):
    correr(Parada, args)
