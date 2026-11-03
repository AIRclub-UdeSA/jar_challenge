"""Fusion node (`fusion`): groups detection's observations into candidates and publishes
them on /jar/candidatas.

ROS wiring only; the math is in candidate_fusion.py and is tested without ROS.

Input:
  /jar/observaciones    geometry_msgs/PoseWithCovarianceStamped, frame `map`, from `deteccion`.
                        position = one box-center estimate; covariance[0] = sigma^2 of it;
                        header.stamp = when it was seen (used to look up the robot pose).
  /tf                   map -> base_footprint at the observation time, for the baseline.

Output:
  /jar/candidatas       equipo_jar_msgs/Candidatas, frame `map`: the full list every time it
                        changes (throttled) and at least once per second. QoS reliable +
                        transient local, depth 1, so a node that starts late gets the last list.

Mission filters these on `observaciones` and on the largest eigenvalue of `covarianza_xy`,
so the count and the covariance must be honest; nothing here decides what gets reported.
"""
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time

import tf2_ros

from equipo_jar.candidate_fusion import DEFAULTS, CandidateFusion
from equipo_jar.comun import QOS_ESTADO, correr, parametros_en_vivo
from equipo_jar_msgs.msg import Candidata, Candidatas

PARAM_DEFAULTS = {
    **DEFAULTS,                            # association, merge, baseline, floor, purge (see candidate_fusion.py)
    #PARAM: robot_frame = base frame whose map pose is the "viewpoint". Support: a frame in
    #       /tf; base_footprint is the odom child on this robot.
    'robot_frame': 'base_footprint',
    #PARAM: publish_min_period_s = minimum time between "changed" publications. Support:
    #       > 0 and < 1 (the 1 Hz timer covers the rest). 0.2 keeps mission fresh without
    #       flooding it at the camera rate.
    'publish_min_period_s': 0.2,
}


class Fusion(Node):
    def __init__(self):
        super().__init__('fusion')
        self.p = parametros_en_vivo(self, PARAM_DEFAULTS)
        self.fusion = CandidateFusion(self.p)     # reads the live dict at call time
        self.t_last_pub = None

        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.create_subscription(
            PoseWithCovarianceStamped, '/jar/observaciones', self.on_observation, 10)
        self.pub_candidatas = self.create_publisher(Candidatas, '/jar/candidatas', QOS_ESTADO)
        # At least once per second, changed or not (interface contract).
        self.create_timer(1.0, self.publish_candidates)
        self.get_logger().info('Fusion ready: /jar/observaciones -> /jar/candidatas')

    def on_observation(self, msg):
        """One observation from detection. Any unexpected error is logged, never fatal."""
        try:
            self._process_observation(msg)
        except Exception as e:  # noqa: BLE001  (deliberate: a crashed node is never restarted)
            self.get_logger().error(f'observation skipped: {type(e).__name__}: {e}', throttle_duration_sec=5.0)

    def _process_observation(self, msg):
        stamp = Time.from_msg(msg.header.stamp)
        t = stamp.nanoseconds * 1e-9
        sigma = max(float(msg.pose.covariance[0]), 1e-6) ** 0.5
        z = (msg.pose.pose.position.x, msg.pose.pose.position.y)

        # Robot pose in map at the observation time. If TF is missing, the observation still
        # counts for the position but not for the baseline.
        robot_pose = None
        try:
            tr = self.tf_buffer.lookup_transform(
                'map', str(self.p['robot_frame']), stamp, timeout=Duration(seconds=0.05))
            robot_pose = (tr.transform.translation.x, tr.transform.translation.y)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f'no robot pose for baseline: {e}', throttle_duration_sec=5.0)

        try:
            c = self.fusion.observe(z, sigma, robot_pose, t)
        except ValueError as e:
            self.get_logger().warn(f'bad observation dropped: {e}')
            return
        self.fusion.purge(t)
        self.get_logger().debug(
            f'obs -> cand {c.id}: mu=({c.mu[0]:.2f}, {c.mu[1]:.2f}) n={c.n_obs}')

        # Publish on change, throttled.
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.fusion.changed and (
                self.t_last_pub is None or now - self.t_last_pub >= float(self.p['publish_min_period_s'])):
            self.publish_candidates()

    def publish_candidates(self):
        now = self.get_clock().now()
        msg = Candidatas()
        msg.header.frame_id = 'map'
        msg.header.stamp = now.to_msg()
        for r in self.fusion.report():
            c = Candidata()
            c.id = int(r['id'])
            c.centro.x = r['x']
            c.centro.y = r['y']
            c.centro.z = 0.0
            c.covarianza_xy = r['cov_xy']
            c.observaciones = int(r['n_obs'])
            c.ultima_vez = Time(nanoseconds=int(r['t_last'] * 1e9)).to_msg()
            msg.candidatas.append(c)
        self.pub_candidatas.publish(msg)
        self.t_last_pub = now.nanoseconds * 1e-9


def main(args=None):
    correr(Fusion, args)
