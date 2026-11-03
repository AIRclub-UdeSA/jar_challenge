"""
Localization node: the scan matcher (scan_matcher.py) against the judge's /map.

Publishes map -> odom on /tf (the only publisher of it), the
pose with its covariance on /equipo/pose, and the scan points the map does not
explain (boxes, decoys) on /equipo/no_mapeado, in the map frame.

For the world model (world_model_node.py, docs/WORLD_MODEL.md §3), and only while
something subscribes:
- /equipo/scan_localizado: each matched scan, every valid return deskewed and in the map
  frame at the matched pose;
- /equipo/pose_continua: map -> base at every odometry stamp, the last scan pose moved by
  odometry and the gyro (`Localizer.pose_at`). Composing map -> odom with wheel odometry
  instead shows heading spikes when the wheels slip.

Each scan is matched at its own stamp, with odometry interpolated there: on the robot
the scan arrives ~136 ms after its stamp, so the node waits for odometry to cover it.
"""
import array
import math

from geometry_msgs.msg import Point, PoseStamped, PoseWithCovarianceStamped, TransformStamped
from nav_msgs.msg import OccupancyGrid, Odometry
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import Imu, LaserScan, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from tf2_ros import Buffer, TransformBroadcaster, TransformException, TransformListener

from equipo_jar.keep_alive import spin
from equipo_jar.log_lines import guarded
from equipo_jar.scan_matcher import DEFAULTS, Localizer, occupancy_from_grid
from equipo_jar_msgs.msg import ScanLocalizado


def stamp_s(header):
    return header.stamp.sec + header.stamp.nanosec * 1e-9


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y ** 2 + q.z ** 2))


def rotation(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class LocalizerNode(Node):

    def __init__(self):
        super().__init__('localizer')
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('transform_tolerance', 0.3)  # s, map -> odom is post-dated
        self.declare_parameter('publish_unexplained', True)
        for key, value in DEFAULTS.items():
            self.declare_parameter(key, value)
        g = lambda k: self.get_parameter(k).value  # noqa: E731
        self.base, self.odom_frame = g('base_frame'), g('odom_frame')
        self.map_frame = g('map_frame')
        self.tolerance = g('transform_tolerance')
        self.publish_unexplained = g('publish_unexplained')
        self.loc = Localizer(**{k: type(v)(g(k)) for k, v in DEFAULTS.items()})
        self.tf = Buffer()
        TransformListener(self.tf, self)
        self.broadcaster = TransformBroadcaster(self)
        self.mount_frame = None
        self.imu_z = {}  # frame -> base z axis expressed in the IMU frame
        self.pending = []
        self.last_event = None
        self.logged_t = None            # the last line for the run's log
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(OccupancyGrid, '/map', self.on_map, latched)
        self.create_subscription(LaserScan, '/scan', self.on_scan, qos_profile_sensor_data)
        self.create_subscription(Odometry, '/odom', self.on_odom, 50)
        self.create_subscription(Imu, '/imu/data', self.on_imu, 10)
        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/equipo/pose', 10)
        self.cloud_pub = self.create_publisher(PointCloud2, '/equipo/no_mapeado', 5)
        self.scan_pub = self.create_publisher(ScanLocalizado, '/equipo/scan_localizado', 10)
        self.track_pub = self.create_publisher(PoseStamped, '/equipo/pose_continua', 50)

    def on_map(self, msg):
        info = msg.info
        self.loc.set_map(occupancy_from_grid(msg.data, info.width, info.height),
                         info.resolution, (info.origin.position.x, info.origin.position.y))
        self.loc.pose = None  # a new map: localize again from the startup search
        self.get_logger().info(f'map {info.width}x{info.height} @ {info.resolution} m')

    def on_odom(self, msg):
        p = msg.pose.pose
        self.loc.add_odom(stamp_s(msg.header), (p.position.x, p.position.y, yaw_of(p.orientation)))
        self.process()
        self.broadcast()
        if self.track_pub.get_subscription_count():
            self.publish_track(msg.header)

    def on_imu(self, msg):
        frame = msg.header.frame_id
        if frame not in self.imu_z:
            try:  # the IMU is mounted upside down on the X3: rotate the rate into the base
                tf = self.tf.lookup_transform(self.base, frame, Time())
            except TransformException:
                return
            self.imu_z[frame] = rotation(tf.transform.rotation)[2]
        w = msg.angular_velocity
        self.loc.add_gyro(stamp_s(msg.header), float(self.imu_z[frame] @ (w.x, w.y, w.z)))

    def on_scan(self, msg):
        if self.mount_frame != msg.header.frame_id:
            try:
                tf = self.tf.lookup_transform(self.base, msg.header.frame_id, Time())
            except TransformException:
                return
            R, t = rotation(tf.transform.rotation), tf.transform.translation
            self.loc.set_laser_mount(R[:2, :2], (t.x, t.y))
            self.mount_frame = msg.header.frame_id
        self.pending = (self.pending + [msg])[-2:]
        self.process()

    def process(self):
        """Match pending scans once odometry covers their stamp (or they are stale)."""
        if not self.loc.odom:
            return
        latest = self.loc.odom[-1][0]
        while self.pending:
            msg = self.pending[0]
            t = stamp_s(msg.header)
            if t > latest and latest - t > -0.3:
                return  # wait for odometry, up to 0.3 s
            self.pending.pop(0)
            try:
                res = self.loc.process_scan(t, msg.ranges, msg.angle_min, msg.angle_increment,
                                            msg.range_min, msg.range_max, msg.time_increment)
            except Exception as e:  # never die mid-run: skip the scan
                self.get_logger().error(f'scan match failed: {e!r}')
                continue
            if res is not None:
                self.publish(msg.header, res)
                if self.scan_pub.get_subscription_count():
                    self.publish_scan(msg, t, res)

    @guarded
    def log_match(self, res):
        """A line a second for the run's log (docs/LOGGING.md): how the matches are going."""
        now = self.get_clock().now()
        if self.logged_t is not None and (now - self.logged_t).nanoseconds < 1e9:
            return
        self.logged_t = now
        x, y, yaw = res['pose']
        self.get_logger().info(
            f"match: {res['event']} pose {x:.3f} {y:.3f} {yaw:.3f} "
            f"inliers {res['inlier_ratio']:.2f} of {res['points']} "
            f"iterations {res['iterations']} wall_offset {res['wall_offset']:.3f}")

    def publish(self, header, res):
        if res['event'] != 'track' or res['event'] != self.last_event:
            x, y, yaw = res['pose']
            self.get_logger().info(
                f"{res['event']}: ({x:.3f}, {y:.3f}, {math.degrees(yaw):.1f} deg), "
                f"{res['inlier_ratio']:.0%} of {res['points']} points on walls")
        self.last_event = res['event']
        self.log_match(res)
        out = PoseWithCovarianceStamped()
        out.header.stamp, out.header.frame_id = header.stamp, self.map_frame
        x, y, yaw = res['pose']
        out.pose.pose.position.x, out.pose.pose.position.y = float(x), float(y)
        out.pose.pose.orientation.z = math.sin(yaw / 2)
        out.pose.pose.orientation.w = math.cos(yaw / 2)
        cov = np.zeros((6, 6))
        cov[np.ix_([0, 1, 5], [0, 1, 5])] = res['cov']
        out.pose.covariance = cov.ravel().tolist()
        self.pose_pub.publish(out)
        if self.publish_unexplained and len(res['unexplained']):
            pts = np.column_stack([res['unexplained'], np.zeros(len(res['unexplained']))])
            self.cloud_pub.publish(point_cloud2.create_cloud_xyz32(
                Header(stamp=header.stamp, frame_id=self.map_frame), pts.astype(np.float32)))

    def publish_scan(self, msg, t, res):
        """The matched scan, at full resolution and in the map frame, for the world model."""
        try:
            pts = self.loc.scan_points(t, msg.ranges, msg.angle_min, msg.angle_increment,
                                       msg.range_min, msg.range_max, msg.time_increment,
                                       limit=0)
            x, y, yaw = res['pose']
            c, s = math.cos(yaw), math.sin(yaw)
            R = np.array([[c, -s], [s, c]])
            sensor = np.array([x, y]) + R @ self.loc.mount[1]
            pts = (pts @ R.T + (x, y)).astype(np.float32)
            out = ScanLocalizado()
            out.header.stamp, out.header.frame_id = msg.header.stamp, self.map_frame
            out.sensor = Point(x=float(sensor[0]), y=float(sensor[1]), z=0.0)
            out.sigma = math.sqrt(max(np.linalg.eigvalsh(res['cov'][:2, :2]).max(), 0.0))
            out.x = array.array('f', pts[:, 0].tobytes())
            out.y = array.array('f', pts[:, 1].tobytes())
            self.scan_pub.publish(out)
        except Exception as e:
            self.get_logger().error(f'localized scan failed: {e!r}')

    def publish_track(self, header):
        """map -> base at this odometry stamp, for poses between scans (camera images)."""
        pose = self.loc.pose_at(stamp_s(header))
        if pose is None:
            return
        out = PoseStamped()
        out.header.stamp, out.header.frame_id = header.stamp, self.map_frame
        out.pose.position.x, out.pose.position.y = float(pose[0]), float(pose[1])
        out.pose.orientation.z = math.sin(pose[2] / 2)
        out.pose.orientation.w = math.cos(pose[2] / 2)
        self.track_pub.publish(out)

    def broadcast(self):
        """
        map -> odom, stamped like AMCL does: from the data (latest odometry), post-dated
        by transform_tolerance, so lookups at the latest odom -> base succeed. Sent on
        every odometry message, so the stamp always moves forward.
        """
        if self.loc.map_to_odom is None or not self.loc.odom:
            return
        x, y, yaw = self.loc.map_to_odom
        tf = TransformStamped()
        t = self.loc.odom[-1][0] + self.tolerance
        tf.header.stamp.sec, tf.header.stamp.nanosec = int(t), int((t % 1) * 1e9)
        tf.header.frame_id, tf.child_frame_id = self.map_frame, self.odom_frame
        tf.transform.translation.x, tf.transform.translation.y = float(x), float(y)
        tf.transform.rotation.z, tf.transform.rotation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        self.broadcaster.sendTransform(tf)


def main():
    rclpy.init()
    node = LocalizerNode()
    spin(node)


if __name__ == '__main__':
    main()
