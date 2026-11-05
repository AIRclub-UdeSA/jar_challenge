"""
World model node (docs/WORLD_MODEL.md): owns a WorldModel (world_model.py) and publishes
what it infers. Its pose inputs come from the `localizer` node:
- /map                       the prior; a new map starts a new model
- /equipo/scan_localizado    every matched scan, deskewed and in the map frame
- /equipo/pose_continua      map -> base at odometry rate: the camera pose at each image stamp
- /cam_1/color/image_raw     with camera_info, and the camera mount from /tf
- /cam_1/depth/color/points  only while the camera sees red and the robot is slow (see
                             __init__): it is the heaviest stream on the robot, the organizers
                             warn it lags the robot over Wi-Fi, and DDS carries nothing without
                             a subscriber

Publishes (latched snapshots, so a late or restarted subscriber catches up):
- /equipo/objetos        equipo_jar_msgs/ObjetoArray: objects, centers, P(victim), viewpoints
- /equipo/busqueda       OccupancyGrid: where an undetected box could still be (0-98)
- /equipo/ocupacion      OccupancyGrid: the posterior map, P(occupied) 0-100
- /equipo/rojo           OccupancyGrid: P(red surface) where the camera has evidence (1-98)
- /equipo/mundo_markers  MarkerArray for RViz
"""
import array
import collections
import math
import time

from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import OccupancyGrid
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from sensor_msgs_py.point_cloud2 import read_points_numpy
from std_msgs.msg import ColorRGBA
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from equipo_jar.keep_alive import spin
from equipo_jar.localizer import rotation, stamp_s, yaw_of
from equipo_jar.world_model import DEFAULTS, WorldModel
from equipo_jar_msgs.msg import Objeto as ObjetoMsg, ObjetoArray, ScanLocalizado

RED, GREY, AMBER = (0.9, 0.15, 0.15), (0.55, 0.55, 0.55), (1.0, 0.65, 0.0)


class WorldModelNode(Node):

    def __init__(self):
        super().__init__('world_model')
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('victimas', 0)          # N, from the launch; 0 = unknown
        self.declare_parameter('mundo_hz', 2.0)        # updates per second
        self.declare_parameter('camara_hz', 10.0)      # images processed per second
        # Depth cloud on demand, so red goes to its measured depth (world_model.add_image):
        # subscribed while images show red and the robot drives slowly (a look sweep, or
        # turning in place: where colour gets decided, and where the latency the cloud adds
        # can't drive the robot into something), dropped profundidad_retener s after the last
        # such image, or as soon as the robot speeds up. Frames taken faster go without depth,
        # as when the cloud is late.
        self.declare_parameter('profundidad', True)
        self.declare_parameter('profundidad_rojo_min', 10)     # red pixel samples to want it
        self.declare_parameter('profundidad_retener', 3.0)     # s subscribed after the last one
        self.declare_parameter('profundidad_v_max', 0.15)      # m/s: a look sweep's (v_lento)
        self.declare_parameter('profundidad_espera', 1.5)      # s without any cloud: use none
        self.declare_parameter('profundidad_par', 0.05)        # s, image-cloud stamp match
        for key, value in DEFAULTS.items():
            if key != 'objetivos_n':
                self.declare_parameter(key, value)
        g = lambda k: self.get_parameter(k).value  # noqa: E731
        self.base, self.frame = g('base_frame'), g('map_frame')
        self.params = {k: type(v)(g(k)) for k, v in DEFAULTS.items() if k != 'objetivos_n'}
        self.params['objetivos_n'] = int(g('victimas'))
        self.camera_period = 1.0 / g('camara_hz')
        self.depth = {k: g('profundidad' + k) for k in
                      ('', '_rojo_min', '_retener', '_espera', '_par', '_v_max')}
        self.cloud_sub = None
        self.clouds = collections.deque(maxlen=8)  # (stamp, msg), newest last
        self.cloud_since = 0.0                      # when the cloud subscription started
        self.last_red = 0.0
        self.waiting = 0.0                          # when the pending image started waiting
        self.depth_used = self.depth_missing = 0
        self.depth_s, self.depth_subs = 0.0, 0     # time subscribed, subscriptions
        self.warned = False
        self.world = None
        self.poses = collections.deque(maxlen=200)  # (t, x, y, yaw) from /equipo/pose_continua
        self.image = None                           # the newest image, waiting for its pose
        self.last_image = 0.0
        self.camera = {}                            # frame -> (R, t) of the camera in the base
        self.K = None
        self.cycles = 0
        self.times = collections.defaultdict(list)
        self.errors = 0
        self.tf = Buffer()
        TransformListener(self.tf, self)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_objects = self.create_publisher(ObjetoArray, '/equipo/objetos', latched)
        self.pub_belief = self.create_publisher(OccupancyGrid, '/equipo/busqueda', latched)
        self.pub_occupancy = self.create_publisher(OccupancyGrid, '/equipo/ocupacion', latched)
        self.pub_color = self.create_publisher(OccupancyGrid, '/equipo/rojo', latched)
        self.pub_markers = self.create_publisher(MarkerArray, '/equipo/mundo_markers', 10)
        self.create_subscription(OccupancyGrid, '/map', self.on_map, latched)
        # Deep enough for the scans that arrive during a slow update (up to ~0.15 s).
        self.create_subscription(ScanLocalizado, '/equipo/scan_localizado', self.on_scan, 30)
        self.create_subscription(PoseStamped, '/equipo/pose_continua', self.on_pose, 50)
        self.create_subscription(Image, '/cam_1/color/image_raw', self.on_image,
                                 qos_profile_sensor_data)
        self.create_subscription(CameraInfo, '/cam_1/color/camera_info', self.on_camera_info,
                                 qos_profile_sensor_data)
        self.create_timer(1.0 / g('mundo_hz'), self.update)
        self.create_timer(0.5, self.drop_depth)
        self.create_timer(30.0, self.report)

    # Inputs.

    def on_map(self, msg):
        info = msg.info
        self.world = self.timed('map', WorldModel, msg.data, info.width, info.height,
                                info.resolution, (info.origin.position.x, info.origin.position.y),
                                **self.params)
        self.image = None

    def on_scan(self, msg):
        if self.world is None:
            return
        pts = np.column_stack([np.frombuffer(msg.x, np.float32),
                               np.frombuffer(msg.y, np.float32)]).astype(float)
        self.timed('scan', self.world.add_scan, np.array([msg.sensor.x, msg.sensor.y]), pts,
                   float(msg.sigma))

    def on_pose(self, msg):
        p = msg.pose
        self.poses.append((stamp_s(msg.header), p.position.x, p.position.y,
                           yaw_of(p.orientation)))
        self.try_image()

    def on_camera_info(self, msg):
        if self.K is None and msg.k[0] > 0:
            self.K = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])

    def on_image(self, msg):
        now = time.monotonic()
        if self.world is None or self.K is None or msg.encoding not in ('rgb8', 'bgr8') \
                or now - self.last_image < self.camera_period:
            return
        if self.image is not None and self.waiting and now - self.waiting < 0.3:
            return  # a red image is waiting for its cloud: don't replace it
        self.image, self.waiting = msg, 0.0
        self.try_image()

    def on_cloud(self, msg):
        self.clouds.append((stamp_s(msg.header), msg))
        self.try_image()

    # Depth on demand.

    def slow(self):
        """Whether the robot drives at most at look-sweep speed (pose stream, last 0.3 s)."""
        if len(self.poses) < 2:
            return True
        t1, x1, y1, _ = self.poses[-1]
        t0, x0, y0, _ = next((p for p in reversed(self.poses) if t1 - p[0] >= 0.3),
                             self.poses[0])
        return math.hypot(x1 - x0, y1 - y0) / max(t1 - t0, 1e-3) <= self.depth['_v_max']

    def want_depth(self):
        self.last_red = time.monotonic()
        if self.cloud_sub is None:
            self.cloud_sub = self.create_subscription(
                PointCloud2, '/cam_1/depth/color/points', self.on_cloud, qos_profile_sensor_data)
            self.cloud_since = time.monotonic()
            self.depth_subs += 1
            self.clouds.clear()

    def drop_depth(self):
        if self.cloud_sub is not None and (
                time.monotonic() - self.last_red > self.depth['_retener'] or not self.slow()):
            self.destroy_subscription(self.cloud_sub)
            self.depth_s += time.monotonic() - self.cloud_since
            self.cloud_sub = None
            self.clouds.clear()

    def cloud_for(self, t):
        """The cloud of the image at t: msg, 'wait' (it may still come) or None (it won't)."""
        tol = self.depth['_par']
        best = min(self.clouds, key=lambda c: abs(c[0] - t), default=None)
        if best is not None and abs(best[0] - t) <= tol:
            return best[1]
        if self.clouds and self.clouds[-1][0] > t + tol:
            return None                       # newer clouds came: this one was dropped
        if not self.clouds and time.monotonic() - self.cloud_since > self.depth['_espera']:
            return None                       # no cloud at all: depth isn't available
        return 'wait'

    def cloud_points(self, msg, R_base, base):
        """The cloud's points in the map frame and their colors, with the base pose (R_base,
        base) at the cloud's own stamp."""
        frame = msg.header.frame_id
        if frame not in self.camera:
            tf = self.tf.lookup_transform(self.base, frame, Time())
            tr = tf.transform.translation
            self.camera[frame] = (rotation(tf.transform.rotation), np.array([tr.x, tr.y, tr.z]))
        R_m, t_m = self.camera[frame]
        pts = read_points_numpy(msg, ('x', 'y', 'z', 'rgb'), skip_nans=True)
        rgb = pts[:, 3].astype(np.float32).view(np.uint32)
        colors = np.column_stack([(rgb >> 16) & 255, (rgb >> 8) & 255, rgb & 255])
        xyz = (pts[:, :3].astype(float) @ R_m.T + t_m) @ R_base.T + base
        return xyz, colors.astype(np.uint8)

    def pose_at(self, t):
        """map -> base at t, interpolated in the pose stream; None if it doesn't cover t."""
        if not self.poses or not self.poses[0][0] <= t <= self.poses[-1][0]:
            return None
        o = np.array(self.poses)
        yaw = np.unwrap(o[:, 3])
        return [np.interp(t, o[:, 0], col) for col in (o[:, 1], o[:, 2], yaw)]

    def try_image(self):
        """The pending image, once the pose stream covers its stamp (dropped if it never will)."""
        msg = self.image
        if msg is None:
            return
        t = stamp_s(msg.header)
        pose = self.pose_at(t)
        if pose is None:
            if self.poses and t < self.poses[0][0]:
                self.image = None
            return
        frame = msg.header.frame_id
        if frame not in self.camera:
            try:
                tf = self.tf.lookup_transform(self.base, frame, Time())
            except TransformException:
                self.image = None
                return
            tr = tf.transform.translation
            self.camera[frame] = (rotation(tf.transform.rotation), np.array([tr.x, tr.y, tr.z]))
        R_cam, t_cam = self.camera[frame]
        rgb = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)
        rgb = rgb[:, :msg.width * 3].reshape(msg.height, msg.width, 3)
        if msg.encoding == 'bgr8':
            rgb = rgb[..., ::-1]
        c, s = math.cos(pose[2]), math.sin(pose[2])
        R_base = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        base = np.array([pose[0], pose[1], 0.0])
        cam = base + R_base @ t_cam
        cloud = None
        if self.depth[''] and self.world.count_red(rgb) >= self.depth['_rojo_min'] \
                and self.slow():
            # Red in view: its color goes where the depth says, never to the first occupied
            # cell (the camera sees through gaps the grid has closed). Wait for the cloud.
            self.want_depth()
            found = self.cloud_for(t)
            if isinstance(found, str):
                self.image, self.waiting = msg, self.waiting or time.monotonic()
                return
            if found is None and self.clouds:
                # Its cloud was dropped: use the frame without depth. Skipping only frames
                # with red in them would bias the color evidence against red; depth is extra
                # evidence for the object color, never a requirement.
                self.depth_missing += 1
            elif found is None and not self.warned:
                self.warned = True
                self.get_logger().warn('no depth cloud: red goes to the first occupied cell')
            if found is not None:
                # Placed with the pose at its own stamp, up to profundidad_par from the
                # image's: at the image's, a cloud taken 30 ms apart in a 1.3 rad/s turn lands
                # 2-4 deg off, its red depth on the columns beside the red.
                pc = self.pose_at(stamp_s(found.header)) or pose
                cc, sc = math.cos(pc[2]), math.sin(pc[2])
                R_cloud = np.array([[cc, -sc, 0.0], [sc, cc, 0.0], [0.0, 0.0, 1.0]])
                try:
                    cloud = self.cloud_points(found, R_cloud, np.array([pc[0], pc[1], 0.0]))
                    self.depth_used += 1
                except (TransformException, ValueError):
                    self.image, self.waiting = None, 0.0
                    return
        self.image, self.waiting = None, 0.0
        self.timed('image', self.world.add_image, cam, R_base @ R_cam, self.K, rgb, cloud)
        self.last_image = time.monotonic()

    # Inference.

    def update(self):
        world = self.world
        if world is None:
            return
        if self.timed('update', world.update) is not None:
            self.timed('publish', self.publish, world, self.cycles % 2 == 0)
        self.cycles += 1

    def timed(self, key, fn, *args, **kwargs):
        t0 = time.perf_counter()
        try:
            out = fn(*args, **kwargs)
        except Exception as e:  # never die mid-run: log and keep going
            self.errors += 1
            if self.errors <= 5 or self.errors % 100 == 0:
                self.get_logger().error(f'{key} failed ({self.errors}): {e!r}')
            return None
        self.times[key].append(time.perf_counter() - t0)
        return out if out is not None else True

    def report(self):
        world = self.world
        if world is None:
            return
        objs = world.objects
        red = sum(o.p_objetivo >= world.p['reporte_p_min'] for o in objs)
        ms = ', '.join(f'{k} {1e3 * sum(v) / len(v):.1f} ms' for k, v in self.times.items() if v)
        depth = 'on' if self.cloud_sub is not None else 'off'
        depth_s = self.depth_s + (time.monotonic() - self.cloud_since
                                  if self.cloud_sub is not None else 0.0)
        self.get_logger().info(f'{len(objs)} objects, {red} red, '
                               f'{world.remaining():.0%} of the arena unseen; {ms}; depth '
                               f'{depth}, {depth_s:.0f} s in {self.depth_subs} subscriptions, '
                               f'{self.depth_used} red images with depth, '
                               f'{self.depth_missing} used without (cloud dropped)')
        self.times.clear()

    # Messages.

    def publish(self, world, grids):
        stamp = self.get_clock().now().to_msg()
        b = world.belief()
        out = ObjetoArray()
        out.header.stamp, out.header.frame_id = stamp, self.frame
        out.busqueda_restante = world.remaining(b)
        for o in world.objects:
            m = ObjetoMsg()
            m.id = int(o.id)
            m.p_objetivo, m.p_rojo = float(o.p_objetivo), float(o.p_rojo)
            m.centro = Point(x=float(o.centro[0]), y=float(o.centro[1]), z=0.0)
            m.cov_centro = [float(o.cov[0, 0]), float(o.cov[0, 1]), float(o.cov[1, 1])]
            m.lado_a, m.lado_b, m.yaw = float(o.lados[0]), float(o.lados[1]), float(o.yaw)
            m.forma_caja = float(o.forma)
            m.scans = int(o.scans)
            m.valor_mirada = float(4 * o.p_objetivo * (1 - o.p_objetivo))
            m.miradores = [Point(x=float(x), y=float(y), z=0.0) for x, y in o.miradores]
            m.celdas = o.cells.astype(np.uint32).tolist()
            out.objetos.append(m)
        self.pub_objects.publish(out)
        self.pub_markers.publish(self.markers(world, out, stamp))
        if grids:
            self.pub_belief.publish(self.grid(world, world.belief_values(b), stamp))
            self.pub_occupancy.publish(self.grid(world, world.occupancy_values(), stamp))
            self.pub_color.publish(self.grid(world, world.color_values(), stamp))

    def grid(self, world, values, stamp):
        g = OccupancyGrid()
        g.header.stamp, g.header.frame_id = stamp, self.frame
        g.info.resolution = float(world.res)
        g.info.height, g.info.width = int(world.shape[0]), int(world.shape[1])
        g.info.origin.position.x = float(world.origin[0])
        g.info.origin.position.y = float(world.origin[1])
        g.info.origin.orientation.w = 1.0
        g.data = array.array('b', values.astype('int8').tobytes())
        return g

    def markers(self, world, objs, stamp):
        arr = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)
        p_min = world.p['reporte_p_min']

        def marker(ns, mid, kind):
            m = Marker()
            m.header.stamp, m.header.frame_id = stamp, self.frame
            m.ns, m.id, m.type, m.action = ns, mid, kind, Marker.ADD
            m.pose.orientation.w = 1.0
            return m

        for o in objs.objetos:
            rgb = RED if o.p_objetivo >= p_min else GREY if o.p_objetivo <= 1 - p_min else AMBER
            box = marker('caja', o.id, Marker.CUBE)
            box.pose.position.x, box.pose.position.y, box.pose.position.z = o.centro.x, o.centro.y, 0.15
            box.pose.orientation.z, box.pose.orientation.w = math.sin(o.yaw / 2), math.cos(o.yaw / 2)
            box.scale.x, box.scale.y, box.scale.z = max(o.lado_a, 0.02), max(o.lado_b, 0.02), 0.3
            box.color = ColorRGBA(r=rgb[0], g=rgb[1], b=rgb[2], a=0.75)
            arr.markers.append(box)
            sigma = math.sqrt(max(o.cov_centro[0], o.cov_centro[2]))
            disc = marker('sigma', o.id, Marker.CYLINDER)
            disc.pose.position.x, disc.pose.position.y, disc.pose.position.z = o.centro.x, o.centro.y, 0.005
            disc.scale.x = disc.scale.y = 4 * sigma
            disc.scale.z = 0.01
            disc.color = ColorRGBA(r=rgb[0], g=rgb[1], b=rgb[2], a=0.3)
            arr.markers.append(disc)
            text = marker('texto', o.id, Marker.TEXT_VIEW_FACING)
            text.pose.position.x, text.pose.position.y, text.pose.position.z = o.centro.x, o.centro.y, 0.5
            text.scale.z = 0.12
            text.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.95)
            text.text = f'#{o.id}  P {o.p_objetivo:.2f}  +-{sigma * 100:.0f} cm'
            arr.markers.append(text)
            if rgb is AMBER and o.miradores:
                views = marker('miradores', o.id, Marker.POINTS)
                views.scale.x = views.scale.y = 0.06
                views.color = ColorRGBA(r=0.2, g=0.8, b=0.9, a=0.9)
                views.points = list(o.miradores)
                arr.markers.append(views)
        status = marker('estado', 0, Marker.TEXT_VIEW_FACING)
        status.pose.position.x = float(world.cx[world.arena.ravel()].min())
        status.pose.position.y = float(world.cy[world.arena.ravel()].max()) + 0.3
        status.pose.position.z = 0.5
        status.scale.z = 0.18
        status.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.95)
        n_red = sum(o.p_objetivo >= p_min for o in objs.objetos)
        n = world.p['objetivos_n']
        status.text = (f'unseen {objs.busqueda_restante:.0%}   objects {len(objs.objetos)}   '
                       f'red {n_red}' + (f'/{n}' if n else ''))
        arr.markers.append(status)
        return arr


def main():
    rclpy.init()
    node = WorldModelNode()
    spin(node)


if __name__ == '__main__':
    main()
