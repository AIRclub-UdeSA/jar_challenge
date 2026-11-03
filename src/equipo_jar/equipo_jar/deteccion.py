"""Detection node (`deteccion`): red in the camera + LiDAR distance -> box center in `map`.

ROS wiring only; the math is in vision.py and is tested without ROS.

Inputs (QoS per the rules, docs/QOS.md):
  /cam_1/color/image_raw      Image           best effort   (320x240 @ 30 Hz on the robot)
  /cam_1/color/camera_info    CameraInfo      best effort   (intrinsics K)
  /scan                       LaserScan       best effort   (1080 rays @ ~7 Hz, primary range)
  /cam_1/depth/color/points   PointCloud2     best effort   (fallback range; unreliable on the robot;
                                                            subscribed only if use_cloud_fallback
                                                            or cloud_check, both off by default)
  /tf, /tf_static             camera, LiDAR and base frames; map -> odom from localization
  /jar/localizacion_lista     std_msgs/Bool   reliable + transient local (localization's "pose
                                              trusted" flag; frames taken while it is false are
                                              skipped at the start, and after turns if
                                              skip_heading_doubt)                vision.skip_for_localization

Output (team-internal topic, defined by the perception module):
  /jar/observaciones          geometry_msgs/PoseWithCovarianceStamped, frame `map`
                              one message per box seen in a frame; position = estimated box
                              center; covariance[0] = covariance[7] = sigma^2 of this single
                              observation; header.stamp = time of the LiDAR scan it came from.
  /jar/deteccion/debug        Image with mask and boxes (only if publish_debug)

Per synchronized (image, scan) pair:
  1. red mask + box-shaped contours                                vision.red_mask, find_blobs
  2. camera rays through the blob's left/right edges -> LiDAR frame,
     at the scan time, through `odom` (robot motion between the two
     timestamps is compensated; with per_ray_time, at the time the
     middle ray of the blob was measured)                           tf2 lookup_transform_full,
                                                                    vision.mid_ray_delay
  3. the LiDAR beams in the central part of that span, grouped by
     range; the largest group is the box face                       vision.central_cluster
     (if none and use_cloud_fallback: depth cloud median, if a
     fresh cloud exists)                                            vision.robust_3d_point
  4. checks: not too far, blob width matches the range, area
     matches the range, the hit is not on a wall of /map            vision.width_is_consistent,
                                                                    area_is_consistent, map_range_along
  5. face -> robot frame -> half a side back = center               vision.center_from_face
  6. center -> `map` at the scan time, publish with sigma(distance) vision.observation_sigma
"""
import numpy as np
from geometry_msgs.msg import PointStamped, PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid
from numpy.lib import recfunctions as rfn
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, LaserScan, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Bool

import message_filters
import tf2_geometry_msgs  # noqa: F401  (registers PointStamped transforms in tf2)
import tf2_ros

from equipo_jar import vision
from equipo_jar.comun import QOS_ESTADO, QOS_MAPA, correr, parametros_en_vivo

# Live-tunable parameters (section `deteccion` of params.yaml). Read them from self.p at use
# time, never cache them: a change made during a run applies to the next frame.
PARAM_DEFAULTS = {
    # Segmentation (see vision.VisionParams for ranges and reasoning)
    'hsv_red_low_1': [0, 100, 60],
    'hsv_red_high_1': [8, 255, 255],
    'hsv_red_low_2': [170, 100, 60],
    'hsv_red_high_2': [180, 255, 255],
    'morphology_kernel_px': 3,
    'min_area_px': 80,
    'aspect_min': 0.4,
    'aspect_max': 1.8,
    'min_solidity': 0.85,
    # Geometry
    'box_side_m': 0.25,                  # sim box 0.25 x 0.25 x 0.30; real one TO BE CONFIRMED
    'max_area_factor': 3.0,              # 0 disables the area-vs-distance check
    # LiDAR (primary range): see vision.central_cluster
    'central_fraction': 0.6,
    'lidar_cluster_depth_m': 0.30,
    'lidar_min_points': 3,
    'min_cluster_coverage': 0.6,
    # Consistency checks against false positives (see vision.width_is_consistent,
    # vision.map_range_along). A false report costs a box (-100) and the fast bonus (-200).
    'width_ratio_min': 0.45,
    'width_ratio_max': 1.8,
    'max_range_m': 3.0,                  # observations farther than this are dropped
    'map_wall_check': True,              # drop a hit that lies on a mapped wall
    'map_wall_tolerance_m': 0.15,        # how close to the map's wall counts as "on the wall"
    'map_box_extent_m': 0.40,            # a mapped obstacle this small is a drawn box, not a wall
    'cloud_check': False,                # drop a LiDAR range the depth cloud contradicts (see params.yaml)
    'cloud_lidar_max_gap_m': 0.5,        # |cloud distance - LiDAR distance| above this = contradiction
    # Cloud (fallback range)
    'use_cloud_fallback': False,         # with cloud_check, decides the subscription; read at startup only
    'cloud_max_age_s': 0.5,
    'erosion_px': 2,
    'z_min_m': 0.6,
    'z_max_m': 4.0,
    'min_3d_points': 15,
    # Observation uncertainty handed to fusion
    'obs_sigma_base_m': 0.05,
    'obs_sigma_per_m': 0.03,
    # Timing and frames
    'processing_rate_hz': 5.0,
    'sync_tolerance_s': 0.1,             # LiDAR period is ~0.14 s; read at startup only
    'sync_queue_size': 60,               # 2 s of 30 Hz images; read at startup only
    'robot_frame': 'base_footprint',
    'fixed_frame': 'odom',               # frame used to bridge image time and scan time
    'per_ray_time': True,                # use the time of the blob's middle LiDAR ray, not the scan stamp
    # Localization's "pose trusted" flag (/jar/localizacion_lista), see vision.skip_for_localization
    'wait_localization_start': True,     # no frames until the flag has been true once...
    'localization_wait_max_s': 10.0,     # ...or this long after /map (mission's start rule)
    'skip_heading_doubt': False,         # also skip frames while the flag is false after a turn
    'publish_debug': False,
}


class Detection(Node):
    def __init__(self):
        super().__init__('deteccion')
        self.p = parametros_en_vivo(self, PARAM_DEFAULTS)

        self.K = None                       # 3x3 intrinsics from camera_info
        self.last_cloud = None              # latest PointCloud2 (fallback range and cross-check)
        self._cloud_cache = None            # (stamp key, (points (H,W,3), frame)) of the last converted cloud
        self.t_last_proc = None
        self.map = None                     # (grid (h, w) int16, resolution, origin_x, origin_y)
        self.t_map = None                   # node clock when /map arrived (t0 of the run)
        self.loc_flag_seen = False          # /jar/localizacion_lista received at least once
        self.loc_ready_once = False         # the flag has been true at least once
        self.loc_ready_now = False          # the flag's last value
        self.rejections = {}                # reason -> count, logged every 10 s
        self.t_last_summary = None
        self.n_published = 0

        #PARAM: cache_time = 10 s of TF history. Support: > the latency between capture and
        #       processing (tenths of a second). Plenty.
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        # The listener gets its own node and thread. On the node's thread, /tf is not received
        # while a lookup waits, so the 0.1 s lookup timeout could never get a late transform.
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, None, spin_thread=True)

        # Image and scan are processed in pairs with close timestamps. The LiDAR runs at
        # ~7 Hz with ~136 ms latency, so the tolerance is wider than for a 30 Hz camera.
        sub_img = message_filters.Subscriber(
            self, Image, '/cam_1/color/image_raw', qos_profile=qos_profile_sensor_data)
        sub_scan = message_filters.Subscriber(
            self, LaserScan, '/scan', qos_profile=qos_profile_sensor_data)
        # The Python synchronizer pairs a message only with partners still queued, so the queue
        # must hold the images taken while a late scan is still in transit.
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [sub_img, sub_scan], queue_size=int(self.p['sync_queue_size']),
            slop=float(self.p['sync_tolerance_s']))
        self.sync.registerCallback(self.on_pair)
        self.create_subscription(
            CameraInfo, '/cam_1/color/camera_info', self.on_camera_info, qos_profile_sensor_data)
        # The cloud only if something reads it: a subscription that exists pulls every cloud
        # over the robot's Wi-Fi (~500 KB at 7-10 Hz), used or not. Image + cloud fill the
        # competition router (physical_rosmaster raphael_router_wifi_2026-10-16.md), and §7.1
        # warns about keeping it on all run.
        if self.p['use_cloud_fallback'] or self.p['cloud_check']:
            self.create_subscription(
                PointCloud2, '/cam_1/depth/color/points', self.on_cloud, qos_profile_sensor_data)
        # The judge's map, for the wall check (transient local: arrives even if published earlier).
        self.create_subscription(OccupancyGrid, '/map', self.on_map, QOS_MAPA)
        # Localization's flag: same QoS as the publisher (reliable + transient local), so the
        # last value arrives even if it was published before this node started.
        self.create_subscription(Bool, '/jar/localizacion_lista', self.on_localization_ready, QOS_ESTADO)

        # Observations are a stream, not a state: default reliable QoS, no transient local.
        self.pub_obs = self.create_publisher(PoseWithCovarianceStamped, '/jar/observaciones', 10)
        self.pub_debug = None
        if self.p['publish_debug']:
            self.pub_debug = self.create_publisher(Image, '/jar/deteccion/debug', 1)

        self.get_logger().info('Detection ready: red + LiDAR -> /jar/observaciones')

    # ------------------------------------------------------------------ simple callbacks

    def on_camera_info(self, msg):
        """camera_info carries K as a flat row-major list of 9."""
        self.K = np.array(msg.k, dtype=float).reshape(3, 3)

    def on_cloud(self, msg):
        self.last_cloud = msg

    def on_map(self, msg):
        """OccupancyGrid -> (grid, resolution, origin). Row = y, column = x (row-major)."""
        grid = np.array(msg.data, dtype=np.int16).reshape(msg.info.height, msg.info.width)
        self.map = (grid, msg.info.resolution, msg.info.origin.position.x, msg.info.origin.position.y)
        if self.t_map is None:
            self.t_map = self.get_clock().now()
        self.get_logger().info(f'Map received: {msg.info.width}x{msg.info.height}')

    def on_localization_ready(self, msg):
        self.loc_flag_seen = True
        self.loc_ready_now = bool(msg.data)
        if msg.data and not self.loc_ready_once:
            self.loc_ready_once = True
            self.get_logger().info('Localization ready: processing frames')

    def _skip_for_localization(self):
        """True if localization does not trust its pose yet (vision.skip_for_localization)."""
        since_map = 0.0
        if self.t_map is not None:
            since_map = (self.get_clock().now() - self.t_map).nanoseconds * 1e-9
        return vision.skip_for_localization(
            self.loc_flag_seen, self.loc_ready_once, self.loc_ready_now, since_map,
            float(self.p['localization_wait_max_s']),
            bool(self.p['wait_localization_start']), bool(self.p['skip_heading_doubt']))

    def _reject(self, reason):
        """Count a rejected blob by reason; the summary goes to the log every 10 s."""
        self.rejections[reason] = self.rejections.get(reason, 0) + 1

    def _log_summary(self, t):
        if self.t_last_summary is None:
            self.t_last_summary = t
            return
        if t - self.t_last_summary >= 10.0:
            parts = ' '.join(f'{k}={v}' for k, v in sorted(self.rejections.items()))
            self.get_logger().info(f'last 10 s: published {self.n_published}; rejected {parts or "none"}')
            self.rejections = {}
            self.n_published = 0
            self.t_last_summary = t

    def _hits_mapped_wall(self, lidar_origin_map, face_map):
        """True if the LiDAR hit lies on a wall of the map (so it is not a box).

        Marches the map from the LiDAR origin towards the hit. If the first occupied cell is
        within `map_wall_tolerance_m` of the hit distance, the LiDAR saw a mapped obstacle.
        A small mapped obstacle (extent <= map_box_extent_m) could be a box the competition map
        draws, so it is not treated as a wall. Without a map the check passes.
        """
        if self.map is None:
            return False
        grid, res, ox, oy = self.map
        d = np.asarray(face_map[:2]) - np.asarray(lidar_origin_map[:2])
        dist = float(np.linalg.norm(d))
        if dist < 1e-6:
            return False
        tol = float(self.p['map_wall_tolerance_m'])
        d_map = vision.map_range_along(grid, res, ox, oy, lidar_origin_map[:2], d, dist + tol)
        if d_map is None or d_map > dist + tol:
            return False                   # the map is free up to the hit: an unmapped object
        if d_map < dist - tol:
            return False                   # the map has a wall BEFORE the hit: the pose is off; do not judge
        hit = (lidar_origin_map[0] + d[0] / dist * d_map, lidar_origin_map[1] + d[1] / dist * d_map)
        extent = vision.mapped_obstacle_extent(grid, res, ox, oy, hit, 0.30)
        return extent > float(self.p['map_box_extent_m'])

    def _vision_params(self):
        """Rebuild the VisionParams dataclass from the live dict (cheap; once per frame)."""
        p = self.p
        return vision.VisionParams(
            hsv_low_1=tuple(int(v) for v in p['hsv_red_low_1']),
            hsv_high_1=tuple(int(v) for v in p['hsv_red_high_1']),
            hsv_low_2=tuple(int(v) for v in p['hsv_red_low_2']),
            hsv_high_2=tuple(int(v) for v in p['hsv_red_high_2']),
            kernel_px=int(p['morphology_kernel_px']), min_area_px=int(p['min_area_px']),
            aspect_min=float(p['aspect_min']), aspect_max=float(p['aspect_max']),
            min_solidity=float(p['min_solidity']), erosion_px=int(p['erosion_px']),
            z_min_m=float(p['z_min_m']), z_max_m=float(p['z_max_m']),
            min_points=int(p['min_3d_points']),
        )

    # ------------------------------------------------------------------ pipeline

    def on_pair(self, img_msg, scan_msg):
        """Main callback: one image and one scan with close timestamps.

        Nothing may kill this node during a run (a crashed node is never restarted), so any
        unexpected error in one frame is logged and the frame is skipped.
        """
        try:
            self._process_pair(img_msg, scan_msg)
        except Exception as e:  # noqa: BLE001  (deliberate: survive any single bad frame)
            self.get_logger().error(f'frame skipped: {type(e).__name__}: {e}', throttle_duration_sec=5.0)

    def _process_pair(self, img_msg, scan_msg):
        img_stamp = Time.from_msg(img_msg.header.stamp)
        scan_stamp = Time.from_msg(scan_msg.header.stamp)
        t = scan_stamp.nanoseconds * 1e-9

        # Rate limit: the camera runs at 30 Hz; fusion needs far fewer observations.
        rate = float(self.p['processing_rate_hz'])
        if self.t_last_proc is not None and (t - self.t_last_proc) < 1.0 / rate:
            return
        self.t_last_proc = t
        self._log_summary(t)
        # Localization does not trust its pose (start-up, or heading fix after a turn):
        # a frame now would put the box in the wrong place. Counted as 'localizing'.
        if self._skip_for_localization():
            self._reject('localizing')
            return
        if self.K is None:
            self.get_logger().warn('no camera_info yet', throttle_duration_sec=5.0)
            return

        vp = self._vision_params()
        try:
            bgr = vision.decode_image(
                img_msg.data, img_msg.height, img_msg.width, img_msg.step, img_msg.encoding)
        except ValueError as e:
            self.get_logger().error(f'{e}', throttle_duration_sec=10.0)
            return
        mask = vision.red_mask(bgr, vp)
        blobs = vision.find_blobs(mask, vp)
        if self.pub_debug is not None:
            self._publish_debug(bgr, mask, blobs, img_msg.header)
        if not blobs:
            return

        cam_frame = img_msg.header.frame_id
        lidar_frame = scan_msg.header.frame_id
        robot_frame = str(self.p['robot_frame'])

        # Camera origin and ray directions expressed in the LiDAR frame AT THE SCAN TIME.
        # lookup_transform_full bridges the two timestamps through the fixed frame (odom), so
        # the robot's motion between image and scan is compensated.
        tr_cam_to_lidar = self._lookup_full(lidar_frame, scan_stamp, cam_frame, img_stamp)
        if tr_cam_to_lidar is None:
            return
        cam_origin_l = self._apply(tr_cam_to_lidar, np.zeros(3))
        scan_pts = vision.scan_to_points(
            np.asarray(scan_msg.ranges), scan_msg.angle_min, scan_msg.angle_increment,
            scan_msg.range_min, scan_msg.range_max)

        side = float(self.p['box_side_m'])

        for blob in blobs:
            x, y, w, h = blob.bbox
            v_mid = y + h / 2.0
            tr, origin, stamp = tr_cam_to_lidar, cam_origin_l, scan_stamp
            # The sweep takes ~0.134 s and ray i is measured at stamp + (n-1-i) * time_increment.
            # Redo the transform at the time of the ray in the middle of the blob. 0 in the simulator.
            if self.p['per_ray_time'] and scan_msg.time_increment > 0.0:
                d_mid = self._apply(tr, vision.pixel_ray(self.K, x + w / 2.0, v_mid)) - origin
                delay = vision.mid_ray_delay(
                    np.arctan2(d_mid[1], d_mid[0]), scan_msg.angle_min, scan_msg.angle_increment,
                    len(scan_msg.ranges), scan_msg.time_increment)
                stamp_ray = scan_stamp + Duration(seconds=delay)
                tr_ray = self._lookup_full(lidar_frame, stamp_ray, cam_frame, img_stamp)
                if tr_ray is not None:     # else: late transform, keep the scan stamp as today
                    tr, stamp = tr_ray, stamp_ray
                    origin = self._apply(tr, np.zeros(3))
            lidar_origin_map = None        # computed once per blob, only if the wall check runs
            # Rays through the left and right edges of the blob, in the LiDAR frame.
            d_left = self._apply(tr, vision.pixel_ray(self.K, x, v_mid)) - origin
            d_right = self._apply(tr, vision.pixel_ray(self.K, x + w, v_mid)) - origin

            # --- primary: the LiDAR beams in the central part of the blob ---
            face_robot = None
            source = 'lidar'
            cluster = vision.central_cluster(
                scan_pts, origin[:2], d_left[:2], d_right[:2],
                float(self.p['central_fraction']), float(self.p['lidar_cluster_depth_m']),
                int(self.p['lidar_min_points']), float(self.p['min_cluster_coverage']))
            if cluster is not None:
                face_xy, _, _ = cluster
                face_lidar = np.array([face_xy[0], face_xy[1], 0.0])
                face_robot = self._transform(face_lidar, lidar_frame, robot_frame, stamp)

            # --- fallback: depth cloud median ---
            if face_robot is None and self.p['use_cloud_fallback']:
                face_robot = self._face_from_cloud(blob, bgr.shape, vp, cam_frame, robot_frame, stamp)
                source = 'cloud'
            if face_robot is None:
                self._reject('no_range')
                continue

            # --- consistency checks: each one alone would have caught the 2026-10-25 false reports
            distance = float(np.linalg.norm(face_robot[:2]))
            if distance > float(self.p['max_range_m']):
                self._reject('too_far')
                continue
            # A blob cut by the image border is narrower than the box for a reason that has
            # nothing to do with the scene (a box passing by at 0.7 m fills the edge of the
            # frame), so the size checks do not apply to it. Its range still comes from the
            # central beams, and the other checks still run.
            at_border = x <= 0 or x + w >= bgr.shape[1]
            if not at_border and not vision.width_is_consistent(
                    w, self.K[0, 0], distance, side,
                    float(self.p['width_ratio_min']), float(self.p['width_ratio_max'])):
                self._reject('width')
                self.get_logger().debug(f'dropped: width {w} px inconsistent at {distance:.2f} m')
                continue
            factor = float(self.p['max_area_factor'])
            if not at_border and factor > 0 and not vision.area_is_consistent(
                    blob.area, distance, self.K[0, 0], self.K[1, 1], side, factor):
                self._reject('area')
                self.get_logger().debug(f'dropped: area {blob.area:.0f} px inconsistent at {distance:.2f} m')
                continue
            # Depth cloud as a second opinion on the LiDAR range (Emiliano-Aguilera, 2026-10-27):
            # on the recorded false reports the cloud inside the red blob was 1-2 m farther
            # than the LiDAR (it saw the real box, the LiDAR saw the wall end), while on real
            # boxes both agree within a few cm. No cloud, or too few valid points, never rejects:
            # the real sensor has gaps of up to a second and ~38 % valid pixels.
            # Skipped when the LiDAR range is below the cloud's own minimum range: there the
            # cloud can only show something farther, which is not a contradiction.
            if source == 'lidar' and self.p['cloud_check'] and distance >= vp.z_min_m:
                face_cloud = self._face_from_cloud(blob, bgr.shape, vp, cam_frame, robot_frame, stamp)
                if face_cloud is not None:
                    d_cloud = float(np.linalg.norm(face_cloud[:2]))
                    if abs(d_cloud - distance) > float(self.p['cloud_lidar_max_gap_m']):
                        self._reject('cloud_disagrees')
                        self.get_logger().debug(
                            f'dropped: LiDAR {distance:.2f} m vs cloud {d_cloud:.2f} m')
                        continue
            face_map = self._transform(face_robot, robot_frame, 'map', stamp)
            if face_map is None:
                self._reject('no_tf_map')
                continue
            if self.p['map_wall_check']:
                if lidar_origin_map is None:
                    lidar_origin_map = self._transform(np.zeros(3), lidar_frame, 'map', stamp)
                if lidar_origin_map is not None and self._hits_mapped_wall(lidar_origin_map, face_map):
                    self._reject('mapped_wall')
                    self.get_logger().debug(f'dropped: hit on a mapped wall at {distance:.2f} m')
                    continue

            # Face -> center (half a side back), then to map at the same time.
            center_robot = vision.center_from_face(face_robot, side)
            center_map = self._transform(center_robot, robot_frame, 'map', stamp)
            if center_map is None:
                self._reject('no_tf_map')
                continue

            sigma = vision.observation_sigma(
                distance, float(self.p['obs_sigma_base_m']), float(self.p['obs_sigma_per_m']))
            self._publish_observation(center_map, sigma, scan_msg.header.stamp)
            self.n_published += 1
            self.get_logger().debug(
                f'obs ({source}): ({center_map[0]:.2f}, {center_map[1]:.2f}) d={distance:.2f} sigma={sigma:.3f}')

    def _face_from_cloud(self, blob, shape, vp, cam_frame, robot_frame, stamp):
        """Fallback: median of the cloud points inside the blob, in the robot frame, or None."""
        cloud = self.last_cloud
        if cloud is None:
            return None
        cloud_stamp = Time.from_msg(cloud.header.stamp)
        if abs((cloud_stamp - stamp).nanoseconds) * 1e-9 > float(self.p['cloud_max_age_s']):
            return None
        # One conversion per cloud message, shared by every blob of the frame.
        key = (cloud.header.stamp.sec, cloud.header.stamp.nanosec)
        if self._cloud_cache is None or self._cloud_cache[0] != key:
            self._cloud_cache = (key, self._cloud_as_image(cloud, shape, cam_frame, cloud_stamp))
        result = self._cloud_cache[1]
        if result is None:
            return None
        points, points_frame = result
        m = vision.blob_mask(blob, shape, vp.erosion_px)
        p_cloud = vision.robust_3d_point(points, m, vp)
        if p_cloud is None:
            return None
        return self._transform(p_cloud, points_frame, robot_frame, cloud_stamp)

    def _cloud_as_image(self, cloud_msg, shape, cam_frame, cloud_stamp):
        """Cloud as a pixel-indexable (H, W, 3) array, plus the frame its points are in.

        - Organized and the same size as the image: direct reshape, points stay in the cloud's
          own frame.
        - Otherwise (the real robot's cloud is unorganized, in `cam_1_depth_frame`, which is
          NOT an optical frame): the points are first moved into the color optical frame with
          TF, then projected with K onto the image grid. The result is in `cam_frame`.
        """
        structured = point_cloud2.read_points(cloud_msg, field_names=('x', 'y', 'z'), skip_nans=False)
        xyz = rfn.structured_to_unstructured(structured).astype(np.float32).reshape(-1, 3)
        height, width = shape[:2]
        if cloud_msg.height == height and cloud_msg.width == width:
            return xyz.reshape(height, width, 3), cloud_msg.header.frame_id
        cloud_frame = cloud_msg.header.frame_id
        if cloud_frame != cam_frame:
            try:
                tr = self.tf_buffer.lookup_transform(
                    cam_frame, cloud_frame, cloud_stamp, timeout=Duration(seconds=0.1))
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException) as e:
                self.get_logger().warn(
                    f'no transform {cloud_frame} -> {cam_frame}: {e}', throttle_duration_sec=5.0)
                return None
            R, t = self._rotation_and_translation(tr)
            xyz = (xyz @ R.T + t).astype(np.float32)      # all points at once, no per-point calls
        return vision.cloud_to_image(xyz, self.K, shape), cam_frame

    @staticmethod
    def _rotation_and_translation(tr):
        """TransformStamped -> (3x3 rotation matrix, translation vector)."""
        q = tr.transform.rotation
        x, y, z, w = q.x, q.y, q.z, q.w
        R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])
        t = np.array([tr.transform.translation.x, tr.transform.translation.y, tr.transform.translation.z])
        return R, t

    # ------------------------------------------------------------------ tf helpers

    def _lookup_full(self, target_frame, target_time, source_frame, source_time):
        """Transform between two frames at two different times, bridged through fixed_frame."""
        try:
            #PARAM: timeout = 0.1 s to wait for a late transform. Support: >= 0. Longer blocks
            #       the callback; 0 drops more frames when tf runs just behind.
            return self.tf_buffer.lookup_transform_full(
                target_frame, target_time, source_frame, source_time,
                str(self.p['fixed_frame']), timeout=Duration(seconds=0.1))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(
                f'no transform {source_frame} -> {target_frame}: {e}', throttle_duration_sec=5.0)
            return None

    def _transform(self, point, source_frame, target_frame, stamp):
        """Transform a (3,) point between frames at one time. None if tf2 cannot."""
        if source_frame == target_frame:
            return np.asarray(point, dtype=float)
        try:
            tr = self.tf_buffer.lookup_transform(
                target_frame, source_frame, stamp, timeout=Duration(seconds=0.1))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(
                f'no transform {source_frame} -> {target_frame}: {e}', throttle_duration_sec=5.0)
            return None
        return self._apply(tr, point)

    @staticmethod
    def _apply(tr, point):
        """Apply a TransformStamped to a (3,) point."""
        ps = PointStamped()
        ps.point.x, ps.point.y, ps.point.z = (float(v) for v in point)
        out = tf2_geometry_msgs.do_transform_point(ps, tr)
        return np.array([out.point.x, out.point.y, out.point.z])

    # ------------------------------------------------------------------ output

    def _publish_observation(self, center_map, sigma, stamp_msg):
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = stamp_msg
        msg.header.frame_id = 'map'
        msg.pose.pose.position.x = float(center_map[0])
        msg.pose.pose.position.y = float(center_map[1])
        msg.pose.pose.position.z = 0.0
        msg.pose.pose.orientation.w = 1.0
        # 6x6 row-major covariance; only x (index 0) and y (index 7) carry information.
        cov = [0.0] * 36
        cov[0] = cov[7] = float(sigma) ** 2
        msg.pose.covariance = cov
        self.pub_obs.publish(msg)

    def _publish_debug(self, bgr, mask, blobs, header):
        """Image for RViz: red-tinted mask, green boxes, cyan centroids."""
        import cv2
        dbg = bgr.copy()
        dbg[mask > 0] = (0.5 * dbg[mask > 0] + (0, 0, 127)).astype(np.uint8)
        for b in blobs:
            x, y, w, h = b.bbox
            cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.circle(dbg, (int(b.centroid[0]), int(b.centroid[1])), 3, (255, 255, 0), -1)
        msg = Image()
        msg.header = header
        msg.height, msg.width = dbg.shape[0], dbg.shape[1]
        msg.encoding = 'bgr8'
        msg.step = dbg.shape[1] * 3
        msg.data = dbg.tobytes()
        self.pub_debug.publish(msg)


def main(args=None):
    correr(Detection, args)
