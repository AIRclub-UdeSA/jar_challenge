"""Tests for vision.py with synthetic images, scans and clouds. No ROS needed."""
import math

import numpy as np
import pytest

from equipo_jar import vision

P = vision.VisionParams()
# Robot camera: 320x240, fx = fy ≈ 271.8, cx ≈ 158.7 (robot contract).
K = np.array([[271.8, 0.0, 158.7], [0.0, 271.8, 120.0], [0.0, 0.0, 1.0]])


def image_with_box(x0=120, y0=90, w=50, h=60, color=(30, 30, 200), background=(180, 180, 180)):
    """Grey 320x240 BGR image with a colored rectangle. Returns (image, bbox)."""
    img = np.full((240, 320, 3), background, np.uint8)
    img[y0:y0 + h, x0:x0 + w] = color
    return img, (x0, y0, w, h)


# ----------------------------------------------------------------------------- mask


def test_mask_detects_red_and_not_background():
    img, (x0, y0, w, h) = image_with_box()
    m = vision.red_mask(img, P)
    assert m[y0 + h // 2, x0 + w // 2] == 255
    assert m[5, 5] == 0


def test_mask_covers_both_red_ranges():
    img_a, _ = image_with_box(color=(0, 0, 255))        # H = 0
    img_b, _ = image_with_box(color=(40, 0, 255))       # H ≈ 175
    assert vision.red_mask(img_a, P)[120, 145] == 255
    assert vision.red_mask(img_b, P)[120, 145] == 255


def test_mask_rejects_distractor_colors():
    # Distractors are blue, yellow, green or neutral (challenge rules).
    for color in [(255, 0, 0), (0, 255, 255), (0, 255, 0), (255, 255, 255), (0, 0, 0), (128, 128, 128)]:
        img, _ = image_with_box(color=color)
        assert not vision.red_mask(img, P).any(), color


def test_mask_rejects_brown_cardboard():
    # Cardboard walls measured at H 10-12, S ~105. This BGR is H 10, S 105, V 150; the upper hue
    # of range 1 is 8 to keep it out, so raising it back to 10 makes this fail.
    img, _ = image_with_box(color=(88, 109, 150))
    assert not vision.red_mask(img, P).any()


def test_opening_removes_isolated_pixel_noise():
    img, _ = image_with_box()
    rng = np.random.default_rng(0)
    for _ in range(150):
        u, v = rng.integers(0, 320), rng.integers(0, 240)
        img[v, u] = (0, 0, 255)
    blobs = vision.find_blobs(vision.red_mask(img, P), P)
    assert len(blobs) == 1


# ----------------------------------------------------------------------------- blobs


def test_blob_bbox_and_centroid():
    img, (x0, y0, w, h) = image_with_box()
    blobs = vision.find_blobs(vision.red_mask(img, P), P)
    assert len(blobs) == 1
    b = blobs[0]
    assert b.bbox[0] == pytest.approx(x0, abs=2)
    assert b.bbox[1] == pytest.approx(y0, abs=2)
    assert b.centroid[0] == pytest.approx(x0 + w / 2, abs=1.5)
    assert b.centroid[1] == pytest.approx(y0 + h / 2, abs=1.5)
    assert b.area == pytest.approx(w * h, rel=0.1)


def test_aspect_filter_rejects_a_stripe():
    img = np.full((240, 320, 3), (180, 180, 180), np.uint8)
    img[100:106, 40:280] = (30, 30, 200)
    assert vision.find_blobs(vision.red_mask(img, P), P) == []


def test_area_filter_rejects_tiny_blobs():
    img, _ = image_with_box(w=6, h=7)
    assert vision.find_blobs(vision.red_mask(img, P), P) == []


def test_blobs_sorted_by_descending_area():
    img, _ = image_with_box(x0=20, y0=20, w=20, h=24)
    img[140:220, 200:270] = (30, 30, 200)
    blobs = vision.find_blobs(vision.red_mask(img, P), P)
    assert len(blobs) == 2 and blobs[0].area > blobs[1].area


# ----------------------------------------------------------------------------- rays


def test_pixel_ray_through_principal_point_is_optical_axis():
    d = vision.pixel_ray(K, K[0, 2], K[1, 2])
    assert d == pytest.approx([0.0, 0.0, 1.0])


def test_pixel_ray_bearing_matches_pinhole():
    # A pixel 100 px right of center: tan(angle) = 100 / fx.
    d = vision.pixel_ray(K, K[0, 2] + 100, K[1, 2])
    assert math.atan2(d[0], d[2]) == pytest.approx(math.atan(100 / 271.8))
    assert np.linalg.norm(d) == pytest.approx(1.0)


# ----------------------------------------------------------------------------- lidar


def synthetic_scan(objects, n=1080, wall=4.0, angle_min=-math.pi):
    """Ranges for a round wall at `wall` m plus boxes given as (bearing, range, half_width_rad)."""
    inc = 2 * math.pi / n
    ranges = np.full(n, wall)
    angles = angle_min + inc * np.arange(n)
    for bearing, r, half in objects:
        d = (angles - bearing + math.pi) % (2 * math.pi) - math.pi
        ranges[np.abs(d) <= half] = r
    return ranges, angle_min, inc


def test_scan_to_points_drops_invalid_and_converts_polar():
    ranges = np.array([1.0, np.inf, np.nan, 2.0, 0.01, 20.0])
    pts = vision.scan_to_points(ranges, 0.0, math.pi / 2, 0.05, 12.0)
    assert pts.shape == (2, 2)
    assert pts[0] == pytest.approx([1.0, 0.0])
    assert pts[1] == pytest.approx([2.0 * math.cos(3 * math.pi / 2), 2.0 * math.sin(3 * math.pi / 2)])


def test_wedge_and_nearest_cluster_pick_the_box_not_the_wall():
    # Box at 1.5 m, bearing 0.4 rad, angular half-width 0.08 rad (~0.24 m wide).
    ranges, a0, inc = synthetic_scan([(0.4, 1.5, 0.08)])
    pts = vision.scan_to_points(ranges, a0, inc, 0.05, 12.0)
    origin = np.zeros(2)
    d_left = np.array([math.cos(0.48), math.sin(0.48)])
    d_right = np.array([math.cos(0.32), math.sin(0.32)])
    sel = vision.points_in_wedge(pts, origin, d_left, d_right, 0.0)
    assert sel.sum() > 20                         # ~0.16 rad / 0.0058 rad per beam
    cluster = vision.nearest_cluster(pts[sel], origin, 0.30, 3)
    assert cluster is not None
    face, count = cluster
    assert np.linalg.norm(face) == pytest.approx(1.5, abs=0.02)
    assert math.atan2(face[1], face[0]) == pytest.approx(0.4, abs=0.01)
    assert count > 20


def test_wedge_handles_parallax_from_an_offset_camera():
    # Camera 0.1 m ahead of the LiDAR: rays are computed from the camera origin, so the same
    # box must still be found when the wedge is defined from that origin.
    ranges, a0, inc = synthetic_scan([(0.0, 1.0, 0.12)])
    pts = vision.scan_to_points(ranges, a0, inc, 0.05, 12.0)
    origin = np.array([0.1, 0.0])
    # From the camera the box (at x=1.0 in lidar frame) is 0.9 m away, same bearing 0.
    d_left = np.array([math.cos(0.1), math.sin(0.1)])
    d_right = np.array([math.cos(-0.1), math.sin(-0.1)])
    sel = vision.points_in_wedge(pts, origin, d_left, d_right, 0.0)
    face, _ = vision.nearest_cluster(pts[sel], origin, 0.30, 3)
    assert face[0] == pytest.approx(1.0, abs=0.02)


def test_wedge_with_only_wall_returns_the_wall_and_cluster_needs_min_points():
    ranges, a0, inc = synthetic_scan([])
    pts = vision.scan_to_points(ranges, a0, inc, 0.05, 12.0)
    sel = vision.points_in_wedge(pts, np.zeros(2), np.array([1.0, 0.05]), np.array([1.0, -0.05]), 0.0)
    cluster = vision.nearest_cluster(pts[sel], np.zeros(2), 0.30, 3)
    assert cluster is not None and np.linalg.norm(cluster[0]) == pytest.approx(4.0, abs=0.01)
    assert vision.nearest_cluster(pts[sel], np.zeros(2), 0.30, 1000) is None
    assert vision.nearest_cluster(np.zeros((0, 2)), np.zeros(2), 0.30, 1) is None


def test_wedge_wraps_around_pi():
    # LiDAR is mounted with yaw = pi: a box in front of the robot is at bearing ~pi in the scan.
    ranges, a0, inc = synthetic_scan([(math.pi, 2.0, 0.06)])
    pts = vision.scan_to_points(ranges, a0, inc, 0.05, 12.0)
    d_left = np.array([math.cos(math.pi + 0.07), math.sin(math.pi + 0.07)])
    d_right = np.array([math.cos(math.pi - 0.07), math.sin(math.pi - 0.07)])
    sel = vision.points_in_wedge(pts, np.zeros(2), d_left, d_right, 0.0)
    face, _ = vision.nearest_cluster(pts[sel], np.zeros(2), 0.30, 3)
    assert face[0] == pytest.approx(-2.0, abs=0.02)


# ----------------------------------------------------------------------------- cloud


def synthetic_cloud(bbox, z_box=1.5, z_background=3.0, shape=(240, 320)):
    height, width = shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u = np.arange(width)[None, :].repeat(height, 0)
    v = np.arange(height)[:, None].repeat(width, 1)
    z = np.full((height, width), z_background, np.float32)
    x0, y0, w, h = bbox
    z[y0:y0 + h, x0:x0 + w] = z_box
    return np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z], axis=2).astype(np.float32)


def test_cloud_median_takes_box_depth_and_survives_nan():
    img, bbox = image_with_box()
    cloud = synthetic_cloud(bbox)
    blob = vision.find_blobs(vision.red_mask(img, P), P)[0]
    m = vision.blob_mask(blob, img.shape, P.erosion_px)
    assert vision.robust_3d_point(cloud, m, P)[2] == pytest.approx(1.5, abs=0.01)
    half = cloud.copy()
    rows, cols = np.nonzero(m)
    half[rows[::2], cols[::2]] = np.nan
    assert vision.robust_3d_point(half, m, P)[2] == pytest.approx(1.5, abs=0.01)
    all_nan = cloud.copy()
    all_nan[m] = np.nan
    assert vision.robust_3d_point(all_nan, m, P) is None


def test_cloud_median_rejects_out_of_range():
    img, bbox = image_with_box()
    blob = vision.find_blobs(vision.red_mask(img, P), P)[0]
    m = vision.blob_mask(blob, img.shape, P.erosion_px)
    assert vision.robust_3d_point(synthetic_cloud(bbox, z_box=0.4), m, P) is None   # below 0.6 m
    assert vision.robust_3d_point(synthetic_cloud(bbox, z_box=6.0), m, P) is None


def test_unorganized_cloud_is_reprojected():
    img, bbox = image_with_box()
    organized = synthetic_cloud(bbox)
    flat = organized.reshape(-1, 3)
    flat = flat[np.random.default_rng(1).permutation(flat.shape[0])]
    re = vision.cloud_to_image(flat, K, img.shape)
    x0, y0, w, h = bbox
    assert re[y0 + h // 2, x0 + w // 2, 2] == pytest.approx(1.5, abs=0.01)
    assert re[5, 5, 2] == pytest.approx(3.0, abs=0.01)


# ----------------------------------------------------------------------------- geometry


def test_center_from_face_shifts_half_a_side_along_the_ray():
    p_face = np.array([2.0, 1.0, 0.0])
    c = vision.center_from_face(p_face, 0.25)
    n = np.array([2.0, 1.0]) / math.hypot(2.0, 1.0)
    assert c[:2] == pytest.approx(p_face[:2] + 0.125 * n)
    assert np.linalg.norm(c[:2] - p_face[:2]) == pytest.approx(0.125)
    assert vision.center_from_face(np.zeros(3), 0.25) == pytest.approx(np.zeros(3))


def test_area_consistency_with_robot_camera():
    fx = fy = 271.8
    L, Z = 0.25, 1.5
    expected = fx * fy * L ** 2 / Z ** 2          # ≈ 2052 px
    assert vision.area_is_consistent(expected, Z, fx, fy, L, 3.0)
    assert vision.area_is_consistent(expected * 1.4, Z, fx, fy, L, 3.0)
    assert not vision.area_is_consistent(expected * 10, Z, fx, fy, L, 3.0)
    assert not vision.area_is_consistent(expected / 10, Z, fx, fy, L, 3.0)
    assert not vision.area_is_consistent(expected, 0.0, fx, fy, L, 3.0)


def test_observation_sigma_grows_with_distance():
    assert vision.observation_sigma(0.0, 0.05, 0.03) == pytest.approx(0.05)
    assert vision.observation_sigma(2.0, 0.05, 0.03) == pytest.approx(0.11)
    assert vision.observation_sigma(-1.0, 0.05, 0.03) == pytest.approx(0.05)


# ----------------------------------------------------------------------------- image decoding


def test_decode_image_bgr8_and_rgb8_with_row_padding():
    img = np.zeros((2, 3, 3), np.uint8)
    img[0, 0] = (10, 20, 30)                      # B, G, R
    img[1, 2] = (40, 50, 60)
    # bgr8, no padding
    out = vision.decode_image(img.tobytes(), 2, 3, 9, 'bgr8')
    assert out.shape == (2, 3, 3) and tuple(out[0, 0]) == (10, 20, 30)
    # rgb8 with 2 bytes of padding per row -> channels swapped, padding dropped
    rows = [img[r].tobytes() + b'\x00\x00' for r in range(2)]
    out = vision.decode_image(b''.join(rows), 2, 3, 11, 'rgb8')
    assert tuple(out[0, 0]) == (30, 20, 10) and tuple(out[1, 2]) == (60, 50, 40)
    with pytest.raises(ValueError):
        vision.decode_image(img.tobytes(), 2, 3, 9, 'mono8')


# ----------------------------------------------------------------------------- false-positive defenses


def corner_scan(box_range=2.9, corner_range=1.46, box_half=None):
    """A distant box straight ahead and a wall corner intruding on the right edge of its blob."""
    box_half = box_half or math.atan(0.125 / box_range)          # ~0.043 rad at 2.9 m
    objects = [(0.0, box_range, box_half),
               (box_half + 0.08, corner_range, 0.10)]            # corner: from +0.02 to +0.22 rad
    ranges, a0, inc = synthetic_scan(objects)
    return vision.scan_to_points(ranges, a0, inc, 0.05, 12.0), box_half


def test_central_cluster_takes_the_box_not_the_corner():
    pts, half = corner_scan()
    origin = np.zeros(2)
    # The blob spans the box (half-angle) plus the corner that overlaps its right edge.
    d_left = np.array([math.cos(half), math.sin(half)])
    d_right = np.array([math.cos(-half), math.sin(-half)])
    # The old rule picked the corner (1.46 m): the false reports of 2026-10-25.
    old = vision.points_in_wedge(pts, origin, d_left, d_right, 0.03)
    face_old, _ = vision.nearest_cluster(pts[old], origin, 0.30, 3)
    assert np.linalg.norm(face_old) == pytest.approx(1.46, abs=0.03)
    # The central rule returns the box.
    res = vision.central_cluster(pts, origin, d_left, d_right, 0.6, 0.30, 3, 0.6)
    assert res is not None
    face, n, n_central = res
    assert np.linalg.norm(face) == pytest.approx(2.9, abs=0.03)
    assert n >= 0.8 * n_central          # at most a stray corner beam at the very edge


def test_central_cluster_rejects_a_span_split_between_two_surfaces():
    # Wall at 1.5 m covering the whole left half of the span, box at 3 m on the right half.
    ranges, a0, inc = synthetic_scan([(0.05, 3.0, 0.05), (-0.05, 1.5, 0.05)])
    pts = vision.scan_to_points(ranges, a0, inc, 0.05, 12.0)
    d_left = np.array([math.cos(0.1), math.sin(0.1)])
    d_right = np.array([math.cos(-0.1), math.sin(-0.1)])
    assert vision.central_cluster(pts, np.zeros(2), d_left, d_right, 1.0, 0.30, 3, 0.6) is None


def test_central_cluster_keeps_both_faces_of_a_diagonal_box():
    # Two faces: ranges 1.50 and 1.65 m across the span, within depth_m of each other.
    n = 1080
    inc = 2 * math.pi / n
    angles = -math.pi + inc * np.arange(n)
    ranges = np.full(n, 4.0)
    ranges[np.abs(angles) <= 0.04] = 1.65
    ranges[(angles > -0.08) & (angles < -0.04)] = 1.5
    pts = vision.scan_to_points(ranges, -math.pi, inc, 0.05, 12.0)
    d_left = np.array([math.cos(0.04), math.sin(0.04)])
    d_right = np.array([math.cos(-0.08), math.sin(-0.08)])
    res = vision.central_cluster(pts, np.zeros(2), d_left, d_right, 1.0, 0.30, 3, 0.6)
    assert res is not None and 1.5 <= np.linalg.norm(res[0]) <= 1.65


def test_central_cluster_needs_min_points():
    pts, half = corner_scan()
    d_left = np.array([math.cos(half), math.sin(half)])
    d_right = np.array([math.cos(-half), math.sin(-half)])
    assert vision.central_cluster(pts, np.zeros(2), d_left, d_right, 0.6, 0.30, 500, 0.6) is None
    assert vision.central_cluster(np.zeros((0, 2)), np.zeros(2), d_left, d_right, 0.6, 0.30, 1, 0.6) is None


def test_width_consistency_catches_a_corner_in_front_of_a_box():
    fx = 271.8
    w_box_at_2_9 = fx * 0.25 / 2.9                    # ~23 px, what the camera sees
    # LiDAR says 1.46 m (the corner): expected 47 px, ratio 0.5 -> rejected at ratio_min 0.6
    assert not vision.width_is_consistent(w_box_at_2_9, fx, 1.46, 0.25, 0.6, 1.8)
    # LiDAR says 2.9 m (the box): accepted
    assert vision.width_is_consistent(w_box_at_2_9, fx, 2.9, 0.25, 0.6, 1.8)
    # Diagonal view (1.4x) accepted, absurd widths rejected
    assert vision.width_is_consistent(w_box_at_2_9 * 1.4, fx, 2.9, 0.25, 0.6, 1.8)
    assert not vision.width_is_consistent(w_box_at_2_9 * 3, fx, 2.9, 0.25, 0.6, 1.8)
    assert not vision.width_is_consistent(10, fx, 0.0, 0.25, 0.6, 1.8)


def grid_with_wall():
    """10 x 10 m map at 0.05 m, origin (-5, -5); a wall along x = 2.0 (column of cells)."""
    g = np.zeros((200, 200), np.int16)
    col = int((2.0 + 5.0) / 0.05)
    g[:, col] = 100
    return g, 0.05, -5.0, -5.0


def test_map_range_along_finds_the_wall_and_free_space():
    g, res, ox, oy = grid_with_wall()
    d = vision.map_range_along(g, res, ox, oy, (0.0, 0.0), (1.0, 0.0), 5.0)
    assert d == pytest.approx(2.0, abs=0.06)
    assert vision.map_range_along(g, res, ox, oy, (0.0, 0.0), (-1.0, 0.0), 3.0) is None   # free
    assert vision.map_range_along(g, res, ox, oy, (0.0, 0.0), (1.0, 0.0), 1.0) is None    # too short
    assert vision.map_range_along(g, res, ox, oy, (0.0, 0.0), (0.0, 0.0), 1.0) is None    # no direction


def test_mapped_obstacle_extent_tells_a_wall_from_a_drawn_box():
    g, res, ox, oy = grid_with_wall()
    assert vision.mapped_obstacle_extent(g, res, ox, oy, (2.0, 0.0), 0.30) > 0.5
    box = np.zeros((200, 200), np.int16)
    c, r = int((0.0 + 5.0) / 0.05), int((0.0 + 5.0) / 0.05)
    box[r - 2:r + 3, c - 2:c + 3] = 100                     # 5 x 5 cells = 0.25 m
    assert vision.mapped_obstacle_extent(box, res, ox, oy, (0.0, 0.0), 0.30) == pytest.approx(0.25)
    assert vision.mapped_obstacle_extent(box, res, ox, oy, (3.0, 3.0), 0.30) == 0.0


# ----------------------------------------------------------------------------- frame gating


def _skip(flag_seen=True, ready_once=False, ready_now=False, since_map_s=0.0,
          wait_max_s=10.0, wait_start=True, skip_doubt=False):
    return vision.skip_for_localization(flag_seen, ready_once, ready_now, since_map_s,
                                        wait_max_s, wait_start, skip_doubt)


def test_gate_waits_at_start_until_the_flag_is_true_once():
    assert _skip(ready_once=False, ready_now=False)
    assert not _skip(ready_once=True, ready_now=True)


def test_gate_gives_up_waiting_after_the_start_timeout():
    # Mission starts moving after its own 10 s: detection must not stay blind.
    assert _skip(since_map_s=9.9)
    assert not _skip(since_map_s=10.0)


def test_gate_never_skips_without_a_localization_node():
    # pruebas.launch.py style: nobody publishes the flag.
    assert not _skip(flag_seen=False)
    assert not _skip(flag_seen=False, skip_doubt=True)


def test_gate_after_turns_only_if_asked():
    # Flag was true once and is false again (heading fix after a turn).
    assert not _skip(ready_once=True, ready_now=False, skip_doubt=False)
    assert _skip(ready_once=True, ready_now=False, skip_doubt=True)
    assert not _skip(ready_once=True, ready_now=True, skip_doubt=True)


def test_gate_start_wait_can_be_turned_off():
    assert not _skip(wait_start=False)
