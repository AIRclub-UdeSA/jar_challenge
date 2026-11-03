import math
from equipo_jar.confirmacion import worst_deviation
from equipo_jar.mapa import cell_value

def yaw_from_quaternion(q):
    yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
    return yaw

def angle_difference(a, b):
    return abs(math.atan2(math.sin(a - b), math.cos(a - b)))

def already_visited(x, y, yaw, visited, dist_min, ang_min):
    for tup in visited:
        xi = tup[0]
        yi = tup[1]
        dx = x - xi
        dy = y - yi
        yawi = tup[2]
        dist = math.hypot(dx, dy)
        ang = angle_difference(yaw, yawi)
        if dist < dist_min and ang < ang_min:
            return True
    return False

def first_pending(poses, visited, dist_min, ang_min, zones):
    for p in poses:
        x = p.pose.position.x
        y = p.pose.position.y
        if near_box(x, y, zones):
            continue
        q = p.pose.orientation
        yaw = yaw_from_quaternion(q)
        seen = already_visited(x, y, yaw, visited, dist_min, ang_min)
        if not seen:
            return p
    return None

def goal_position(plan_x, plan_y, robot_pose, radius):
    in_place = False
    if robot_pose is None:
        return (plan_x, plan_y, in_place)
    x, y, _ = robot_pose
    dist = math.hypot(x - plan_x, y - plan_y)
    if dist < radius:
        in_place = True
        return (x, y, in_place)
    return (plan_x, plan_y, in_place)

def box_zones(cands, min_obs, clearance, sigmas):
    zones = []
    for c in cands:
        if c.observaciones < min_obs or not math.isfinite(c.centro.x) or not math.isfinite(c.centro.y):
            continue
        radius = clearance + sigmas * worst_deviation(c.covarianza_xy)
        zones.append((c.centro.x, c.centro.y, radius))
    return zones

def near_box(x, y, zones):
    for z in zones:
        dist = math.hypot(x - z[0], y - z[1])
        if dist < z[2]:
            return True
    return False

def return_verdict(error, tries, tolerance, max_tries):
    if error <= tolerance:
        return 'ok'
    if tries < max_tries:
        return 'retry'
    return 'give_up'

def goal_blocked(costmap, x, y, blocked_value):
    if costmap is None:
        return False
    value = cell_value(costmap, x, y)
    return value is not None and value >= blocked_value

def start_ready(localization_ready, elapsed_s, max_wait_s):
    return localization_ready or elapsed_s >= max_wait_s