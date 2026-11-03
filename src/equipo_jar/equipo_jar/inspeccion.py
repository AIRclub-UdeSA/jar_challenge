import math
from equipo_jar.mapa import is_free, line_is_clear
from equipo_jar.confirmacion import near_any

def viewpoint_for(grid, cx, cy, robot_x, robot_y, distance, clearance, n_angles, min_shift, box_margin,
                cell_grid, reachable, blocked=None, tried=()):
    # blocked(x, y): optional; True skips that spot (mission: occupied in Nav2's costmap).
    # tried: (x, y) spots this candidate was already looked at from without a result, or that
    # Nav2 did not reach; a spot closer than min_shift to one of them is the same view.
    best = None
    best_dist = None
    for k in range(n_angles):
        angle = 2 * math.pi * k / n_angles
        c = math.cos(angle)
        s = math.sin(angle)
        vx = cx + distance * c  
        vy = cy + distance * s
        ex = cx + box_margin * c
        ey = cy + box_margin * s
        if not is_free(grid, vx, vy, clearance):
            continue
        if not line_is_clear(grid, vx, vy, ex, ey):
            continue
        if cell_grid.cell(vx, vy) not in reachable:
            continue
        if blocked is not None and blocked(vx, vy):
            continue
        d = math.hypot(vx - robot_x, vy - robot_y)
        if d < min_shift:
            continue
        if near_any(vx, vy, tried, min_shift - 1e-9):
            continue
        if best is None or d < best_dist:
            yaw = math.atan2(cy - vy, cx - vx)
            best = (vx, vy, yaw)
            best_dist = d
    return best

def pick_doubtful(cands, reported_ids, reported_points, inspected_points, min_obs, dist_min, robot_x, robot_y, failed, max_attempts):
    best = None
    best_dist = None
    remaining = []
    for c in cands:
        if c.id in reported_ids or near_any(c.centro.x, c.centro.y, reported_points, dist_min):
            continue
        if near_any(c.centro.x, c.centro.y, inspected_points, dist_min):
            continue
        if not math.isfinite(c.centro.x) or not math.isfinite(c.centro.y):
            continue
        if c.observaciones < min_obs:
            continue
        if c.id in failed:
            last_obs, attempts = failed[c.id]
            if attempts >= max_attempts or c.observaciones <= last_obs:
                continue
        remaining.append(c)
    for c in remaining:
        dist = math.hypot(c.centro.x - robot_x, c.centro.y - robot_y)
        if best is None or dist < best_dist:
            best = c
            best_dist = dist
    return best

def forget_failed_inspection(inspected_points, failed, cand_id, x, y, observations):
    if (x, y) in inspected_points:
        inspected_points.remove((x, y))
    _, attempts = failed.get(cand_id, (0, 0))
    failed[cand_id] = (observations, attempts + 1)
    return attempts + 1

def unhelpful_inspection(inspected_points, failed, cand_id, x, y):
    return forget_failed_inspection(inspected_points, failed, cand_id, x, y, 0)

def inspection_outcome(cands, cand_id, reported_ids, reported_points, dist_min, now, look_until):
    if cand_id in reported_ids:
        return 'reported'
    current = None
    for c in cands or []:
        if c.id == cand_id:
            current = c
    if current is None:
        return 'gone'
    if near_any(current.centro.x, current.centro.y, reported_points, dist_min):
        return 'reported'
    if now < look_until:
        return 'looking'
    return 'unhelpful'
