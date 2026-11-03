import math

def worst_deviation(cov_xy):
    #cov_xy = [a, b, b, d] matriz de covarianza escrita como vector
    a = cov_xy[0] 
    b = cov_xy[1]
    d = cov_xy[3]
    lambda_max = (a + d) / 2 + math.sqrt(((a - d) / 2) ** 2 + b ** 2) 
    return math.sqrt(max(lambda_max, 0.0))

def is_confirmed(obs, cov_xy, min_obs, max_deviation):
    if obs >= min_obs and worst_deviation(cov_xy) <= max_deviation:
        return True
    return False

def near_any(x, y, points, dist_min):
    for p in points:
        xi, yi = p
        dx = abs(x - xi)
        dy = abs(y - yi)
        dist = math.hypot(dx, dy)
        if dist <= dist_min:
            return True
    return False

def select_to_report(cands, reported_ids, reported_points, n, min_obs, max_deviation, dist_min):
    remaining = n - len(reported_points)
    if remaining <= 0:
        return []
    ordered = sorted(cands, key=lambda c: worst_deviation(c.covarianza_xy))
    selected = []
    selected_points = []
    for c in ordered:
        confirmed = is_confirmed(c.observaciones, c.covarianza_xy, min_obs, max_deviation)
        if ((not confirmed) or (c.id in reported_ids) 
            or near_any(c.centro.x, c.centro.y, reported_points + selected_points, dist_min)
            or (not math.isfinite(c.centro.x)) or (not math.isfinite(c.centro.y))):
            continue
        selected.append(c)
        selected_points.append((c.centro.x, c.centro.y))
        if len(selected) == remaining:
            break
    return selected
