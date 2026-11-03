"""Order in which to visit places, from their pairwise distances. No ROS."""


def shortest_path_order(D):
    """Order of the nodes of the symmetric distance matrix D as an open path from node 0.

    Nearest neighbour first, then 2-opt (reverse a stretch while that shortens the path).
    """
    n = len(D)
    order, left = [0], set(range(1, n))
    while left:
        nxt = min(left, key=lambda j: D[order[-1]][j])
        order.append(nxt)
        left.remove(nxt)
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                before = D[order[i - 1]][order[i]]
                after = D[order[i - 1]][order[j]]
                if j + 1 < n:  # the path is open: the last node has no edge after it
                    before += D[order[j]][order[j + 1]]
                    after += D[order[i]][order[j + 1]]
                if after < before - 1e-9:
                    order[i:j + 1] = order[i:j + 1][::-1]
                    improved = True
    return order


def latency(T, value, fixed, order):
    """Sum over the nodes of value x arrival time, along order from node 0. T: travel times,
    fixed: time spent at each node."""
    t, total = 0.0, 0.0
    for a, b in zip(order, order[1:]):
        t += T[a][b] + fixed[b]
        total += value[b] * t
    return total


def latency_order(T, value, fixed, order, keep=1):
    """order improved for latency(): the most valuable nodes early, without detours that
    delay all the others more than they gain. Or-opt: move one node elsewhere while that
    lowers it. The first keep nodes stay where they are."""
    order = list(order)
    best = latency(T, value, fixed, order)
    improved = True
    while improved:
        improved = False
        for i in range(keep, len(order)):
            node = order[i]
            rest = order[:i] + order[i + 1:]
            for j in range(keep, len(order)):
                if j == i:
                    continue
                trial = rest[:j] + [node] + rest[j:]
                cost = latency(T, value, fixed, trial)
                if cost < best - 1e-9:
                    order, best, improved = trial, cost, True
                    break
            if improved:
                break
    return order
