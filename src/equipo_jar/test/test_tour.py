from itertools import permutations

import numpy as np

from equipo_jar.tour import shortest_path_order


def length(D, order):
    return sum(D[a][b] for a, b in zip(order, order[1:]))


def random_matrix(n, seed):
    pts = np.random.default_rng(seed).random((n, 2))
    return np.hypot(*(pts[:, None] - pts[None]).transpose(2, 0, 1))


def test_tiny_inputs():
    assert shortest_path_order(np.zeros((1, 1))) == [0]
    assert shortest_path_order(np.array([[0.0, 2.0], [2.0, 0.0]])) == [0, 1]


def test_it_visits_every_node_once_starting_at_zero():
    order = shortest_path_order(random_matrix(30, 1))
    assert order[0] == 0 and sorted(order) == list(range(30))


def test_it_walks_a_line_in_order_instead_of_zigzagging():
    xs = np.array([0.0, 3.0, 1.0, 4.0, 2.0])  # robot at 0, goals scattered along a line
    D = np.abs(xs[:, None] - xs[None])
    assert length(D, shortest_path_order(D)) == 4.0


def test_it_is_close_to_the_brute_force_optimum_on_small_instances():
    # 2-opt is a heuristic: a few percent over the optimum on average is expected
    ratios = []
    for seed in range(30):
        D = random_matrix(7, seed)
        best = min(length(D, (0, *p)) for p in permutations(range(1, 7)))
        ratios.append(length(D, shortest_path_order(D)) / best)
    assert np.mean(ratios) < 1.05 and max(ratios) < 1.25


def test_latency_order_brings_a_valuable_node_forward_for_a_small_detour():
    from equipo_jar.tour import latency, latency_order
    # Robot at 0; small-value goals at 1 and 2, a valuable one just past them at 2.5... and
    # one far away on the other side, at -1.
    xs = np.array([0.0, 1.0, 2.0, 2.5, -1.0])
    T = np.abs(xs[:, None] - xs[None])
    fixed = np.zeros(5)
    value = np.array([0.0, 0.01, 0.01, 0.01, 1.0])
    short = shortest_path_order(T)
    assert short[1] == 4 or short[-1] == 4
    better = latency_order(T, value, fixed, short)
    assert better[1] == 4  # worth the 2 s detour
    assert latency(T, value, fixed, better) <= latency(T, value, fixed, short)
    # Worth little: no detour, the short path stays.
    same = latency_order(T, np.array([0.0, 1.0, 1.0, 1.0, 0.01]), fixed, short)
    assert length(T, same) == length(T, short)


def test_latency_order_keeps_the_first_nodes():
    from equipo_jar.tour import latency_order
    xs = np.array([0.0, 5.0, 1.0, 2.0])
    T = np.abs(xs[:, None] - xs[None])
    order = latency_order(T, np.ones(4), np.zeros(4), [0, 1, 2, 3], keep=2)
    assert order[:2] == [0, 1] and sorted(order) == [0, 1, 2, 3]
