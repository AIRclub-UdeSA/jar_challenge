import math
from types import SimpleNamespace

import pytest

from equipo_jar.confirmacion import (
    near_any, worst_deviation, select_to_report, is_confirmed)


def test_deviation_round_ellipse():
    assert worst_deviation([0.01, 0.0, 0.0, 0.01]) == pytest.approx(0.1)


def test_deviation_ellipse_stretched_in_x():
    assert worst_deviation([0.04, 0.0, 0.0, 0.0025]) == pytest.approx(0.2)


def test_deviation_tilted_ellipse_is_larger_than_in_x_or_y():
    # Mirando solo a y d daría 0.1; en la diagonal (45°) el desvío es mayor.
    assert worst_deviation([0.01, 0.009, 0.009, 0.01]) == pytest.approx(math.sqrt(0.019))


def test_deviation_with_negative_variances_does_not_crash():
    # Un bug en fusión no tiene que tirar abajo el nodo de misión.
    assert worst_deviation([-0.01, 0.0, 0.0, -0.01]) == 0.0


def test_not_confirmed_with_few_observations():
    assert not is_confirmed(2, [0.0001, 0.0, 0.0, 0.0001], 5, 0.1)


def test_not_confirmed_if_too_uncertain():
    assert not is_confirmed(20, [0.04, 0.0, 0.0, 0.04], 5, 0.1)


def test_confirmed_with_enough_observations_and_precision():
    assert is_confirmed(20, [0.0025, 0.0, 0.0, 0.0025], 5, 0.1)


def test_confirmed_with_exactly_min_observations():
    assert is_confirmed(5, [0.0025, 0.0, 0.0, 0.0025], 5, 0.1)


# --- Paso 4: near_any y select_to_report ---

# Umbrales fijos para los tests de select_to_report.
MIN_OBS = 5
MAX_DEVIATION = 0.1
DIST_MIN = 0.4
PRECISE = [0.0025, 0.0, 0.0, 0.0025]   # desvío 0.05 m: pasa MAX_DESVIO


def cand(id_, x, y, observaciones=20, cov=PRECISE):
    """Candidata falsa con los mismos campos que equipo_jar_msgs/Candidata."""
    return SimpleNamespace(
        id=id_, centro=SimpleNamespace(x=x, y=y),
        observaciones=observaciones, covarianza_xy=cov)


def select(cands, reported_ids=None, reported_points=None, n=5):
    return select_to_report(
        cands, reported_ids or set(), reported_points or [], n,
        MIN_OBS, MAX_DEVIATION, DIST_MIN)


def ids(candidatas):
    return [c.id for c in candidatas]


def test_near_with_no_points_is_false():
    assert not near_any(1.0, 1.0, [], 0.4)


def test_near_within_distance():
    assert near_any(1.0, 1.0, [(5.0, 5.0), (1.3, 1.0)], 0.4)


def test_far_outside_distance():
    assert not near_any(1.0, 1.0, [(1.5, 1.0)], 0.4)


def test_selects_a_confirmed_one():
    assert ids(select([cand(1, 1.0, 1.0)])) == [1]


def test_does_not_select_an_unconfirmed_one():
    assert select([cand(1, 1.0, 1.0, observaciones=2)]) == []


def test_never_more_than_n():
    cands = [cand(1, 0.0, 0.0), cand(2, 2.0, 0.0), cand(3, 4.0, 0.0)]
    assert len(select(cands, n=2)) == 2


def test_counts_reports_already_used():
    cands = [cand(1, 0.0, 0.0), cand(2, 2.0, 0.0), cand(3, 4.0, 0.0)]
    assert len(select(cands, reported_points=[(9.0, 9.0)], n=3)) == 2


def test_no_reports_left_selects_nothing():
    assert select([cand(1, 0.0, 0.0)], reported_points=[(9.0, 9.0)], n=1) == []


def test_skips_reported_id():
    assert select([cand(1, 1.0, 1.0)], reported_ids={1}, reported_points=[(9.0, 9.0)]) == []


def test_skips_same_box_with_another_id():
    # Fusión la recreó con id nuevo, a 10 cm de un punto ya reportado.
    assert select([cand(7, 1.1, 1.0)], reported_ids={1}, reported_points=[(1.0, 1.0)]) == []


def test_does_not_select_same_box_twice_in_one_call():
    assert len(select([cand(1, 1.0, 1.0), cand(2, 1.1, 1.0)])) == 1


def test_with_one_report_selects_most_precise():
    less_precise = cand(1, 0.0, 0.0, cov=[0.0064, 0.0, 0.0, 0.0064])   # desvío 0.08
    more_precise = cand(2, 3.0, 0.0, cov=[0.0001, 0.0, 0.0, 0.0001])     # desvío 0.01
    assert ids(select([less_precise, more_precise], n=1)) == [2]


def test_does_not_modify_input_lists():
    reported_ids = {9}
    reported_points = [(9.0, 9.0)]
    select([cand(1, 0.0, 0.0), cand(2, 2.0, 0.0)], reported_ids, reported_points)
    assert reported_ids == {9}
    assert reported_points == [(9.0, 9.0)]


@pytest.mark.parametrize('x, y', [(math.nan, 1.0), (1.0, math.inf)])
def test_skips_non_finite_center(x, y):
    # armar_reporte rechaza NaN/inf con ValueError: elegirla tiraría abajo el nodo.
    assert select([cand(1, x, y)]) == []
