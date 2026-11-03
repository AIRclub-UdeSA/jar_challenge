"""Watch on AMCL: when to move it to a better pose found with the LiDAR."""
from equipo_jar.vigia import Vigia

P = {'vigia_puntaje_bajo': 0.5, 'vigia_puntaje_bueno': 0.7, 'vigia_mejora_minima': 0.2,
     'vigia_distancia_minima_m': 0.10, 'vigia_coincidencia_m': 0.05, 'vigia_confirmaciones': 2}
AMCL = (1.0, 2.0, 0.5)


def test_searches_only_when_the_score_is_low():
    v = Vigia(P)
    assert not v.hace_falta_buscar(0.85)
    assert v.hace_falta_buscar(0.3)


def test_moves_amcl_after_two_checks_at_the_same_place():
    v = Vigia(P)
    assert v.revisar(AMCL, 0.2, (1.3, 2.0, 0.5, 0.85)) is None
    assert v.revisar(AMCL, 0.2, (1.32, 2.01, 0.5, 0.86)) == (1.32, 2.01, 0.5)
    assert v.reubicaciones == 1


def test_does_not_move_for_a_small_or_unclear_gain():
    v = Vigia(P)
    for _ in range(3):
        assert v.revisar(AMCL, 0.45, (1.3, 2.0, 0.5, 0.6)) is None      # not good enough
        assert v.revisar(AMCL, 0.6, (1.3, 2.0, 0.5, 0.75)) is None      # gain < 0.2
        assert v.revisar(AMCL, 0.2, (1.03, 2.0, 0.5, 0.85)) is None     # same place
    assert v.reubicaciones == 0


def test_a_different_place_each_time_restarts_the_count():
    v = Vigia(P)
    assert v.revisar(AMCL, 0.2, (1.3, 2.0, 0.5, 0.85)) is None
    assert v.revisar(AMCL, 0.2, (0.7, 2.0, 0.5, 0.85)) is None
    assert v.revisar(AMCL, 0.2, (0.71, 2.0, 0.5, 0.85)) is not None


def test_driving_between_checks_forgets_the_candidate():
    v = Vigia(P)
    assert v.revisar(AMCL, 0.2, (1.3, 2.0, 0.5, 0.85)) is None
    v.descartar()
    assert v.revisar(AMCL, 0.2, (1.3, 2.0, 0.5, 0.85)) is None


def test_a_heading_error_alone_counts():
    v = Vigia(P)
    assert v.revisar(AMCL, 0.3, (1.0, 2.0, 0.7, 0.85)) is None
    assert v.revisar(AMCL, 0.3, (1.0, 2.0, 0.7, 0.85)) == (1.0, 2.0, 0.7)
