import pytest

from equipo_jar.seguimiento import Seguidor, cara_a_centro


def test_cara_a_centro_corre_alejandose_del_robot():
    # Robot en el origen mirando a +x, cara de la caja a 1 m: el centro queda 0,125 m más lejos.
    assert cara_a_centro(1.0, 0.0, 0.0, 0.0, 0.125) == pytest.approx((1.125, 0.0))
    # Desde la diagonal, el corrimiento sigue la recta robot -> objeto.
    x, y = cara_a_centro(1.0, 1.0, 0.0, 0.0, 0.125)
    assert x == pytest.approx(1.0 + 0.125 / 2 ** 0.5)
    assert y == pytest.approx(1.0 + 0.125 / 2 ** 0.5)


def test_cara_a_centro_con_robot_encima_no_divide_por_cero():
    assert cara_a_centro(1.0, 2.0, 1.0, 2.0, 0.125) == (1.0, 2.0)


def test_confirma_una_sola_vez_al_llegar_al_minimo():
    s = Seguidor(radio_asociacion=0.3, observaciones_minimas=3, tiempo_olvido=5.0)
    resultados = [s.observar(1.0 + 0.01 * i, 2.0, t=i) for i in range(6)]
    assert [r is not None for r in resultados] == [False, False, True, False, False, False]
    assert len(s.tracks) == 1


def test_promedia_las_observaciones():
    s = Seguidor(0.3, 10, 5.0)
    for x in (0.9, 1.0, 1.1):
        s.observar(x, 2.0, t=0)
    assert s.tracks[0].x == pytest.approx(1.0)
    assert s.tracks[0].observaciones == 3


def test_cajas_separadas_son_tracks_distintos():
    s = Seguidor(0.3, 10, 5.0)
    s.observar(0.0, 0.0, t=0)
    s.observar(1.0, 0.0, t=0)
    s.observar(0.05, 0.0, t=0)
    assert len(s.tracks) == 2
    assert sorted(tr.observaciones for tr in s.tracks) == [1, 2]


def test_olvida_candidatos_viejos_pero_no_confirmados():
    s = Seguidor(0.3, 2, tiempo_olvido=5.0)
    s.observar(0.0, 0.0, t=0)          # ruido: una sola vez
    s.observar(3.0, 0.0, t=0)
    s.observar(3.0, 0.0, t=1)          # confirmado
    s.olvidar(t=10)
    assert [(tr.x, tr.confirmado) for tr in s.tracks] == [(3.0, True)]


def test_una_caja_ya_confirmada_no_genera_otro_reporte():
    s = Seguidor(0.3, 2, 5.0)
    s.observar(1.0, 1.0, t=0)
    assert s.observar(1.0, 1.0, t=1) is not None
    # Volver a pasar por la misma caja mucho después no crea un track nuevo.
    assert all(s.observar(1.02, 0.98, t=100 + i) is None for i in range(5))
    assert len(s.tracks) == 1
