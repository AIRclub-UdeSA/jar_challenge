import math

from equipo_jar.politica_reportes import PoliticaReportes


def obj(x, y, n=5, confirmado=True):
    return {'x': x, 'y': y, 'n': n, 'confirmado': confirmado}


def test_reporta_confirmados_una_sola_vez():
    p = PoliticaReportes(3, 0.35)
    assert p.nuevos([obj(1, 1), obj(2, 2, confirmado=False)]) == [(1, 1)]
    # El mismo objetivo, un poco corrido: no se repite.
    assert p.nuevos([obj(1.1, 1.0)]) == []
    assert p.restantes == 2


def test_nunca_mas_de_los_cupos():
    p = PoliticaReportes(2, 0.35)
    hechos = p.nuevos([obj(0, 0, n=3), obj(1, 0, n=9), obj(2, 0, n=6)])
    assert hechos == [(1, 0), (2, 0)]   # los de más observaciones primero
    assert p.completo()
    assert p.nuevos([obj(5, 5)]) == []


def test_dos_confirmados_pegados_cuentan_como_uno():
    p = PoliticaReportes(3, 0.35)
    assert p.nuevos([obj(0, 0, n=8), obj(0.2, 0, n=5)]) == [(0, 0)]


def test_sobrantes_gasta_candidatos_sin_confirmar():
    p = PoliticaReportes(3, 0.35)
    p.nuevos([obj(0, 0)])
    candidatos = [obj(3, 3, n=1, confirmado=False), obj(0.1, 0, n=4, confirmado=False),
                  obj(4, 4, n=2, confirmado=False), obj(6, 6, n=0, confirmado=False)]
    assert p.sobrantes(candidatos) == [(4, 4), (3, 3)]
    assert p.completo()


def test_ignora_valores_no_finitos():
    p = PoliticaReportes(2, 0.35)
    assert p.nuevos([obj(math.nan, 0)]) == []
    assert p.restantes == 2
