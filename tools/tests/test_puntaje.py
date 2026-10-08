"""Tests del puntaje: cada caso sale de una frase del reglamento (§7.2, §8)."""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from puntaje import Corrida, Reporte, puntuar  # noqa: E402

# Cinco víctimas separadas por más de 1,0 m (§6.1).
V = [(0.0, 0.0), (2.0, 0.0), (4.0, 0.0), (0.0, 2.0), (2.0, 2.0)]


def r(t, x, y, valido=True):
    return Reporte(t=t, x=x, y=y, valido=valido)


def exactos(t=60.0):
    """Un reporte perfecto para cada víctima."""
    return [r(t + 10 * i, x, y) for i, (x, y) in enumerate(V)]


def test_cinco_exactos_dan_500_y_el_bono_de_200():
    res = puntuar(Corrida(V, exactos()))
    assert res.puntos_victimas == 500 and res.bono_rapido and res.total == 700


def test_radios_100_50_0():
    reps = [r(1, 0.25, 0), r(2, 2.30, 0), r(3, 4.51, 0)]   # 0,25 m / 0,30 m / 0,51 m
    res = puntuar(Corrida(V, reps))
    assert [p.puntos for p in res.por_victima[:3]] == [100, 50, 0]


def test_borde_de_0_50_puntua_50_y_de_0_25_puntua_100():
    res = puntuar(Corrida(V, [r(1, 0.5, 0), r(2, 2.0, 0.25)]))
    assert [p.puntos for p in res.por_victima[:2]] == [50, 100]


def test_reporte_a_mas_de_0_5_no_puntua_pero_gasta_un_reporte():
    reps = [r(1, 0.6, 0)] + exactos()[1:] + [r(90, 0.0, 0.0)]   # 1 malo + 4 buenos + 1 extra
    res = puntuar(Corrida(V, reps))
    assert res.por_victima[0].puntos == 0
    assert res.reportes_contados == 5 and res.reportes_ignorados == 1


def test_solo_se_cuentan_N_reportes_los_demas_se_ignoran():
    reps = [r(i, 100 + i, 100) for i in range(5)] + exactos()   # 5 malos y después 5 buenos
    res = puntuar(Corrida(V, reps))
    assert res.puntos_victimas == 0 and res.reportes_ignorados == 5


def test_repetir_una_victima_gasta_reportes_y_solo_cuenta_el_mas_cercano():
    reps = [r(10, 0.4, 0), r(20, 0.1, 0), r(30, 0.3, 0)]        # tres veces la víctima 1
    res = puntuar(Corrida(V, reps))
    assert res.por_victima[0].puntos == 100
    assert res.por_victima[0].indice_reporte == 1
    assert res.puntos_victimas == 100


def test_repetir_no_permite_el_bono_porque_faltan_reportes_para_otras_victimas():
    reps = [r(10, 0.0, 0.0), r(20, 0.0, 0.0)] + exactos()[1:4]  # 2 sobre V1, 1 sobre V2..V4, ninguna V5
    res = puntuar(Corrida(V, reps))
    assert not res.bono_rapido


def test_reporte_mal_formado_gasta_un_reporte_y_no_puntua():
    reps = [r(5, 0.0, 0.0, valido=False)] + exactos()[1:]
    res = puntuar(Corrida(V, reps))
    assert res.por_victima[0].puntos == 0 and res.reportes_contados == 5
    assert not res.bono_rapido


def test_bono_de_5_minutos_es_estricto_antes_de_300_segundos():
    justo = exactos(); justo[-1] = r(299.9, *V[-1])
    tarde = exactos(); tarde[-1] = r(300.0, *V[-1])
    assert puntuar(Corrida(V, justo)).bono_rapido
    assert not puntuar(Corrida(V, tarde)).bono_rapido


def test_un_solo_reporte_tarde_pierde_el_bono_pero_conserva_sus_puntos():
    reps = exactos(); reps[2] = r(400.0, *V[2])
    res = puntuar(Corrida(V, reps))
    assert not res.bono_rapido and res.puntos_victimas == 500 and res.total == 500


def test_bono_de_regreso_pide_done_cerca_de_la_largada_y_una_victima_puntuada():
    reps = [r(50, 0.0, 0.0)]
    base = dict(victimas=V, reportes=reps, hizo_done=True)
    assert puntuar(Corrida(**base, distancia_a_largada_en_done=0.25)).bono_regreso
    assert not puntuar(Corrida(**base, distancia_a_largada_en_done=0.26)).bono_regreso
    assert not puntuar(Corrida(V, [], hizo_done=True, distancia_a_largada_en_done=0.0)).bono_regreso
    assert not puntuar(Corrida(V, reps, hizo_done=False, distancia_a_largada_en_done=0.0)).bono_regreso


def test_colisiones_y_salida_restan_y_el_minimo_es_cero():
    res = puntuar(Corrida(V, [r(1, 0.0, 0.0)], colisiones=2, salio_del_laberinto=True))
    assert res.total == 0                      # 100 - 100 - 50 -> mínimo 0
    assert res.penalizacion_colisiones == 100 and res.penalizacion_salida == 50


def test_sin_reportes_el_puntaje_es_cero():
    assert puntuar(Corrida(V, [])).total == 0


def test_un_reporte_entre_dos_victimas_se_asigna_a_la_mas_cercana_y_a_una_sola():
    cerca = [(0.0, 0.0), (0.9, 0.0)]           # dos víctimas a 0,9 m: caso que el reglamento evita
    res = puntuar(Corrida(cerca, [r(1, 0.45, 0.0), r(2, 0.5, 0.0)]))
    # El primer reporte (a 0,45 de ambas) va a la víctima 1; el segundo, a 0,4 de la 2, es de la 2.
    assert [p.indice_reporte for p in res.por_victima] == [0, 1]


def test_el_resumen_se_puede_imprimir():
    assert 'total' in puntuar(Corrida(V, exactos())).resumen()


def test_el_veredicto_de_un_juez_manda_sobre_la_distancia():
    reps = [r(50, 0.0, 0.0)]
    si = Corrida(V, reps, hizo_done=True, regreso_confirmado=True)
    no = Corrida(V, reps, hizo_done=True, regreso_confirmado=False,
                 distancia_a_largada_en_done=0.0)
    assert puntuar(si).bono_regreso
    assert not puntuar(no).bono_regreso
    # Igual necesita /done y una víctima puntuada.
    assert not puntuar(Corrida(V, reps, hizo_done=False, regreso_confirmado=True)).bono_regreso
    assert not puntuar(Corrida(V, [], hizo_done=True, regreso_confirmado=True)).bono_regreso


def test_una_corrida_anulada_por_vuelco_queda_en_cero_pero_conserva_el_desglose():
    res = puntuar(Corrida(V, exactos(), anulada=True))
    assert res.total == 0 and res.anulada and res.puntos_victimas == 500


def test_tiempo_del_ultimo_reporte_que_puntua_ignora_los_que_no_puntuan():
    reps = [r(10, 0.0, 0.0), r(40, 2.0, 0.0), r(200, 9.0, 9.0)]   # el último no puntúa
    assert puntuar(Corrida(V, reps)).tiempo_ultimo_reporte_que_puntua == 40
    assert puntuar(Corrida(V, [r(5, 9, 9)])).tiempo_ultimo_reporte_que_puntua is None
