"""Tabla de posiciones: mejor corrida o suma, desempate por tiempo y penalizaciones (§4.1, §8.2)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ranking import armar, cargar, finalistas  # noqa: E402


def corrida(equipo, total, tiempo=100.0, ultimo=50.0, col=0, sal=0, anulada=False, viol=()):
    return {'equipo': equipo, 'termino_por': 'done', 'mapa_publicado': True, 'violaciones': list(viol),
            'tiempo_total_s': tiempo, 'tiempo_ultimo_reporte_que_puntua_s': ultimo,
            'puntaje': {'total': total, 'penalizacion_colisiones': col, 'penalizacion_salida': sal,
                        'anulada': anulada}}


def test_por_defecto_cuenta_la_mejor_corrida_de_cada_equipo():
    filas = armar([corrida('A', 300), corrida('A', 500), corrida('B', 400)])
    assert [(f['equipo'], f['puntos']) for f in filas] == [('A', 500), ('B', 400)]


def test_con_criterio_suma_se_suman_las_corridas():
    filas = armar([corrida('A', 300), corrida('A', 100), corrida('B', 350)], criterio='suma')
    assert [(f['equipo'], f['puntos']) for f in filas] == [('A', 400), ('B', 350)]


def test_por_defecto_gana_el_de_menos_tiempo_total():
    filas = armar([corrida('A', 400, tiempo=200), corrida('B', 400, tiempo=150)])
    assert [f['equipo'] for f in filas] == ['B', 'A']


def test_con_desempate_ultimo_reporte_cambia_el_orden():
    cs = [corrida('A', 400, tiempo=150, ultimo=120), corrida('B', 400, tiempo=200, ultimo=60)]
    assert [f['equipo'] for f in armar(cs)] == ['A', 'B']                                # por tiempo total
    assert [f['equipo'] for f in armar(cs, desempate='ultimo_reporte')] == ['B', 'A']    # por último reporte


def test_si_empatan_en_tiempo_desempatan_las_penalizaciones():
    filas = armar([corrida('A', 400, col=100), corrida('B', 400, col=50)])   # mismos tiempos
    assert [f['equipo'] for f in filas] == ['B', 'A']


def test_la_salida_del_laberinto_no_cuenta_en_el_desempate_por_penalizaciones():
    filas = armar([corrida('A', 400, col=50, sal=50), corrida('B', 400, col=100)])   # §8.2: solo colisiones
    assert [f['equipo'] for f in filas] == ['A', 'B']


def test_si_siguen_empatados_cuenta_la_otra_corrida_de_la_clasificacion():
    filas = armar([corrida('A', 400), corrida('A', 100), corrida('B', 400), corrida('B', 250)])
    assert [f['equipo'] for f in filas] == ['B', 'A']
    assert not filas[0]['empate']


def test_empate_total_comparte_posicion_y_queda_marcado():
    filas = armar([corrida('A', 400), corrida('B', 400), corrida('C', 100)])
    assert [f['posicion'] for f in filas] == [1, 1, 3]
    assert filas[0]['empate'] and filas[1]['empate'] and not filas[2]['empate']


def test_finalistas_avisa_si_el_empate_cruza_el_corte():
    filas = armar([corrida('A', 500), corrida('B', 400), corrida('C', 300), corrida('D', 300)])
    elegidos, aviso = finalistas(filas, 3)
    assert [f['equipo'] for f in elegidos] == ['A', 'B', 'C']
    assert aviso and 'C' in aviso and 'D' in aviso
    assert finalistas(armar([corrida('A', 500), corrida('B', 400)]), 3)[1] is None


def test_una_corrida_anulada_cuenta_con_cero_y_deja_un_aviso():
    filas = armar([corrida('A', 0, anulada=True), corrida('B', 100)])
    assert filas[1]['equipo'] == 'A' and 'corrida anulada' in filas[1]['avisos']


def test_las_violaciones_quedan_como_aviso_para_que_las_revise_un_juez():
    filas = armar([corrida('A', 400, viol=['movió el robot antes del mapa'])])
    assert 'violaciones (revisar)' in filas[0]['avisos']


def test_cargar_lee_los_resultados_y_saca_el_equipo_de_la_carpeta(tmp_path):
    for nombre, total in (('20261104_101500_los_osos', 300), ('20261104_103000_los_osos', 500)):
        d = tmp_path / nombre
        d.mkdir()
        c = corrida('', total)
        c.pop('equipo')
        (d / 'resultado.json').write_text(json.dumps(c))
    cs = cargar(tmp_path)
    assert {c['equipo'] for c in cs} == {'los_osos'}
    assert armar(cs)[0]['puntos'] == 500
