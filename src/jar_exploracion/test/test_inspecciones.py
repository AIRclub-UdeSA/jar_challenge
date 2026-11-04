from jar_exploracion.inspecciones import Inspecciones


def cand(ident=1, estado='sin_inspeccionar', rojo=0, no_rojo=0):
    return {'id': ident, 'x': 1.0, 'y': 2.0, 'estado': estado, 'rojo': rojo, 'no_rojo': no_rojo}


def test_apagado_cada_visita_gasta_un_intento_aunque_no_vote():
    i = Inspecciones(intentos_max=2)
    i.visitado(1, [cand()], (0, 0))
    assert i.pendientes([cand()]) == {1: (1.0, 2.0)}
    i.visitado(1, [cand()], (0, 0))
    assert i.pendientes([cand()]) == {}
    assert i.dudosos([cand()]) == [1]
    assert i.revisar_dudosos([cand()]) == []       # sin seguir_dudosos no se revisa


def test_una_visita_sin_votos_no_gasta_el_intento_hasta_el_tope():
    i = Inspecciones(intentos_max=2, seguir_dudosos=True, visitas_sin_voto_max=2)
    i.visitado(1, [cand()], (0, 0))
    assert i.intentos.get(1, 0) == 0
    i.visitado(1, [cand()], (0, 0))                # dos seguidas sin votar: ahora sí cuenta
    assert i.intentos[1] == 1
    assert i.puntos_sin_decidir == []              # sin votos, el mismo punto sirve


def test_una_visita_con_votos_sin_decidir_gasta_el_intento_y_no_repite_el_punto():
    i = Inspecciones(intentos_max=2, seguir_dudosos=True)
    i.visitado(1, [cand(rojo=1, no_rojo=1)], (0.5, 0.5))
    assert i.intentos[1] == 1
    assert i.puntos_sin_decidir == [(0.5, 0.5)]
    # Los votos de antes no cuentan como nuevos en la visita siguiente.
    i.visitado(1, [cand(rojo=1, no_rojo=1)], (0.9, 0.5))
    assert i.intentos[1] == 1 and i.sin_voto[1] == 1


def test_un_candidato_que_se_decidio_cuenta_el_intento():
    i = Inspecciones(intentos_max=2, seguir_dudosos=True)
    i.visitado(1, [cand(estado='descartado', no_rojo=4)], (0, 0))
    assert i.intentos[1] == 1
    i.visitado(2, [], (0, 0))                       # borrado: también
    assert i.intentos[2] == 1


def test_los_dudosos_se_revisan_una_sola_vez_al_final():
    i = Inspecciones(intentos_max=2, seguir_dudosos=True)
    lista = [cand(1, rojo=1, no_rojo=1), cand(2, estado='descartado'), cand(3)]
    i.agotar(1)
    i.agotar(2)
    assert i.dudosos(lista) == [1]
    assert i.revisar_dudosos(lista) == [1]
    assert i.pendientes(lista) == {1: (1.0, 2.0), 3: (1.0, 2.0)}
    i.visitado(1, [cand(1, rojo=1, no_rojo=2)], (0, 0))
    assert i.dudosos(lista) == [1]
    assert i.revisar_dudosos(lista) == []           # ya se revisó


def test_uno_mas_devuelve_un_intento_solo_si_los_gasto():
    i = Inspecciones(intentos_max=2)
    i.uno_mas(1)
    assert i.intentos[1] == 0
    i.agotar(1)
    i.uno_mas(1)
    assert i.intentos[1] == 1


def test_un_descarte_lejos_del_candidato_no_gasta_el_intento_hasta_el_tope():
    i = Inspecciones(intentos_max=2, seguir_dudosos=True, descarte_lejos_m=1.0,
                     descartes_lejos_max=3)
    lista = [cand()]                                # en (1.0, 2.0)
    i.descartado(1, lista, (1.0, 0.0))              # a 2 m: el camino falló, no la mirada
    i.descartado(1, lista, (1.0, 0.0))
    assert i.intentos.get(1, 0) == 0
    i.descartado(1, lista, (1.0, 0.0))              # el tercero sí cuenta
    assert i.intentos[1] == 1
    i.descartado(1, lista, (1.0, 1.5))              # cerca: cuenta siempre
    assert i.intentos[1] == 2


def test_sin_seguir_dudosos_todo_descarte_gasta_el_intento():
    i = Inspecciones(intentos_max=2)
    i.descartado(1, [cand()], (1.0, 0.0))
    assert i.intentos[1] == 1


def test_un_dudoso_cerca_del_robot_se_mira_de_paso_una_vez():
    i = Inspecciones(intentos_max=2, seguir_dudosos=True, dudoso_de_paso_m=1.0)
    lista = [cand(1), cand(2, estado='descartado')]
    i.agotar(1)
    i.agotar(2)
    assert i.de_paso(lista, (5.0, 5.0)) == []       # lejos
    assert i.de_paso(lista, (1.0, 2.5)) == [1]      # el descartado no es dudoso
    assert i.pendientes(lista) == {1: (1.0, 2.0)}
    i.agotar(1)
    assert i.de_paso(lista, (1.0, 2.5)) == []       # una sola vez
    assert Inspecciones(seguir_dudosos=False).de_paso(lista, (1.0, 2.5)) == []
