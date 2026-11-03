import numpy as np

from equipo_jar.detector_scan import filtrar_por_profundidad


def test_descarta_lo_que_paso_por_el_borde_hacia_atras():
    # Cara de la caja a ~1 m (rayos 3-6) y, pegados al borde, dos rayos que pegaron en la
    # pared de atrás a 2,5 m (2 y 7): todos caen sobre píxeles rojos.
    rangos = np.array([3.0, 3.0, 2.5, 1.0, 0.98, 1.02, 1.05, 2.5, 3.0, 3.0])
    es_rojo = np.zeros(10, dtype=bool)
    es_rojo[2:8] = True
    filtrado = filtrar_por_profundidad(rangos, es_rojo, 0.3)
    assert list(np.flatnonzero(filtrado)) == [3, 4, 5, 6]


def test_cada_grupo_se_evalua_por_separado():
    # Dos cajas a distinta distancia: ninguna se descarta por estar lejos de la otra.
    rangos = np.array([1.0, 1.05, 5.0, 2.0, 2.1, 5.0])
    es_rojo = np.array([True, True, False, True, True, False])
    assert list(filtrar_por_profundidad(rangos, es_rojo, 0.3)) == list(es_rojo)


def test_un_grupo_que_cruza_el_principio_y_el_final_del_scan_es_uno_solo():
    # Con el lidar a 180°, el rayo 0 y el último quedan adelante del robot: son vecinos.
    rangos = np.array([1.0, 2.6, 5.0, 5.0, 5.0, 0.95])
    es_rojo = np.array([True, True, False, False, False, True])
    # El rayo 1 está 1,65 m detrás del más cercano del grupo (el 5, del otro extremo).
    assert list(np.flatnonzero(filtrar_por_profundidad(rangos, es_rojo, 0.3))) == [0, 5]


def test_con_tolerancia_cero_no_filtra_y_sin_rojo_no_falla():
    rangos = np.array([1.0, 3.0])
    es_rojo = np.array([True, True])
    assert list(filtrar_por_profundidad(rangos, es_rojo, 0.0)) == [True, True]
    assert not filtrar_por_profundidad(rangos, np.zeros(2, dtype=bool), 0.3).any()
