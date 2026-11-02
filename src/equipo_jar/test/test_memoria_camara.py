import numpy as np

from equipo_jar.memoria_camara import (
    MAX_VECES, ROJO, ROJO_VISTO, SIN_ROJO, SIN_VER, MemoriaCamara, veredicto)


def grilla(memoria):
    return memoria.datos().reshape(memoria.alto, memoria.ancho)


def test_cada_imagen_suma_una_vez_por_celda():
    m = MemoriaCamara(4, 5)
    m.anotar([1, 1, 2], [3, 3, 0], [False, False, False])  # dos rayos en la misma celda
    g = grilla(m)
    assert g[1, 3] == 1 and g[2, 0] == 1 and g.sum() == 2
    m.anotar([1], [3], [False])
    assert grilla(m)[1, 3] == 2


def test_las_veces_se_saturan():
    m = MemoriaCamara(2, 2)
    for _ in range(150):
        m.anotar([0], [0], [False])
    assert grilla(m)[0, 0] == MAX_VECES


def test_el_rojo_queda_para_siempre():
    m = MemoriaCamara(2, 2)
    m.anotar([0], [1], [True])
    for _ in range(10):
        m.anotar([0], [1], [False])
    assert grilla(m)[0, 1] == ROJO


def test_datos_en_el_orden_de_occupancy_grid():
    m = MemoriaCamara(3, 4)
    m.anotar([2], [1], [False])
    datos = m.datos()
    assert datos.dtype == np.int8 and len(datos) == 12
    assert datos[2 * 4 + 1] == 1  # fila 2 (y), columna 1 (x)


def objeto_en(celdas, alto=10, ancho=10):
    mascara = np.zeros((alto, ancho), dtype=bool)
    for f, c in celdas:
        mascara[f, c] = True
    return mascara


OBJETO = objeto_en([(5, 4), (5, 5), (5, 6)])


def test_sin_vistas_no_se_sabe():
    assert veredicto(np.zeros((10, 10)), OBJETO, 3, 3) == SIN_VER


def test_visto_varias_veces_sin_rojo():
    vistas = np.zeros((10, 10), dtype=int)
    vistas[5, 4:7] = 3
    assert veredicto(vistas, OBJETO, 3, 3) == SIN_ROJO


def test_pocas_veces_o_pocas_celdas_no_alcanza():
    vistas = np.zeros((10, 10), dtype=int)
    vistas[5, 4:7] = 2                      # cada celda, 2 imágenes: hacen falta 3
    assert veredicto(vistas, OBJETO, 3, 3) == SIN_VER
    vistas[5, 4:6] = 5                      # solo 2 celdas con 3 o más
    assert veredicto(vistas, OBJETO, 3, 3) == SIN_VER


def test_un_rojo_pegado_al_objeto_alcanza():
    vistas = np.zeros((10, 10), dtype=int)
    vistas[5, 4:7] = 20
    vistas[6, 7] = ROJO                     # en diagonal a una celda del objeto
    assert veredicto(vistas, OBJETO, 3, 3) == ROJO_VISTO


def test_lo_visto_lejos_del_objeto_no_cuenta():
    vistas = np.zeros((10, 10), dtype=int)
    vistas[0:2, :] = 30                     # una pared bien vista, a 3 celdas del objeto
    assert veredicto(vistas, OBJETO, 3, 3) == SIN_VER


def test_un_objeto_sin_celdas_no_se_sabe():
    assert veredicto(np.full((10, 10), 50), np.zeros((10, 10), bool), 3, 3) == SIN_VER
