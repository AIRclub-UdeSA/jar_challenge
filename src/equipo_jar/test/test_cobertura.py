import numpy as np
import pytest

from equipo_jar.cobertura import Cobertura
from equipo_jar.memoria_camara import ROJO, ROJO_VISTO, SIN_ROJO, SIN_VER

RES = 0.05


def sala(ancho_m=4.0, alto_m=3.0):
    """Una sala vacía con paredes alrededor; el origen del mapa es la esquina."""
    grilla = np.zeros((int(alto_m / RES), int(ancho_m / RES)), dtype=np.int16)
    grilla[0, :] = grilla[-1, :] = 100
    grilla[:, 0] = grilla[:, -1] = 100
    return grilla


def cobertura(grilla, largada=(2.0, 1.5), despeje_meta=0.30):
    return Cobertura(grilla, RES, 0.0, 0.0, largada, 0.17, despeje_meta)


def ver_objeto(c, wx, wy, veces=10):
    """Simula ver un objeto desde dos lados distintos."""
    for pose in ((0.5, 1.5, 0.0), (3.5, 1.5, 0.0)):
        for _ in range(veces):
            c.acumular_lidar(pose, np.array(wx), np.array(wy))


def caja(cx, cy, lado=0.2):
    s = np.arange(-lado / 2, lado / 2, RES)
    xx, yy = np.meshgrid(cx + s, cy + s)
    return xx.ravel().tolist(), yy.ravel().tolist()


def test_celda_y_xy_son_inversas():
    c = cobertura(sala())
    assert c.xy(*c.celda(1.5, 2.0)) == pytest.approx((1.5, 2.0))


def test_los_rayos_no_atraviesan_paredes():
    grilla = sala()
    grilla[:, int(2.5 / RES)] = 100     # pared que parte la sala en dos
    c = cobertura(grilla)
    visibles = c.rayos(1.0, 1.5, 3.0)
    assert visibles[c.celda(2.0, 1.5)]
    assert not visibles[c.celda(3.0, 1.5)]


def test_meta_de_cobertura_respeta_el_despeje():
    c = cobertura(sala())
    _, x, y = c.meta_cobertura(c.distancias_desde(2.0, 1.5))
    assert c.despeje_en(x, y) >= 0.30


def test_todo_visto_no_hay_meta():
    c = cobertura(sala())
    c.visto[:] = True
    assert c.meta_cobertura(c.distancias_desde(2.0, 1.5)) is None


def test_lista_negra_saca_al_candidato():
    c = cobertura(sala())
    distancias = c.distancias_desde(2.0, 1.5)
    k, _, _ = c.meta_cobertura(distancias)
    c.lista_negra.add(k)
    assert c.meta_cobertura(distancias)[0] != k


def test_objeto_visto_de_dos_lados_se_confirma():
    c = cobertura(sala())
    ver_objeto(c, *caja(2.0, 1.5))
    objetos = c.objetos_nuevos()
    assert len(objetos) == 1
    assert objetos[0] == pytest.approx((2.0, 1.5), abs=0.1)
    assert c.despeje_en(2.0, 1.5) == 0.0


def test_objeto_visto_de_un_solo_lado_no_se_confirma():
    c = cobertura(sala())
    wx, wy = caja(2.0, 1.5)
    for _ in range(10):
        c.acumular_lidar((0.5, 1.5, 0.0), np.array(wx), np.array(wy))
    assert c.objetos_nuevos() == []


def test_algo_largo_no_es_un_objeto():
    c = cobertura(sala())
    ver_objeto(c, *caja(2.0, 1.5, lado=0.8))
    assert c.objetos_nuevos() == []


def test_lo_pegado_a_una_pared_del_mapa_no_cuenta():
    c = cobertura(sala())
    ver_objeto(c, *caja(2.0, 0.15))
    assert c.objetos_nuevos() == []


def test_meta_de_inspeccion_queda_lejos_y_mira_al_objeto():
    c = cobertura(sala())
    ver_objeto(c, *caja(2.0, 1.5))
    c.objetos_nuevos()
    x, y, yaw = c.meta_inspeccion(2.0, 1.5, c.distancias_desde(0.6, 1.5))
    assert np.hypot(x - 2.0, y - 1.5) >= 0.8
    assert c.despeje_en(x, y) >= 0.30
    assert np.cos(yaw) * (2.0 - x) + np.sin(yaw) * (1.5 - y) > 0.0


def test_sin_lugar_con_despeje_no_hay_inspeccion():
    c = cobertura(sala(), despeje_meta=2.0)
    assert c.meta_inspeccion(2.0, 1.5, c.distancias_desde(2.0, 1.5)) is None


def test_arranca_aunque_el_robot_este_pegado_a_una_pared():
    c = cobertura(sala())
    assert len(c.distancias_desde(0.1, 1.5)) > 100


def test_olvidar_objetos_no_recupera_metas_abandonadas():
    c = cobertura(sala())
    c.lista_negra.add(3)
    ver_objeto(c, *caja(2.0, 1.5))
    c.objetos_nuevos()
    c.olvidar_objetos()
    assert not c.nuevos.any()
    assert c.lista_negra == {3}


def test_fase_fina_vuelve_a_recorrer():
    c = cobertura(sala())
    c.marcar_visto(2.0, 1.5)
    assert c.porcentaje_visto() > 0.0
    c.pasar_a_fase_fina()
    assert c.porcentaje_visto() == 0.0
    assert c.fase == 2


def test_el_margen_contra_las_paredes_se_puede_cambiar():
    # Un objeto a 0,25-0,40 m de la pared de abajo: con el margen de 0,20 es un objeto; con
    # 0,40 m, todo lo que ve el lidar de él queda dentro del margen.
    normal = cobertura(sala())
    ver_objeto(normal, *caja(2.0, 0.35))
    assert len(normal.objetos_nuevos()) == 1
    amplio = Cobertura(sala(), RES, 0.0, 0.0, (2.0, 1.5), 0.17, 0.30, margen_pared_m=0.40)
    ver_objeto(amplio, *caja(2.0, 0.35))
    assert amplio.objetos_nuevos() == []


def camara_vacia(c):
    return np.zeros((c.alto, c.ancho), dtype=np.int16)


def test_lo_que_vio_la_camara_de_un_objeto():
    c = cobertura(sala())
    ver_objeto(c, *caja(2.0, 1.5))
    c.objetos_nuevos()
    camara = camara_vacia(c)
    assert c.veredicto_camara(2.0, 1.5, camara, 3, 3) == SIN_VER
    camara[c.cuenta > 0] = 5                # la cámara vio el objeto en 5 imágenes, sin rojo
    assert c.veredicto_camara(2.0, 1.5, camara, 3, 3) == SIN_ROJO
    i, j = c.celda(2.0, 1.5)
    camara[i, j] = ROJO                     # en una imagen vio rojo
    assert c.veredicto_camara(2.0, 1.5, camara, 3, 3) == ROJO_VISTO


def test_la_camara_viendo_solo_paredes_no_descarta_al_objeto():
    c = cobertura(sala())
    ver_objeto(c, *caja(2.0, 1.5))
    camara = camara_vacia(c)
    camara[c.ocupado] = 50                  # vio todas las paredes, nunca el objeto
    assert c.veredicto_camara(2.0, 1.5, camara, 3, 3) == SIN_VER


@pytest.mark.parametrize('x, y', [(-1.0, -1.0), (10.0, 10.0), (0.02, 0.02)])
def test_veredicto_de_la_camara_en_los_bordes_del_mapa(x, y):
    c = cobertura(sala())
    assert c.veredicto_camara(x, y, camara_vacia(c), 3, 3) == SIN_VER
