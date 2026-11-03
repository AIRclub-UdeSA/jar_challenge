"""Contratos M4: orientación, oclusión, evidencia y marcos del mapa."""
from dataclasses import replace
import math

import numpy as np
import pytest

from equipo_jar.cobertura_visual import CoberturaVisual, ParametrosCobertura, linea_libre
from equipo_jar.navegacion.grilla import Grilla


def mapa():
    return Grilla(.1, 0., 0., 0., 30, 30, np.zeros((30, 30), dtype=bool))


def test_planear_no_acredita_y_dar_la_espalda_cambia_la_vista():
    coverage = CoberturaVisual(mapa())
    east = coverage.predecir((1.55, 1.55, 0.))
    west = coverage.predecir((1.55, 1.55, math.pi))
    assert east.any() and west.any()
    assert not (east & west).any()
    assert not coverage.observada.any()


def test_camara_invalida_duplicada_o_atrasada_no_acredita():
    coverage = CoberturaVisual(mapa())
    pose = (1.55, 1.55, 0.)
    assert coverage.registrar(pose, 1., False) == 0
    assert coverage.registrar(pose, float('nan'), True) == 0
    assert coverage.registrar(pose, 1., True) > 0
    assert coverage.ganancia(pose) == 0
    before = coverage.observada.copy()
    assert coverage.registrar((1.55, 1.55, math.pi), 1., True) == 0
    assert coverage.registrar((1.55, 1.55, math.pi), .9, True) == 0
    assert np.array_equal(before, coverage.observada)
    assert coverage.registrar((1.55, 1.55, math.pi), 2., True) > 0


def test_pared_y_oclusiones_vivas_no_se_atraviesan():
    grid = mapa()
    coverage = CoberturaVisual(grid)
    wall = np.zeros_like(grid.ocupada)
    wall[:, 16] = True
    pose = (.55, 1.55, 0.)
    assert (coverage.puntos[coverage.predecir(pose), 0] > 1.7).any()
    assert not (coverage.puntos[coverage.predecir(pose, wall), 0] > 1.7).any()
    grid.ocupada[:, 16] = True
    assert not (coverage.puntos[coverage.predecir(pose), 0] > 1.7).any()


def test_no_corta_esquinas_ni_ve_fuera_del_mapa():
    grid = mapa()
    grid.ocupada[0, 1] = True
    assert not linea_libre(grid, (.05, .05), (.25, .25), grid.ocupada)
    assert not linea_libre(grid, (-.1, .05), (.25, .25), grid.ocupada)
    assert linea_libre(grid, (.25, .25), (.05, .25), grid.ocupada)


def test_rango_minimo_y_maximo_y_campo_visual():
    p = ParametrosCobertura(.5, 1., math.pi / 2, .1)
    coverage = CoberturaVisual(mapa(), p)
    pose = (1.05, 1.05, 0.)
    delta = coverage.puntos[coverage.predecir(pose)] - pose[:2]
    distance = np.linalg.norm(delta, axis=1)
    assert len(distance) > 0
    assert np.all(distance >= .5) and np.all(distance <= 1.)
    assert np.all(np.abs(np.arctan2(delta[:, 1], delta[:, 0])) <= math.pi / 4)


def test_mapa_con_origen_rotado_conserva_visibilidad():
    grid = mapa()
    a = CoberturaVisual(grid)
    rotated = replace(grid, origen_x=4., origen_y=-2., origen_yaw=math.pi / 2)
    b = CoberturaVisual(rotated)
    assert np.array_equal(a.predecir((1.55, 1.55, .2)),
                          b.predecir((4. - 1.55, -2. + 1.55, .2 + math.pi / 2)))


def test_mapa_ocupado_no_inventa_cobertura():
    grid = mapa()
    grid.ocupada[:] = True
    coverage = CoberturaVisual(grid)
    assert coverage.ganancia((1., 1., 0.)) == 0
    assert coverage.registrar((1., 1., 0.), 1., True) == 0


@pytest.mark.parametrize('kwargs', [dict(alcance_min_m=-1),
    dict(alcance_max_m=.05), dict(campo_horizontal_rad=0),
    dict(paso_muestreo_m=float('nan'))])
def test_parametros_invalidos(kwargs):
    with pytest.raises(ValueError):
        ParametrosCobertura(**kwargs)


def test_pose_y_capa_invalidas():
    coverage = CoberturaVisual(mapa())
    with pytest.raises(ValueError):
        coverage.predecir((0., 0., float('nan')))
    with pytest.raises(ValueError):
        coverage.predecir((0., 0., 0.), np.zeros((3, 3)))


def test_inspecciona_pared_cercana_sin_ver_a_traves():
    grid = mapa()
    grid.ocupada[:, 15:17] = True
    coverage = CoberturaVisual(grid)
    visible = coverage.predecir((1.15, 1.55, 0.))
    assert (visible & coverage.objetivos_pared).any()
    assert not (coverage.puntos[visible, 0] > 1.7).any()
    assert coverage.ganancia((1.15, 1.55, 0.)) > 0
    # RGB puede inspeccionar a <0,65 m; la confirmación RGB-D es otra política.
    assert np.any(np.linalg.norm(coverage.puntos[visible]-(1.15, 1.55), axis=1) < .65)
    before = int(np.count_nonzero(coverage.observada))
    assert coverage.registrar((1.15, 1.55, 0.), 1., True) > 0
    assert int(np.count_nonzero(coverage.observada)) > before
    assert coverage.ganancia((1.15, 1.55, 0.)) == 0


def test_rayos_vectorizados_equivalen_al_dda_conservador():
    from equipo_jar.cobertura_visual import lineas_libres
    rng = np.random.default_rng(734)
    grid = replace(mapa(), origen_x=-2., origen_y=.7, origen_yaw=.37)
    grid.ocupada[:] = rng.random(grid.ocupada.shape) < .12
    cells = [(f, c) for f in range(grid.alto) for c in range(grid.ancho)]
    targets = np.asarray([grid.a_mundo(*cell) for cell in cells])
    allowed = rng.random(len(cells)) < .5
    for cell in ((0, 0), (15, 15), (29, 29), (7, 19)):
        origin = grid.a_mundo(*cell)
        for offset in (0., .013):
            origin = (origin[0]+offset, origin[1]-offset)
            expected = [linea_libre(grid, origin, point, grid.ocupada, bool(allow))
                        for point, allow in zip(targets, allowed)]
            actual = lineas_libres(grid, origin, targets, grid.ocupada, allowed)
            assert np.array_equal(actual, expected)


def test_ver_pared_de_canto_no_acredita_inspeccion_frontal():
    grid = mapa()
    grid.ocupada[:, 15:17] = True
    coverage = CoberturaVisual(grid)
    faces = coverage.objetivos_pared & (coverage.normales[:, 0] < -.9)
    faces &= (coverage.puntos[:, 1] > 1.2) & (coverage.puntos[:, 1] < 1.9)
    assert faces.any()
    grazing = (1.15, .15, math.atan2(1.4, .4))
    assert not (coverage.predecir(grazing) & faces).any()
    assert (coverage.predecir((1.15, 1.55, 0.)) & faces).any()


def test_no_acredita_cara_opuesta_de_pared_delgada():
    grid = mapa()
    grid.ocupada[:, 15] = True
    coverage = CoberturaVisual(grid)
    rear = coverage.objetivos_pared & (coverage.normales[:, 0] > .9)
    assert rear.any()
    coverage.registrar((1.15, 1.55, 0.), 1., True)
    assert not coverage.observada[rear].any()
    assert (coverage.predecir((1.95, 1.55, math.pi)) & rear).any()


def test_cotas_en_lote_con_rayos_parciales_y_cambio_de_obstaculos():
    g=mapa();g.ocupada[8:22,16]=True
    cov=CoberturaVisual(g)
    poses=[(1.05,1.55,k*math.pi/8) for k in range(16)]
    for cambio in range(3):
        cotas=cov.cotas_ganancia(poses)
        exactas=np.array([cov.ganancia(p) for p in poses])
        assert np.all(cotas>=exactas)
        assert np.all(cov.cotas_ganancia(poses)>=exactas)
        if cambio==0:cov.registrar(poses[0],1.,True)
        else:g.ocupada[8:22,16]=False
