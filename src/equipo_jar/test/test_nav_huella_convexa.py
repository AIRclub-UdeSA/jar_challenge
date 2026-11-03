"""Contención física y heurística equivalente; sin geometría de escenarios."""
import math
import numpy as np
from equipo_jar.navegacion.guardia import huella_convexa_x3
from equipo_jar.navegacion.grilla import Grilla
from equipo_jar.navegacion.orientado import _heuristica_grilla
from equipo_jar.navegacion.planificador import _dijkstra

def test_huella_contiene_chasis_y_circulos_completos_de_ruedas():
    footprint=huella_convexa_x3(-.19,.13,.07,.08,.0845,.033,.1)
    polygon=np.asarray(footprint['corners_base_m'])
    for a,b in zip(polygon,np.roll(polygon,-1,axis=0)):
        edge=b-a;normal=np.array([-edge[1],edge[0]])/np.linalg.norm(edge)
        for x in (-.181,.119):
            for y in (-.0693,.0693):
                assert normal@ (np.array([x,y])-a)>=-1e-12
        for x in (-.08,.08):
            for y in (-.0845,.0845):
                assert normal@(np.array([x,y])-a)>=.0325-1e-12
    assert footprint['radius_m']<.21
    assert footprint['margin_m']==.1

def test_heuristica_nativa_equivale_a_dijkstra_sin_atajos_diagonales():
    rng=np.random.default_rng(923)
    for _ in range(5):
        occupied=rng.random((23,29))<.15;occupied[5,5]=False
        grid=Grilla(.05,0.,0.,0.,29,23,occupied)
        expected,_,_=_dijkstra(grid,np.where(occupied,np.inf,0.),(5,5))
        got=_heuristica_grilla(~occupied,(5,5))
        assert np.array_equal(np.isinf(expected),np.isinf(got))
        finite=np.isfinite(expected)
        assert np.allclose(expected[finite],got[finite],atol=1e-10)

def test_heuristica_nativa_no_cruza_esquina_cerrada():
    got=_heuristica_grilla(np.array([[True,False],[False,True]]),(0,0))
    assert math.isinf(got[1,1])


def test_huella_contiene_ruedas_visuales_en_todas_las_rotaciones():
    footprint=huella_convexa_x3(-.19,.13,.07,.08,.0845,.033,.1,.018)
    polygon=np.asarray(footprint['corners_base_m'])
    for a,b in zip(polygon,np.roll(polygon,-1,axis=0)):
        edge=b-a;normal=np.array([-edge[1],edge[0]])/np.linalg.norm(edge)
        for x in (-.08,.08):
            for y in (-.0845,.0845):
                for dx in (-.0325,.0325):
                    for dy in (-.0176,.0176):
                        assert normal@(np.array([x+dx,y+dy])-a)>=-1e-12


def test_margen_navegacion_mantiene_frenado_y_detiene_una_invasion():
    from equipo_jar.navegacion.guardia import limitar_velocidad
    footprint=huella_convexa_x3(-.19,.13,.07,.08,.0845,.033,.075,.018)
    points=[(.203,0.)]*30  # 7,3 cm del frente: dentro del margen de 7,5 cm.
    for vx,vy,w in ((.1,0.,0.),(.1,.05,.2),(0.,0.,.2)):
        x,y,diag=limitar_velocidad(points,vx,vy,footprint,.5,.15,wz=w)
        assert x==y==0. and diag['bloqueado']


def test_heuristica_se2_es_optimista_y_el_barrido_decide_la_colision():
    free=np.array([[True,False],[False,True]])
    relaxed=_heuristica_grilla(free,(0,0),permitir_diagonales_optimistas=True)
    assert relaxed[1,1]==math.sqrt(2)
    assert math.isinf(_heuristica_grilla(free,(0,0))[1,1])


def test_filtro_kernel_equivale_a_distancias_completas_incluso_en_bordes():
    from equipo_jar.navegacion.guardia import poligono_barrido,distancias_poligono
    from equipo_jar.navegacion.orientado import _colision_con_margen
    footprint=huella_convexa_x3(-.19,.13,.07,.08,.0845,.033,.075,.018)
    rng=np.random.default_rng(709)
    for angle in (0.,.4,1.3,-2.1):
        hull,error=poligono_barrido(footprint,angle,.175)
        for margin in (.075+error,.11):
            points=np.vstack([rng.uniform(-.8,.8,(1200,2)),np.array(hull),
                              np.array(hull)+np.array([margin,0.])])
            expected=distancias_poligono(points,hull)<=margin
            assert np.array_equal(_colision_con_margen(points,hull,margin),expected)
