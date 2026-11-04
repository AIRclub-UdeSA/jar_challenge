"""Vistas desde datos recibidos; sin fixtures de competencia."""
import math
from dataclasses import replace
import numpy as np
import pytest
from equipo_jar.navegacion.grilla import Grilla
from equipo_jar.vistas_percepcion import espacio_libre, proponer_vistas

def test_vistas_revelan_otra_cara_sin_confundir_superficie_con_centro():
    g=Grilla(.05,-3.,-3.,0.,120,120,np.zeros((120,120),bool))
    s=dict(superficie_x=.9,superficie_y=.2,normal_x=-1.,normal_y=0.)
    views=proponer_vistas([s],(0.,0.),g,espacio_libre(g))
    assert views
    for v in views:
        delta=np.array([v['x']-.9,v['y']-.2])
        assert math.dist((v['x'],v['y']),(.9,.2))>=.89
        assert abs(delta[1])>.38
        assert -delta[0]>0
        assert math.isclose(math.atan2(.2-v['y'],.9-v['x']),v['yaw'])

def test_no_propone_en_mapa_ocupado_ni_al_otro_lado_de_pared():
    occ=np.ones((80,80),bool)
    g=Grilla(.05,-2.,-2.,0.,80,80,occ)
    s=dict(superficie_x=.9,superficie_y=.2,normal_x=-1.,normal_y=0.)
    assert not proponer_vistas([s],(0.,0.),g,espacio_libre(g))
    occ[:]=False;occ[:,35:40]=True
    v=proponer_vistas([s],(-1.,0.),g,espacio_libre(g))
    assert all(x['x']>0 for x in v) # no ray crosses wall x[-.25,0]

def test_normal_invalida_y_distancia_invalida():
    g=Grilla(.1,-3.,-3.,0.,60,60,np.zeros((60,60),bool))
    s=dict(superficie_x=1.,superficie_y=0.,normal_x=0.,normal_y=0.)
    assert not proponer_vistas([s],(0.,0.),g,espacio_libre(g))
    with pytest.raises(ValueError):
        proponer_vistas([], (0,0),g,espacio_libre(g),separacion=0)

def test_origen_rotado_conserva_vistas_y_despeje():
    g=Grilla(.05,-3.,-3.,0.,120,120,np.zeros((120,120),bool))
    s=dict(superficie_x=.9,superficie_y=.2,normal_x=-1.,normal_y=0.)
    a=proponer_vistas([s],(0.,0.),g,espacio_libre(g))
    rotated=replace(g,origen_x=3.,origen_y=-3.,origen_yaw=math.pi/2)
    b=proponer_vistas([s],(0.,0.),rotated,espacio_libre(rotated))
    assert a==b


def test_abanico_estrecho_propone_sin_bajar_margen_del_planificador():
    occupied = np.ones((80, 120), bool)
    occupied[28:52, 4:116] = False
    grid = Grilla(.025, -1., -1., 0., 120, 80, occupied)
    surface = dict(superficie_x=1.5, superficie_y=0., normal_x=-1., normal_y=0.)
    distance = espacio_libre(grid)
    assert not proponer_vistas([surface], (0., 0.), grid, distance)
    views = proponer_vistas([surface], (0., 0.), grid, distance,
                           despeje=.16, angulos=(-math.pi/36, math.pi/36))
    assert views
    assert all(distance[grid.a_celda(v['x'], v['y'])] > .16 for v in views)
    assert all(0 < abs(v['y']) < .2 for v in views)


def test_superficie_medida_permite_sondeo_de_vista_si_mapa_cierra_hueco():
    g = Grilla(.05, -3., -3., 0., 120, 120, np.zeros((120, 120), bool))
    g.ocupada[:, 60] = True
    surface = dict(superficie_x=.6, superficie_y=0., normal_x=-1., normal_y=0.)
    kwargs = dict(despeje=.2, angulos=(-math.pi/36, math.pi/36))
    assert not proponer_vistas([surface], (-1., 0.), g, espacio_libre(g), **kwargs)
    views = proponer_vistas([surface], (-1., 0.), g, espacio_libre(g),
                           comprobar_vision=False, **kwargs)
    assert views
    assert all(v['x'] < 0 and not g.ocupada[g.a_celda(v['x'], v['y'])] for v in views)


def test_rango_observado_permite_otras_vistas_fuera_de_ranura():
    occupied = np.ones((120, 180), bool)
    occupied[10:110, 10:65] = False
    occupied[57:63, 65:170] = False
    g = Grilla(.025, -.5, -1.5, 0., 180, 120, occupied)
    surface = dict(superficie_x=3.2, superficie_y=0., normal_x=-1., normal_y=0.)
    kwargs = dict(despeje=.16, angulos=(-math.pi/36, math.pi/36),
                  comprobar_vision=False, desplazamiento_min=.10)
    assert not proponer_vistas([surface], (.6, 0.), g, espacio_libre(g), **kwargs)
    views = proponer_vistas([surface], (.6, 0.), g, espacio_libre(g),
                           radios_adicionales=(2.6,), **kwargs)
    assert views
    assert all(v['x'] < 1.125 for v in views)
    assert all(math.dist((v['x'], v['y']), (.6, 0.)) > .10 for v in views)


def test_vista_oblicua_conserva_eje_observado_en_pasillo_estrecho():
    # Cara oblicua: el abanico sobre su normal sale del corredor aunque la
    # observación actual y un pequeño cambio lateral sean alcanzables.
    occ = np.ones((100, 140), bool)
    occ[44:56, 5:125] = False
    g = Grilla(.025, -1., -1.25, 0., 140, 100, occ)
    surface = dict(superficie_x=1.8, superficie_y=0.,
                   normal_x=-.5, normal_y=math.sqrt(.75))
    args = dict(despeje=.04, comprobar_vision=False,
                radios_adicionales=(1.8,), desplazamiento_min=.05,
                angulos=(-math.radians(2.5), math.radians(2.5)))
    assert not proponer_vistas([surface], (0., 0.), g, espacio_libre(g), **args)
    views = proponer_vistas([surface], (0., 0.), g, espacio_libre(g),
                           desde_observador=True, **args)
    assert views
    assert any(math.dist((v['x'], v['y']), (0., 0.)) < .15 for v in views)
