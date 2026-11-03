"""Una pose arbitraria no debe excluir orientaciones libres de un pasillo."""
import math
import numpy as np
from equipo_jar.navegacion.grilla import Grilla
from equipo_jar.navegacion.guardia import huella_convexa_x3
from equipo_jar.navegacion.orientado import planificar_orientado
from equipo_jar.navegacion.tipos import Pose2D

def test_pasillo_estrecho_admite_alinearse_con_mapa_desde_rumbo_arbitrario():
    occupied=np.zeros((60,120),dtype=bool)
    occupied[:26,40:80]=True
    occupied[33:,40:80]=True
    grid=Grilla(.05,0.,0.,0.,120,60,occupied)
    fp=huella_convexa_x3(-.19,.13,.07,.08,.0845,.033,.04)
    fp['heading_uncertainty_rad']=.1
    start=Pose2D(1.,1.475,.20)
    route=planificar_orientado(grid,occupied,start,(5.,1.475),fp)
    assert route is not None
    # Comprobación independiente de las paredes horizontales, incluyendo
    # todo el intervalo de incertidumbre al atravesar el centro del pasillo.
    corners=np.asarray(fp['corners_base_m'])
    checked=0
    for a,b,ha,hb in zip(route.puntos,route.puntos[1:],route.orientaciones,route.orientaciones[1:]):
        delta=math.atan2(math.sin(hb-ha),math.cos(hb-ha))
        for t in np.linspace(0.,1.,11):
            x,y=(1-t)*np.asarray(a)+t*np.asarray(b)
            if not 2.25<=x<=3.75:continue
            checked+=1
            for error in np.linspace(-.1,.1,11):
                h=ha+t*delta+error
                ys=y+corners[:,0]*math.sin(h)+corners[:,1]*math.cos(h)
                assert ys.min()>=1.3+.04-1e-8
                assert ys.max()<=1.65-.04+1e-8
    assert checked>0
