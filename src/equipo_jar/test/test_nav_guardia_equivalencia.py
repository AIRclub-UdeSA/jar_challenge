"""La fase amplia conserva contactos y distancia mínima de la evaluación exhaustiva."""
import math
import numpy as np
import pytest
from equipo_jar.navegacion.guardia import (comprobar_barrido,poligono_barrido,
    huella_desde_parametros,distancias_poligono,puntos_desde_scan)

def scalar_distances(points,hull):
    result=[]
    for x,y in points:
        inside=True;best=math.inf
        for (ax,ay),(bx,by) in zip(hull,hull[1:]+hull[:1]):
            ex,ey=bx-ax,by-ay;dx,dy=x-ax,y-ay
            inside=inside and ex*dy-ey*dx>=-1e-12
            t=max(0.,min(1.,(dx*ex+dy*ey)/(ex*ex+ey*ey)))
            best=min(best,math.hypot(dx-t*ex,dy-t*ey))
        result.append(0. if inside else best)
    return result

@pytest.mark.parametrize("margin",[.10,.04])
@pytest.mark.parametrize("alpha",[-2.,0.,.8])
def test_caja_no_descarta_contactos_ni_minimo(alpha,margin):
    rng=np.random.default_rng(416)
    cloud=rng.uniform(-3,3,(1000,2)).tolist()
    footprint=huella_desde_parametros(-.19,.13,.12,margin)
    hull,error=poligono_barrido(footprint,alpha,.3)
    c,s=math.cos(alpha),math.sin(alpha)
    projected=[(c*x+s*y,-s*x+c*y) for x,y in cloud]
    expected=scalar_distances(projected,hull)
    assert distancias_poligono(projected,hull)==pytest.approx(expected,abs=1e-12)
    result=comprobar_barrido(cloud,alpha,.3,footprint)
    blocked=[p for p,gap in zip(cloud,expected) if gap<=margin+error+1e-12]
    assert len(result['blocking_points_base_and_along_cross_m'])==len(blocked)
    assert result['minimum_clearance_m']==pytest.approx(min(expected)-margin-error,abs=1e-12)
    assert result['clear']==(not blocked)

def test_scan_vectorizado_conserva_extrinsecos_y_descarte():
    ranges=[float('inf'),float('nan'),.1,.6,1.,5.,10.]
    expected=[]
    for i,r in enumerate(ranges):
        if math.isfinite(r) and .2<=r<10:
            a=.5-.7+i*.1;expected.append((.043+r*math.cos(a),.02+r*math.sin(a)))
    actual=puntos_desde_scan(ranges,-.7,.1,.2,10.,(.043,.02,.5))
    assert np.asarray(actual)==pytest.approx(np.asarray(expected),abs=1e-12)
