"""La liberación vectorizada conserva la referencia escalar de sectores."""
import bisect,math
import numpy as np
import pytest
from equipo_jar.navegacion.grilla import Grilla
from equipo_jar.navegacion.costos import CapaViva

def escalar(layer,points,origin,increment):
    rays=sorted((math.atan2(y-origin[1],x-origin[0]),math.hypot(x-origin[0],y-origin[1])) for x,y in points)
    rays=[(a+offset,r) for offset in (-2*math.pi,0.,2*math.pi) for a,r in rays]
    angles=[a for a,r in rays];cleared=set();radius=layer.grilla.resolucion/math.sqrt(2)
    for f,c in np.argwhere(np.isfinite(layer.marca)):
        x,y=layer.grilla.a_mundo(int(f),int(c));x-=origin[0];y-=origin[1]
        distance=math.hypot(x,y)
        if distance<=radius:continue
        angle=math.atan2(y,x);width=math.asin(radius/distance)
        lo=bisect.bisect_left(angles,angle-width-increment);hi=bisect.bisect_right(angles,angle+width+increment)
        sector=rays[lo:hi]
        if len(sector)<2 or sector[0][0]>angle-width or sector[-1][0]<angle+width:continue
        if any(b[0]-a[0]>1.5*increment for a,b in zip(sector,sector[1:])):continue
        if min(r for a,r in sector)>distance+radius+1e-9:cleared.add((int(f),int(c)))
    return cleared

@pytest.mark.parametrize('yaw',[0.,.4])
@pytest.mark.parametrize('missing',[False,True])
def test_liberacion_conserva_todas_las_oclusiones_y_huecos(yaw,missing):
    rng=np.random.default_rng(391)
    g=Grilla(.05,-2.,-2.,yaw,100,80,np.zeros((80,100),bool))
    layer=CapaViva(g,np.ones((80,100)),600.,.1)
    cells=np.column_stack((rng.integers(10,70,160),rng.integers(10,90,160)))
    layer.marca[cells[:,0],cells[:,1]]=0.
    before=set(map(tuple,np.argwhere(np.isfinite(layer.marca))))
    angles=np.arange(-math.pi,math.pi,.006)
    if missing:angles=angles[(angles<-.04)|(angles>.09)]
    ranges=rng.uniform(.4,3.,len(angles));origin=(.1,.2)
    points=np.column_stack((origin[0]+ranges*np.cos(angles),origin[1]+ranges*np.sin(angles)))
    expected=escalar(layer,points,origin,.006)
    layer._liberar_observadas(1.,points,origin,.006)
    remaining=set(map(tuple,np.argwhere(np.isfinite(layer.marca))))
    assert before-remaining==expected
    assert expected and remaining
