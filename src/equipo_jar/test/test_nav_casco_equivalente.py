"""Equivalencia geométrica con un casco de referencia independiente de Qhull."""
import math
import numpy as np
import pytest
from equipo_jar.navegacion.guardia import poligono_barrido,huella_convexa_x3,distancias_poligono

def referencia(fp,alpha,distance):
    theta=fp['heading_uncertainty_rad']
    steps=max(1,math.ceil(2*theta/.04));points=[]
    for i in range(steps+1):
        angle=-theta+2*theta*i/steps-alpha
        c,s=math.cos(angle),math.sin(angle)
        for x,y in fp['corners_base_m']:
            u,v=c*x-s*y,s*x+c*y
            points.extend(((u,v),(u+distance,v)))
    points=sorted(set(points))
    def cross(a,b,c):
        return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    lower=[];upper=[]
    for chain,ordered in ((lower,points),(upper,reversed(points))):
        for point in ordered:
            while len(chain)>1 and cross(chain[-2],chain[-1],point)<=0:chain.pop()
            chain.append(point)
    return lower[:-1]+upper[:-1]

@pytest.mark.parametrize('theta',[0.,.1,.2,.55])
@pytest.mark.parametrize('distance',[0.,.1,.25])
def test_casco_conserva_soporte_y_distancias(theta,distance):
    rng=np.random.default_rng(234)
    fp=huella_convexa_x3(-.19,.13,.07,.08,.0845,.033,.075)
    fp['heading_uncertainty_rad']=theta
    for alpha in (-math.pi,-.7,0.,1.4,math.pi):
        old=referencia(fp,alpha,distance)
        new,error=poligono_barrido(fp,alpha,distance)
        axes=rng.normal(size=(128,2));axes/=np.linalg.norm(axes,axis=1)[:,None]
        np.testing.assert_allclose((np.asarray(old)@axes.T).max(axis=0),
                                   (np.asarray(new)@axes.T).max(axis=0),rtol=0,atol=1e-12)
        points=rng.uniform(-.5,.5,(150,2))
        np.testing.assert_allclose(distancias_poligono(points,old),
                                   distancias_poligono(points,new),rtol=0,atol=1e-12)
        steps=max(1,math.ceil(2*theta/.04))
        assert error==fp['radius_m']*(1-math.cos(theta/steps))
