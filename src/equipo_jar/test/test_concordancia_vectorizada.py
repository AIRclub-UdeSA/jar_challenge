"""Equivalencia del muestreo por bloques con el ray casting escalar."""
import math
from types import SimpleNamespace as N
import numpy as np
import pytest
from equipo_jar.localizacion_activa import _primera_ocupada, _primeras_ocupadas

@pytest.mark.parametrize("giro",[0.,.37,-1.2])
def test_vectorizacion_conserva_primera_celda_y_fuera_de_mapa(giro):
    rng=np.random.default_rng(414)
    data=np.where(rng.random((31,43))<.15,100,0).ravel().tolist()
    mapa=N(data=data,info=N(width=43,height=31,resolution=.05,
        origin=N(position=N(x=-.8,y=-1.1),orientation=N(x=0.,y=0.,
        z=math.sin(giro/2),w=math.cos(giro/2)))))
    rayos=[(float(rng.uniform(-1.2,1.5)),float(rng.uniform(-1.5,1.5)),
            float(rng.uniform(-math.pi,math.pi)),float(rng.uniform(.01,4.))) for _ in range(103)]
    esperados=[_primera_ocupada(mapa,*r) for r in rayos]
    assert _primeras_ocupadas(mapa,rayos)==pytest.approx(esperados,abs=1e-12)
    assert _primeras_ocupadas(mapa,[])==[]

@pytest.mark.parametrize("giro",[0.,.37,-1.2])
def test_extremos_vectorizados_conservan_criterio_escalar(giro):
    from equipo_jar.localizacion_activa import concordancia_mapa_simple, _concordancia_mapa_simple_escalar
    rng=np.random.default_rng(415)
    data=np.where(rng.random((31,43))<.15,100,0).ravel().tolist()
    mapa=N(data=data,info=N(width=43,height=31,resolution=.05,
        origin=N(position=N(x=-.8,y=-1.1),orientation=N(x=0.,y=0.,
        z=math.sin(giro/2),w=math.cos(giro/2)))))
    rangos=list(enumerate(rng.uniform(.01,4.,317)))
    for pose in ((0.,0.,0.),(-1.,-1.,.5),(1.2,-.5,-1.)):
        assert concordancia_mapa_simple(mapa,rangos,-math.pi,.02,pose)==pytest.approx(
            _concordancia_mapa_simple_escalar(mapa,rangos,-math.pi,.02,pose))


def test_diagnostico_simple_se_reutiliza_sin_cambiar_valor():
    from equipo_jar.localizacion_activa import concordancia_mapa,concordancia_mapa_simple
    mapa=N(data=[100]*400,info=N(width=20,height=20,resolution=.05,
        origin=N(position=N(x=-.5,y=-.5),orientation=N(x=0.,y=0.,z=0.,w=1.))))
    rangos=[(i,.3) for i in range(30)];diag={}
    concordancia_mapa(mapa,rangos,-1.,.05,(0.,0.,0.),diagnostico=diag)
    assert diag['concordancia_simple']==concordancia_mapa_simple(mapa,rangos,-1.,.05,(0.,0.,0.))[0]
