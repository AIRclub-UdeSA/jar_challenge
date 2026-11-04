"""La caché conserva aciertos y cruces; se invalida con cada entrada."""
import math
from types import SimpleNamespace as N
import numpy as np
import pytest
import equipo_jar.localizacion_activa as module

def mapa(giro=0.,empty=False):
    rng=np.random.default_rng(916)
    data=np.where(rng.random((45,67))<.13,100,0)
    if empty:data[:]=0
    return N(data=data.ravel().tolist(),info=N(width=67,height=45,resolution=.05,
      origin=N(position=N(x=-.7,y=-1.1),orientation=N(x=0.,y=0.,z=math.sin(giro/2),w=math.cos(giro/2)))))

@pytest.mark.parametrize('angle',[0.,.37,-1.2])
@pytest.mark.parametrize('empty',[False,True])
def test_cache_conserva_calidad_completa(angle,empty):
    m=mapa(angle,empty);accelerated=module.EvaluadorMapa(m)
    rng=np.random.default_rng(917)
    for pose in ((0.,0.,0.),(-1.,-1.,.5),(1.2,-.5,-1.)):
        rays=list(enumerate(rng.uniform(.01,4.,317)))
        diag={}
        expected=module.concordancia_mapa(m,rays,-math.pi,.02,pose,.1,.4,diagnostico=diag)
        actual,detail=accelerated.calcular(rays,-math.pi,.02,pose,.1,.4)
        assert actual==expected and detail==diag

def test_cache_cambia_con_cada_entrada_y_no_reutiliza_mapa_anterior(monkeypatch):
    original=module.concordancia_mapa;calls=[]
    def record(*args,**kwargs):
        calls.append(1);return original(*args,**kwargs)
    monkeypatch.setattr(module,'concordancia_mapa',record)
    evaluator=module.EvaluadorMapa(mapa())
    args=([(0,.5)],0.,.01,(0.,0.,0.),.1,.4)
    first=evaluator.calcular(*args)
    first[1]['extra']='mutado afuera'
    assert 'extra' not in evaluator.calcular(*args)[1]
    assert len(calls)==1
    variants=[([(0,.6)],0.,.01,(0.,0.,0.),.1,.4),
              ([(0,.6)],.02,.01,(0.,0.,0.),.1,.4),
              ([(0,.6)],.02,.02,(0.,0.,0.),.1,.4),
              ([(0,.6)],.02,.02,(.01,0.,0.),.1,.4),
              ([(0,.6)],.02,.02,(.01,0.,0.),.11,.4),
              ([(0,.6)],.02,.02,(.01,0.,0.),.11,.5)]
    for n,variant in enumerate(variants,2):
        evaluator.calcular(*variant);assert len(calls)==n
    module.EvaluadorMapa(mapa(empty=True)).calcular(*variants[-1])
    assert len(calls)==8

