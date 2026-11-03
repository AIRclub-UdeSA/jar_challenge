"""Publicar primera posición RGB-D admitida, sin otro veto temporal."""
import math
import pytest
from equipo_jar.percepcion import Deteccion,ParametrosPercepcion,RegistroVictimas


def detection(x=1.,error=.1):return Deteccion(x,.2,error,(.2,.2),100)


def test_primer_frame_valido_reporta_sin_espera():
    r=RegistroVictimas(1,ParametrosPercepcion())
    sent=r.observar([detection()],.01)
    assert len(sent)==1 and math.isclose(sent[0].centro[0],1.)
    assert len(sent[0].observaciones)==1
    assert r.reportes==1


def test_misma_caja_no_reporta_dos_veces_al_volver():
    r=RegistroVictimas(3,ParametrosPercepcion());assert len(r.observar([detection()],1.))==1
    for t in [1.,2.,50.]:assert not r.observar([detection(1.04)],t)
    assert r.reportes==1


@pytest.mark.parametrize('error',[.151,float('nan'),float('inf'),0.])
def test_reporte_inmediato_conserva_gate_de_posicion(error):
    r=RegistroVictimas(1,ParametrosPercepcion())
    assert not r.observar([detection(error=error)],1.) and not r.victimas


def test_no_inventa_coordenada_invalida():
    r=RegistroVictimas(1,ParametrosPercepcion())
    assert not r.observar([detection(float('nan'))],1.)


def test_nunca_supera_cantidad_reportes_del_juez():
    r=RegistroVictimas(1,ParametrosPercepcion())
    assert len(r.observar([detection(1.),detection(2.)],1.))==1
    assert not r.observar([detection(2.)],2.) and r.reportes==1


def test_misma_superficie_no_se_convierte_en_otra_caja_por_salto_de_centro():
    r=RegistroVictimas(3,ParametrosPercepcion())
    first=Deteccion(-.908,.893,.1,(.2,.2),100,(-1.03,.78))
    shifted=Deteccion(-1.249,.757,.11,(.43,.43),80,(-1.025,.785))
    assert len(r.observar([first],1.))==1
    assert not r.observar([shifted],2.) and r.reportes==1
    other=Deteccion(1.,-1.9,.1,(.2,.2),100,(.9,-1.9))
    assert len(r.observar([other],3.))==1
    assert r.reportes==2
