from dataclasses import replace
import math
import pytest
from equipo_jar.percepcion import Deteccion,ParametrosPercepcion,RegistroVictimas,detectar_cajas
from test_percepcion import caras

P=replace(ParametrosPercepcion(),error_reporte_max_m=.5)
def d(x=1.,error=.217,surface=None):return Deteccion(x,.2,error,(.2,.2),100,surface)


def test_preferida_se_publica_sin_espera():
    r=RegistroVictimas(3,P);assert len(r.observar([d(error=.1)],1.))==1
    assert not r.pendientes()


def test_conserva_mejor_estimacion_y_la_publica_una_vez_al_vaciar():
    r=RegistroVictimas(3,P)
    for t,x,error in [(1.,1.,.3),(2.,1.02,.217),(3.,1.03,.4)]:assert not r.observar([d(x,error)],t)
    assert len(r.pendientes())==1 and math.isclose(r.pendientes()[0].centro[0],1.02)
    sent=r.reportar_pendientes();assert len(sent)==1 and sent[0].reportada_por_respaldo
    assert math.isclose(sent[0].error_m,.217)
    assert not r.reportar_pendientes() and not r.observar([d()],4.)


def test_mejora_bajo15_publica_inmediatamente_sin_respaldo():
    r=RegistroVictimas(3,P);assert not r.observar([d()],1.)
    sent=r.observar([d(1.01,.1)],2.)
    assert len(sent)==1 and not sent[0].reportada_por_respaldo
    assert not r.reportar_pendientes()


def test_respaldo_respeta_N_y_prioriza_mejor_calidad():
    r=RegistroVictimas(1,P);assert not r.observar([d(1.,.4),d(2.,.2)],1.)
    sent=r.reportar_pendientes();assert len(sent)==1 and sent[0].centro[0]==2.
    assert not r.pendientes() and not r.reportar_pendientes()


def test_posiciones_inciertas_cercanas_no_se_fusionan_por_sumar_errores():
    r=RegistroVictimas(2,P);r.observar([d(1.,.4),d(1.5,.4)],1.)
    assert len(r.pendientes())==2


@pytest.mark.parametrize('error',[.501,float('nan'),float('inf')])
def test_error_fuera_de_limite_no_se_guarda_ni_publica(error):
    r=RegistroVictimas(3,P);assert not r.observar([d(error=error)],1.)
    assert not r.pendientes() and not r.reportar_pendientes()


def test_detector_conserva_centro_valido_por_encima_del_objetivo_preferido():
    xyz,rgb=caras();diag={}
    ds=detectar_cajas(xyz,rgb,(0,0,0),(0,0,0,1),.06,.056,P,diagnostico=diag)
    assert ds and .15<ds[0].error_estimado_m<=.5
    r=RegistroVictimas(3,P);assert not r.observar(ds,1.)
    assert r.reportar_pendientes()
