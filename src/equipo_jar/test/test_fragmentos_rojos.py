"""Rojo ocluido no deja de ser víctima por medir menos de 9cm visibles."""
import math
import numpy as np
import pytest
from equipo_jar.percepcion import ParametrosPercepcion,detectar_cajas,RegistroVictimas


def fragmento(color=(220,15,20)):
    x,z=np.meshgrid(np.linspace(.985,1.015,30),np.linspace(.03,.36,50))
    xyz=np.column_stack((x.ravel(),np.full(x.size,.075),z.ravel()))
    return xyz,np.tile(np.array(color,np.uint8),(len(xyz),1))

def estimate(std=.005):
    xyz,rgb=fragmento();diag={}
    d=detectar_cajas(xyz-np.array([1.,-1.,0.]),rgb,(1.,-1.,0.),(0.,0.,0.,1.),std,std,ParametrosPercepcion(),
                    diagnostico=diag,asumir_base_cuadrada=True)
    return d,diag

def test_fragmento_3cm_no_se_rechaza_por_tamano_y_no_se_usa_como_lado_completo():
    d,diag=estimate()
    assert len(d)==1 and d[0].error_estimado_m<=.15
    assert math.dist((d[0].x,d[0].y),(1.,.2))<.15
    inference=diag['inferencias_base_cuadrada'][0]
    assert inference['lado_observado_m']<.04
    assert inference['usando_prior_tamano']
    assert inference['lado_estimado_m']>inference['lado_observado_m']
    assert not diag['rechazos']

def test_fragmento_rojo_con_covarianza_alta_sigue_sin_reportar():
    d,diag=estimate(.1)
    assert not d and diag['rechazos'][0]['motivo']=='INCERTIDUMBRE'
    assert ParametrosPercepcion().error_objetivo_m==.15

def test_fragmento_estable_confirma_una_vez_sin_consumir_reportes_repetidos():
    d,_=estimate();reg=RegistroVictimas(2,ParametrosPercepcion());reports=[]
    for stamp in (0.,.5,1.,1.5,2.):reports.extend(reg.observar(d,stamp))
    assert len(reports)==1 and reg.reportes==1
    assert math.dist(reports[0].centro,(1.,.2))<.15

@pytest.mark.parametrize('color',[(20,20,220),(130,80,45),(220,180,110)])
def test_fragmentos_no_rojos_no_son_victimas(color):
    xyz,rgb=fragmento(color)
    assert not detectar_cajas(xyz,rgb,(1.,-1.,0.),(0.,0.,0.,1.),.005,.005,
                             ParametrosPercepcion(),asumir_base_cuadrada=True)
