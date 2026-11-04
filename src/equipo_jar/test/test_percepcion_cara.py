import math
import numpy as np
import pytest
from equipo_jar.percepcion import detectar_cajas,ParametrosPercepcion
from test_percepcion import caras

P=ParametrosPercepcion()

@pytest.mark.parametrize('side',[.12,.15,.25,.4])
@pytest.mark.parametrize('angle',[0.,.3,.9,-1.2])
def test_cara_estima_tamano_variable_solo_con_prior_explicito(side,angle):
    center=np.array([1.,.2]);xyz,rgb=caras(tuple(center),(side,side),angle)
    xyz=xyz[:len(xyz)//2];rgb=rgb[:len(rgb)//2]
    camera=np.r_[center+2*np.array([math.sin(angle),-math.cos(angle)]),0.]
    raw=xyz-camera
    assert not detectar_cajas(raw,rgb,camera,(0,0,0,1),.005,.005,P)
    diag={};d=detectar_cajas(raw,rgb,camera,(0,0,0,1),.005,.005,P,
                           diagnostico=diag,asumir_base_cuadrada=True)
    assert len(d)==1 and math.dist((d[0].x,d[0].y),center)<.02
    assert diag['inferencias_base_cuadrada'][0]['supuesto']=='base_cuadrada'


def test_prior_no_reporta_con_covarianza_alta_y_expone_centro_para_pausa():
    xyz,rgb=caras();xyz=xyz[:720];rgb=rgb[:720];diag={}
    d=detectar_cajas(xyz,rgb,(0,0,0),(0,0,0,1),.1,.1,P,
                    diagnostico=diag,asumir_base_cuadrada=True)
    assert not d
    assert diag['rechazos'][0]['motivo']=='INCERTIDUMBRE'
    assert len(diag['rechazos'][0]['centro'])==2


def test_dos_caras_rectangulares_no_se_reemplazan_por_prior():
    xyz,rgb=caras(sides=(.19,.35));d=detectar_cajas(xyz,rgb,(0,0,0),(0,0,0,1),.005,.005,P,
                                                asumir_base_cuadrada=True)
    assert len(d)==1 and math.dist((d[0].x,d[0].y),(1.,.2))<.02


def test_documenta_ambiguedad_de_una_cara_rectangular():
    xyz,rgb=caras(sides=(.15,.6));xyz=xyz[:720];rgb=rgb[:720]
    d=detectar_cajas(xyz,rgb,(0,0,0),(0,0,0,1),.005,.005,P,asumir_base_cuadrada=True)
    # Ningún prior cuadrado puede garantizar profundidad de una caja rectangular.
    assert len(d)==1 and math.dist((d[0].x,d[0].y),(1.,.2))>.15

@pytest.mark.parametrize('color',[(15,30,220),(130,80,45),(220,180,110)])
def test_prior_no_convierte_azul_o_madera_en_victimas(color):
    xyz,rgb=caras(color=color);assert not detectar_cajas(xyz[:720],rgb[:720],
        (0,0,0),(0,0,0,1),.005,.005,P,asumir_base_cuadrada=True)

@pytest.mark.parametrize('side',[.15,.25])
def test_prior_oculto_comun_acota_error_en_fragmento_lateral(side):
    from dataclasses import replace
    p=replace(P,lado_prior_oculto_m=.20)
    center=np.array([1.,.2]);xyz,rgb=caras(tuple(center),(side,side))
    xyz=xyz[:len(xyz)//2];rgb=rgb[:len(rgb)//2]
    # Una franja de la cara; su ancho no determina el fondo de la caja.
    keep=xyz[:,0]<1.-side/2+.06;xyz,rgb=xyz[keep],rgb[keep]
    diag={};ds=detectar_cajas(xyz,rgb,(0,0,0),(0,0,0,1),.005,.005,p,
                            diagnostico=diag,asumir_base_cuadrada=True)
    assert len(ds)==1 and math.dist((ds[0].x,ds[0].y),center)<.15
    assert diag['inferencias_base_cuadrada'][0]['usando_prior_tamano']
    assert p.error_objetivo_m==.15 and p.lado_min_m==.09


def test_prior_oculto_no_modifica_geometria_con_dos_caras():
    from dataclasses import replace
    xyz,rgb=caras(sides=(.15,.25))
    p=replace(P,lado_prior_oculto_m=.20)
    ds=detectar_cajas(xyz,rgb,(0,0,0),(0,0,0,1),.005,.005,p,asumir_base_cuadrada=True)
    assert len(ds)==1 and math.dist((ds[0].x,ds[0].y),(1.,.2))<.02


def test_prior_oculto_no_actua_sin_opt_in_cuadrado():
    from dataclasses import replace
    xyz,rgb=caras();xyz,rgb=xyz[:720],rgb[:720]
    assert not detectar_cajas(xyz,rgb,(0,0,0),(0,0,0,1),.005,.005,
                            replace(P,lado_prior_oculto_m=.20))


def test_prior_oculto_fuera_de_rango_se_rechaza():
    with pytest.raises(ValueError):ParametrosPercepcion(lado_prior_oculto_m=.8)
