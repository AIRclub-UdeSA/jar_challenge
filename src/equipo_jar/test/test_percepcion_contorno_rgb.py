import copy
import numpy as np
import pytest
from equipo_jar.percepcion import ParametrosPercepcion
from equipo_jar.percepcion_cara import inferir_base_cuadrada,regiones_desde_mascara


def scene(side):
    y,z=np.meshgrid(np.linspace(-.035,.035,20),np.linspace(.03,.27,20))
    points=np.column_stack((np.ones(y.size),y.ravel(),z.ravel()))
    pixels=round(500*side)
    rgb=dict(k=[500.,0.,320.,0.,500.,240.,0.,0.,1.],translation=[0.,0.,.15],
             quaternion=[-.5,.5,-.5,.5],width=640,height=480,
             regions=[dict(left=round(320-pixels/2+.5),top=165,width=pixels,height=150)])
    return points,rgb

@pytest.mark.parametrize('side',[.15,.25])
def test_contorno_recupera_ancho_aunque_nube_solo_tiene_tira_de_7cm(side):
    points,rgb=scene(side);p=ParametrosPercepcion()
    partial=inferir_base_cuadrada(points,(0.,0.),p)
    assert partial is not None and partial[1]==(p.lado_min_m,p.lado_min_m)
    # RGB completo sigue recuperando el tamaño, sin confundir prior con medida.
    h=inferir_base_cuadrada(points,(0.,0.),p,rgb)
    assert h is not None
    center,sides,_=h
    assert np.linalg.norm(center-np.array([1+side/2,0.]))<.005
    assert abs(sides[0]-side)<.003


def test_borde_de_imagen_no_se_toma_como_ancho_completo():
    points,rgb=scene(.25);rgb['regions']=[dict(left=0,top=165,width=400,height=150)]
    assert inferir_base_cuadrada(points,(0.,0.),ParametrosPercepcion(),rgb) is None


def test_region_no_correspondiente_no_se_asocia_al_fragmento():
    points,rgb=scene(.25);rgb['regions'][0]['left']=50
    assert inferir_base_cuadrada(points,(0.,0.),ParametrosPercepcion(),rgb) is None


def test_dos_regiones_ambiguas_no_se_mezclan():
    points,rgb=scene(.25);rgb['regions']*=2
    assert inferir_base_cuadrada(points,(0.,0.),ParametrosPercepcion(),rgb) is None


def test_calibracion_invalida_no_infiere_dimension():
    points,rgb=scene(.25);rgb['k'][0]=0.
    assert inferir_base_cuadrada(points,(0.,0.),ParametrosPercepcion(),rgb) is None


def test_regiones_de_color_rechazan_motas():
    mask=np.zeros((50,50),bool);mask[::8,::8]=True
    assert not regiones_desde_mascara(mask,30)
    mask[10:20,10:25]=True
    regions=regiones_desde_mascara(mask,30)
    assert len(regions)==1 and regions[0]['width']==15


def test_feedback_de_fragmento_recortado_no_autoriza_geometria():
    from equipo_jar.percepcion_cara import realimentacion_cara_rgb
    points,rgb=scene(.25)
    rgb['regions']=[dict(left=0,top=165,width=400,height=150)]
    feedback=realimentacion_cara_rgb(points,rgb)
    assert feedback['recortada'] and feedback['calidad_visual']>0
    assert inferir_base_cuadrada(points,(0.,0.),ParametrosPercepcion(),rgb) is None
