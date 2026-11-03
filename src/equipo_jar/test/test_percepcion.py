"""Escenas sintéticas: no sustituyen imágenes ni encuentros físicos de Gazebo."""
from dataclasses import replace
from types import SimpleNamespace as NS
import math

import cv2
import numpy as np
import pytest

from equipo_jar.percepcion import (ParametrosPercepcion, Deteccion, RegistroVictimas,
    detectar_cajas, hay_region_roja, leer_imagen, leer_nube, mascara_roja, transformar)

P = ParametrosPercepcion()


def caras(center=(1., .2), sides=(.25, .30), angle=0., color=(220, 15, 20)):
    """Dos caras adyacentes completas, fondo conocido sólo por el evaluador."""
    a, b = sides
    u, z = np.meshgrid(np.linspace(-a/2, a/2, 30), np.linspace(.03, .28, 24))
    front = np.column_stack((u.ravel(), np.full(u.size, -b/2), z.ravel()))
    v, z = np.meshgrid(np.linspace(-b/2, b/2, 30), np.linspace(.03, .28, 24))
    side = np.column_stack((np.full(v.size, -a/2), v.ravel(), z.ravel()))
    xyz = np.vstack((front, side))
    c, s = math.cos(angle), math.sin(angle)
    xyz[:, :2] = xyz[:, :2] @ np.array([[c, s], [-s, c]])
    xyz[:, :2] += center
    return xyz, np.tile(np.array(color, dtype=np.uint8), (len(xyz), 1))


@pytest.mark.parametrize('angle', [0., .2, .7, 1.1, -1.4])
@pytest.mark.parametrize('center,sides', [((1., .2), (.25,.3)), ((1.5,-.3), (.19,.35))])
def test_centro_de_dos_caras_rotadas(angle, center, sides):
    xyz, rgb = caras(center, sides, angle)
    d = detectar_cajas(xyz, rgb, (0,0,0), (0,0,0,1), .015, .01, P)
    assert len(d) == 1
    assert math.dist((d[0].x,d[0].y), center) < .025


@pytest.mark.parametrize('color', [(15,30,220), (130,80,45), (220,180,110),
                                   (110,110,110), (220,220,15), (15,180,30)])
def test_azul_madera_y_otros_distractores_no_son_rojos(color):
    xyz, rgb = caras(color=color)
    assert not detectar_cajas(xyz,rgb,(0,0,0),(0,0,0,1),.01,.01,P)


def test_rojo_en_ambos_extremos_del_tono():
    rgb = cv2.cvtColor(np.array([[[0,240,180],[179,240,180],[90,240,180]]],np.uint8),
                       cv2.COLOR_HSV2RGB)
    assert mascara_roja(rgb,P).tolist() == [[True,True,False]]


def test_region_coherente_y_no_motas_sueltas():
    image = np.zeros((50,50,3),np.uint8)
    image[::5,::5] = [220,0,0]
    assert not hay_region_roja(image,P)
    image[10:20,10:20] = [220,0,0]
    assert hay_region_roja(image,P)


def test_una_cara_no_permite_inventar_fondo():
    xyz,rgb = caras()
    assert not detectar_cajas(xyz[:720],rgb[:720],(0,0,0),(0,0,0,1),.01,.01,P)


def test_cara_unica_solicita_evidencia_sin_publicar_un_centro_inventado():
    xyz,rgb=caras();diagnostic={}
    assert not detectar_cajas(xyz[:720],rgb[:720],(0,0,0),(0,0,0,1),.01,.01,P,
                              diagnostico=diagnostic)
    assert diagnostic['vistas_incompletas']
    assert diagnostic['incertidumbre_alta']==0


def test_covarianza_y_vista_incompleta_son_diagnosticos_distintos():
    xyz,rgb=caras();diagnostic={}
    assert not detectar_cajas(xyz,rgb,(0,0,0),(0,0,0,1),.1,.1,P,
                              diagnostico=diagnostic)
    assert not diagnostic['vistas_incompletas']
    assert diagnostic['incertidumbre_alta']==1


def test_pared_roja_extensa_se_rechaza_por_geometria():
    xyz,rgb = caras(sides=(1.8,.3))
    assert not detectar_cajas(xyz,rgb,(0,0,0),(0,0,0,1),.01,.01,P)


def test_covarianza_alta_no_se_convierte_en_reporte():
    xyz,rgb=caras()
    assert not detectar_cajas(xyz,rgb,(0,0,0),(0,0,0,1),.1,.1,P)


def test_nan_y_distancias_invalidas_se_descartan():
    xyz,rgb=caras()
    xyz[:] = np.nan
    assert not detectar_cajas(xyz,rgb,(0,0,0),(0,0,0,1),.01,.01,P)


def test_transformacion_3d_no_suponemos_marco_optico():
    xyz=np.array([[1.,0.,0.]])
    np.testing.assert_allclose(transformar(xyz,(2,3,4),(0,0,math.sqrt(.5),math.sqrt(.5))), [[2,4,4]])
    with pytest.raises(ValueError):transformar(xyz,(0,0,0),(0,0,0,0))


@pytest.mark.parametrize('big', [False,True])
@pytest.mark.parametrize('datatype', [6,7])
def test_lectura_nube_respeta_rgb_bits_padding_y_endian(big,datatype):
    endian='>' if big else '<'
    dtype=np.dtype(dict(names=['x','y','z','rgb'],formats=[endian+'f4']*3+[endian+'u4'],
                        offsets=[0,4,8,16],itemsize=24))
    buffer=bytearray(2*56)
    rows=np.ndarray((2,2),dtype=dtype,buffer=buffer,strides=(56,24))
    rows['x']=1;rows['y']=2;rows['z']=3;rows['rgb']=0xDC0F14
    msg=NS(fields=[NS(name=n,offset=dtype.fields[n][1],datatype=datatype if n=='rgb' else 7,count=1)
                   for n in dtype.names],point_step=24,row_step=56,width=2,height=2,
           is_bigendian=big,data=buffer)
    xyz,rgb=leer_nube(msg)
    assert xyz.shape==(4,3)
    assert np.all(rgb==[220,15,20])
    msg.data=buffer[:-1]
    with pytest.raises(ValueError):leer_nube(msg)


def test_imagen_bgr_con_padding():
    msg=NS(encoding='bgr8',width=2,height=1,step=8,data=bytes([10,20,220,30,40,210,0,0]))
    assert leer_imagen(msg).tolist()==[[[220,20,10],[210,40,30]]]


def detection(x=1., y=.2):return Deteccion(x,y,.1,(.25,.3),100)


def test_confirma_una_vez_y_recuerda_al_volver():
    r=RegistroVictimas(2,P)
    assert len(r.observar([detection()],1.))==1
    assert not r.observar([detection(1.01)],1.5)
    assert not r.observar([detection(.99)],2.)
    assert not r.observar([detection(1.02)],80.)
    assert len(r.victimas)==r.reportes==1


def test_no_contar_estampas_repetidas_ni_dos_regiones_como_dos_vistas():
    r=RegistroVictimas(1,P)
    r.observar([detection(),detection()],1.)
    for _ in range(8):assert not r.observar([detection()],1.)
    assert len(r.victimas[0].observaciones)==1


def test_cantidad_limita_reportes_y_mantiene_identidades_separadas():
    r=RegistroVictimas(1,P)
    for stamp in [1.,1.5,2.]:r.observar([detection(),detection(2.)],stamp)
    assert r.reportes==1
    assert len(r.victimas)==2


def test_identidad_ambigua_no_gasta_intento():
    r=RegistroVictimas(2,P)
    r.observar([detection(.8),detection(1.3)],1.)
    for stamp in [2.,3.,4.]:assert not r.observar([detection(1.05)],stamp)
    assert r.reportes==2


def test_no_se_agrega_veto_espacial_ni_espera_a_deteccion_admitida():
    r=RegistroVictimas(1,P)
    sent=r.observar([detection()],1.)
    assert len(sent)==1 and np.allclose(sent[0].centro,(1.,.2))
    assert not r.observar([detection(1.05)],2.)


def test_parametros_invalidos():
    with pytest.raises(ValueError):replace(P,conexion_m=0.)
    with pytest.raises(ValueError):RegistroVictimas(-1,P)


def test_parche_rojizo_de_pared_pasa_hsv_pero_no_geometria():
    xyz,rgb=caras(color=(200,45,20))
    # Parche de madera/iluminación que cae deliberadamente dentro de rojo.
    assert mascara_roja(rgb,P).all()
    assert not detectar_cajas(xyz[:720],rgb[:720],(0,0,0),(0,0,0,1),.01,.01,P)

def test_normal_de_cara_apunta_a_la_camara_y_no_inventa_centro():
    xyz,rgb=caras()
    diag={}
    assert not detectar_cajas(xyz[:720],rgb[:720],(0,0,0),(0,0,0,1),
                              .01,.01,P,diagnostico=diag)
    s=diag['vistas_incompletas'][0]
    surface=np.array([s['superficie_x'],s['superficie_y']])
    normal=np.array([s['normal_x'],s['normal_y']])
    assert np.isclose(np.linalg.norm(normal),1)
    assert normal @ -surface > 0


def test_fragmento_rojo_oculto_menor_a_caja_solicita_vista_sin_reportar():
    xyz, rgb = caras(sides=(.055, .30))
    diagnostic = {}
    assert not detectar_cajas(xyz[:720], rgb[:720], (0, 0, 0), (0, 0, 0, 1),
                              .01, .01, P, diagnostico=diagnostic)
    assert diagnostic['vistas_incompletas']
    assert diagnostic['rechazos'][0]['motivo'] == 'EXTENSION_OCULTA_O_PEQUENA'
