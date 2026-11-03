"""La guardia cubre el giro futuro y puede distinguir una pared lateral segura."""
import math
import pytest
from equipo_jar.seguridad_giro import comprobar_giro, limites_barrido_giro
from equipo_jar.localizacion_activa import LocalizacionActiva, Observacion

PARAMS=(-.19,.13,.12,math.radians(20)+.12,.10)


def test_pared_lateral_fuera_del_barrido_permita_observar_girando():
    assert comprobar_giro([(0.,-.32)]*60,*PARAMS)['seguro']


def test_obstaculo_frontal_rechaza_giro():
    assert not comprobar_giro([(.20,0.)]*60,*PARAMS)['seguro']


def test_barrido_futuro_no_solo_huella_estatica():
    points=[(-.13,-.27)]*60
    assert comprobar_giro(points,-.19,.13,.12,0.,.10)['seguro']
    assert not comprobar_giro(points,*PARAMS)['seguro']


def test_extremos_analiticos_cubren_rotaciones_intermedias():
    bounds=limites_barrido_giro(*PARAMS)
    for i in range(501):
        a=PARAMS[3]*i/500
        for x in (-.19,.13):
            for y in (-.12,.12):
                u=x*math.cos(a)-y*math.sin(a)
                v=x*math.sin(a)+y*math.cos(a)
                assert bounds[0]-1e-12 <= u-.10 and u+.10 <= bounds[1]+1e-12
                assert bounds[2]-1e-12 <= v-.10 and v+.10 <= bounds[3]+1e-12


@pytest.mark.parametrize('points',[[],[(0.,-.32)]*29,[(float('nan'),0.)]*60])
def test_datos_insuficientes_no_habilitan_giro(points):
    assert not comprobar_giro(points,*PARAMS)['seguro']


@pytest.mark.parametrize('parameters',[
    (.19,.13,.12,.4,.1),(-.19,.13,-.12,.4,.1),
    (-.19,.13,.12,.4,0.),(-.19,.13,.12,float('nan'),.1)])
def test_geometria_invalida_rechazada(parameters):
    with pytest.raises(ValueError):limites_barrido_giro(*parameters)


def obs(seguro, distancia=.32):
    return Observacion(0.,(0.,0.,0.),1,.1,.1,1.,distancia,seguro)


def test_guardia_geometrica_segura_permite_giro_antes_bloqueado_por_radio():
    c=LocalizacionActiva()
    assert c.paso(0.,True,obs(True))==0.
    assert c.estado=='GIRANDO'
    assert c.paso(.1,True,obs(True))>0


def test_guardia_insegura_no_se_ignora_aunque_rango_minimo_sea_grande():
    c=LocalizacionActiva()
    assert c.paso(0.,True,obs(False,1.))==0.
    assert c.estado=='VERIFICANDO_QUIETO'


def test_obstaculo_nuevo_detiene_giro_sin_rearme():
    c=LocalizacionActiva()
    c.paso(0.,True,obs(True))
    assert c.paso(.1,True,obs(True))>0
    assert c.paso(.2,True,obs(False,1.))==0
    assert c.estado=='INSUFICIENTE'
    assert c.paso(.3,True,obs(True))==0


def test_esquina_de_la_caja_no_es_un_obstaculo_en_el_barrido():
    lo,hi,bot,top=limites_barrido_giro(*PARAMS)
    point=(lo+.001,top-.001)
    assert lo<point[0]<hi and bot<point[1]<top
    assert comprobar_giro([point]*60,*PARAMS)['seguro']


def test_casco_cubre_giro_continuo_y_margen_alrededor_de_cada_esquina():
    points=[]
    for i in range(47):
        a=PARAMS[3]*i/46
        for x in (-.19,.13):
            for y in (-.12,.12):
                u=x*math.cos(a)-y*math.sin(a)
                v=x*math.sin(a)+y*math.cos(a)
                for j in range(12):
                    phi=j*math.pi/6
                    points.append((u+.09999*math.cos(phi),v+.09999*math.sin(phi)))
    guard=comprobar_giro(points,*PARAMS)
    assert guard['puntos_bloqueantes']==len(points)
    assert 0<guard['cota_discretizacion_m']<.0001


def test_guardia_de_giro_cero_conserva_margen():
    guard=comprobar_giro([(.229,0.)]*60,-.19,.13,.12,0.,.10)
    assert not guard['seguro'] and guard['cota_discretizacion_m']==0.
