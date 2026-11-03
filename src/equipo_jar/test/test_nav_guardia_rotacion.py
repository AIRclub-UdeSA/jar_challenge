"""La guardia debe incluir el giro y su curvatura, incluso sin traslación."""
import math

from equipo_jar.navegacion.guardia import huella_desde_parametros, limitar_velocidad


def points():
    return [(3*math.cos(i*math.pi/30),3*math.sin(i*math.pi/30)) for i in range(60)]


def test_giro_solo_bloquea_punto_que_alcanza_la_esquina_al_rotar():
    footprint=huella_desde_parametros(-.19,.13,.12,.10)
    cloud=points()+[(-.08,-.29)]
    _,_,diagnostic=limitar_velocidad(cloud,0.,0.,footprint,.5,.15,wz=.5)
    assert diagnostic['bloqueado'] or diagnostic['escala_angular']<1.


def test_giro_y_traslacion_comparten_escala_y_reserva_de_curvatura():
    footprint=huella_desde_parametros(-.19,.13,.12,.10)
    vx,vy,diagnostic=limitar_velocidad(points(),.2,.1,footprint,.5,.15,wz=.2)
    assert not diagnostic['bloqueado'] and vx>0 and vy>0
    assert diagnostic['escala_angular']==diagnostic['escala']
    assert diagnostic['reserva_curvatura_m']>0
