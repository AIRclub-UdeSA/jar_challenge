import math
from dataclasses import replace
from equipo_jar.navegacion.guardia import limitar_velocidad,huella_desde_parametros
from equipo_jar.navegacion import guardia
from equipo_jar.navegacion.ida_vuelta import IdaVuelta,EntradaNavegacion
from equipo_jar.navegacion.costos import ParametrosCosto
from equipo_jar.navegacion.planificador import Ruta
from equipo_jar.navegacion.tipos import Pose2D
from test_mision import nueva

def wall(x):return [(x,-.4+i*.02) for i in range(41)]
def footprint():return dict(huella_desde_parametros(-.19,.13,.12,.04),heading_uncertainty_rad=0.)

def test_intrusion_de_buffer_solo_permite_alejarse_del_contacto():
    fp=footprint();points=wall(.165)
    assert limitar_velocidad(points,.02,0.,fp,.5,.15)[2]['bloqueado']
    r=guardia.recuperar_margen(points,fp,.5,.15)
    assert r is not None and r[0]<0.
    assert abs(r[1])<1e-8
    assert r[2]['separacion_final_m']>r[2]['separacion_inicial_m']>.02
    for t in (0.,.2,.5,1.,1.16):
        assert .165-(.13+r[0]*t)>=.035-1e-10
    assert fp['margin_m']==.04

def test_no_escapa_cerca_de_contacto_ni_sin_espacio_que_mejore():
    assert guardia.recuperar_margen(wall(.145),footprint(),.5,.15) is None
    assert guardia.recuperar_margen(wall(.165)+wall(-.225),footprint(),.5,.15) is None

def test_navegador_recupera_hacia_afuera_y_limita_la_duracion():
    _,original=nueva();fp=footprint()
    nav=IdaVuelta(original.grilla,ParametrosCosto(.12,fp['radius_m'],.04,.25,5.),
                  original.seguidor_ida.p,fp,replace(original.p,regreso_automatico=False))
    nav.largada=(.7,.7,0.);nav.meta=(1.4,.7);nav.ruta=Ruta([(.7,.7),(1.4,.7)],.7,.7)
    nav.estado='YENDO';nav.seguidor_ida.reiniciar(nav.ruta,0.)
    e=EntradaNavegacion(1.,Pose2D(.7,.7,0.),True,wall(.165),[(.865,y+.7) for _,y in wall(.165)],estampa_scan=(1,0))
    cmd=nav._avanzar(e)
    assert cmd.es_cero() and nav.ultima_guardia['esperando_frenado']
    # Avanza el scan dos segundos, aunque el reloj de pared sólo avanzó uno.
    e=replace(e,ahora=2.,estampa_scan=(3,0))
    cmd=nav._avanzar(e)
    assert cmd.vx<0. and cmd.wz==0. and nav.ultima_guardia['recuperacion_margen']
    e=replace(e,ahora=7.,estampa_scan=(7,0))
    cmd=nav._avanzar(e)
    assert cmd.es_cero() and nav.estado=='BLOQUEADO'


def test_recuperacion_no_agota_presupuesto_por_rtf_bajo():
    _,original=nueva();fp=footprint()
    nav=IdaVuelta(original.grilla,ParametrosCosto(.12,fp['radius_m'],.04,.25,5.),
                  original.seguidor_ida.p,fp,replace(original.p,regreso_automatico=False))
    nav.largada=(.7,.7,0.);nav.meta=(1.4,.7)
    nav.ruta=Ruta([(.7,.7),(1.4,.7)],.7,.7)
    nav.estado='YENDO';nav.seguidor_ida.reiniciar(nav.ruta,0.)
    e=EntradaNavegacion(1.,Pose2D(.7,.7,0.),True,wall(.165),[],estampa_scan=(1,0))
    assert nav._avanzar(e).es_cero()
    # Dos segundos de sensores, diecisiete de pared: el frenado terminó
    # y todavía queda presupuesto de recuperación medido por sensores.
    cmd=nav._avanzar(replace(e,ahora=18.,estampa_scan=(3,0)))
    assert cmd.vx<0 and cmd.wz==0.
    assert nav.ultima_guardia['recuperacion_margen']
    # El límite de pared sigue evitando recuperación indefinida sin reloj.
    assert nav._avanzar(replace(e,ahora=23.,estampa_scan=(3,0))).es_cero()
