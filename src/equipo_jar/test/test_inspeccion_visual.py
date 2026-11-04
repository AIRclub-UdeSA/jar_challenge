"""Inspección por observaciones, sin coordenadas ni tamaños de mundos reales."""
import math
import numpy as np
from equipo_jar.vistas_percepcion import proponer_ajustes,espacio_libre
from equipo_jar.navegacion.grilla import Grilla
from equipo_jar.coordinador_mision import CoordinadorMision
from equipo_jar.mision import elegir_vista,ParametrosMision,clave_vista
from test_mision import nueva
from test_mision_inspecciones import Pendiente


def surface(quality=.2):
    return dict(superficie_x=1.7,superficie_y=.7,normal_x=-1.,normal_y=0.,calidad_visual=quality)

def setup():
    _,nav=nueva();m=CoordinadorMision(nav,0.,executor=Pendiente());m.ultima_pose=(.7,.7,0.)
    return m,nav

def test_ajustes_cortos_con_optica_correcta_y_destinos_libres():
    g=Grilla(.025,0.,0.,0.,120,120,np.zeros((120,120),bool))
    g.ocupada[:,50:55]=True # La observación acredita rojo detrás de rendija no resuelta en mapa.
    ext=(.04,.025,.03)
    vs=proponer_ajustes(surface(),(.7,.7,0.),g,espacio_libre(g),ext,(0.,1.))
    assert vs
    for v in vs:
        assert .099<=math.dist((v['x'],v['y']),(.7,.7))<=.301
        assert not g.ocupada[g.a_celda(v['x'],v['y'])]
        c,s=math.cos(v['yaw']),math.sin(v['yaw']);cx=v['x']+c*ext[0]-s*ext[1];cy=v['y']+s*ext[0]+c*ext[1]
        bearing=math.atan2(.7-cy,1.7-cx)
        assert abs(math.atan2(math.sin(bearing-v['yaw']-ext[2]),math.cos(bearing-v['yaw']-ext[2])))<.001
    assert any(v['prioridad_visual']>.9 for v in vs)
    assert any(v['prioridad_visual']<-.9 for v in vs)

def test_recuerda_superficie_al_perder_rojo_y_no_la_convierte_en_reporte():
    m,nav=setup();m.percepcion(dict(reportes=0,diagnostico=dict(vistas_incompletas=[surface()])))
    assert m.pendientes_inspeccion and any(v.get('inspeccion_local') for v in m.sugeridas)
    m.percepcion(dict(motivo='SIN_REGION_ROJA',reportes=0))
    assert len(m.pendientes_inspeccion)==1 and m.reportes==0
    m.percepcion(dict(reportes=1,reportadas=[dict(x=1.72,y=.72)]))
    assert not m.pendientes_inspeccion
    assert not any(math.dist(m.objetivo(v),(1.72,.72))<.3 for v in m.sugeridas)

def test_realimentacion_continua_direccion_que_mejora_y_revierte_si_empeora():
    m,nav=setup();m.actualizar_inspeccion(surface(.2));m.estado='OBSERVANDO'
    m.ultima_pose=(.7,.8,0.);m.actualizar_inspeccion(surface(.4))
    assert np.dot(m.pendientes_inspeccion[0]['direccion'],(0.,1.))>.99
    m.ultima_pose=(.7,.9,0.);m.actualizar_inspeccion(surface(.1))
    assert np.dot(m.pendientes_inspeccion[0]['direccion'],(0.,1.))<-.99

def test_sin_rojo_en_vista_local_es_realimentacion_negativa_no_objetivo_resuelto():
    m,nav=setup();m.actualizar_inspeccion(surface(.4));m.estado='OBSERVANDO'
    m.vista=dict(x=.7,y=.8,yaw=0.,mirar_x=1.7,mirar_y=.7,inspeccion_local=True)
    m.ultima_pose=(.7,.8,0.);m.percepcion(dict(motivo='SIN_REGION_ROJA'))
    assert len(m.pendientes_inspeccion)==1
    assert m.pendientes_inspeccion[0]['direccion'][1]<-.99
    assert m.reportes==0

def test_fallo_local_no_elimina_ajuste_vecino_necesario_para_rendija():
    m,nav=setup();m.vista=dict(x=.7,y=.8,yaw=0.,inspeccion_local=True)
    m._descartar(fallo=True)
    assert clave_vista((.7,.8,0.)) in m.descartadas
    assert (*clave_vista((.7,.85,.02))[:2],None) not in m.descartadas

def test_inspecciones_agotadas_conservan_objetivo_pendiente_sin_bucle():
    m,nav=setup();s=surface();v=dict(x=.7,y=.8,yaw=0.,mirar_x=1.7,mirar_y=.7)
    for _ in range(m.p.max_solicitudes_vista):m.anotar_intento(v)
    m.percepcion(dict(diagnostico=dict(vistas_incompletas=[s])))
    assert len(m.pendientes_inspeccion)==1 and not m.sugeridas
    assert not m.resumen()['objetivo_cumplido']

def test_selector_prefiere_ajuste_en_direccion_que_mejoro_imagen():
    _,nav=nueva()
    class NoGain:
        def ganancia(self,pose):return 0
    a=dict(x=.7,y=.8,yaw=0.,mirar_x=1.7,mirar_y=.7,inspeccion_local=True,prioridad_visual=1.)
    b=dict(a,y=.6,prioridad_visual=-1.)
    v=elegir_vista(nav.grilla,nav.costos,(.7,.7,0.),(.7,.7,0.),NoGain(),set(),600.,ParametrosMision(),[b,a])
    assert v['y']==.8 and v['inspeccion_local']


def test_rojo_rgb_sin_profundidad_centra_camara_sin_inventar_reporte():
    m,nav=setup();m.estado='YENDO_VISTA';m.vista=dict(x=1.2,y=1.2,yaw=0.,sugerida=False)
    obs=dict(estampa_s=1.,pose_camara=[.74,.725,0.],campo_horizontal_rad=1.,
             region_roja=True,regiones_rojas=[dict(desvio_rad=-.25,area_relativa=.01)])
    m.observar(obs,1.,(.04,.025,0.))
    assert m.estado=='BUSCANDO_VISTA' and m.pista_rgb
    assert math.dist((m.pista_rgb['x'],m.pista_rgb['y']),(.7,.7))<1e-9
    assert m.pista_rgb['yaw']==-.25
    assert m.reportes==0 and not m.pendientes_inspeccion

def test_centrado_rgb_ya_intentado_no_se_repite_en_mismo_lugar():
    m,nav=setup();m.centrados_rgb=[(.7,.7,-.25)]
    obs=dict(estampa_s=1.,pose_camara=[.74,.725,0.],campo_horizontal_rad=1.,
             region_roja=True,regiones_rojas=[dict(desvio_rad=-.25,area_relativa=.01)])
    m.observar(obs,1.,(.04,.025,0.))
    assert m.pista_rgb is None

def test_rgb_ambiguo_o_desvio_fuera_de_imagen_no_manda_giro():
    m,nav=setup()
    obs=dict(estampa_s=1.,pose_camara=[.74,.725,0.],campo_horizontal_rad=1.,
             region_roja=True,regiones_rojas=[dict(desvio_rad=float('nan')),dict(desvio_rad=2.)])
    m.observar(obs,1.,(.04,.025,0.))
    assert m.pista_rgb is None


def test_no_recentra_region_alineada_con_victima_ya_reportada():
    m,nav=setup();m.reportadas=[(1.7,.7)]
    obs=dict(estampa_s=1.,pose_camara=[.74,.725,-.25],campo_horizontal_rad=1.,
             region_roja=True,regiones_rojas=[dict(desvio_rad=.225,area_relativa=.01)])
    m.observar(obs,1.,(.04,.025,0.))
    assert m.pista_rgb is None

def test_otro_rumbo_rojo_no_se_descarta_por_existir_una_reportada():
    m,nav=setup();m.reportadas=[(1.7,.7)]
    obs=dict(estampa_s=1.,pose_camara=[.74,.725,0.],campo_horizontal_rad=1.,
             region_roja=True,regiones_rojas=[dict(desvio_rad=-.35,area_relativa=.01)])
    m.observar(obs,1.,(.04,.025,0.))
    assert m.pista_rgb is not None and m.pista_rgb['yaw']==-.35

def test_rgb_actualiza_calidad_de_ajuste_aunque_no_haya_nube_aceptada():
    m,nav=setup();m.actualizar_inspeccion(surface(.2));m.estado='OBSERVANDO'
    m.ultima_pose=(.7,.8,0.);m.vista=dict(x=.7,y=.8,yaw=0.,mirar_x=1.7,mirar_y=.7,inspeccion_local=True)
    obs=dict(estampa_s=1.,pose_camara=[.74,.825,0.],campo_horizontal_rad=1.,region_roja=True,
             regiones_rojas=[dict(desvio_rad=-.13,semiancho_rad=.05,calidad_visual=.4,area_relativa=.01)])
    m.observar(obs,1.,(.04,.025,0.))
    assert m.pendientes_inspeccion[0]['direccion'][1]>.99
    assert m.pendientes_inspeccion[0]['calidad_mejor']==.4


def test_centrado_no_admisible_no_interrumpe_viaje_en_cada_frame():
    m,nav=setup();m.estado='YENDO_VISTA'
    regular=dict(x=1.5,y=.7,yaw=0.,sugerida=False)
    m.vista=regular.copy()
    obs=dict(estampa_s=1.,pose_camara=[.74,.725,0.],campo_horizontal_rad=1.,
        region_roja=True,regiones_rojas=[dict(desvio_rad=-.48,semiancho_rad=.03,
                                            area_relativa=.01)])
    m.observar(obs,1.,(.04,.025,0.))
    assert m.estado=='BUSCANDO_VISTA' and len(m.solicitudes_centrado_rgb)==1
    # El selector rechaza giro y elige otra ruta; pista sigue pendiente.
    m.estado='YENDO_VISTA';m.vista=regular.copy()
    for k in range(2,8):m.observar(dict(obs,estampa_s=float(k)),float(k),(.04,.025,0.))
    assert m.estado=='YENDO_VISTA' and m.pista_rgb
    assert len(m.solicitudes_centrado_rgb)==1 and not m.centrados_rgb


def test_posicion_nueva_permite_reconsiderar_giro_pendiente():
    m,nav=setup();m.estado='YENDO_VISTA';m.vista=dict(x=1.5,y=.7,yaw=0.,sugerida=False)
    obs=dict(estampa_s=1.,pose_camara=[.74,.725,0.],campo_horizontal_rad=1.,
             region_roja=True,regiones_rojas=[dict(desvio_rad=-.25,area_relativa=.01)])
    m.observar(obs,1.,(.04,.025,0.))
    m.estado='YENDO_VISTA';m.vista=dict(x=1.5,y=.7,yaw=0.,sugerida=False)
    m.observar(dict(obs,estampa_s=2.,pose_camara=[1.04,.725,0.]),2.,(.04,.025,0.))
    assert m.estado=='BUSCANDO_VISTA' and len(m.solicitudes_centrado_rgb)==2
