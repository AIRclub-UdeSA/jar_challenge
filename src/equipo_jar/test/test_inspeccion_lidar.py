"""Un obstáculo observado puede pedir mirada RGB sin ser una víctima conocida."""
from dataclasses import replace
import math
import numpy as np
from equipo_jar.vistas_percepcion import proponer_inspecciones_lidar,espacio_libre
from equipo_jar.navegacion.grilla import Grilla
from equipo_jar.navegacion.ida_vuelta import EntradaNavegacion
from equipo_jar.navegacion.tipos import Pose2D
from equipo_jar.mision import ParametrosMision,elegir_vista
from equipo_jar.coordinador_mision import CoordinadorMision
from test_mision import nueva
from test_mision_inspecciones import Pendiente


def setup(occluded=True):
    _,nav=nueva();nav.largada=(.7,.7,0.)
    if occluded:
        nav.grilla.ocupada[30,:]=True
        nav.capa.distancia_mapa=espacio_libre(nav.grilla)
    m=CoordinadorMision(nav,0.,executor=Pendiente());m.ultima_pose=(.7,.7,0.);m.extrinseca=(0.,0.,0.)
    m.estado='YENDO_VISTA';m.vista=dict(x=1.5,y=.7,yaw=0.,sugerida=False)
    points=[(.64,1.8),(.66,1.8),(.68,1.8)]
    e=EntradaNavegacion(1.,Pose2D(.7,.7,0.),True,[],points,estampa_scan=(1,0))
    return m,nav,e


def test_abertura_real_pide_mirada_y_misma_estampa_no_duplica_solicitud():
    m,nav,e=setup();m.actualizar_pistas_lidar(e)
    assert len(m.pistas_lidar)==1 and m.estado=='BUSCANDO_VISTA'
    m.actualizar_pistas_lidar(e)
    assert len(m.solicitudes_lidar)==1
    v=m.pistas_lidar[0]
    assert v['pista_lidar'] and v['puntos_soporte']==3 and v['mapa_ocluye']
    assert math.dist((v['x'],v['y']),(.7,.7))<1e-9
    assert v['yaw']>1.5 and m.reportes==0


def test_pared_conocida_no_pide_inspeccion_y_no_se_modifica_mapa():
    occ=np.zeros((80,80),bool);occ[40,:]=True
    g=Grilla(.05,0.,0.,0.,80,80,occ);before=g.ocupada.copy()
    assert not proponer_inspecciones_lidar([(.98,2.01),(1.02,2.01)],(1.,1.,0.),g,espacio_libre(g),.1)
    np.testing.assert_array_equal(g.ocupada,before)


def test_rendija_cerrada_en_raster_no_impide_pedir_mirada_sin_cruzarla():
    occ=np.zeros((80,80),bool);occ[30,:]=True
    g=Grilla(.05,0.,0.,0.,80,80,occ);before=g.ocupada.copy()
    views=proponer_inspecciones_lidar([(.98,2.),(1.02,2.)],(1.,1.,0.),g,espacio_libre(g),.1)
    assert views and views[0]['y']==1.
    np.testing.assert_array_equal(g.ocupada,before)


def test_objeto_fuera_de_alcance_o_nan_no_crea_pista():
    _,nav=nueva()
    for points in [[(.7,3.5),(.72,3.5)],[(float('nan'),1.)]]:
        assert not proponer_inspecciones_lidar(points,(.7,.7,0.),nav.grilla,nav.distancia,.1)


def test_ya_reportada_no_detiene_exploracion_por_lidar():
    m,nav,e=setup();m.reportadas=[(.66,1.8)]
    for k in range(3):m.actualizar_pistas_lidar(replace(e,ahora=1.+k*.2,estampa_scan=(k,0)))
    assert not m.pistas_lidar and m.estado=='YENDO_VISTA'


def test_no_repite_intento_en_mismo_objeto_y_posicion_pero_admite_otra_vista():
    m,nav,e=setup();m.intentos_lidar=[dict(point=(.66,1.8),anchor=(.7,.7))]
    for k in range(3):m.actualizar_pistas_lidar(replace(e,ahora=1.+k*.2,estampa_scan=(k,0)))
    assert not m.pistas_lidar
    m.ultima_pose=(1.1,.7,0.)
    m.actualizar_pistas_lidar(replace(e,ahora=1.7,estampa_scan=(4,0)))
    assert m.pistas_lidar


def test_no_interrumpe_si_objeto_ya_esta_en_campo_de_camara():
    m,nav,e=setup();e=replace(e,puntos_mapa=[(1.8,.68),(1.8,.72)])
    for k in range(3):m.actualizar_pistas_lidar(replace(e,ahora=1.+k*.2,estampa_scan=(k,0)))
    assert not m.pistas_lidar and m.estado=='YENDO_VISTA'


def test_presupuesto_inspeccion_impide_perseguir_objeto_indefinidamente():
    m,nav,e=setup();v=dict(x=.7,y=.7,yaw=1.6,mirar_x=.66,mirar_y=1.8)
    for _ in range(m.p.max_solicitudes_vista):m.anotar_intento(v)
    for k in range(3):m.actualizar_pistas_lidar(replace(e,ahora=1.+k*.2,estampa_scan=(k,0)))
    assert not m.pistas_lidar and m.reportes==0


def test_selector_conserva_origen_lidar_sin_inventar_deteccion():
    m,nav,e=setup()
    for k in range(2):m.actualizar_pistas_lidar(replace(e,ahora=1.+k*.2,estampa_scan=(k,0)))
    class NoGain:
        def ganancia(self,pose):return 0
    v=elegir_vista(nav.grilla,nav.costos,(.7,.7,0.),(.7,.7,0.),NoGain(),set(),600.,ParametrosMision(),m.pistas_lidar)
    assert v['pista_lidar'] and v['sugerida'] and v['puntos_soporte']==3


def test_un_solo_rayo_por_hueco_real_no_se_descarta_por_no_parecer_caja():
    occ=np.zeros((80,80),bool);occ[30,:]=True
    g=Grilla(.05,0.,0.,0.,80,80,occ)
    views=proponer_inspecciones_lidar([(1.,2.)],(1.,1.,0.),g,espacio_libre(g),.1)
    assert len(views)==1 and views[0]['puntos_soporte']==1 and views[0]['mapa_ocluye']


def test_un_punto_aislado_en_scan_abierto_no_interrumpe_sin_persistencia():
    m,nav,e=setup(occluded=False);m.actualizar_pistas_lidar(replace(e,puntos_mapa=[(.66,1.8)]))
    assert not m.pistas_lidar and m.reportes==0


def test_pista_por_rendija_se_conserva_si_siguiente_scan_ya_no_la_ve():
    m,nav,e=setup();nav.grilla.ocupada[30,:]=True
    nav.capa.distancia_mapa=espacio_libre(nav.grilla)
    m.actualizar_pistas_lidar(replace(e,puntos_mapa=[(.66,1.8)]))
    assert len(m.pistas_lidar)==1 and m.pistas_lidar[0]['mapa_ocluye']
    m.actualizar_pistas_lidar(replace(e,ahora=1.2,estampa_scan=(1,200000000),puntos_mapa=[]))
    assert len(m.pistas_lidar)==1 and m.reportes==0


def test_rojo_observado_prevalece_sobre_mirada_a_obstaculo_sin_clasificar():
    m,nav,e=setup();m.vista=dict(x=.7,y=.7,yaw=1.6,sugerida=True,pista_lidar=True)
    obs=dict(estampa_s=1.,pose_camara=[.74,.725,0.],campo_horizontal_rad=1.,
             region_roja=True,regiones_rojas=[dict(desvio_rad=-.25,area_relativa=.01)])
    m.observar(obs,1.,(.04,.025,0.))
    assert m.estado=='BUSCANDO_VISTA' and m.pista_rgb and m.reportes==0


def test_pista_rechazada_no_cancela_mismo_viaje_en_cada_scan():
    m,nav,e=setup()
    for k in range(2):m.actualizar_pistas_lidar(replace(e,ahora=1.+k*.2,estampa_scan=(k,0)))
    assert m.estado=='BUSCANDO_VISTA' and len(m.solicitudes_lidar)==1
    # El planificador rechazó la orientación de la pista y eligió seguir viaje.
    m.estado='YENDO_VISTA';m.vista=dict(x=1.5,y=.7,yaw=0.,sugerida=False)
    for k in range(2,8):m.actualizar_pistas_lidar(replace(e,ahora=1.+k*.2,estampa_scan=(k,0)))
    assert m.estado=='YENDO_VISTA' and m.pistas_lidar
    assert len(m.solicitudes_lidar)==1


def test_si_pose_exacta_no_admite_mirada_se_elige_otro_punto_cercano():
    m,nav,e=setup()
    for k in range(2):m.actualizar_pistas_lidar(replace(e,ahora=1.+k*.2,estampa_scan=(k,0)))
    class NoGain:
        def ganancia(self,pose):return 0
    v=elegir_vista(nav.grilla,nav.costos,(.7,.7,0.),(.7,.7,0.),NoGain(),set(),600.,
        ParametrosMision(),m.vistas_lidar(),filtro_pose=lambda x,y,yaw:math.dist((x,y),(.7,.7))>.05)
    assert v and v['pista_lidar']
    assert .05<math.dist((v['x'],v['y']),(.7,.7))<=.301


def test_fallo_de_giro_lidar_no_descarta_toda_vecindad():
    from equipo_jar.mision import clave_vista
    m,nav,e=setup();m.vista=dict(x=.7,y=.7,yaw=1.6,pista_lidar=True)
    m._descartar(fallo=True)
    assert clave_vista((.7,.7,1.6)) in m.descartadas
    assert (*clave_vista((.75,.7,1.7))[:2],None) not in m.descartadas


def test_superficie_abierta_persistente_no_desplaza_cobertura():
    m,nav,e=setup(occluded=False)
    for k in range(5):
        m.actualizar_pistas_lidar(replace(e,ahora=1.+k*.2,estampa_scan=(k,0)))
    assert m.estado=='YENDO_VISTA' and not m.pistas_lidar
    assert not m.solicitudes_lidar and m.reportes==0


def test_presupuesto_ciego_de_superficie_no_se_renueva_al_cambiar_de_lugar():
    m,nav,e=setup()
    m.intentos_lidar=[dict(point=(.66,1.8),anchor=(x,.7))
                     for x in (-.5,0.,1.5)]
    m.actualizar_pistas_lidar(e)
    assert not m.pistas_lidar and m.estado=='YENDO_VISTA'
    m.ultima_pose=(1.1,.7,0.)
    m.actualizar_pistas_lidar(replace(e,ahora=1.2,estampa_scan=(2,0)))
    assert not m.pistas_lidar


def test_nueva_evidencia_roja_conserva_prioridad_tras_presupuesto_lidar():
    m,nav,e=setup()
    m.intentos_lidar=[dict(point=(.66,1.8),anchor=(x,.7))
                     for x in (-.5,0.,1.5)]
    m.actualizar_pistas_lidar(e)
    assert not m.pistas_lidar
    obs=dict(estampa_s=1.,pose_camara=[.74,.725,0.],campo_horizontal_rad=1.,
             region_roja=True,regiones_rojas=[dict(desvio_rad=-.25,area_relativa=.01)])
    m.observar(obs,1.,(.04,.025,0.))
    assert m.estado=='BUSCANDO_VISTA' and m.pista_rgb and m.reportes==0


def test_rojo_recortado_se_centra_aunque_haya_inspeccion_pendiente():
    m,nav,e=setup()
    m.pendientes_inspeccion=[dict(superficie_x=.66,superficie_y=1.8,
                                normal_x=0.,normal_y=-1.)]
    m.vista=dict(x=.7,y=.7,yaw=1.6,sugerida=True,inspeccion_local=True)
    obs=dict(estampa_s=1.,pose_camara=[.74,.725,0.],campo_horizontal_rad=1.,
        region_roja=True,regiones_rojas=[dict(desvio_rad=-.48,semiancho_rad=.03,
                                            area_relativa=.01)])
    m.observar(obs,1.,(.04,.025,0.))
    assert m.estado=='BUSCANDO_VISTA' and m.pista_rgb['recorte_rgb']
    assert m.reportes==0


def test_inspeccion_ciega_completa_sin_rojo_no_se_repite_desde_otro_lugar():
    m,nav,e=setup();m.estado='OBSERVANDO';m.t_en_meta=0.
    m.datos_visuales_en_meta=2;m.ultima_visual=10.;m.rojo_en_vista=False
    m.vista=dict(x=.7,y=.7,yaw=1.6,sugerida=True,pista_lidar=True,
                 mirar_x=.66,mirar_y=1.8,regreso_m=1.)
    m.paso(10.,10.,replace(e,ahora=10.),3)
    assert m.superficies_lidar_sin_rojo==[(.66,1.8)] and not m.pistas_lidar
    m.ultima_pose=(1.1,.7,0.)
    m.actualizar_pistas_lidar(replace(e,ahora=10.2,estampa_scan=(2,0)))
    assert not m.pistas_lidar
    # Nueva evidencia roja sí puede reclamar inspección de esta región.
    obs=dict(estampa_s=2.,pose_camara=[1.14,.725,0.],campo_horizontal_rad=1.,
             region_roja=True,regiones_rojas=[dict(desvio_rad=-.25,area_relativa=.01)])
    m.observar(obs,10.2,(.04,.025,0.));assert m.pista_rgb
