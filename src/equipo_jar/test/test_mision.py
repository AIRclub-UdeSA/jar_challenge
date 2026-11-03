"""M4 conserva largada, guarda giros y elige vistas sin conocer víctimas."""
from concurrent.futures import Future
from dataclasses import replace
import math

import numpy as np
import pytest

from equipo_jar.cobertura_visual import CoberturaVisual
from equipo_jar.coordinador_mision import CoordinadorMision
from equipo_jar.mision import ParametrosMision, clave_vista, elegir_vista
from equipo_jar.navegacion.ida_vuelta import EntradaNavegacion, IdaVuelta
from equipo_jar.navegacion.tipos import Pose2D
from test_nav_ida_vuelta import Mundo, P, P_COSTO, P_SEG, HUELLA, DT


def nueva():
    mundo = Mundo(np.zeros((60, 60), bool))
    nav = IdaVuelta(mundo.grilla, P_COSTO, P_SEG, HUELLA,
                   replace(P, regreso_automatico=False))
    return mundo, nav


def step(nav, mundo, pose, k, lista=True):
    base, mapa = mundo.puntos(pose)
    entrada = EntradaNavegacion(k*DT, pose, lista, base, mapa, .01)
    cmd = nav.paso(entrada)
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    return Pose2D(pose.x+(c*cmd.vx-s*cmd.vy)*DT,
                  pose.y+(s*cmd.vx+c*cmd.vy)*DT,
                  pose.yaw+cmd.wz*DT), cmd


def test_dos_vistas_con_giro_y_regreso_conservan_largada():
    mundo, nav = nueva()
    pose = Pose2D(.7, .7, 0.)
    original = (pose.x, pose.y)
    nav.continuar_meta((1.3, .7), math.pi/2)
    k = 0
    for target, heading in [((1.3, .7), math.pi/2), ((1.3, 1.3), math.pi)]:
        if k:
            nav.continuar_meta(target, heading)
        for _ in range(600):
            pose, cmd = step(nav, mundo, pose, k)
            k += 1
            if nav.estado in ('EN_META', 'FALLO'):
                break
        assert nav.estado == 'EN_META', nav.resumen()
        assert math.dist((pose.x, pose.y), target) < .08
        assert abs(math.atan2(math.sin(pose.yaw-heading),
                              math.cos(pose.yaw-heading))) < .065
        for _ in range(10):
            pose, cmd = step(nav, mundo, pose, k)
            k += 1
            assert cmd.es_cero()
            assert nav.estado == 'EN_META'
        assert nav.largada[:2] == original
    nav.pedir_regreso()
    for _ in range(800):
        pose, cmd = step(nav, mundo, pose, k)
        k += 1
        if nav.estado in ('EN_LARGADA', 'FALLO'):
            break
    assert nav.estado == 'EN_LARGADA', nav.resumen()
    assert math.dist((pose.x, pose.y), original) <= .055


def test_sin_localizacion_no_arranca_regreso():
    mundo, nav = nueva()
    pose = Pose2D(.7, .7, 0.)
    nav.continuar_meta((1.2, .7), .5)
    step(nav, mundo, pose, 0)
    nav.pedir_regreso()
    _, cmd = step(nav, mundo, pose, 1, lista=False)
    assert cmd.es_cero() and nav.solicitud_regreso


def test_metas_invalidas_no_sobrescriben_meta():
    _, nav = nueva()
    nav.continuar_meta((1., 1.), 0.)
    with pytest.raises(ValueError):
        nav.continuar_meta((float('nan'), 1.), 0.)
    assert nav.meta == (1., 1.)
    with pytest.raises(ValueError):
        nav.continuar_meta((1., 1.), float('nan'))


def test_elige_rumbo_y_descarta_sala_inaccesible():
    mundo, nav = nueva()
    nav.grilla.ocupada[:, 30] = True
    nav.costos[:, 30] = math.inf
    cov = CoberturaVisual(nav.grilla)
    p = replace(ParametrosMision(), max_candidatas=8)
    result = elegir_vista(nav.grilla, nav.costos, (.7, .7, 0.),
                          (.7, .7, 0.), cov, set(), 600., p)
    assert result is not None and result['x'] < 1.5
    assert result['ganancia'] > 0
    assert math.isfinite(result['regreso_m'])


def test_no_gasta_reserva_en_otra_vista():
    mundo, nav = nueva()
    cov = CoberturaVisual(nav.grilla)
    p = replace(ParametrosMision(), max_candidatas=4)
    assert elegir_vista(nav.grilla, nav.costos, (.7, .7, 0.),
                        (.7, .7, 0.), cov, set(), 40., p) is None


class ExecutorInmediato:
    def submit(self, funcion, *args):
        f = Future()
        try:
            f.set_result(funcion(*args))
        except Exception as error:
            f.set_exception(error)
        return f
    def shutdown(self, **kwargs):
        pass


def test_n_completo_vuelve_y_no_finaliza_hasta_llegar():
    mundo, nav = nueva()
    pose = Pose2D(.7, .7, 0.)
    entrada = EntradaNavegacion(0., pose, True, [], [])
    nav.paso(entrada)
    m = CoordinadorMision(nav, 0., executor=ExecutorInmediato())
    m.percepcion(dict(reportes=2))
    m.paso(1., 1., entrada, 2)
    assert m.estado == 'REGRESANDO'
    assert not m.finalizar
    nav.estado = 'EN_LARGADA'
    m.paso(2., 2., entrada, 2)
    assert m.finalizar and m.estado == 'FINALIZADA'


def test_camara_no_disponible_no_manda_metas_y_retorna_acotado():
    mundo, nav = nueva()
    pose = Pose2D(.7, .7, 0.)
    entrada = EntradaNavegacion(0., pose, True, [], [])
    nav.paso(entrada)
    m = CoordinadorMision(nav, 0., executor=ExecutorInmediato())
    m.paso(1., 1., entrada, 2)
    assert nav.meta is None
    m.paso(18., 18., entrada, 2)
    assert m.estado == 'REGRESANDO'


def test_reloj_retrocede_cancela_sin_done():
    mundo, nav = nueva()
    entrada = EntradaNavegacion(0., Pose2D(.7, .7, 0.), True, [], [])
    nav.paso(entrada)
    m = CoordinadorMision(nav, 10., executor=ExecutorInmediato())
    m.paso(9., 1., entrada, 2)
    assert m.estado == 'FALLO_RELOJ'
    assert not m.finalizar and nav.estado == 'FALLO'


def test_observacion_repetida_no_cambia_rumbo_de_cobertura():
    _, nav = nueva()
    m = CoordinadorMision(nav, 0., executor=ExecutorInmediato())
    obs = dict(estampa_s=1., pose_camara=[1.5, 1.5, 0.],
               campo_horizontal_rad=1.)
    m.observar(obs, 1., (0., 0., 0.))
    before = m.cobertura.observada.copy()
    obs['pose_camara'][2] = math.pi
    m.observar(obs, 2., (0., 0., 0.))
    assert np.array_equal(before, m.cobertura.observada)
    assert m.ultima_visual == 1.


@pytest.mark.parametrize('kwargs', [dict(duracion_s=30.), dict(observacion_s=0.),
                                    dict(max_candidatas=1.2)])
def test_parametros_de_mision_invalidos(kwargs):
    with pytest.raises(ValueError):
        ParametrosMision(**kwargs)

def test_sugerencia_m3_alcanzable_tiene_prioridad_sobre_cobertura_general():
    _, nav = nueva()
    coverage = CoberturaVisual(nav.grilla)
    p = replace(ParametrosMision(), max_candidatas=4)
    vista = dict(x=1.2, y=1.2, yaw=math.pi/2)
    result = elegir_vista(nav.grilla, nav.costos, (.7, .7, 0.),
                          (.7, .7, 0.), coverage, set(), 600., p, [vista])
    assert result['sugerida']
    assert (result['x'], result['y']) == (1.2, 1.2)


def test_sugerencia_inaccesible_no_bloquea_exploracion_general():
    _, nav = nueva()
    coverage = CoberturaVisual(nav.grilla)
    p = replace(ParametrosMision(), max_candidatas=4)
    result = elegir_vista(nav.grilla, nav.costos, (.7, .7, 0.),
                          (.7, .7, 0.), coverage, set(), 600., p,
                          [dict(x=30., y=30., yaw=0.)])
    assert result is not None and not result['sugerida']


def test_no_persigue_superficie_de_una_victima_ya_reportada():
    _, nav = nueva()
    m = CoordinadorMision(nav, 0., executor=ExecutorInmediato())
    a = dict(x=.7, y=.7, yaw=0., mirar_x=1.1, mirar_y=1.)
    b = dict(x=1.7, y=1.7, yaw=0., mirar_x=2.1, mirar_y=2.)
    m.percepcion(dict(reportes=1, reportadas=[dict(x=1., y=1.)],
                      vistas_sugeridas=[a, b]))
    assert m.sugeridas == [b]

def test_lotes_no_descartan_posiciones_de_observacion():
    _, nav = nueva()
    class GananciaLocal:
        def ganancia(self, pose):
            return 50 if math.dist(pose[:2], (1.025, 2.025)) < 1e-6 else 0
    p = replace(ParametrosMision(), max_candidatas=1)
    diag = {}
    vista = elegir_vista(nav.grilla, nav.costos, (.7, .7, 0.),
        (.7, .7, 0.), GananciaLocal(), set(), 600., p, diagnostico=diag)
    assert vista is not None
    assert math.dist((vista['x'], vista['y']), (1.025, 2.025)) < 1e-6
    assert diag['vistas_evaluadas'] > p.max_candidatas


def test_agotamiento_refina_y_reinicia_busqueda_si_faltan_victimas():
    _, nav = nueva()
    entrada = EntradaNavegacion(0., Pose2D(.7, .7, 0.), True, [], [])
    nav.paso(entrada)
    m = CoordinadorMision(nav, 0., executor=ExecutorInmediato())
    m.observar(dict(estampa_s=1., pose_camara=[.7, .7, 0.],
                   campo_horizontal_rad=1.), 1., (0., 0., 0.))
    for expected in ('BUSCANDO_VISTA', 'BUSCANDO_VISTA', 'BUSCANDO_VISTA'):
        m.estado = 'PLANIFICANDO'
        m.futuro = Future()
        m.futuro.set_result(dict(vista=None, diagnostico={}))
        m.paso(1., 1., entrada, 2)
        assert m.estado == expected
        assert not m.finalizar
    assert m.refinamientos == 1
    assert not m.resumen()['objetivo_cumplido']
    assert 'incompleta' in m.motivo
    assert m.reinicios_busqueda==1 and not nav.solicitud_regreso


def test_m4_reutiliza_huella_m2_para_pasar_corredor_sin_bajar_margen():
    from equipo_jar.coordinador_mision import resolver_busqueda
    from equipo_jar.navegacion.costos import ParametrosCosto, mapa_costos
    from equipo_jar.navegacion.guardia import huella_convexa_x3
    occ = np.ones((60, 100), bool)
    occ[8:52, 8:30] = False
    occ[8:52, 60:92] = False
    occ[26:34, 30:60] = False
    world = Mundo(occ)
    footprint = huella_convexa_x3(-.19, .13, .070, .080, .0845, .033, .04, .018)
    footprint['heading_uncertainty_rad'] = .1
    p_costs = ParametrosCosto(.12, footprint['radius_m'], .04, .25, 5.)
    class VerOtraSala:
        def ganancia(self, pose):
            return 50 if pose[0] > 3.2 and abs(pose[2]) < .01 else 0
    cov = VerOtraSala()
    extra = np.zeros_like(occ)
    p = ParametrosMision()
    args = (world.grilla, p_costs, extra, (.9, 1.5, 0.), (.9, 1.5, 0.),
            cov, set(), 600., p, [], (0., 0., 0.), footprint)
    result = resolver_busqueda(*args)
    assert result['vista'] is not None, result['diagnostico']
    assert not result['vista'].get('validacion_se2',False)
    assert result['diagnostico']['sondeos']==[]
    result = resolver_busqueda(*args, sondear_se2=True)
    assert result['vista'] is not None, result['diagnostico']
    assert not result['vista'].get('validacion_se2',False)
    assert result['diagnostico']['sondeos']==[]
    assert footprint['margin_m'] == .04
    assert result['vista']['camino_m'] > 2.


@pytest.mark.parametrize('seed', range(3))
def test_poda_por_cota_coincide_con_busqueda_exhaustiva(seed):
    _, nav = nueva()
    rng = np.random.default_rng(seed)
    nav.grilla.ocupada[15:22, 27] = True
    cov = CoberturaVisual(nav.grilla)
    cov.observada[:] = rng.random(len(cov.observada)) < .35
    class Exhaustiva:
        def ganancia(self, pose):
            return cov.ganancia(pose)
    p = replace(ParametrosMision(), separacion_vistas_m=.5)
    args = (nav.grilla, nav.costos, (.7, .7, .3), (.7, .7, .3))
    expected, actual, diag = [], [], {}
    a = elegir_vista(*args, Exhaustiva(), set(), 600., p,
                     alternativas=expected, limite_alternativas=4)
    b = elegir_vista(*args, cov, set(), 600., p,
                     alternativas=actual, limite_alternativas=4, diagnostico=diag)
    assert a == b
    assert expected == actual
    assert diag['vistas_con_rayos'] < diag['vistas_evaluadas']


def test_sondeo_acotado_sin_vista_no_declara_agotada_la_busqueda():
    _, nav = nueva()
    entrada = EntradaNavegacion(0., Pose2D(.7, .7, 0.), True, [], [])
    nav.paso(entrada)
    m = CoordinadorMision(nav, 0., executor=ExecutorInmediato())
    m.observar(dict(estampa_s=1., pose_camara=[.7, .7, 0.],
                   campo_horizontal_rad=1.), 1., (0., 0., 0.))
    m.estado = 'PLANIFICANDO'
    m.futuro = Future()
    m.futuro.set_result(dict(vista=None, diagnostico=dict(
        descartes_sondeo=[[5, 7, 0]], mas_sondeos=True)))
    m.paso(1., 1., entrada, 2)
    assert m.estado == 'BUSCANDO_VISTA'
    assert (5, 7, 0) in m.sondeos_pendientes_descartados
    assert not m.finalizar and not nav.solicitud_regreso
    assert m.refinamientos == 0


def test_m4_busca_otra_vista_si_m3_no_propone_en_corredor():
    from equipo_jar.navegacion.costos import distancia_a_ocupado
    _, nav = nueva()
    nav.grilla.ocupada[:] = True
    nav.grilla.ocupada[20:32, 2:58] = False
    nav.distancia = distancia_a_ocupado(nav.grilla.ocupada, nav.grilla.resolucion)
    m = CoordinadorMision(nav, 0., executor=ExecutorInmediato())
    m.ultima_pose = (.6, 1.3, 0.)
    margen_antes = nav.huella['margin_m']
    m.percepcion(dict(reportes=0, vistas_sugeridas=[], diagnostico=dict(
        vistas_incompletas=[dict(superficie_x=2., superficie_y=1.3,
                                normal_x=-1., normal_y=0.)])))
    assert m.sugeridas
    assert all(v['mirar_x'] == 2. for v in m.sugeridas)
    assert nav.huella['margin_m'] == margen_antes
    assert nav.meta is None


def test_mision_orienta_camara_durante_traslacion_mecanum():
    mundo, nav = nueva()
    pose = Pose2D(.7, .7, 0.)
    nav.continuar_meta((1.7, .7), math.pi/2)
    for k in range(30):
        pose, cmd = step(nav, mundo, pose, k)
    assert nav.estado == 'YENDO', nav.resumen()
    assert nav.seguidor_ida.yaw_mantener == math.pi/2
    assert pose.yaw > .15
    assert pose.x < 1.5


def test_limite_de_otras_vistas_es_por_objetivo_y_no_global():
    _, nav = nueva()
    m = CoordinadorMision(nav, 0., executor=ExecutorInmediato())
    a = dict(x=.8, y=.8, yaw=0., mirar_x=1.2, mirar_y=.8)
    b = dict(x=1.8, y=1.8, yaw=0., mirar_x=2.2, mirar_y=1.8)
    for _ in range(3):
        m.anotar_intento(a)
    m.uso_sugeridas = 3
    m.percepcion(dict(vistas_sugeridas=[a, b]))
    assert m.intentos_objetivo(a) == 3
    assert m.intentos_objetivo(b) == 0
    assert m.sugeridas == [b]


def test_candidato_rojo_interrumpe_una_sola_vez_vista_general():
    mundo, nav = nueva()
    pose = Pose2D(.7, .7, 0.)
    nav.continuar_meta((1.7, .7), 0.)
    step(nav, mundo, pose, 0)
    largada = nav.largada
    m = CoordinadorMision(nav, 0., executor=ExecutorInmediato())
    m.ultima_pose = (.7, .7, 0.)
    m.estado = 'YENDO_VISTA'
    general = dict(x=1.7, y=.7, yaw=0., sugerida=False)
    m.vista = general
    hint = dict(x=.8, y=1., yaw=0., mirar_x=1.4, mirar_y=1.)
    m.percepcion(dict(vistas_sugeridas=[hint]))
    assert m.estado == 'BUSCANDO_VISTA'
    assert nav.estado == 'ESPERANDO_META' and nav.meta is None
    _, cmd = step(nav, mundo, pose, 1)
    assert cmd.es_cero() and nav.largada == largada
    m.estado, m.vista = 'YENDO_VISTA', general
    m.percepcion(dict(vistas_sugeridas=[hint]))
    assert m.estado == 'YENDO_VISTA'


def test_deteccion_valida_pausa_para_confirmar_y_preserva_largada():
    mundo, nav = nueva()
    pose = Pose2D(.7, .7, 0.)
    nav.continuar_meta((1.7, .7), 0.)
    step(nav, mundo, pose, 0)
    largada = nav.largada
    m = CoordinadorMision(nav, 0., executor=ExecutorInmediato())
    m.ultima_pose = (.7, .7, 0.)
    m.estado = 'YENDO_VISTA'
    m.vista = dict(x=1.7, y=.7, yaw=0., sugerida=False)
    m.observar(dict(estampa_s=1., pose_camara=[.75, .7, 0.], campo_horizontal_rad=1.),
               1., (.05, 0., 0.))
    m.ultimo_reloj = 1.
    m.percepcion(dict(detecciones=[dict(x=1.3, y=1.2)], reportes=0))
    assert m.estado == 'CONFIRMANDO'
    _, cmd = step(nav, mundo, pose, 1)
    assert cmd.es_cero() and nav.largada == largada
    entrada = EntradaNavegacion(1.5, pose, True, [], [])
    m.paso(1.5, 1.5, entrada, 2)
    assert m.estado == 'CONFIRMANDO'
    m.percepcion(dict(reportes=1, reportadas=[dict(x=1.3, y=1.2)],
                      detecciones=[dict(x=1.3, y=1.2)]))
    m.paso(2., 2., entrada, 2)
    assert m.estado != 'CONFIRMANDO'
    assert nav.largada == largada and not m.finalizar


def test_clave_vista_no_oculta_paralaje_y_normaliza_vuelta_completa():
    assert clave_vista((1., 1., math.pi)) == clave_vista((1., 1., -math.pi))
    assert clave_vista((1., 1., .1)) == clave_vista((1., 1., .1+2*math.pi))
    assert clave_vista((1., 1., 0.)) != clave_vista((1.08, 1., 0.))


def test_filtro_pose_descarta_colision_con_margen_sin_autorizar_ruta():
    from equipo_jar.coordinador_mision import filtro_huella_estatica
    from equipo_jar.navegacion.guardia import poligono_barrido, distancias_poligono
    from equipo_jar.navegacion.grilla import Grilla
    occupied = np.zeros((60, 60), bool)
    occupied[:, 30] = True
    g = Grilla(.05, 0., 0., 0., 60, 60, occupied)
    fp = dict(corners_base_m=[(-.19,-.12),(-.19,.12),(.13,-.12),(.13,.12)],
              radius_m=.23, margin_m=.04, heading_uncertainty_rad=.10)
    libre = filtro_huella_estatica(g, occupied, fp)
    assert libre(.7, 1.4, 0.)
    assert not libre(1.4, 1.4, 0.)
    assert not libre(.02, 1.4, math.pi)
    hull, error = poligono_barrido(fp, 0., 0.)
    points = np.array([g.a_mundo(int(f), int(c)) for f,c in np.argwhere(occupied)])
    for x in np.linspace(1.1, 1.9, 9):
        for yaw in np.linspace(-math.pi, math.pi, 17):
            rot = np.array([[math.cos(yaw),-math.sin(yaw)],
                            [math.sin(yaw),math.cos(yaw)]])
            collision = np.any(distancias_poligono((points-(x,1.4))@rot,hull)
                               <= fp['margin_m']+error+g.resolucion/math.sqrt(2))
            assert libre(x,1.4,yaw) == (not collision)


def test_busqueda_admite_rumbos_intermedios_del_planificador_m2():
    _, nav = nueva()
    class VistaIntermedia:
        def ganancia(self, pose):
            return 10 if abs(pose[2]-math.pi/16) < 1e-9 else 0
    chosen = elegir_vista(nav.grilla, nav.costos, (.7,.7,0.), (.7,.7,0.),
                         VistaIntermedia(), set(), 600., ParametrosMision(), rumbos=32)
    assert chosen is not None and chosen['yaw'] == pytest.approx(math.pi/16)


def test_vista_alcanzable_avanza_sin_sondeos_especulativos(monkeypatch):
    from equipo_jar.coordinador_mision import resolver_busqueda
    _,nav=nueva()
    def prohibido(*args,**kwargs):
        raise AssertionError('Una vista XY válida no necesita sondear otra ruta')
    monkeypatch.setattr('equipo_jar.coordinador_mision.planificar_orientado',prohibido)
    result=resolver_busqueda(nav.grilla,nav.p_costos,np.zeros_like(nav.grilla.ocupada),
        (.7,.7,0.),(.7,.7,0.),CoberturaVisual(nav.grilla),set(),600.,
        ParametrosMision(),[],(0.,0.,0.),nav.huella)
    assert result['vista'] is not None
    assert result['diagnostico']['sondeos']==[]


def test_plazo_busqueda_retrocede_por_historial_en_vez_de_quedar_quieto():
    _,nav=nueva();pose=Pose2D(1.7,.7,0.)
    nav.largada=(.7,.7,0.)
    nav.estado='ESPERANDO_META'
    m=CoordinadorMision(nav,0.,executor=ExecutorInmediato())
    m.observar(dict(estampa_s=1.,pose_camara=[1.75,.7,0.],campo_horizontal_rad=1.),21.,(.05,0.,0.))
    m.estado='PLANIFICANDO';m.futuro=Future();m.t_plan=0.
    e=EntradaNavegacion(1.,pose,True,[],[])
    m.paso(1.,21.,e,2)
    assert m.estado=='YENDO_VISTA'
    assert m.vista['retroceso'] and nav.meta==(.7,.7)
    assert not m.finalizar and len(m.retrocesos_usados)==1


def test_sondeos_sucesivos_tienen_limite_total_aunque_cada_uno_termine():
    _,nav=nueva();pose=Pose2D(1.7,.7,0.)
    nav.largada=(.7,.7,0.);nav.estado='ESPERANDO_META'
    m=CoordinadorMision(nav,0.,executor=ExecutorInmediato())
    m.observar(dict(estampa_s=1.,pose_camara=[1.75,.7,0.],campo_horizontal_rad=1.),30.,(.05,0.,0.))
    m.estado='PLANIFICANDO';m.futuro=Future();m.t_plan=29.;m.t_busqueda_total=(0.,0.)
    m.paso(1.,30.,EntradaNavegacion(1.,pose,True,[],[]),2)
    assert m.estado=='YENDO_VISTA' and m.vista['retroceso']


def test_vistas_laterales_m3_no_se_desplazan_por_respaldo_frontal():
    _,nav=nueva();m=CoordinadorMision(nav,0.,executor=ExecutorInmediato())
    m.ultima_pose=(.7,.7,0.);m.extrinseca=(.05,0.,0.)
    lateral=dict(x=.8,y=1.,yaw=.8,mirar_x=1.2,mirar_y=1.2)
    m.percepcion(dict(vistas_sugeridas=[lateral],diagnostico=dict(vistas_incompletas=[
        dict(superficie_x=1.2,superficie_y=1.2,normal_x=-1.,normal_y=0.)])))
    assert m.sugeridas==[lateral]


def test_ganancia_local_pequena_no_bloquea_explorar_otra_sala():
    from equipo_jar.coordinador_mision import resolver_busqueda
    from equipo_jar.navegacion.costos import ParametrosCosto
    from equipo_jar.navegacion.guardia import huella_convexa_x3
    occ=np.ones((60,100),bool);occ[8:52,8:30]=False;occ[8:52,60:92]=False;occ[26:34,30:60]=False
    world=Mundo(occ)
    footprint=huella_convexa_x3(-.19,.13,.070,.080,.0845,.033,.04,.018)
    footprint['heading_uncertainty_rad']=.1
    costs=ParametrosCosto(.12,footprint['radius_m'],.04,.25,5.)
    class Ganancia:
        def ganancia(self,pose):
            return 50 if pose[0]>3.2 and abs(pose[2])<.01 else 2
    result=resolver_busqueda(world.grilla,costs,np.zeros_like(occ),(.9,1.5,0.),(.9,1.5,0.),
        Ganancia(),set(),600.,ParametrosMision(),[],(0.,0.,0.),footprint)
    assert result['vista'] is not None
    assert result['vista']['x']>3.2
    assert not result['vista'].get('validacion_se2',False)
    assert result['diagnostico']['sondeos']==[]


def test_region_sin_conexion_de_huella_no_se_sondea_por_cada_vista(monkeypatch):
    from equipo_jar.coordinador_mision import resolver_busqueda
    from equipo_jar.navegacion.grilla import Grilla
    from equipo_jar.navegacion.costos import ParametrosCosto
    from equipo_jar.navegacion.guardia import huella_convexa_x3
    occ=np.ones((120,200),bool);occ[16:104,16:60]=False;occ[16:104,120:184]=False;occ[52:68,60:120]=False
    # El disco inscripto pasa por 0,325 m; el cuerpo completo con margen no.
    occ[52:54,85:95]=True;occ[67,85:95]=True
    g=Grilla(.025,0.,0.,0.,200,120,occ)
    fp=huella_convexa_x3(-.19,.13,.070,.080,.0845,.033,.04,.018)
    fp['heading_uncertainty_rad']=.1
    class Ganancia:
        def ganancia(self,pose):return 50 if pose[0]>3.2 else 2
    def prohibido(*args,**kwargs):raise AssertionError('No sondear una región desconectada')
    monkeypatch.setattr('equipo_jar.coordinador_mision.planificar_orientado',prohibido)
    result=resolver_busqueda(g,ParametrosCosto(.12,fp['radius_m'],.04,.25,5.),np.zeros_like(occ),
        (.9,1.5,0.),(.9,1.5,0.),Ganancia(),set(),600.,ParametrosMision(),[],(0.,0.,0.),fp)
    assert result['vista'] is not None and result['vista']['x']<1.5
    assert result['diagnostico']['celdas_descartadas_conectividad']>0
