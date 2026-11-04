"""Una víctima reportada es un obstáculo de planificación desde el reporte, no sólo al verla el LiDAR."""
import math
from concurrent.futures import Future

from equipo_jar.coordinador_mision import CoordinadorMision
from equipo_jar.navegacion.costos import CapaViva, mapa_costos
from test_mision import nueva
from test_nav_costos import P, pasillo


class Pendiente:
    def submit(self, *args): return Future()


def test_marca_fija_no_vence_y_lidar_no_la_acorta():
    g = pasillo(); _, distancia = mapa_costos(g, P)
    capa = CapaViva(g, distancia, vida_s=5.0, tolerancia_mapa_m=0.10)
    capa.registrar_fija([(1.0, 0.55)])
    capa.actualizar(1.0, [(1.0, 0.55)])
    capa.registrar_bloqueo(2.0, [(1.0, 0.55)])
    assert capa.ocupadas(1000.)[g.a_celda(1.0, 0.55)]


def test_marca_fija_ignora_paredes_del_mapa_y_no_finitos():
    g = pasillo(); _, distancia = mapa_costos(g, P)
    capa = CapaViva(g, distancia, vida_s=5.0, tolerancia_mapa_m=0.10)
    capa.registrar_fija([(1.0, 0.0), (float('nan'), .5), (-9., .5)])
    assert not capa.ocupadas(0.).any()


def preparada():
    _, nav = nueva(); nav.largada = (.7, .7, 0.)
    m = CoordinadorMision(nav, 0., executor=Pendiente())
    return m, nav


def test_reporte_marca_la_caja_en_la_planificacion():
    m, nav = preparada(); m.ultima_pose = (.7, .7, 0.)
    m.percepcion(dict(reportes=1, reportadas=[dict(x=1.6, y=1.6)]))
    ocup = nav.capa.ocupadas(1e6)
    assert ocup[nav.grilla.a_celda(1.6, 1.6)]
    assert ocup[nav.grilla.a_celda(1.6+.09, 1.6)] and ocup[nav.grilla.a_celda(1.6, 1.6-.09)]
    assert not ocup[nav.grilla.a_celda(1.6+.2, 1.6)]


def test_no_marca_bajo_el_robot_y_pospone_hasta_alejarse():
    m, nav = preparada(); m.ultima_pose = (1.6, 1.6, 0.)
    m.percepcion(dict(reportes=1, reportadas=[dict(x=1.6, y=1.6)]))
    assert not nav.capa.ocupadas(1e6).any()
    assert not m.cajas_planificacion
    m.ultima_pose = (.7, .7, 0.)
    m.percepcion(dict(reportes=1, reportadas=[dict(x=1.62, y=1.6)]))
    assert nav.capa.ocupadas(1e6).any() and len(m.cajas_planificacion) == 1
    # Una vez insertada, la evidencia libre prevalece sobre repetir estadoM3.
    nav.capa.marca.fill(-math.inf)
    m.percepcion(dict(reportes=1, reportadas=[dict(x=1.62, y=1.6)]))
    assert not nav.capa.ocupadas(1e6).any() and len(m.cajas_planificacion)==1


def test_reporte_invalido_no_marca():
    m, nav = preparada(); m.ultima_pose = (.7, .7, 0.)
    m.percepcion(dict(reportes=1, reportadas=[dict(x=float('nan'), y=1.0), 'x']))
    assert not nav.capa.ocupadas(1e6).any()


def test_rayos_libres_retiran_una_caja_mal_ubicada():
    from test_nav_memoria_obstaculos import capa, fan
    world, layer, cell = capa()
    layer.registrar_fija([(1.5, 1.)])
    assert layer.ocupadas(1e6)[cell]
    layer.actualizar(1., fan(2.), origen=(.5, 1.), incremento_angular=.005)
    assert not layer.ocupadas(1.)[cell]


def test_caja_oculta_por_otro_obstaculo_se_conserva():
    from test_nav_memoria_obstaculos import capa, fan
    world, layer, cell = capa()
    layer.registrar_fija([(1.5, 1.)])
    layer.actualizar(1., fan(.5), origen=(.5, 1.), incremento_angular=.005)
    assert layer.ocupadas(1e6)[cell]
