"""Cupo sobrante: hipótesis de último recurso y entrega antes de /done."""
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import yaml
from builtin_interfaces.msg import Time
from rclpy.task import Future
from std_srvs.srv import Trigger

from equipo_jar.deteccion_victimas import DeteccionVictimas, Grupo, campo_distancia_paredes
from equipo_jar.exploracion import APROXIMANDO, FINALIZANDO, TERMINADO, Exploracion
from equipo_jar.reportes_finales import lugares_probables


def mapa():
    libre = np.ones((24, 44), dtype=bool)
    libre[[0, -1], :] = False
    libre[:, [0, -1]] = False
    return dict(libre=libre, res=0.1, ox=-2., oy=-1., alto=24, ancho=44, frame='map',
                dist=campo_distancia_paredes(libre, 0.1))


def detector():
    n = DeteccionVictimas.__new__(DeteccionVictimas)
    params = Path(__file__).resolve().parents[1] / 'params' / 'victimas.yaml'
    n.cfg = yaml.safe_load(params.read_text())['deteccion_victimas']['ros__parameters']
    n.mapa, n.t_mapa = mapa(), 100.
    n.visto = n.mapa['libre'].copy()
    n.visto[:, 24:] = False
    n.recorrido = np.zeros_like(n.visto)
    n.grupos, n.grupos_objetos = [], []
    n.reportes, n.n_max, n.reportes_cerrados = 0, 3, False
    n.pub, n.pub_pendientes, n.publicar_debug = Mock(), Mock(), Mock()
    n.get_logger = lambda: Mock()
    n.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(to_msg=lambda: Time()))
    return n


def test_estima_en_piso_no_visto_sin_cruzarse_con_huella_azules_ni_reportes():
    n = detector()
    n.recorrido[:, 25:28] = True
    excluidos = [(1.2, 0.)]
    puntos = lugares_probables(n.mapa, n.visto, n.recorrido, excluidos, 4, 0.1, 0.75)
    assert len(puntos) == 4
    for i, (x, y) in enumerate(puntos):
        f, c = n.a_celda(x, y)
        assert n.mapa['libre'][f, c] and not n.visto[f, c] and not n.recorrido[f, c]
        assert n.mapa['dist'][f, c] >= 0.1
        assert all(math.hypot(x - px, y - py) >= 0.75 - 1e-8
                   for px, py in excluidos + puntos[:i])


@pytest.mark.parametrize('sin_margen', [False, True])
def test_no_inventa_posiciones_si_no_hay_lugar_posible(sin_margen):
    m = mapa()
    recorrido = np.ones_like(m['libre'])
    margen = 100. if sin_margen else 0.1
    assert lugares_probables(m, None, recorrido, [], 3, margen, 0.75) == []


def test_cierre_prioriza_deteccion_debil_y_completa_el_cupo_sin_duplicar():
    n = detector()
    n.cfg['segundos_reporte_final'] = float('inf')  # El cierre ocurre antes del temporizador de 480 s.
    reportada = Grupo(-1.4, -0.2, 101., 1.)
    n.grupos.append(reportada)
    n.reportar(reportada, 'confirmada')
    duplicada, debil = Grupo(-1.35, -0.2, 102., 1.8), Grupo(-0.5, 0.8, 102., 1.8)
    descartada = Grupo(0.5, 0.5, 102., 1.8)
    descartada.descartado = True
    n.grupos += [duplicada, debil, descartada]
    n.grupos_objetos = [Grupo(1.1, -0.1, 102., 1.)]
    respuesta = n.cerrar_reportes(Trigger.Request(), Trigger.Response())
    assert respuesta.success and n.reportes == n.n_max == 3
    assert debil.reportado and not duplicada.reportado and not descartada.reportado
    mensajes = [c.args[0] for c in n.pub.publish.call_args_list]
    assert (mensajes[1].point.x, mensajes[1].point.y) == debil.posicion()
    ultimo = mensajes[-1]
    assert math.hypot(ultimo.point.x - 1.1, ultimo.point.y + 0.1) >= 0.75
    assert all(m.header.frame_id == 'map' and m.point.z == 0. for m in mensajes)
    n.cerrar_reportes(Trigger.Request(), Trigger.Response())
    n.revisar_reportes()
    assert n.pub.publish.call_count == 3  # Reintentos y temporizadores no consumen más reportes.


def test_puede_estimar_sin_ninguna_deteccion_y_deja_constancia_en_log():
    n = detector()
    logger = Mock()
    n.get_logger = lambda: logger
    respuesta = n.cerrar_reportes(Trigger.Request(), Trigger.Response())
    assert respuesta.success and n.reportes == 3
    assert all(getattr(g, 'estimado', False) for g in n.grupos)
    assert any('sin confirmación visual' in c.args[0] for c in logger.info.call_args_list)


def test_sin_estimaciones_habilitadas_solo_reporta_detecciones_pendientes():
    n = detector()
    n.cfg['estimar_al_cerrar'] = False
    n.cerrar_reportes(Trigger.Request(), Trigger.Response())
    n.pub.publish.assert_not_called()
    assert n.reportes_cerrados


def test_no_publica_nada_despues_de_done_ni_sin_mapa():
    n = detector()
    n.cb_done(None)
    n.cerrar_reportes(Trigger.Request(), Trigger.Response())
    n.pub.publish.assert_not_called()
    n.mapa = None
    assert not n.cerrar_reportes(Trigger.Request(), Trigger.Response()).success


def test_cobertura_sin_rviz_llega_al_detector_y_rechaza_un_origen_incompatible():
    d = detector()
    n = Exploracion.__new__(Exploracion)
    m = d.mapa
    n.cfg = {'publicar_debug': False}
    n.grilla = SimpleNamespace(**{k: m[k] for k in ('res', 'ox', 'oy', 'alto', 'ancho', 'libre', 'frame')},
                               bloques_conocidos=np.zeros_like(m['libre']))
    n.visto = d.visto.copy()
    n.get_clock = d.get_clock
    n.pub_visto = Mock(publish=Mock(side_effect=d.cb_visto))
    d.visto = None
    n.publicar_visto()
    np.testing.assert_array_equal(d.visto, n.visto)
    mensaje = n.pub_visto.publish.call_args.args[0]
    mensaje.info.origin.position.x += 1.
    anterior = d.visto.copy()
    mensaje.data = [0] * len(mensaje.data)
    d.cb_visto(mensaje)
    np.testing.assert_array_equal(d.visto, anterior)


def explorador(monkeypatch, servicio=True, elapsed=120.):
    reloj = {'t': 100.}
    monkeypatch.setattr('equipo_jar.exploracion.time.monotonic', lambda: reloj['t'])
    n = Exploracion.__new__(Exploracion)
    n.cfg = dict(publicar_done=True, cantidad_victimas=3, espera_reportes_finales_s=0.7,
                 duracion_corrida_s=600., done_forzado_s=599.)
    n.estado, n.reportes, n.motivo_regreso = APROXIMANDO, 1, 'exploración terminada'
    n.cancelar_actual, n.publicar_visto = Mock(), Mock()
    n.log_estado = Mock()
    n.pub_cmd, n.pub_done = Mock(), Mock()
    n.cobertura, n.tiempo_transcurrido = lambda: 0.8, lambda: elapsed
    n.get_logger = lambda: Mock()
    futuro = Future()
    n.reportes_finales_cli = Mock(service_is_ready=lambda: servicio, call_async=Mock(return_value=futuro))
    return n, reloj, futuro


def test_done_espera_a_los_reportes_y_el_cierre_no_vuelve_a_mover_el_robot(monkeypatch):
    n, reloj, futuro = explorador(monkeypatch)
    eventos = []
    n.pub_done.publish.side_effect = lambda _: eventos.append('done')
    n.terminar('en la largada')
    assert n.estado == FINALIZANDO
    n.pub_done.publish.assert_not_called()
    n.publicar_visto.assert_called_once()
    n.ciclo()  # La espera de cierre no intenta explorar ni recuperar Nav2.
    eventos.append('reporte final')
    futuro.set_result(Trigger.Response(success=True, message='3/3 reportes publicados'))
    n.tick_rapido()
    n.pub_done.publish.assert_not_called()
    reloj['t'] += 0.2
    n.tick_rapido()
    assert n.estado == TERMINADO and eventos == ['reporte final', 'done']
    n.terminar('repetido')
    n.pub_done.publish.assert_called_once()
    for llamada in n.pub_cmd.publish.call_args_list:
        assert llamada.args[0].linear.x == llamada.args[0].linear.y == llamada.args[0].angular.z == 0.


@pytest.mark.parametrize('servicio', [True, False])
def test_detector_ausente_o_sin_respuesta_no_bloquea_done(monkeypatch, servicio):
    n, reloj, _ = explorador(monkeypatch, servicio=servicio)
    n.terminar('fin')
    reloj['t'] += 0.8
    n.tick_rapido()
    assert n.estado == TERMINADO
    n.pub_done.publish.assert_called_once()


def test_no_consulta_el_detector_si_ya_se_usaron_todos_los_reportes(monkeypatch):
    n, _, _ = explorador(monkeypatch)
    n.reportes = 3
    n.terminar('fin')
    assert n.estado == TERMINADO
    n.reportes_finales_cli.call_async.assert_not_called()
    n.pub_done.publish.assert_called_once()


def test_plazo_final_reduce_la_espera_del_detector(monkeypatch):
    n, reloj, _ = explorador(monkeypatch, elapsed=599.6)
    n.terminar('plazo')
    assert n.cierre_reportes['limite'] <= reloj['t'] + 0.11
    reloj['t'] += 0.2
    n.tick_rapido()
    assert n.estado == TERMINADO
