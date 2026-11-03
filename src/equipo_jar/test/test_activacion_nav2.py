"""Regresión de corrida_1341: acciones visibles, pero servidores aún inactivos."""
from types import SimpleNamespace
from unittest.mock import Mock

from geometry_msgs.msg import Twist
from lifecycle_msgs.msg import State, Transition
from nav2_msgs.action import Spin
from rclpy.task import Future
from rclpy.time import Time

from equipo_jar.exploracion import (
    ACTIVANDO_NAV2, ELIGIENDO, LOCALIZANDO, NAVEGANDO, OBSERVANDO_OBJETO, Exploracion,
)


def _listo(resultado):
    futuro = Future()
    futuro.set_result(resultado)
    return futuro


def _nodo(monkeypatch):
    reloj = {'t': 100.0}
    monkeypatch.setattr('equipo_jar.exploracion.time.monotonic', lambda: reloj['t'])
    n = Exploracion.__new__(Exploracion)
    n.cfg = dict(giro_inicial=True)
    n.estado, n.t_mapa = LOCALIZANDO, 80.0
    n.log_estado = Mock()
    n.pose_robot = Mock(return_value=(0.0, 0.0, 0.0))
    n.publicar_pose_inicial = Mock()
    n.t_ult_revision_lc = n.t_ultimo_init = 0.0
    n.lc, n.estados_lc, n.consultas_lc, n.transiciones_lc = {}, {}, {}, {}
    estados = {k: State.PRIMARY_STATE_UNCONFIGURED for k in n.NODOS_NAV2}
    estados.update(amcl=State.PRIMARY_STATE_ACTIVE,
                   controller_server=State.PRIMARY_STATE_INACTIVE,
                   smoother_server=State.PRIMARY_STATE_INACTIVE)
    cambios = []

    def crear_cliente(tipo, nombre):
        nodo, servicio = nombre.strip('/').split('/')

        def llamar(req):
            if servicio == 'get_state':
                return _listo(SimpleNamespace(current_state=SimpleNamespace(id=estados[nodo])))
            cambios.append((nodo, req.transition.id))
            estados[nodo] = (State.PRIMARY_STATE_INACTIVE if req.transition.id == Transition.TRANSITION_CONFIGURE
                              else State.PRIMARY_STATE_ACTIVE)
            return _listo(SimpleNamespace(success=True))
        return SimpleNamespace(service_is_ready=lambda: True, call_async=llamar)

    n.create_client = crear_cliente
    n.nav = n.nav_ruta = n.spin_cli = Mock(server_is_ready=lambda: True)
    n.get_logger = lambda: Mock()
    n.get_clock = lambda: SimpleNamespace(now=lambda: Time(nanoseconds=int(reloj['t'] * 1e9)))
    n.pub_cmd, n.pub_vel = Mock(), Mock()
    n.goal_id, n.goal_handle, n.solicitud_goal = 0, None, None
    n.cancelacion_en_curso, n.goal_heredado, n.goal_actual = None, None, None
    n.objeto_observado, n.motivo_regreso, n.pendiente = None, None, None
    n.n_atascos, n.lista_negra = 0, []
    n.hist, n.hist_global = [], []
    n.iniciar_escape, n.limpiar_costmaps, n.girar = Mock(), Mock(), Mock()
    return n, reloj, estados, cambios


def test_no_arranca_solo_porque_existen_las_acciones_y_completa_la_activacion(monkeypatch):
    n, reloj, estados, cambios = _nodo(monkeypatch)
    # Mismo fallo del log: el rescate configura nodos y aparecen los action servers.
    n.ciclo()
    assert n.estado == LOCALIZANDO
    n.girar.assert_not_called()
    assert estados['bt_navigator'] == State.PRIMARY_STATE_INACTIVE
    reloj['t'] += 1.1
    n.ciclo()
    assert n.estado == LOCALIZANDO  # Todavía falta confirmar el estado ACTIVE por GetState.
    reloj['t'] += 1.1
    n.ciclo()
    assert n.nav2_activo()
    n.girar.assert_called_once()
    n.publicar_pose_inicial.assert_not_called()  # No reiniciar AMCL mientras espera Nav2.
    for nodo in n.NODOS_NAV2:
        assert estados[nodo] == State.PRIMARY_STATE_ACTIVE
    assert ('behavior_server', Transition.TRANSITION_ACTIVATE) in cambios
    assert ('bt_navigator', Transition.TRANSITION_ACTIVATE) in cambios


def test_rechazo_de_goal_recupera_nav2_sin_falso_atasco_ni_lista_negra(monkeypatch):
    n, reloj, estados, _ = _nodo(monkeypatch)
    n.estado = NAVEGANDO
    n.goal_actual = dict(x=1.0, y=0.0)
    n.cb_goal_aceptado(_listo(SimpleNamespace(accepted=False)), n.goal_id)
    assert n.estado == ACTIVANDO_NAV2
    assert not n.lista_negra and n.n_atascos == 0
    n.iniciar_escape.assert_not_called()
    n.pub_cmd.publish.assert_called_with(Twist())
    for _ in range(3):
        reloj['t'] += 1.1
        n.ciclo()
        if n.estado != ACTIVANDO_NAV2:
            break
    assert n.estado == ELIGIENDO and n.nav2_activo()


def test_giro_rechazado_no_da_objeto_por_observado_y_se_reanuda(monkeypatch):
    n, reloj, estados, _ = _nodo(monkeypatch)
    n.objeto_observado = dict(punto=(0.0, 1.0), inicio=99.0, fase='frenando', giros=1)
    cliente = SimpleNamespace(send_goal_async=lambda goal: _listo(SimpleNamespace(accepted=False)))
    goal = Spin.Goal(target_yaw=1.57)
    n.comportamiento(cliente, goal, OBSERVANDO_OBJETO)
    assert n.estado == ACTIVANDO_NAV2 and n.objeto_observado is not None
    assert n.objeto_observado['giros'] == 0
    for _ in range(3):
        reloj['t'] += 1.1
        n.ciclo()
        if n.estado != ACTIVANDO_NAV2:
            break
    assert n.estado == OBSERVANDO_OBJETO
    assert n.objeto_observado['inicio'] == reloj['t']


def test_una_consulta_vieja_no_pisa_el_estado_activo_actual(monkeypatch):
    n, reloj, estados, _ = _nodo(monkeypatch)
    actual = _listo(SimpleNamespace(current_state=SimpleNamespace(id=State.PRIMARY_STATE_ACTIVE)))
    vieja = _listo(SimpleNamespace(current_state=SimpleNamespace(id=State.PRIMARY_STATE_INACTIVE)))
    n.consultas_lc['bt_navigator'] = (actual, reloj['t'])
    n._estado_ciclo_de_vida('bt_navigator', actual)
    n._estado_ciclo_de_vida('bt_navigator', vieja)
    assert n.estados_lc['bt_navigator'][0] == State.PRIMARY_STATE_ACTIVE


def test_no_duplica_una_transicion_todavia_pendiente(monkeypatch):
    n, reloj, estados, cambios = _nodo(monkeypatch)
    n.revisar_ciclo_de_vida(['controller_server'])
    n.transiciones_lc['controller_server'] = (Future(), reloj['t'])
    estados['controller_server'] = State.PRIMARY_STATE_INACTIVE
    n.revisar_ciclo_de_vida(['controller_server'])
    assert cambios == [('controller_server', Transition.TRANSITION_ACTIVATE)]
