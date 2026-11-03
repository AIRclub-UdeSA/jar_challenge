"""Atasco incipiente, recuperación y cambios de parámetros sin bloquear navegación."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml
from rcl_interfaces.msg import SetParametersResult
from rcl_interfaces.srv import GetParameters, SetParameters
from rclpy.task import Future

from equipo_jar.exploracion import (
    Exploracion, NAVEGANDO, VOLVIENDO, OBSERVANDO_OBJETO, RELOCALIZANDO, TERMINADO,
)
from equipo_jar.inflacion_adaptativa import InflacionAdaptativa, PARAMETROS_INFLACION


class Cliente:
    def __init__(self):
        self.disponible = True
        self.solicitudes = []
        self.futuros = []
        self.remove_pending_request = Mock()

    def service_is_ready(self):
        return self.disponible

    def call_async(self, solicitud):
        self.solicitudes.append(solicitud)
        futuro = Future()
        self.futuros.append(futuro)
        return futuro


@pytest.fixture
def entorno(monkeypatch):
    reloj = dict(ros=0.0, real=0.0)
    monkeypatch.setattr('equipo_jar.inflacion_adaptativa.time.monotonic', lambda: reloj['real'])
    clientes = {}

    def crear(tipo, ruta):
        assert tipo in (SetParameters, GetParameters)
        clientes[ruta] = Cliente()
        return clientes[ruta]

    nodo = SimpleNamespace(create_client=crear, get_logger=lambda: logger,
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=int(reloj['ros'] * 1e9))))
    logger = Mock()
    # Estos tests usan el radio normal fijo; la lectura desde Nav2 se prueba aparte.
    cfg = dict(PARAMETROS_INFLACION, distancia_atascado=0.12, tiempo_atascado=10.0,
               inflacion_desde_nav2=False, inflacion_normal_m=0.21, inflacion_minima_m=0.15,
               paso_inflacion_m=0.03)
    control = InflacionAdaptativa(nodo, cfg)
    return control, reloj, clientes


def aceptar(control):
    for estado in control.costmaps:
        futuro = estado['futuro']
        if futuro is not None and not futuro.done():
            futuro.set_result(SetParameters.Response(results=[SetParametersResult(successful=True)]))
    control.sincronizar()


def paso(entorno, t, pose=(0.0, 0.0, 0.0), **kwargs):
    control, reloj, _ = entorno
    reloj.update(ros=t, real=t)
    control.paso(pose, kwargs.pop('navegando', True), meta=kwargs.pop('meta', 1), **kwargs)
    aceptar(control)


def radios(cliente):
    return [s.parameters[0].value.double_value for s in cliente.solicitudes]


def test_reduce_antes_del_atasco_completo_en_ambos_costmaps_y_respeta_minimo(entorno):
    control, _, clientes = entorno
    for t in (0.0, 3.5):
        paso(entorno, t)
        assert control.objetivo == 0.21
    paso(entorno, 4.0)
    assert control.objetivo == 0.18
    paso(entorno, 6.5)
    assert control.objetivo == 0.18
    for t in (7.0, 10.0, 15.0, 20.0):
        paso(entorno, t)
        assert control.objetivo == 0.15
    assert set(clientes) == {'/global_costmap/global_costmap/set_parameters',
                             '/local_costmap/local_costmap/set_parameters'}
    for cliente in clientes.values():
        assert radios(cliente) == [0.21, 0.18, 0.15]
        assert all(len(s.parameters) == 1 and s.parameters[0].name ==
                   'inflation_layer.inflation_radius' for s in cliente.solicitudes)


def test_girar_dentro_de_una_accion_no_cuenta_como_trasladarse(entorno):
    paso(entorno, 0.0)
    paso(entorno, 4.0, (0.0, 0.0, 2.5))
    assert entorno[0].objetivo == 0.18


def test_pausas_de_camara_o_giros_explicitos_no_cuentan_como_atasco(entorno):
    paso(entorno, 0.0)
    paso(entorno, 3.0)
    paso(entorno, 50.0, navegando=False)
    paso(entorno, 51.0)
    paso(entorno, 54.5)
    assert entorno[0].objetivo == 0.21
    paso(entorno, 55.0)
    assert entorno[0].objetivo == 0.18


def test_avance_normal_y_desvios_con_progreso_nav2_no_reducen(entorno):
    for t in range(12):
        # La distancia recta a una guía puede aumentar al rodear una pared.
        paso(entorno, float(t), (0.0, t * 0.15, 0.0), progreso=-5.0 + t * 0.15)
    assert entorno[0].objetivo == 0.21


def test_oscilaciones_sin_progreso_nav2_reducen_aunque_haya_movimiento(entorno):
    for t in range(8):
        paso(entorno, float(t), (0.25 * (t % 2), 0.0, 0.0), progreso=-2.0)
    assert entorno[0].objetivo == 0.15


def test_cambiar_de_goal_no_oculta_falta_de_traslacion(entorno):
    for t in range(8):
        paso(entorno, float(t), progreso=-2.0, meta=t)
    assert entorno[0].objetivo == 0.15


def test_restaura_tras_salir_del_lugar_y_no_por_un_avance_minimo(entorno):
    for t in (0.0, 4.0, 7.0):
        paso(entorno, t)
    paso(entorno, 8.0, (0.15, 0.0, 0.0))
    assert entorno[0].objetivo == 0.15
    paso(entorno, 9.0, (0.41, 0.0, 0.0))  # todavía no pasó el mínimo entre cambios
    assert entorno[0].objetivo == 0.15
    paso(entorno, 10.0, (0.55, 0.0, 0.0))
    assert entorno[0].objetivo == 0.21
    for cliente in entorno[2].values():
        assert radios(cliente)[-1] == 0.21


def test_no_restaura_por_oscilacion_alejada_sin_mejorar_el_camino(entorno):
    for t in (0.0, 4.0, 7.0):
        paso(entorno, t, progreso=-2.0)
    paso(entorno, 10.0, (0.6, 0.0, 0.0), progreso=-2.0)
    assert entorno[0].objetivo == 0.15


def test_localizacion_mala_fin_y_reinicio_del_reloj_restauran(entorno):
    paso(entorno, 0.0)
    paso(entorno, 4.0)
    paso(entorno, 5.0, restaurar=True)
    assert entorno[0].objetivo == 0.21
    paso(entorno, 6.0)
    paso(entorno, 10.0)
    assert entorno[0].objetivo == 0.18
    paso(entorno, 0.0)
    assert entorno[0].objetivo == 0.21
    paso(entorno, 0.5)
    paso(entorno, 4.5)
    assert entorno[0].objetivo == 0.18


def test_no_asume_cambio_hasta_respuesta_y_restaura_tras_respuesta_atrasada(entorno):
    control, reloj, clientes = entorno
    control.paso(None, False)
    aceptar(control)
    reloj.update(ros=4.0, real=4.0)
    control.reducir((0.0, 0.0), 'atasco')
    assert all(e['aplicado'] == 0.21 for e in control.costmaps)
    control.restaurar('fin')  # no solapa dos solicitudes al mismo servidor
    assert all(radios(c) == [0.21, 0.18] for c in clientes.values())
    aceptar(control)  # ACK de 0.18: ahora debe enviar 0.21
    assert all(radios(c) == [0.21, 0.18, 0.21] for c in clientes.values())
    aceptar(control)
    assert all(e['aplicado'] == 0.21 for e in control.costmaps)


def test_reintenta_servidor_ausente_sin_reenviar_al_que_ya_acepto(entorno):
    control, reloj, clientes = entorno
    local = clientes['/local_costmap/local_costmap/set_parameters']
    local.disponible = False
    paso(entorno, 0.0)
    paso(entorno, 4.0)
    assert radios(local) == []
    local.disponible = True
    paso(entorno, 4.5)
    assert radios(local) == [0.18]
    assert radios(clientes['/global_costmap/global_costmap/set_parameters']) == [0.21, 0.18]


@pytest.mark.parametrize('resultado', ['rechazo', 'excepcion', 'timeout'])
def test_reintenta_rechazos_excepciones_y_timeout_sin_bucle_de_solicitudes(entorno, resultado):
    control, reloj, _ = entorno
    control.paso(None, False)
    estado = control.costmaps[0]
    futuro = estado['futuro']
    if resultado == 'rechazo':
        futuro.set_result(SetParameters.Response(results=[SetParametersResult(successful=False, reason='ocupado')]))
    elif resultado == 'excepcion':
        futuro.set_exception(RuntimeError('fallo de servicio'))
    else:
        reloj['real'] = 2.0
    control.sincronizar()
    assert estado['aplicado'] is None
    for _ in range(5):
        control.sincronizar()
    assert len(estado['cli'].solicitudes) == 1
    reloj['real'] += 2.0
    control.sincronizar()
    assert len(estado['cli'].solicitudes) == 2


@pytest.mark.parametrize('estado', [NAVEGANDO, VOLVIENDO, OBSERVANDO_OBJETO, RELOCALIZANDO, TERMINADO])
def test_integracion_solo_mide_acciones_aceptadas_y_usa_odometria(estado):
    n = SimpleNamespace(estado=estado, goal_handle=object(), goal_id=3, loc_mala_desde=None,
        cfg=dict(odom_frame='odom', base_frame='base_footprint'),
        goal_actual=dict(restante_nav2=1.5), lookup=Mock(return_value=(0.1, 0.2, 0.0)), inflacion=Mock())
    Exploracion.actualizar_inflacion(n)
    nav = estado in (NAVEGANDO, VOLVIENDO)
    assert n.inflacion.paso.call_args.args[1] == nav
    if nav:
        n.lookup.assert_called_once_with('odom', 'base_footprint')
        assert n.inflacion.paso.call_args.kwargs['progreso'] == -1.5
    else:
        n.lookup.assert_not_called()
    if estado in (RELOCALIZANDO, TERMINADO):
        assert n.inflacion.paso.call_args.kwargs['restaurar']
    n.goal_handle = None
    Exploracion.actualizar_inflacion(n)
    assert not n.inflacion.paso.call_args.args[1]


def test_configuracion_normal_coincide_con_los_costmaps():
    params = Path(__file__).resolve().parents[1] / 'params'
    cfg = yaml.safe_load((params / 'exploracion.yaml').read_text())['exploracion']['ros__parameters']
    nav = yaml.safe_load((params / 'nav2_exploracion.yaml').read_text())
    for nombre in ('global_costmap', 'local_costmap'):
        assert cfg['inflacion_normal_m'] == nav[nombre][nombre]['ros__parameters']['inflation_layer']['inflation_radius']


def test_se_puede_desactivar_sin_tocar_parametros(entorno):
    cfg = dict(entorno[0].cfg, inflacion_adaptativa=False)
    control = InflacionAdaptativa(entorno[0].nodo, cfg)
    control.paso((0.0, 0.0, 0.0), True)
    control.reducir((0.0, 0.0), 'atasco')
    assert not control.costmaps and control.objetivo == 0.21


@pytest.mark.parametrize('cambio', [dict(inflacion_minima_m=0.25), dict(paso_inflacion_m=0.0),
    dict(tiempo_pre_atasco_s=11.0), dict(distancia_restaurar_inflacion_m=0.01)])
def test_rechaza_umbrales_inconsistentes(entorno, cambio):
    with pytest.raises(ValueError):
        InflacionAdaptativa(entorno[0].nodo, dict(entorno[0].cfg, **cambio))


def _nodo_lectura(monkeypatch, reloj):
    monkeypatch.setattr('equipo_jar.inflacion_adaptativa.time.monotonic', lambda: reloj['real'])
    clientes, logger = {}, Mock()

    def crear(tipo, ruta):
        clientes[ruta] = Cliente()
        return clientes[ruta]
    nodo = SimpleNamespace(create_client=crear, get_logger=lambda: logger,
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=int(reloj['ros'] * 1e9))))
    return nodo, clientes


def test_radio_normal_se_lee_de_nav2_y_no_se_pisa_antes(monkeypatch):
    from rcl_interfaces.msg import ParameterValue
    reloj = dict(ros=0.0, real=0.0)
    nodo, clientes = _nodo_lectura(monkeypatch, reloj)
    cfg = dict(PARAMETROS_INFLACION, distancia_atascado=0.12, tiempo_atascado=10.0)
    control = InflacionAdaptativa(nodo, cfg)
    lectura = clientes['/global_costmap/global_costmap/get_parameters']
    control.sincronizar()
    assert control.objetivo is None and len(lectura.solicitudes) == 1
    assert lectura.solicitudes[0].names == ['inflation_layer.inflation_radius']
    assert not clientes['/global_costmap/global_costmap/set_parameters'].solicitudes  # nada pisado
    control.reducir((0.0, 0.0), 'atasco')                         # sin radio conocido no reduce
    assert control.objetivo is None
    lectura.futuros[0].set_result(GetParameters.Response(values=[ParameterValue(type=3, double_value=0.31)]))
    control.sincronizar()
    assert control.objetivo == 0.31 and cfg['inflacion_normal_m'] == 0.31
    assert radios(clientes['/global_costmap/global_costmap/set_parameters']) == [0.31]
    control.reducir((0.0, 0.0), 'atasco')
    assert control.objetivo == pytest.approx(0.29)
    control.restaurar('avanzó')
    assert control.objetivo == 0.31                               # vuelve al valor de Nav2


def test_minimo_no_supera_al_radio_de_nav2(monkeypatch):
    from rcl_interfaces.msg import ParameterValue
    reloj = dict(ros=0.0, real=0.0)
    nodo, clientes = _nodo_lectura(monkeypatch, reloj)
    cfg = dict(PARAMETROS_INFLACION, distancia_atascado=0.12, tiempo_atascado=10.0)
    control = InflacionAdaptativa(nodo, cfg)
    control.sincronizar()
    clientes['/global_costmap/global_costmap/get_parameters'].futuros[0].set_result(
        GetParameters.Response(values=[ParameterValue(type=3, double_value=0.22)]))
    control.sincronizar()
    assert control.objetivo == 0.22 and cfg['inflacion_minima_m'] == 0.22


def test_sin_respuesta_de_nav2_usa_el_respaldo(monkeypatch):
    reloj = dict(ros=0.0, real=0.0)
    nodo, clientes = _nodo_lectura(monkeypatch, reloj)
    cfg = dict(PARAMETROS_INFLACION, distancia_atascado=0.12, tiempo_atascado=10.0)
    control = InflacionAdaptativa(nodo, cfg)
    clientes['/global_costmap/global_costmap/get_parameters'].disponible = False
    control.sincronizar()
    assert control.objetivo is None
    reloj['real'] = cfg['espera_radio_nav2_s'] + 0.1
    control.sincronizar()
    assert control.objetivo == cfg['inflacion_normal_m']
