"""Detección de obstáculos nuevos y tiempo real de observación con cámara."""

import math
from collections import deque
import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import TransformStamped
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

from equipo_jar.exploracion import OBSERVANDO_OBJETO, Exploracion, Grilla
from equipo_jar.observacion_objetos import (
    ObjetosFueraMapa, centros_obstaculos, objetos_del_scan, leer_observaciones,
)


def test_objeto_persistente_se_confirma_y_no_se_inspecciona_dos_veces():
    mascara = np.zeros((12, 12), bool)
    mascara[4:6, 6:8] = True
    mascara[9, 9] = True  # celda suelta: ruido
    centros = centros_obstaculos(mascara, (0.0, 0.0), 0.1)
    assert len(centros) == 1
    assert centros[0] == pytest.approx((0.7, 0.5))
    rastreador = ObjetosFueraMapa()
    for t in (0.0, 0.2, 0.5):
        rastreador.actualizar(centros, t)
    assert rastreador.confirmados(0.5) == centros
    rastreador.finalizar(centros[0], 0.5, True)
    rastreador.actualizar(centros, 0.7)
    assert not rastreador.confirmados(0.7)


def test_costmap_evita_obstaculos_sin_convertir_ecos_en_recuerdos(monkeypatch):
    mapa = OccupancyGrid()
    mapa.info.width = mapa.info.height = 20
    mapa.info.resolution = 0.1
    datos = np.zeros((20, 20), dtype=np.int8)
    datos[0, :] = 100
    mapa.data = datos.ravel().tolist()
    g = Grilla(mapa, 0.0, 0.0)
    costmap = OccupancyGrid()
    costmap.info = mapa.info
    costo = np.zeros_like(datos)
    costo[0:2, 4:6] = 100  # pared conocida y eco demasiado cercano
    costo[10:12, 10:12] = 100  # obstáculo en piso libre
    costmap.data = costo.ravel().tolist()
    reloj = {'t': 0.0}
    monkeypatch.setattr('equipo_jar.exploracion.time.monotonic', lambda: reloj['t'])
    n = SimpleNamespace(grilla=g, cfg={'costo_obstaculo': 99, 'inspeccionar_objetos': True,
        'separacion_pared_objeto_m': 0.12},
        dist_pared=Exploracion.campo_distancia_paredes(g),
        objetos_fuera_mapa=ObjetosFueraMapa(), estado='ELIGIENDO', loc_mala_desde=None,
        registrar_objeto_local=Mock())
    for t in (0.0, 0.2, 0.5):
        reloj['t'] = t
        Exploracion.cb_costmap(n, costmap)
    assert not g.transitable[10, 10]
    assert n.objetos_fuera_mapa.confirmados(0.5) == []


def test_objeto_nuevo_tapa_el_piso_detras_en_la_cobertura_de_camara():
    mapa = OccupancyGrid()
    mapa.info.width = mapa.info.height = 12
    mapa.info.resolution = 0.1
    mapa.data = [0] * 144
    g = Grilla(mapa, 0.0, 0.0)
    extra = np.zeros((12, 12), bool)
    extra[5, 7] = True
    g.actualizar_obstaculos(extra)
    n = SimpleNamespace(grilla=g, ray_dx=np.array([[1.0, 2.0, 3.0]]),
                        ray_dy=np.zeros((1, 3)))
    _, _, visible = Exploracion.rayos_desde(n, 5, 5)
    assert visible.tolist() == [[True, False, False]]


def test_objeto_nuevo_interrumpe_goal_y_reserva_observacion(monkeypatch):
    monkeypatch.setattr('equipo_jar.exploracion.time.monotonic', lambda: 1.0)
    rastreador = ObjetosFueraMapa()
    for t in (0.0, 0.2, 0.5):
        rastreador.actualizar([(1.0, 0.0)], t)
    n = SimpleNamespace(
        cfg={'inspeccionar_objetos': True, 'rango_vision': 2.5,
             'tiempo_observacion_objeto_s': 2.5, 'distancia_observacion_ideal_m': 0.85},
        objeto_observado=None, objetos_fuera_mapa=rastreador,
        t_ultima_imagen=1.0, pose_camara=Mock(return_value=(0.0, 0.0, 0.0)),
        linea_libre_al_objeto=Mock(return_value=True), cancelar_actual=Mock(),
        observacion_visual=Mock(return_value=None),
        goal_actual={'x': 2.0}, pendiente=None, pub_cmd=Mock(), get_logger=lambda: Mock())
    assert Exploracion.considerar_objeto_nuevo(n, (0.0, 0.0, 0.0))
    n.cancelar_actual.assert_called_once()
    assert n.goal_actual is None
    assert n.estado == OBSERVANDO_OBJETO
    assert n.objeto_observado['punto'] == (1.0, 0.0)


def test_pose_camara_usa_eje_optico_no_yaw_del_frame():
    q = SimpleNamespace(x=0.0, y=math.sin(math.pi / 4), z=0.0,
                        w=math.cos(math.pi / 4))
    transform = SimpleNamespace(
        translation=SimpleNamespace(x=1.0, y=2.0), rotation=q)
    n = SimpleNamespace(grilla=SimpleNamespace(frame='map'), frame_camara='camara_optica',
                        tf_buffer=SimpleNamespace(lookup_transform=Mock(
                            return_value=SimpleNamespace(transform=transform))))
    x, y, rumbo = Exploracion.pose_camara(n)
    assert (x, y) == (1.0, 2.0)
    assert rumbo == 0.0


def _objeto(**cambios):
    o = dict(id='azul-0', color='azul', tipo='caja', nuevo=False, x=0.85, y=0.0,
             n=10, completo=True, confirmado=True, reportado=False, distancia=0.85, error_angular=0.0)
    o.update(cambios)
    return o


def _nodo_observando(monkeypatch, punto=(0.85, 0.0)):
    reloj = {'t': 1.0}
    monkeypatch.setattr('equipo_jar.exploracion.time.monotonic', lambda: reloj['t'])
    n = Exploracion.__new__(Exploracion)
    n.cfg = dict(inspeccionar_objetos=True, rango_vision=2.0, tiempo_observacion_objeto_s=0.8,
        distancia_observacion_min_m=0.45, distancia_observacion_max_m=1.60,
        distancia_observacion_ideal_m=0.85, tolerancia_centrar_objeto_deg=25.0,
        imagenes_observacion_minimas=2, timeout_observacion_objeto_s=10.0,
        porcentaje_vel_observacion=25.0, lado_bloque_objeto_m=0.30, margen_regreso_s=25.0,
        radio_visto_objeto_m=0.45, margen_transito=0.17, margen_candidato=0.22,
        peso_lejos=0.5)
    mapa = OccupancyGrid()
    mapa.header.frame_id = 'map'
    mapa.info.width = mapa.info.height = 80
    mapa.info.resolution = 0.1
    mapa.info.origin.position.x = mapa.info.origin.position.y = -4.0
    mapa.data = [0] * 6400
    n.grilla = Grilla(mapa, 0.17, 0.22)
    n.mapa_msg, n.bloques_objetos = mapa, []
    n.visto = np.zeros((80, 80), dtype=bool)
    n.sectorizador = None
    n.objeto_observado = None
    n.observaciones_vision = None
    n.ultimo_stamp_vision = 0
    n.objetos_fuera_mapa = ObjetosFueraMapa()
    n.scans_objetos = deque(maxlen=15)
    n.limpiar_costmaps = Mock()
    n.ruta_sector, n.ruta_region = [], None
    for t in (0.0, 0.2, 0.5):
        n.objetos_fuera_mapa.actualizar([punto], t)
    n.get_clock = lambda: SimpleNamespace(now=lambda: Time(nanoseconds=int((100 + reloj['t']) * 1e9)))
    n.t_ultima_imagen = 1.0
    n.vel_base, n.vel_giro = (0.0, 0.0), 0.0
    n.fov = math.radians(70)
    n.frame_camara = 'camara_optica'
    n.pose_camara = Mock(return_value=(0.0, 0.0, 0.0))
    n.pose_robot = Mock(return_value=(0.0, 0.0, 0.0))
    n.cancelar_actual = Mock(return_value=None)
    n.goal_actual, n.pendiente = None, None
    n.pub_cmd, n.pub_vel = Mock(), Mock()
    n.pub_mapa_objetos = Mock()
    n.get_logger = lambda: Mock()
    n.spin_cli, n.girar = Mock(), Mock()
    n.enviar_goal = Mock(return_value=True)
    n.tiempo_restante = lambda: 500.0
    n.segundos_para_volver = lambda x, y: 10.0
    n.terminar_observacion_objeto = Mock()
    assert n.considerar_objeto_nuevo(n.pose_robot())
    return n, reloj


def _fotograma(n, reloj, t, objetos=(), stamp=None):
    reloj['t'] = t
    n.t_ultima_imagen = t
    stamp = int((100 + t) * 1e9) if stamp is None else stamp
    n.cb_observaciones(String(data=json.dumps(dict(version=1, frame='map', stamp_ns=stamp,
                                                   objetos=list(objetos)))))
    n.paso_observacion_objeto()


def test_mirada_breve_termina_con_lidar_aunque_no_haya_analisis_de_camara(monkeypatch):
    n, reloj = _nodo_observando(monkeypatch)
    for t in (1.4, 1.7, 1.9, 2.1, 2.6):
        reloj['t'] = t
        n.t_ultima_imagen = t
        n.paso_observacion_objeto()
    n.terminar_observacion_objeto.assert_called_once()
    assert n.terminar_observacion_objeto.call_args.args[0] is True
    assert 'mirada completada' in n.terminar_observacion_objeto.call_args.args[1]


@pytest.mark.parametrize('objetos', [[], [_objeto()]])
def test_una_mirada_corta_con_o_sin_color_alcanza(monkeypatch, objetos):
    n, reloj = _nodo_observando(monkeypatch)
    for t in (1.4, 1.7, 1.9, 2.1, 2.3):
        _fotograma(n, reloj, t, objetos)
    n.terminar_observacion_objeto.assert_not_called()
    _fotograma(n, reloj, 2.6, objetos)
    n.terminar_observacion_objeto.assert_called_once()
    assert n.terminar_observacion_objeto.call_args.args[0] is True
    assert ('azul' in n.terminar_observacion_objeto.call_args.args[1]) == bool(objetos)


def test_espera_parada_real_y_no_hace_giros_alternados(monkeypatch):
    n, reloj = _nodo_observando(monkeypatch)
    _fotograma(n, reloj, 1.4, [_objeto()])
    _fotograma(n, reloj, 1.7, [_objeto()])
    assert n.objeto_observado['desde'] == 1.7
    n.vel_base = (0.10, 0.0)
    _fotograma(n, reloj, 1.8, [_objeto()])
    assert n.objeto_observado['desde'] is None
    n.vel_base = (0.0, 0.0)
    _fotograma(n, reloj, 1.9, [_objeto()])
    _fotograma(n, reloj, 2.2, [_objeto(error_angular=0.8)])
    n.girar.assert_called_once_with(0.8, OBSERVANDO_OBJETO)
    for t, angulo in ((2.3, -0.4), (2.6, 0.4), (2.8, -0.4), (3.5, 0.4)):
        _fotograma(n, reloj, t, [_objeto(error_angular=angulo)])
    assert n.girar.call_count == 1
    n.terminar_observacion_objeto.assert_called_once()


def test_fotogramas_repetidos_y_previos_a_la_parada_no_cuentan(monkeypatch):
    n, reloj = _nodo_observando(monkeypatch)
    _fotograma(n, reloj, 1.4)
    # La foto llegó después de frenar, pero se tomó antes de estabilizarse.
    _fotograma(n, reloj, 2.0, stamp=101_300_000_000)
    assert n.objeto_observado['imagenes'] == 0
    _fotograma(n, reloj, 2.2)
    assert n.objeto_observado['imagenes'] == 1
    for t in (2.3, 2.4, 2.5):
        _fotograma(n, reloj, t, stamp=102_200_000_000)
    assert n.objeto_observado['imagenes'] == 1
    reloj['t'] = 2.9  # sin análisis nuevo: no bloquea al LiDAR
    n.paso_observacion_objeto()
    assert n.objeto_observado['imagenes'] == 1
    n.terminar_observacion_objeto.assert_called_once()


def test_ajusta_distancia_con_nav2_lento_y_espera_resultado(monkeypatch):
    from action_msgs.msg import GoalStatus
    n, reloj = _nodo_observando(monkeypatch, (1.9, 0.0))
    _fotograma(n, reloj, 1.6)
    _fotograma(n, reloj, 2.0)
    n.enviar_goal.assert_called_once()
    x, y, yaw, _, _ = n.enviar_goal.call_args.args
    assert 0.45 <= math.hypot(1.9 - x, y) <= 1.60
    assert yaw == pytest.approx(math.atan2(-y, 1.9 - x))
    assert n.estado == OBSERVANDO_OBJETO and n.objeto_observado['fase'] == 'acercando'
    assert n.pub_vel.publish.call_args.args[0].speed_limit == 25.0
    _fotograma(n, reloj, 2.2)
    assert n.objeto_observado['imagenes'] == 0
    n.resultado_goal(GoalStatus.STATUS_SUCCEEDED)
    assert n.objeto_observado['fase'] == 'frenando'
    assert n.objeto_observado['quieto_desde'] is None


def test_espera_cancelacion_y_abandona_si_nav2_falla(monkeypatch):
    from action_msgs.msg import GoalStatus
    n, reloj = _nodo_observando(monkeypatch)
    futuro = SimpleNamespace(done=lambda: False)
    n.objeto_observado['cancelacion'] = futuro
    _fotograma(n, reloj, 2.0, [_objeto(error_angular=0.8)])
    n.girar.assert_not_called()
    assert n.objeto_observado['imagenes'] == 0
    n.objeto_observado['fase'] = 'acercando'
    n.resultado_goal(GoalStatus.STATUS_ABORTED)
    assert n.terminar_observacion_objeto.call_args.args[0] is False


def test_no_exige_encuadre_perfecto_ni_confirmacion_prolongada(monkeypatch):
    n, reloj = _nodo_observando(monkeypatch)
    for t in (1.4, 1.7, 1.9, 2.6):
        _fotograma(n, reloj, t, [_objeto(completo=False, confirmado=False)])
    n.enviar_goal.assert_not_called()
    n.terminar_observacion_objeto.assert_called_once()


def test_camara_solo_complementa_objeto_confirmado_por_lidar(monkeypatch):
    n, reloj = _nodo_observando(monkeypatch)
    n.objeto_observado = None
    n.objetos_fuera_mapa = ObjetosFueraMapa()
    n.scans_objetos = deque(maxlen=15)
    n.limpiar_costmaps = Mock()
    n.ruta_sector, n.ruta_region = [], None
    _fotograma(n, reloj, 1.8, [_objeto(n=1)])
    assert not n.considerar_objeto_nuevo(n.pose_robot())
    _fotograma(n, reloj, 2.0, [_objeto(n=2)])
    assert not n.considerar_objeto_nuevo(n.pose_robot())
    for t in (1.5, 1.8, 2.0):
        n.objetos_fuera_mapa.actualizar([(0.85, 0.0)], t)
    assert n.considerar_objeto_nuevo(n.pose_robot())
    assert n.objeto_observado['color'] == 'azul'


def test_color_puede_cambiar_sin_mover_posicion_lidar(monkeypatch):
    n, reloj = _nodo_observando(monkeypatch)
    n.objeto_observado['color'] = 'azul'
    for t in (1.4, 1.7, 1.9, 2.6):
        _fotograma(n, reloj, t, [_objeto(id='objeto-0', x=1.10, color='verde', tipo='objeto')])
    assert n.objeto_observado['punto'] == (0.85, 0.0)
    assert n.objeto_observado['color'] == 'verde' and n.objeto_observado['color_confirmado']


def test_lectura_lidar_corrige_posicion_aunque_camara_falte(monkeypatch):
    n, reloj = _nodo_observando(monkeypatch)
    n.t_ultima_imagen = None
    for t in (1.4, 1.7):
        reloj['t'] = t
        n.objetos_fuera_mapa.actualizar([(0.97, 0.0)], t)
        n.paso_observacion_objeto()
    assert n.objeto_observado['punto'] == (0.97, 0.0)
    assert (0.85, 0.0) in n.objeto_observado['puntos']
    assert (0.97, 0.0) in n.objeto_observado['puntos']


def test_lidar_distingue_objeto_nuevo_de_pared_conocida():
    mapa = OccupancyGrid()
    mapa.info.width = mapa.info.height = 40
    mapa.info.resolution = 0.1
    datos = np.zeros((40, 40), dtype=np.int8)
    datos[:, 20] = 100  # pared conocida en x=2
    mapa.data = datos.ravel().tolist()
    g = Grilla(mapa, 0.17, 0.22)
    pared = Exploracion.campo_distancia_paredes(g)
    scan = LaserScan(angle_min=-0.1, angle_increment=0.05, range_min=0.1,
                     range_max=5.0, ranges=[1.0] * 5)
    objetos = objetos_del_scan(scan, (0.0, 1.0, 0.0), g, pared)
    assert len(objetos) == 1 and objetos[0] == pytest.approx((1.0, 1.0), abs=0.01)
    scan.ranges = [2.01 / math.cos(a) for a in (-0.1, -0.05, 0.0, 0.05, 0.1)]
    assert objetos_del_scan(scan, (0.0, 1.0, 0.0), g, pared) == []


def test_scan_directo_confirma_objeto_sin_costmap_y_no_cuenta_repetidos(monkeypatch):
    n, reloj = _nodo_observando(monkeypatch)
    n.objetos_fuera_mapa = ObjetosFueraMapa()
    n.scans_objetos = deque(maxlen=15)
    n.limpiar_costmaps = Mock()
    n.ruta_sector, n.ruta_region = [], None
    n.cfg.update(timeout_scan_s=0.5, separacion_pared_objeto_m=0.12)
    n.loc_mala_desde = None
    n.t_scan_objetos, n.ultimo_stamp_scan_objetos = -float('inf'), 0
    n.dist_pared = Exploracion.campo_distancia_paredes(n.grilla)
    tf = TransformStamped()
    tf.transform.rotation.w = 1.0
    n.tf_buffer = SimpleNamespace(lookup_transform=Mock(return_value=tf))
    scan = LaserScan(angle_min=-0.1, angle_increment=0.05, range_min=0.1,
                     range_max=5.0, ranges=[0.85] * 5)
    scan.header.frame_id = 'laser'
    for t in (1.0, 1.3, 1.6):
        reloj['t'] = t
        scan.header.stamp = n.get_clock().now().to_msg()
        n.cb_scan(scan)
    puntos = n.objetos_fuera_mapa.confirmados(1.6)
    # El LiDAR ve la cara a 0.85 m; el centro del objeto queda media caja más allá.
    assert len(puntos) == 1 and puntos[0] == pytest.approx((0.85 + 0.12, 0.0), abs=0.01)
    reloj['t'] = 1.9
    n.cb_scan(scan)
    assert n.objetos_fuera_mapa.rastros[0]['n'] == 3


def test_caja_que_deja_de_verse_conserva_localizacion_lidar_y_no_bloquea(monkeypatch):
    n, reloj = _nodo_observando(monkeypatch)
    for t in (1.4, 1.7):
        _fotograma(n, reloj, t, [_objeto()])
    for t in (1.9, 2.1, 2.6):
        _fotograma(n, reloj, t)
    n.terminar_observacion_objeto.assert_called_once()
    assert n.terminar_observacion_objeto.call_args.args[0] is True
    assert 'sin color confirmado' in n.terminar_observacion_objeto.call_args.args[1]


def test_objeto_lateral_se_encara_aunque_la_imagen_no_lo_haya_detectado(monkeypatch):
    n, reloj = _nodo_observando(monkeypatch, (0.0, 0.85))
    n.t_ultima_imagen = None  # No condiciona el giro hacia la posición del láser.
    for t in (1.4, 1.7):
        reloj['t'] = t
        n.paso_observacion_objeto()
    n.girar.assert_called_once_with(math.pi / 2, OBSERVANDO_OBJETO)
    assert n.objeto_observado['desde'] is None
    n.terminar_observacion_objeto.assert_not_called()
    # El giro realmente ejecutado pone el eje óptico hacia el objeto.
    n.pose_camara.return_value = (0.0, 0.0, math.pi / 2)
    for t in (1.8, 2.1, 3.0):
        reloj['t'] = t
        n.t_ultima_imagen = t
        n.paso_observacion_objeto()
    assert n.terminar_observacion_objeto.call_args.args[0] is True


def test_no_afirma_que_miro_si_el_giro_no_movio_la_camara(monkeypatch):
    n, reloj = _nodo_observando(monkeypatch, (0.0, 0.85))
    for t in (1.4, 1.7, 1.8, 2.1, 3.0):
        reloj['t'] = t
        n.t_ultima_imagen = t
        n.paso_observacion_objeto()
    n.girar.assert_called_once()
    assert n.terminar_observacion_objeto.call_args.args[0] is False


@pytest.mark.parametrize('cambio', [dict(frame='odom'), dict(stamp_ns=0),
    dict(objetos=[_objeto(x=float('nan'))]), dict(objetos=[_objeto(color='invalido')])])
def test_rechaza_observaciones_invalidas(cambio):
    mensaje = dict(version=1, frame='map', stamp_ns=100, objetos=[_objeto()])
    mensaje.update(cambio)
    assert leer_observaciones(json.dumps(mensaje), 'map') is None


def test_finalizar_restablece_velocidad_y_no_repite_objeto(monkeypatch):
    n, reloj = _nodo_observando(monkeypatch)
    n.marcar_alrededor = Mock()
    Exploracion.terminar_observacion_objeto(n, True, 'caja azul confirmada')
    assert n.objeto_observado is None
    assert n.pub_vel.publish.call_args.args[0].speed_limit == 100.0
    assert not n.objetos_fuera_mapa.confirmados(reloj['t'])


def test_sin_camara_tambien_inspecciona_una_vez_y_guarda_el_obstaculo(monkeypatch):
    n, reloj = _nodo_observando(monkeypatch)
    n.objeto_observado = None
    n.pose_camara.return_value = None
    n.t_ultima_imagen = None
    assert n.considerar_objeto_nuevo(n.pose_robot())
    for t in (1.4, 1.7, 2.6):
        reloj['t'] = t
        n.paso_observacion_objeto()
    n.terminar_observacion_objeto.assert_called_once()
    observado, motivo = n.terminar_observacion_objeto.call_args.args
    Exploracion.terminar_observacion_objeto(n, observado, motivo)
    assert not n.considerar_objeto_nuevo(n.pose_robot())
    f, c = n.grilla.a_celda(0.85, 0.0)
    assert n.grilla.bloques_conocidos[f, c]
    assert n.grilla.obstaculos_extra[f, c]


def test_fallo_no_reencola_y_memoria_se_asocia_a_posiciones_e_id():
    rastreador = ObjetosFueraMapa()
    rastreador.finalizar((1.96, -0.88), 0.0, False,
                        aliases=[(2.19, -1.11)], id_visual='azul-0')
    rastreador.finalizar((2.43, -0.51), 0.0, False)
    # Posiciones de los reintentos de corrida_1324.log, mucho después de 10 s.
    puntos = [(1.91, -0.90), (1.98, -0.88), (2.16, -0.88), (2.43, -0.51), (3.33, -1.07)]
    for t in (100.0, 100.3, 100.6):
        rastreador.actualizar(puntos, t)
    # El ID evita reinspección incluso si la localización visual salta lejos.
    rastreador.actualizar_vision([_objeto(id='azul-0', x=4.0, y=0.0)], 100.6)
    assert rastreador.confirmados(100.6) == [(3.33, -1.07)]


@pytest.mark.parametrize('observado', [True, False])
def test_mapa_del_equipo_retiene_objeto_tras_limpiar_costmap(monkeypatch, observado):
    from pathlib import Path
    import yaml
    n, reloj = _nodo_observando(monkeypatch)
    Exploracion.terminar_observacion_objeto(n, observado, 'mirada cerrada')
    reloj['t'] = 100.0
    n.grilla.actualizar_obstaculos(np.zeros_like(n.grilla.libre))
    n.publicar_mapa_objetos()
    f, c = n.grilla.a_celda(0.85, 0.0)
    indice = f * n.grilla.ancho + c
    mapa = n.pub_mapa_objetos.publish.call_args.args[0]
    assert mapa.data[indice] == 100 and n.mapa_msg.data[indice] == 0
    assert not n.grilla.transitable[f, c]
    for t in (100.0, 100.3, 100.6):
        n.objetos_fuera_mapa.actualizar([(0.85, 0.0)], t)
    assert not n.considerar_objeto_nuevo(n.pose_robot())
    perfil = Path(__file__).resolve().parents[1] / 'params' / 'nav2_exploracion.yaml'
    nav = yaml.safe_load(perfil.read_text())
    global_cfg = nav['global_costmap']['global_costmap']['ros__parameters']
    assert global_cfg['static_layer']['map_topic'] == '/exploracion/mapa_objetos'
    assert 'static_layer' not in nav['local_costmap']['local_costmap']['ros__parameters']['plugins']


def test_actualizar_grilla_no_borra_memoria_de_objetos(monkeypatch):
    n, reloj = _nodo_observando(monkeypatch)
    Exploracion.terminar_observacion_objeto(n, True, 'mirada cerrada')
    memoria = n.objetos_fuera_mapa
    n.angulos = np.linspace(-math.pi, math.pi, 72, endpoint=False)
    n.portales_fallidos = set()
    n.publicar_mapa_amcl = Mock()
    nuevo = OccupancyGrid()
    nuevo.header.frame_id = 'map'
    nuevo.info.width, nuevo.info.height, nuevo.info.resolution = 90, 90, 0.1
    nuevo.info.origin.position.x = nuevo.info.origin.position.y = -4.0
    nuevo.data = [0] * 8100
    n.cb_mapa(nuevo)
    assert n.objetos_fuera_mapa is memoria
    f, c = n.grilla.a_celda(0.85, 0.0)
    assert n.grilla.bloques_conocidos[f, c]
    assert not n.considerar_objeto_nuevo(n.pose_robot())


def test_confirmacion_de_cajas_tampoco_revisita_un_objeto_guardado(monkeypatch):
    n, reloj = _nodo_observando(monkeypatch)
    n.cfg.update(confirmar_cajas=True, intentos_confirmacion=1)
    n.objetos_fuera_mapa.finalizar((0.85, 0.0), 0.0, True)
    n.pendientes_vision, n.t_pendientes = [(0.85, 0.0, 10)], reloj['t']
    n.intentos_confirmar = []
    assert n.elegir_confirmacion(n.pose_robot()) is None
