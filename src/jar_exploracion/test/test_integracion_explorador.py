"""Prueba callbacks y control con mensajes reales y transporte ROS interceptado."""

import math
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
from nav_msgs.msg import OccupancyGrid
from rclpy.time import Time
from rclpy.clock import ClockType

from jar_exploracion import explorador
from jar_exploracion.explorador import Explorador


@pytest.fixture
def nodo(monkeypatch):
    """Ejecuta la inicialización real sin DDS, procesos ni movimiento físico."""
    parametros = {}
    monkeypatch.setattr(explorador.Node, '__init__', lambda *args, **kwargs: None)
    monkeypatch.setattr(Explorador, 'declare_parameter',
                        lambda self, name, default: parametros.setdefault(
                            name, SimpleNamespace(value=default)))
    monkeypatch.setattr(Explorador, 'get_parameter', lambda self, name: parametros[name])
    monkeypatch.setattr(Explorador, 'create_publisher', lambda *args, **kwargs: Mock())
    monkeypatch.setattr(Explorador, 'create_subscription', lambda *args, **kwargs: Mock())
    monkeypatch.setattr(Explorador, 'create_timer', lambda *args, **kwargs: Mock())
    reloj = Mock()
    reloj.now.return_value = Time(seconds=100, clock_type=ClockType.ROS_TIME)
    monkeypatch.setattr(Explorador, 'get_clock', lambda self: reloj)
    monkeypatch.setattr(Explorador, 'get_logger', lambda self: Mock())
    buffer = Mock()
    monkeypatch.setattr(explorador, 'Buffer', lambda: buffer)
    monkeypatch.setattr(explorador, 'TransformListener', lambda *args: None)
    n = Explorador()
    yield n, buffer, reloj
    n.ejecutor_plan.shutdown(wait=True, cancel_futures=True)


def mapa():
    """Mapa libre de un metro con marco y metadatos válidos."""
    msg = OccupancyGrid()
    msg.header.frame_id = 'map'
    msg.info.width = msg.info.height = 20
    msg.info.resolution = 0.05
    msg.info.origin.orientation.w = 1.0
    msg.data = [0] * 400
    return msg


def test_mapa_repetido_conserva_objetivo_y_mapa_nuevo_frena(nodo):
    """Una retransmisión no interrumpe; un cambio invalida inmediatamente el camino."""
    n, _, _ = nodo
    msg = mapa()
    n.recibir_mapa(msg)
    n.objetivo_mision = (1, 1)
    n.camino = [(10, 10), (10, 11)]
    n.recibir_mapa(msg)
    assert n.objetivo_mision == (1, 1)
    assert n.camino is not None
    msg.data[210] = 100
    n.recibir_mapa(msg)
    assert n.objetivo_mision is None
    assert n.camino is None
    assert not n.libre_fino[n.celda_de(0.525, 0.525, n.info_mapa)]
    assert n.pub_cmd_vel.publish.call_args.args[0].linear.x == 0.0


@pytest.mark.parametrize('defecto', ['frame', 'resolucion', 'datos'])
def test_mapa_invalido_no_sustituye_mapa_bueno(nodo, defecto):
    """Mensajes malformados se rechazan sin destruir el estado existente."""
    n, _, _ = nodo
    n.recibir_mapa(mapa())
    anterior = n.libre_fino.copy()
    msg = mapa()
    if defecto == 'frame':
        msg.header.frame_id = 'otro_robot'
    elif defecto == 'resolucion':
        msg.info.resolution = 0.0
    else:
        msg.data = [0]
    n.recibir_mapa(msg)
    assert np.array_equal(anterior, n.libre_fino)


def test_tf_vencida_o_nan_no_permite_control(nodo):
    """Una posición vieja o no finita no se usa para ordenar movimiento."""
    n, buffer, _ = nodo
    tf = TransformStamped()
    tf.header.stamp = Time(seconds=99).to_msg()
    tf.transform.rotation.w = 1.0
    buffer.lookup_transform.return_value = tf
    assert n.obtener_pose_robot() is None
    tf.header.stamp = Time(seconds=100).to_msg()
    tf.transform.translation.x = float('nan')
    assert n.obtener_pose_robot() is None
    tf.transform.translation.x = 0.5
    assert n.obtener_pose_robot() == (0.5, 0.0, 0.0)


def test_pose_corregida_frena_y_descarta_plan_anterior(nodo):
    """La nueva confirmación no reutiliza un camino de la localización anterior."""
    n, _, _ = nodo
    n.estado = explorador.ESTADO_EXPLORANDO
    n.camino = [(1, 1)]
    n.objetivo_mision = (1, 1)
    n.velocidad_ordenada = 0.2
    msg = PoseWithCovarianceStamped()
    msg.header.frame_id = 'map'
    n.recibir_pose_inicial(msg)
    assert n.estado == explorador.ESTADO_INICIALIZANDO
    assert n.camino is None
    assert n.objetivo_mision is None
    assert n.velocidad_ordenada == 0.0


def test_oscilar_no_reinicia_timeout(nodo):
    """Moverse de lado sin reducir el camino no evita el detector de atasco."""
    n, _, reloj = nodo
    assert not n.objetivo_sin_avance((1, 1), 0., 0., 2.0)
    for segundo in range(1, 15):
        reloj.now.return_value = Time(seconds=100 + segundo)
        assert not n.objetivo_sin_avance((1, 1), (-1.)**segundo * 0.1, 0., 2.0)
    reloj.now.return_value = Time(seconds=115)
    assert n.objetivo_sin_avance((1, 1), 0., 0., 2.0)
    assert not n.objetivo_sin_avance((1, 1), 0., 0., 1.9)


def test_control_completo_avanza_y_frena_si_scan_invalido(nodo):
    """El comando publicado pasa por el control final y no sobrevive a perder el láser."""
    from sensor_msgs.msg import LaserScan

    n, buffer, _ = nodo
    msg = mapa()
    n.recibir_mapa(msg)
    cobertura = mapa()
    cobertura.data[10 * 20 + 14] = 100
    n.recibir_grilla(cobertura)
    scan = LaserScan()
    scan.header.frame_id = 'laser'
    scan.header.stamp = Time(seconds=100).to_msg()
    scan.range_min = 0.05
    scan.range_max = 10.0
    scan.angle_increment = 0.1
    scan.ranges = [float('inf')] * 20
    n.recibir_scan(scan)
    tf = TransformStamped()
    tf.transform.rotation.w = 1.0
    buffer.lookup_transform_full.return_value = tf
    n.actuar((0.5, 0.5, 0.0))
    if n.futuro_plan is not None:
        n.futuro_plan.result(timeout=5)
        n.actuar((0.5, 0.5, 0.0))
    comando = n.pub_cmd_vel.publish.call_args.args[0]
    assert 0.0 < comando.linear.x <= 0.3
    assert np.isfinite(comando.linear.y)
    scan.ranges = [float('nan')] * 20
    n.recibir_scan(scan)
    n.actuar((0.5, 0.5, 0.0))
    comando = n.pub_cmd_vel.publish.call_args.args[0]
    assert comando.linear.x == comando.linear.y == 0.0


def test_plan_pendiente_no_parpadea_ni_reinicia_calculo(nodo, monkeypatch):
    """Mantiene el destino durante varios ticks y arranca al completar el mismo futuro."""
    from concurrent.futures import Future
    from sensor_msgs.msg import LaserScan
    from visualization_msgs.msg import Marker
    from jar_exploracion.planificador import a_estrella

    n, buffer, _ = nodo
    n.recibir_mapa(mapa())
    cobertura = mapa()
    cobertura.data[10 * 20 + 14] = 100
    n.recibir_grilla(cobertura)
    scan = LaserScan()
    scan.header.frame_id = 'laser'
    scan.header.stamp = Time(seconds=100).to_msg()
    scan.range_min = 0.05
    scan.range_max = 10.0
    scan.angle_increment = 0.1
    scan.ranges = [float('inf')] * 20
    n.recibir_scan(scan)
    tf = TransformStamped()
    tf.transform.rotation.w = 1.0
    buffer.lookup_transform_full.return_value = tf
    pendiente = Future()
    submit = Mock(return_value=pendiente)
    monkeypatch.setattr(n.ejecutor_plan, 'submit', submit)
    n.pub_objetivo.publish.reset_mock()
    for _ in range(10):
        n.actuar((0.5, 0.5, 0.0))
        assert n.pub_cmd_vel.publish.call_args.args[0].linear.x == 0.0
    assert submit.call_count == 1
    assert all(c.args[0].action == Marker.ADD for c in n.pub_objetivo.publish.call_args_list)
    fin = n.clave_plan[1]
    pendiente.set_result(a_estrella(n.libre_fino, n.celda_de(0.5, 0.5, n.info_mapa), fin))
    n.actuar((0.5, 0.5, 0.0))
    assert n.pub_cmd_vel.publish.call_args.args[0].linear.x > 0.0
    assert submit.call_count == 1


@pytest.mark.parametrize('cubiertas, termina', [(97, False), (98, True), (99, True)])
def test_termina_segun_porcentaje_sin_contar_paredes(nodo, cubiertas, termina):
    """El umbral se calcula sobre libres y no exige visitar cada celda pendiente."""
    n, _, _ = nodo
    n.estado = explorador.ESTADO_EXPLORANDO
    cobertura = mapa()
    cobertura.data = [0] * cubiertas + [100] * (100 - cubiertas) + [-1] * 300
    n.recibir_grilla(cobertura)
    assert n.porcentaje_cobertura() == cubiertas
    n.transicionar((0.5, 0.5, 0.0))
    assert (n.estado == explorador.ESTADO_REGRESANDO) == termina


def test_100_por_ciento_opcional_y_grilla_sin_libres(nodo):
    """Cero celdas libres no significa misión completa y 100 conserva el criterio estricto."""
    n, _, _ = nodo
    assert n.porcentaje_cobertura() is None
    n.porcentaje_objetivo = 100.0
    n.estado = explorador.ESTADO_EXPLORANDO
    cobertura = mapa()
    cobertura.data = [0] * 99 + [100] + [-1] * 300
    n.recibir_grilla(cobertura)
    n.transicionar((0.5, 0.5, 0.0))
    assert n.estado == explorador.ESTADO_EXPLORANDO


def test_umbral_cancela_camino_y_frena(nodo, monkeypatch):
    """El siguiente ciclo publica terminado, elimina el destino y deja velocidad cero."""
    from concurrent.futures import Future

    n, _, _ = nodo
    n.estado = explorador.ESTADO_EXPLORANDO
    n.recibir_mapa(mapa())
    cobertura = mapa()
    cobertura.data = [0] * 98 + [100] * 2 + [-1] * 300
    n.recibir_grilla(cobertura)
    n.camino = [(10, 10), (10, 11)]
    n.objetivo_mision = (1, 1)
    n.origen = (0.5, 0.5)
    n.futuro_plan = Future()
    pendiente = n.futuro_plan
    monkeypatch.setattr(n, 'obtener_pose_robot', lambda: (0.5, 0.5, 0.0))
    n.controlar()
    assert n.estado == explorador.ESTADO_TERMINADO
    assert pendiente.cancelled()
    assert n.camino is None and n.objetivo_mision is None
    assert n.pub_estado.publish.call_args.args[0].data == 'terminado'
    assert n.pub_cmd_vel.publish.call_args.args[0].linear.x == 0.0


def test_caja_fuera_del_mapa_replanifica_por_un_costado_y_expira(nodo):
    """Una caja vista por lidar corta la ruta directa y A* encuentra un rodeo."""
    from concurrent.futures import Future
    from sensor_msgs.msg import LaserScan
    from jar_exploracion.planificador import a_estrella, segmento_transitable

    n, _, reloj = nodo
    msg = mapa()
    msg.info.width = msg.info.height = 80
    msg.data = [0] * 6400
    n.recibir_mapa(msg)
    inicio = n.celda_de(0.8, 2.0, n.info_mapa)
    fin = n.celda_de(3.2, 2.0, n.info_mapa)
    n.camino = a_estrella(n.libre_fino, inicio, fin)
    assert segmento_transitable(n.libre_fino, inicio, fin)
    pendiente = Future()
    n.futuro_plan = pendiente
    n.recibir_scan(LaserScan())
    # Cara de una caja a x=1.7 m (dentro de alcance_obstaculos_scan),
    # coordenadas relativas a la base en (0.8, 2).
    impactos = np.array([(0.9, y) for y in np.linspace(-0.125, 0.125, 11)])
    n.actualizar_obstaculos((0.8, 2.0, 0.0), impactos)
    assert pendiente.cancelled()
    assert n.camino is None
    assert not segmento_transitable(n.libre_fino, inicio, fin)
    rodeo = a_estrella(n.libre_fino, inicio, fin)
    assert rodeo
    assert max(abs(f - inicio[0]) for f, _ in rodeo) >= 10
    assert all(segmento_transitable(n.libre_fino, a, b)
               for a, b in zip(rodeo, rodeo[1:]))
    # No modifica el mapa original ni renueva la memoria con un scan repetido.
    assert np.all(n.datos_mapa == 0)
    reloj.now.return_value = Time(seconds=109)
    n.actualizar_obstaculos((0.8, 2.0, 0.0), impactos)
    assert segmento_transitable(n.libre_fino, inicio, fin)


def test_impactos_rotados_y_pose_corregida_descarta_obstaculos(nodo):
    """Los impactos usan la pose del robot y no sobreviven a una relocalización."""
    import math
    from sensor_msgs.msg import LaserScan

    n, _, _ = nodo
    n.recibir_mapa(mapa())
    n.recibir_scan(LaserScan())
    n.actualizar_obstaculos((0.5, 0.5, math.pi / 2), np.array([(0.3, 0.)]))
    celda = n.celda_de(0.5, 0.8, n.info_mapa)
    assert n.datos_planificacion[celda] == 100
    msg = PoseWithCovarianceStamped()
    msg.header.frame_id = 'map'
    n.recibir_pose_inicial(msg)
    n.recibir_scan(LaserScan())
    n.actualizar_obstaculos((0.5, 0.5, 0.), np.empty((0, 2)))
    assert not n.obstaculos_temporales
    assert n.datos_planificacion[celda] == 0


def test_guarda_origen_al_iniciar_y_lo_conserva_al_corregir_pose(nodo):
    """El origen es la pose asentada de salida, no una corrección durante la misión."""
    n, _, reloj = nodo
    n.recibir_mapa(mapa())
    n.recibir_grilla(mapa())
    msg = PoseWithCovarianceStamped()
    msg.header.frame_id = 'map'
    n.recibir_pose_inicial(msg)
    n.transicionar((0.4, 0.4, 0.0))
    assert n.origen is None
    reloj.now.return_value = Time(seconds=104, clock_type=ClockType.ROS_TIME)
    n.transicionar((0.5, 0.5, 0.0))
    assert n.origen == (0.5, 0.5)
    n.recibir_pose_inicial(msg)
    reloj.now.return_value = Time(seconds=108, clock_type=ClockType.ROS_TIME)
    n.transicionar((0.7, 0.7, 0.0))
    assert n.origen == (0.5, 0.5)


def test_regresa_antes_de_terminar_y_no_cambia_por_cobertura(nodo):
    """Completar cobertura lejos del origen inicia regreso y solo llegar termina."""
    n, _, _ = nodo
    n.recibir_mapa(mapa())
    n.origen = (0.3, 0.3)
    n.estado = explorador.ESTADO_EXPLORANDO
    n.recibir_grilla(mapa())
    n.transicionar((0.7, 0.7, 0.0))
    assert n.estado == explorador.ESTADO_REGRESANDO
    n.objetivo_celda = (-1, -1)
    n.camino = [(15, 15), (12, 12)]
    n.recibir_grilla(mapa())
    assert n.camino is not None
    n.transicionar(None)
    assert n.estado == explorador.ESTADO_REGRESANDO
    n.transicionar((0.35, 0.3, 0.0))
    assert n.estado == explorador.ESTADO_TERMINADO


def test_regreso_controla_hacia_origen_con_scan_valido(nodo):
    """El regreso usa A* y publica movimiento hacia el origen guardado."""
    from sensor_msgs.msg import LaserScan

    n, buffer, _ = nodo
    n.recibir_mapa(mapa())
    n.recibir_grilla(mapa())
    n.estado = explorador.ESTADO_REGRESANDO
    n.origen = (0.3, 0.5)
    scan = LaserScan()
    scan.header.frame_id = 'laser'
    scan.header.stamp = Time(seconds=100).to_msg()
    scan.range_min = 0.05
    scan.range_max = 10.0
    scan.angle_increment = 0.1
    scan.ranges = [float('inf')] * 20
    n.recibir_scan(scan)
    tf = TransformStamped()
    tf.transform.rotation.w = 1.0
    buffer.lookup_transform_full.return_value = tf
    n.actuar((0.7, 0.5, 0.0))
    if n.futuro_plan is not None:
        n.futuro_plan.result(timeout=5)
        n.actuar((0.7, 0.5, 0.0))
    assert n.pub_cmd_vel.publish.call_args.args[0].linear.x < 0.0
    scan.ranges = [float('nan')] * 20
    n.recibir_scan(scan)
    n.actuar((0.7, 0.5, 0.0))
    cmd = n.pub_cmd_vel.publish.call_args.args[0]
    assert cmd.linear.x == cmd.linear.y == 0.0


def preparar_mirada(nodo, ancho_pasillo=None):
    """Mapa con paredes en el borde (y opcionalmente un pasillo) y scan válido."""
    from sensor_msgs.msg import LaserScan

    n, buffer, _ = nodo
    msg = mapa()
    for i in range(20):
        for j in (0, 19):
            msg.data[i * 20 + j] = msg.data[j * 20 + i] = 100
    if ancho_pasillo is not None:
        # Paredes horizontales a ambos lados de y = 0.5 m.
        medio = int(ancho_pasillo / 0.05 / 2)
        for col in range(20):
            msg.data[(10 - medio - 1) * 20 + col] = 100
            msg.data[(10 + medio) * 20 + col] = 100
    n.recibir_mapa(msg)
    n.recibir_grilla(mapa())
    n.estado = explorador.ESTADO_EXPLORANDO
    scan = LaserScan()
    scan.header.frame_id = 'laser'
    scan.header.stamp = Time(seconds=100).to_msg()
    scan.range_min = 0.05
    scan.range_max = 10.0
    scan.angle_increment = 0.1
    scan.ranges = [float('inf')] * 20
    n.recibir_scan(scan)
    tf = TransformStamped()
    tf.transform.rotation.w = 1.0
    buffer.lookup_transform_full.return_value = tf
    # Mirada vencida: la última vuelta fue a un tramo de distancia.
    n.posicion_ultima_mirada = (0.5 - n.distancia_entre_miradas, 0.5)
    return n


def test_no_gira_al_empezar_a_explorar(nodo):
    n = preparar_mirada(nodo)
    n.posicion_ultima_mirada = None
    assert not n.mirar_alrededor((0.5, 0.5, 0.0))
    assert not n.mirada.activa


def test_mirada_gira_en_el_lugar_en_espacio_abierto(nodo):
    """Después de un tramo mira alrededor: solo velocidad angular."""
    n = preparar_mirada(nodo)
    assert n.mirar_alrededor((0.5, 0.5, 0.0))
    cmd = n.pub_cmd_vel.publish.call_args.args[0]
    assert cmd.angular.z > 0.0
    assert cmd.linear.x == cmd.linear.y == 0.0
    assert n.mirada.activa


def test_mirada_no_gira_en_pasillo_angosto(nodo):
    """Si el círculo de giro no entra, la mirada queda pendiente y se sigue explorando."""
    n = preparar_mirada(nodo, ancho_pasillo=0.4)
    assert not n.mirar_alrededor((0.5, 0.5, 0.0))
    assert not n.mirada.activa
    assert n.recorrido_desde_mirada >= n.distancia_entre_miradas


def test_mirada_no_gira_al_regresar_ni_desactivada(nodo):
    n = preparar_mirada(nodo)
    n.estado = explorador.ESTADO_REGRESANDO
    assert not n.mirar_alrededor((0.5, 0.5, 0.0))
    n.estado = explorador.ESTADO_EXPLORANDO
    n.distancia_entre_miradas = 0.0
    assert not n.mirar_alrededor((0.5, 0.5, 0.0))


def test_mirada_no_consume_el_timeout_sin_avance(nodo):
    """Girar varios segundos no hace que el objetivo se marque como sin avance."""
    n = preparar_mirada(nodo)
    n.seguimiento_avance = ((3, 3), 0.5, 0.5, 0.0, 1.0)
    assert n.mirar_alrededor((0.5, 0.5, 0.0))
    assert n.seguimiento_avance is None


def test_impactos_junto_a_pared_conocida_no_son_obstaculos(nodo):
    """La pared vista con error de localización no engorda; una caja lejos sí se marca."""
    n = preparar_mirada(nodo)
    n.ultimo_scan = object()
    pose = (0.5, 0.5, 0.0)
    # Desde (0.5, 0.5): un impacto a 6 cm de la pared del borde y otro en el medio.
    n.actualizar_obstaculos(pose, [(-0.39, 0.0), (0.2, 0.0)])
    marcadas = [n.celda_a_mundo(f, c, n.info_mapa) for f, c in n.obstaculos_temporales]
    assert len(marcadas) == 1
    x, y = marcadas[0]
    assert abs(x - 0.7) < 0.05 and abs(y - 0.5) < 0.05


def test_obstaculos_del_scan_usan_margen_menor(nodo):
    """Una caja cerca del robot bloquea menos que si tuviera el margen de las paredes."""
    n = preparar_mirada(nodo)
    datos = n.datos_mapa.copy()
    fila, col = n.celda_de(0.5, 0.5, n.info_mapa)
    datos[fila - 1:fila + 2, col - 1:col + 2] = 100
    con_margen_scan = n.espacio_libre(datos, 0.0).sum()
    n.margen_obstaculos_scan = n.margen_seguridad
    con_margen_pared = n.espacio_libre(datos, 0.0).sum()
    assert con_margen_scan > con_margen_pared


def test_frenado_usa_el_margen_de_obstaculos_del_scan(nodo, monkeypatch):
    """El control en vivo no exige más margen a una caja que el planificador."""
    n = preparar_mirada(nodo)
    usados = []
    original = explorador.regular_traslacion

    def espia(velocidad, puntos, largo, ancho, margen, *args, **kwargs):
        usados.append(margen)
        return original(velocidad, puntos, largo, ancho, margen, *args, **kwargs)

    monkeypatch.setattr(explorador, 'regular_traslacion', espia)
    n.recibir_grilla(mapa())
    cobertura = mapa()
    cobertura.data[10 * 20 + 14] = 100
    n.recibir_grilla(cobertura)
    n.actuar((0.5, 0.5, 0.0))
    if n.futuro_plan is not None:
        n.futuro_plan.result(timeout=5)
        n.actuar((0.5, 0.5, 0.0))
    assert usados and all(m == n.margen_obstaculos_scan for m in usados)


def test_sale_del_margen_de_una_pared_en_vez_de_quedarse_bloqueado(nodo):
    """Pegado a la pared de abajo, ordena alejarse hacia +y despacio."""
    n = preparar_mirada(nodo)
    x, y = 0.5, 0.17
    assert not n.libre_fino[n.celda_fina_segura(x, y)]
    n.pub_cmd_vel.reset_mock()
    assert n.salir_del_margen(x, y, 0.0, np.zeros((0, 2)))
    cmd = n.pub_cmd_vel.publish.call_args[0][0]
    assert cmd.linear.y > 0.0
    assert math.hypot(cmd.linear.x, cmd.linear.y) <= n.velocidad_escape + 1e-9


def test_no_sale_del_margen_si_el_scan_lo_impide(nodo):
    """Caja arriba y pared abajo, pegadas al chasis: ningún paso aleja."""
    n = preparar_mirada(nodo)
    lados = np.linspace(-0.2, 0.2, 21)
    encierro = np.array([[x, 0.13] for x in lados] + [[x, -0.13] for x in lados]
                        + [[0.195, y] for y in lados] + [[-0.195, y] for y in lados])
    assert not n.salir_del_margen(0.5, 0.17, 0.0, encierro)


def test_encerrado_por_el_scan_se_aleja_de_lo_que_ve(nodo):
    """Sin celda libre cerca, se mueve en la dirección que más lo aleja del scan."""
    n = preparar_mirada(nodo)
    n.libre_fino[:] = False
    caja = np.array([[0.2, y] for y in np.linspace(-0.1, 0.1, 11)])
    n.pub_cmd_vel.reset_mock()
    assert n.salir_del_margen(0.5, 0.5, 0.0, caja)
    cmd = n.pub_cmd_vel.publish.call_args[0][0]
    assert cmd.linear.x < 0.0


def test_mirada_bloqueada_un_momento_se_retoma_sin_reiniciar(nodo, monkeypatch):
    """Un impacto pasajero en el círculo frena el giro pero no empieza otra vuelta."""
    n = preparar_mirada(nodo)
    _, _, reloj = nodo
    assert n.mirar_alrededor((0.5, 0.5, 0.0))
    yaw_inicial = n.mirada.yaw_inicial
    monkeypatch.setattr(explorador, 'giro_seguro', lambda *args: False)
    reloj.now.return_value = Time(seconds=101, clock_type=ClockType.ROS_TIME)
    assert n.mirar_alrededor((0.5, 0.5, 1.0))
    cmd = n.pub_cmd_vel.publish.call_args.args[0]
    assert cmd.angular.z == 0.0 and n.mirada.activa
    monkeypatch.setattr(explorador, 'giro_seguro', lambda *args: True)
    reloj.now.return_value = Time(seconds=102, clock_type=ClockType.ROS_TIME)
    assert n.mirar_alrededor((0.5, 0.5, 1.0))
    assert n.mirada.yaw_inicial == yaw_inicial


def test_mirada_bloqueada_se_abandona_y_no_reinicia_enseguida(nodo, monkeypatch):
    n = preparar_mirada(nodo)
    _, _, reloj = nodo
    assert n.mirar_alrededor((0.5, 0.5, 0.0))
    monkeypatch.setattr(explorador, 'giro_seguro', lambda *args: False)
    reloj.now.return_value = Time(seconds=101, clock_type=ClockType.ROS_TIME)
    assert n.mirar_alrededor((0.5, 0.5, 1.0))
    reloj.now.return_value = Time(seconds=101 + explorador.ESPERA_MIRADA_S,
                                  clock_type=ClockType.ROS_TIME)
    assert not n.mirar_alrededor((0.5, 0.5, 1.0))
    assert not n.mirada.activa
    monkeypatch.setattr(explorador, 'giro_seguro', lambda *args: True)
    assert not n.mirar_alrededor((0.5, 0.5, 1.0))
    assert not n.mirada.activa


def test_mirada_solo_si_hubo_cobertura_nueva(nodo):
    """Tras 1.5 m sin cubrir celdas nuevas no hace falta girar."""
    n = preparar_mirada(nodo)
    n.cubiertas_en_ultima_mirada = n.celdas_cubiertas
    assert not n.mirar_alrededor((0.5, 0.5, 0.0))
    n.celdas_cubiertas += 1
    assert n.mirar_alrededor((0.5, 0.5, 0.0))


def test_deriva_de_giro_no_reinicia_la_busqueda_de_observacion(nodo):
    """Unos grados de desvío respecto del eje no recalculan el espacio libre."""
    n = preparar_mirada(nodo)
    libre = n.libre_fino
    n.puntos_observacion[(1, 1)] = (0.5, 0.5)
    n.actualizar_orientacion_huella(0.05)
    assert n.libre_fino is libre
    assert n.puntos_observacion == {(1, 1): (0.5, 0.5)}
    n.actualizar_orientacion_huella(math.pi / 2)
    assert n.libre_fino is not libre


def test_obstaculos_del_scan_no_borran_la_busqueda_de_observacion(nodo):
    """Un impacto nuevo cambia el espacio libre sin perder lo ya buscado."""
    n = preparar_mirada(nodo)
    n.puntos_observacion[(1, 1)] = (0.5, 0.5)
    n.cache_busquedas[((1, 1), 10.0)] = np.array([[5, 5]])
    n.actualizar_obstaculos((0.5, 0.5, 0.0), np.array([[0.2, 0.0]]))
    assert not n.libre_fino[n.celda_fina_segura(0.7, 0.5)]
    assert n.puntos_observacion == {(1, 1): (0.5, 0.5)}
    assert ((1, 1), 10.0) in n.cache_busquedas


def test_el_temblor_de_la_pose_no_suma_recorrido(nodo):
    """Ir y venir un centímetro muchas veces no cuenta como un tramo."""
    n = preparar_mirada(nodo)
    n.posicion_ultima_mirada = None
    for i in range(400):
        assert not n.mirar_alrededor((0.5 + 0.01 * (i % 2), 0.5, 0.0))
    assert n.recorrido_desde_mirada < 0.02


def test_impactos_lejanos_no_son_obstaculos_temporales(nodo):
    n = preparar_mirada(nodo)
    n.alcance_obstaculos_scan = 0.3
    n.actualizar_obstaculos((0.5, 0.5, 0.0), np.array([[0.35, 0.0], [0.2, 0.0]]))
    assert len(n.obstaculos_temporales) == 1


def test_impacto_suelto_no_es_obstaculo_hasta_repetirse(nodo):
    """Con impactos_minimos_scan, una celda vista en un solo scan no bloquea."""
    from sensor_msgs.msg import LaserScan
    n = preparar_mirada(nodo)
    n.impactos_minimos_scan = 3
    libre = n.libre_fino.copy()
    impacto = np.array([[0.2, 0.0]])
    for vez in range(3):
        n.recibir_scan(LaserScan())
        n.actualizar_obstaculos((0.5, 0.5, 0.0), impacto)
        assert np.array_equal(n.libre_fino, libre) == (vez < 2)


def test_con_camino_vigente_no_vuelve_a_elegir_destino(nodo, monkeypatch):
    """Mientras el camino hacia la zona sigue libre, el destino no se reevalúa."""
    n = preparar_mirada(nodo)
    cobertura = mapa()
    cobertura.data[10 * 20 + 14] = 100
    n.recibir_grilla(cobertura)
    zona = (10, 14)
    n.objetivo_mision = n.objetivo_celda = zona
    n.destino_actual = (0.72, 0.52)
    n.camino = [n.celda_de(0.6, 0.5, n.info_mapa), n.celda_de(0.72, 0.52, n.info_mapa)]

    def no_llamar(*args):
        raise AssertionError('no debería reelegir el destino')

    monkeypatch.setattr(n, 'punto_seguro_en_zona', no_llamar)
    monkeypatch.setattr(n, 'punto_para_observar', no_llamar)
    n.actuar((0.5, 0.5, 0.0))
    assert n.objetivo_celda == zona


def test_costo_centra_el_camino_en_un_pasillo(nodo):
    """En un pasillo con holgura, el costo es mínimo en el medio y crece hacia las paredes."""
    n = preparar_mirada(nodo, ancho_pasillo=0.6)
    costos = [n.costo_fino[n.celda_de(0.5, y, n.info_mapa)] for y in (0.5, 0.45, 0.4)]
    assert costos[0] <= costos[1] <= costos[2]
    assert costos[0] < costos[2]


def test_regreso_por_tiempo_y_done_en_la_largada(nodo):
    """Vencido tiempo_regreso_s vuelve; en la largada publica /done una sola vez."""
    n = preparar_mirada(nodo)
    _, _, reloj = nodo
    n.origen = (0.5, 0.5)
    n.tiempo_regreso_s = 30.0
    n.momento_mapa = Time(seconds=100, clock_type=ClockType.ROS_TIME)
    reloj.now.return_value = Time(seconds=131, clock_type=ClockType.ROS_TIME)
    n.revisar_reloj()
    n.transicionar((2.0, 2.0, 0.0))
    assert n.estado == explorador.ESTADO_REGRESANDO
    n.transicionar((0.5, 0.5, 0.0))
    assert n.estado == explorador.ESTADO_TERMINADO


def test_con_todas_las_victimas_vuelve(nodo):
    n = preparar_mirada(nodo)
    n.recibir_victimas_completas(None)
    n.transicionar((2.0, 2.0, 0.0))
    assert n.estado == explorador.ESTADO_REGRESANDO


def test_quieto_mucho_tiempo_fuerza_una_mirada(nodo, monkeypatch):
    """El corte de 60 s del jurado es en tiempo real: el temporizador usa el reloj de pared."""
    n = preparar_mirada(nodo)
    pared = [1000.0]
    monkeypatch.setattr(explorador.time, 'monotonic', lambda: pared[0])
    n.posicion_ultima_mirada = (0.5, 0.5)
    n.cubiertas_en_ultima_mirada = n.celdas_cubiertas
    n.velocidad_ordenada = 0.0
    n.revisar_reloj()
    pared[0] += n.espera_inmovil_s
    n.revisar_reloj()
    assert n.forzar_mirada
    assert n.mirar_alrededor((0.5, 0.5, 0.0))
    assert n.mirada.activa


def test_marcas_del_scan_que_encierran_al_robot_se_borran(nodo):
    """Si el mapa lo deja libre y solo lo encierran marcas del scan, se liberan."""
    n = preparar_mirada(nodo)
    x, y = 0.5, 0.5
    n.actualizar_obstaculos((x, y, 0.0), np.array([[0.17, 0.0], [-0.17, 0.0]]))
    assert not n.libre_fino[n.celda_fina_segura(x, y)]
    assert n.liberar_marcas_cercanas(x, y)
    assert n.libre_fino[n.celda_fina_segura(x, y)]
    assert not n.obstaculos_temporales
