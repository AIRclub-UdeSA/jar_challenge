"""Regresiones de pérdida y recuperación acotada de localización."""
import math
import time

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

from equipo_jar.localizacion_activa import LocalizacionActiva, Observacion
from equipo_jar.nodo_equipo import NodoEquipo


def medida(secuencia=10, coincidencia=1.0):
    return Observacion(0.4, (0.0, 0.0, 0.4), secuencia, 0.1, 0.1,
                       coincidencia, 1.0)


def lista(**kwargs):
    control = LocalizacionActiva(**kwargs)
    control.paso(0, True, medida(1))
    control.paso(1, True, medida(2))
    control.paso(2, True, Observacion(0.8, (0, 0, 0.8), 3, .1, .1, 1, 1))
    for secuencia in (4, 5, 6):
        control.paso(2+secuencia*.1, True, medida(secuencia))
    assert control.estado == 'LISTA'
    return control


def perder(control, desde):
    """Falta de datos sostenida: desde M2 un ciclo no basta (plazo_falta_datos = 2 s)."""
    control.paso(desde, True, None)
    control.paso(desde + 2.1, True, None)


def test_pierde_habilitacion_ante_datos_faltantes():
    control = lista()
    perder(control, 97.9)
    assert control.estado == 'RECUPERANDO'


def test_pierde_habilitacion_ante_mala_concordancia_o_covarianza():
    for tipo in ('mapa', 'covarianza', 'invalida'):
        control = lista()
        m = medida(7, 0.1 if tipo == 'mapa' else 1.0)
        if tipo == 'covarianza':
            m.std_yaw = 0.3
        if tipo == 'invalida':
            m.pose = (math.nan, 0, 0)
        assert control.paso(100, True, m) == 0
        assert not control.utilizable  # se detiene de inmediato
        assert control.paso(101.1, True, m) == 0  # y recupera si persiste más de 1 s
        assert control.estado == 'RECUPERANDO'


def test_exige_tres_poses_nuevas_y_no_reutiliza_la_pose_retendida():
    control = lista()
    perder(control, 97.9)
    for instante in (100.1, 100.2, 100.3):
        assert control.paso(instante, True, medida(6)) == 0
    assert control.estado == 'RECUPERANDO'
    for secuencia in (7, 8):
        assert control.paso(100+secuencia*.1, True, medida(secuencia)) == 0
        assert control.estado == 'RECUPERANDO'
    assert control.paso(101, True, medida(9)) == 0
    assert control.estado == 'LISTA'
    assert control.recuperaciones == 1


def test_datos_perdidos_interrumpen_las_muestras_de_recuperacion():
    control = lista()
    perder(control, 97.9)
    control.paso(101, True, medida(7))
    control.paso(102, True, medida(8))
    control.paso(103, True, None)
    control.paso(104, True, medida(9))
    assert control.estado == 'RECUPERANDO'
    control.paso(105, True, medida(10))
    control.paso(106, True, medida(11))
    assert control.estado == 'LISTA'


def test_recuperacion_persistente_reinicia_ventana_sin_moverse():
    control = lista(plazo_recuperacion=5)
    perder(control, 97.9)
    assert control.paso(105, True, None) == 0
    assert control.estado == 'RECUPERANDO' and not control.utilizable
    for secuencia in (9, 10, 11):
        assert control.paso(106+secuencia*.1, True, medida(secuencia)) == 0
    assert control.utilizable


def test_presupuesto_de_episodios_no_bloquea_tras_primera_lista():
    control = lista(max_recuperaciones=1)
    perder(control, 97.9)
    for secuencia in (7, 8, 9):
        control.paso(100+secuencia*.1, True, medida(secuencia))
    assert control.estado == 'LISTA'
    perder(control, 102)
    assert control.estado == 'RECUPERANDO' and not control.utilizable


def test_pose_vencida_se_rechaza_aunque_scan_y_odom_sean_recientes(monkeypatch):
    rclpy.init()
    node = NodoEquipo()
    try:
        scan = LaserScan()
        scan.range_max = 10.0
        scan.ranges = [1.0]*60
        node.al_scan(scan)
        node.al_odom(Odometry())
        pose = PoseWithCovarianceStamped()
        pose.header.frame_id = 'map'
        pose.pose.pose.orientation.w = 1.0
        node.al_pose(pose)
        node.mapa = object()
        node.t_pose = time.monotonic()-4.0

        def no_lookup(*args, **kwargs):
            raise AssertionError('Una pose vencida no debe llegar a la consulta TF')

        monkeypatch.setattr(node.tf_buffer, 'lookup_transform', no_lookup)
        assert node.observacion(time.monotonic()) is None
    finally:
        node.destroy_node()
        rclpy.shutdown()

def test_pose_retrasada_no_renueva_frescura_ni_reemplaza_la_actual():
    rclpy.init()
    node = NodoEquipo()
    try:
        actual = PoseWithCovarianceStamped()
        actual.header.stamp.sec = 10
        actual.header.frame_id = 'map'
        node.al_pose(actual)
        recibido = node.t_pose
        retrasada = PoseWithCovarianceStamped()
        retrasada.header.stamp.sec = 9
        retrasada.header.frame_id = 'map'
        node.al_pose(retrasada)
        assert node.secuencia_pose == 1
        assert node.pose.header.stamp.sec == 10
        assert node.t_pose == recibido
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_un_ciclo_sin_observacion_no_gasta_recuperacion():
    control = lista()
    assert control.utilizable
    assert control.paso(100, True, None) == 0
    assert control.estado == 'LISTA' and not control.utilizable
    control.paso(100.1, True, medida(7))
    assert control.estado == 'LISTA' and control.utilizable
    assert control.recuperaciones == 0


def test_falta_de_datos_de_mas_de_2_s_gasta_una_recuperacion():
    control = lista()
    control.paso(100, True, None)
    control.paso(101.9, True, None)
    assert control.estado == 'LISTA' and not control.utilizable
    control.paso(102.1, True, None)
    assert control.estado == 'RECUPERANDO' and control.recuperaciones == 1


def test_calidad_mala_por_0_5_s_no_gasta():
    control = lista()
    control.paso(100, True, medida(7, 0.1))
    control.paso(100.5, True, medida(8, 0.1))
    assert control.estado == 'LISTA' and not control.utilizable
    control.paso(100.6, True, medida(9))
    assert control.utilizable and control.recuperaciones == 0


def test_calidad_mala_por_1_2_s_gasta():
    control = lista()
    control.paso(100, True, medida(7, 0.1))
    control.paso(101.2, True, medida(8, 0.1))
    assert control.estado == 'RECUPERANDO' and control.recuperaciones == 1


def test_no_utilizable_fuera_de_lista():
    assert not LocalizacionActiva().utilizable



def test_cuatro_recuperaciones_tras_lista_siempre_rehabilitan_sin_movimiento():
    control = lista(max_recuperaciones=2)
    secuencia = 7
    for episodio in range(4):
        t = 10.+episodio*5
        assert control.paso(t, True, None) == 0
        assert not control.utilizable
        assert control.paso(t+2.1, True, None) == 0
        assert control.estado == 'RECUPERANDO' and not control.utilizable
        for dt in (2.2, 2.3, 2.4):
            assert control.paso(t+dt, True, medida(secuencia)) == 0
            secuencia += 1
        assert control.utilizable
    assert control.recuperaciones == 4


def test_contador_se_reinicia_solo_tras_30s_continuos_utilizables():
    control = lista()
    perder(control, 10.)
    for secuencia, t in [(7, 12.2), (8, 12.3), (9, 12.4)]:
        control.paso(t, True, medida(secuencia))
    control.paso(42.3, True, medida(10))
    assert control.recuperaciones == 1
    control.paso(42.4, True, medida(11))
    assert control.recuperaciones == 0


def test_falla_breve_interrumpe_los_30s_continuos():
    control = lista()
    perder(control, 10.)
    for secuencia, t in [(7, 12.2), (8, 12.3), (9, 12.4)]:
        control.paso(t, True, medida(secuencia))
    control.paso(40., True, None)
    control.paso(40.1, True, medida(10))
    control.paso(69.9, True, medida(11))
    assert control.recuperaciones == 1
    control.paso(70.2, True, medida(12))
    assert control.recuperaciones == 0


def test_adquisicion_inicial_sigue_fallando_y_no_se_rearma():
    control = LocalizacionActiva(presupuesto=1.)
    control.paso(0., True, None)
    control.paso(2., True, None)
    assert control.estado == 'INSUFICIENTE'
    for secuencia in (1, 2, 3, 4):
        assert control.paso(3+secuencia, True, medida(secuencia)) == 0
        assert not control.utilizable and control.estado == 'INSUFICIENTE'



def test_nodo_publica_cmd_cero_mientras_recupera_tras_lista():
    from rclpy.parameter import Parameter
    from test_mision import nueva
    from test_percepcion_nodo import Captura
    rclpy.init()
    node = None
    try:
        node = NodoEquipo(parameter_overrides=[Parameter('modo_navegacion', value='mision')])
        _, node.navegacion = nueva()
        node.mapa = node.mapa_navegacion = object()
        node.control = lista(max_recuperaciones=1)
        perder(node.control, 10.)
        node.observacion = lambda ahora: None
        node.datos_navegacion = lambda *args: None
        node.pub_cmd_vel = Captura()
        for _ in range(4):
            node.ciclo_de_control()
            cmd = node.pub_cmd_vel.mensajes[-1]
            assert cmd.linear.x == cmd.linear.y == cmd.angular.z == 0.
            assert not node.control.utilizable and node.control.estado == 'RECUPERANDO'
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()
