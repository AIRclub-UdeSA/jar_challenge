"""Integración ROS M4: exclusión antes del mapa, óptica y /done detenido."""
import json
import math
import time

import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo, Image

from test_nav_nodo import nodo, preparar_lista, mapa_libre, cerrar, Captura
from test_percepcion_nodo import node, preparar


def test_m4_espera_mapa():
    n = nodo(modo_navegacion='mision')
    try:
        preparar_lista(n)
        n.pub_done = Captura()
        n.ciclo_de_control()
        assert n.navegacion is None and n.mision is None
        assert not n.pub_cmd_vel.mensajes and not n.pub_done.mensajes
    finally:
        cerrar(n)


def test_m4_camara_ausente_espera_sin_meta():
    n = nodo(modo_navegacion='mision')
    try:
        n.al_recibir_mapa(mapa_libre())
        preparar_lista(n)
        n.ciclo_de_control()
        assert n.mision is not None
        assert not n.navegacion.p.regreso_automatico
        assert n.navegacion.meta is None
        assert n.mision.estado == 'ESPERANDO_CAMARA'
        assert n.pub_cmd_vel.mensajes[-1].linear.x == 0.
    finally:
        cerrar(n)


def test_done_exige_quietud_y_es_unico():
    n = nodo(modo_navegacion='mision')
    try:
        n.al_recibir_mapa(mapa_libre())
        preparar_lista(n)
        n.ciclo_de_control()
        n.pub_done = Captura()
        n.mision.finalizar = True
        n.mision.estado = 'FINALIZADA'
        n.navegacion.estado = 'EN_LARGADA'
        n.navegacion.largada = (1.5, 1., 0.)
        n.odom = Odometry()
        n.odom.twist.twist.linear.x = .1
        n.ciclo_de_control()
        assert not n.pub_done.mensajes
        n.odom.twist.twist.linear.x = 0.
        n.t_fin_quieto = time.monotonic()-.6
        n.ciclo_de_control()
        assert len(n.pub_done.mensajes) == 1
        n.ciclo_de_control()
        assert len(n.pub_done.mensajes) == 1
        assert all(m.linear.x == m.linear.y == m.angular.z == 0.
                   for m in n.pub_cmd_vel.mensajes)
    finally:
        cerrar(n)


def test_imagen_calibrada_publica_rumbo_optico_y_no_gasta_reporte(node):
    preparar(node)
    node.pub_observacion = Captura()
    info = CameraInfo()
    info.header.frame_id = 'camara_test'
    info.width = 100
    info.k[0] = 100.
    node.al_info_camara(info)
    tf = TransformStamped()
    tf.header.frame_id = 'map'
    tf.child_frame_id = 'camara_test'
    # Óptico Z hacia map X; óptico Y hacia abajo.
    tf.transform.rotation.x = -.5
    tf.transform.rotation.y = .5
    tf.transform.rotation.z = -.5
    tf.transform.rotation.w = .5
    node.tf_buffer.set_transform_static(tf, 'test')
    msg = Image()
    msg.header.frame_id = 'camara_test'
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.encoding = 'rgb8'
    msg.width = msg.height = 10
    msg.step = 30
    msg.data = bytes([0, 0, 220]*100)
    node.al_imagen(msg)
    assert len(node.pub_observacion.mensajes) == 1
    data = json.loads(node.pub_observacion.mensajes[0].data)
    assert abs(data['pose_camara'][2]) < 1e-9
    assert math.isclose(data['campo_horizontal_rad'], 2*math.atan(.5))
    assert not data['region_roja']
    assert not node.pub_reportes.mensajes


def test_imagen_sin_tf_no_acredita_vista(node):
    preparar(node)
    node.pub_observacion = Captura()
    info = CameraInfo()
    info.header.frame_id = 'sin_tf'
    info.width = 100
    info.k[0] = 100.
    node.al_info_camara(info)
    msg = Image()
    msg.header.frame_id = 'sin_tf'
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.encoding = 'rgb8'
    msg.width = msg.height = 10
    msg.step = 30
    msg.data = bytes([0, 0, 220]*100)
    node.al_imagen(msg)
    assert not node.pub_observacion.mensajes


def test_done_detiene_tambien_percepcion_sin_gastar_mas_intentos(node):
    from std_msgs.msg import Empty
    from test_percepcion_nodo import nube
    preparar(node)
    msg = nube(node)
    node.al_done(Empty())
    node.al_nube(msg)
    assert not node.disponible()
    assert not node.pub_reportes.mensajes
    assert not node.registro.victimas


def test_reloj_real_es_monotono_y_simulacion_usa_clock():
    n = nodo(modo_navegacion='mision')
    try:
        assert abs(n.reloj_mision()-time.monotonic()) < .05
    finally:
        cerrar(n)
    n = nodo(modo_navegacion='mision', use_sim_time=True)
    try:
        assert n.reloj_mision() == n.get_clock().now().nanoseconds/1e9
    finally:
        cerrar(n)
