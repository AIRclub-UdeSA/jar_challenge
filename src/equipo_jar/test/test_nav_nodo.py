"""Integración de la navegación en nodo_equipo: árbitro, prueba en seco y mapa único."""
import json
import math
import time

import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.parameter import Parameter

from equipo_jar.localizacion_activa import Observacion
from equipo_jar.navegacion.tipos import Pose2D
from equipo_jar.nodo_equipo import NodoEquipo


class Captura:
    def __init__(self):
        self.mensajes = []

    def publish(self, mensaje):
        self.mensajes.append(mensaje)


class M1Lista:
    """M1 ya localizado: no gira y la localización es utilizable."""
    estado, motivo, giros, recuperaciones, utilizable = 'LISTA', '', 0, 0, True

    def paso(self, *args):
        return 0.0

    def fallar(self, motivo):
        pass


def mapa_libre(ancho=60, alto=40):
    """3 x 2 m libre; afuera cuenta como ocupado."""
    mapa = OccupancyGrid()
    mapa.info.resolution = 0.05
    mapa.info.width, mapa.info.height = ancho, alto
    mapa.info.origin.orientation.w = 1.0
    mapa.data = [0] * (ancho * alto)
    return mapa


def anillo(radio=0.9):
    return [(radio * math.cos(i * math.pi / 30), radio * math.sin(i * math.pi / 30))
            for i in range(60)]


def nodo(**parametros):
    rclpy.init()
    node = NodoEquipo(parameter_overrides=[Parameter(k, value=v)
                                           for k, v in parametros.items()])
    node.pub_cmd_vel, node.pub_estado = Captura(), Captura()
    node.pub_navegacion, node.pub_propuesto, node.pub_ruta = Captura(), Captura(), Captura()
    return node


def preparar_lista(node, pose=(1.5, 1.0, 0.0)):
    node.control = M1Lista()
    observacion = Observacion(pose[2], pose, 1, 0.02, 0.02, 1.0, 0.9, True)

    def observar(ahora):
        node.puntos_base = anillo()
        node.puntos_mapa = [(pose[0] + x, pose[1] + y) for x, y in anillo()]
        return observacion

    node.observacion = observar
    # Estos tests aíslan arbitraje y misión. La compensación TF de sensores
    # se verifica con estampas distintas en test_nav_temporal.py.
    node.datos_navegacion = lambda ahora, obs: (
        (Pose2D(*pose), anillo(), [(pose[0]+x,pose[1]+y) for x,y in anillo()])
        if obs is not None else None)


def esperar_comando(node,publicador):
    limite=time.monotonic()+2.
    while time.monotonic()<limite:
        node.ciclo_de_control()
        if publicador.mensajes and publicador.mensajes[-1].linear.x>0.:return
        time.sleep(.01)

def cerrar(node):
    node.destroy_node()
    rclpy.shutdown()


def test_no_mueve_antes_de_map_con_navegacion_activa():
    node = nodo(modo_navegacion='meta_externa')
    try:
        preparar_lista(node)
        node.ciclo_de_control()
        assert not node.pub_cmd_vel.mensajes
        assert node.navegacion is None
    finally:
        cerrar(node)


def test_inactivo_no_mueve_tras_lista():
    node = nodo()
    try:
        node.al_recibir_mapa(mapa_libre())
        preparar_lista(node)
        for _ in range(20):
            node.ciclo_de_control()
        assert node.pub_cmd_vel.mensajes
        assert all(m.linear.x == m.linear.y == m.angular.z == 0.0
                   for m in node.pub_cmd_vel.mensajes)
        assert node.navegacion is None
    finally:
        cerrar(node)


def test_meta_externa_mueve_hacia_la_meta():
    node = nodo(modo_navegacion='meta_externa')
    try:
        node.al_recibir_mapa(mapa_libre())
        preparar_lista(node)
        node.ciclo_de_control()
        node.navegacion.fijar_meta((2.3, 1.0))
        esperar_comando(node,node.pub_cmd_vel)
        assert node.pub_cmd_vel.mensajes[-1].linear.x > 0.0
        estado = json.loads(node.pub_navegacion.mensajes[-1].data)
        assert estado['estado'] == 'YENDO'
        assert 'ciclo_p95_ms' in estado
        assert node.pub_ruta.mensajes and len(node.pub_ruta.mensajes[-1].poses) >= 2
    finally:
        cerrar(node)


def test_prueba_en_seco_publica_cero_y_propuesto():
    node = nodo(modo_navegacion='meta_externa', prueba_en_seco=True)
    try:
        node.al_recibir_mapa(mapa_libre())
        preparar_lista(node)
        node.ciclo_de_control()
        node.navegacion.fijar_meta((2.3, 1.0))
        esperar_comando(node,node.pub_propuesto)
        assert all(m.linear.x == m.linear.y == m.angular.z == 0.0
                   for m in node.pub_cmd_vel.mensajes)
        assert node.pub_propuesto.mensajes[-1].linear.x > 0.0
    finally:
        cerrar(node)


def test_mapa_repetido_no_reconstruye_grilla():
    node = nodo(modo_navegacion='meta_externa')
    try:
        node.al_recibir_mapa(mapa_libre())
        preparar_lista(node)
        node.ciclo_de_control()
        primera = node.navegacion
        node.al_recibir_mapa(mapa_libre(80, 80))
        node.ciclo_de_control()
        assert node.navegacion is primera
        # La navegación puede refinar celdas; sigue conservando el primer
        # mapa de 3 m, no el mapa posterior de 4 m.
        assert node.navegacion.grilla.ancho * node.navegacion.grilla.resolucion == 3.
    finally:
        cerrar(node)


def test_ida_vuelta_prueba_elige_meta_desde_el_mapa():
    node = nodo(modo_navegacion='ida_vuelta_prueba', meta_distancia_min_m=0.5,
                meta_distancia_max_m=1.2)
    try:
        node.al_recibir_mapa(mapa_libre())
        preparar_lista(node)
        for _ in range(3):
            node.ciclo_de_control()
        assert node.navegacion.meta is not None
        assert node.navegacion.estado == 'YENDO'
    finally:
        cerrar(node)


def test_modo_invalido_falla_al_iniciar():
    try:
        nodo(modo_navegacion='otra_cosa')
    except ValueError:
        pass
    else:
        raise AssertionError('Un modo desconocido debe rechazarse')
    finally:
        rclpy.shutdown()


def test_sin_meta_automatica_no_queda_esperando_indefinidamente():
    node=nodo(modo_navegacion='ida_vuelta_prueba')
    try:
        node.al_recibir_mapa(mapa_libre(20,20))
        preparar_lista(node,pose=(.5,.5,0.))
        node.ciclo_de_control()
        assert node.navegacion.estado=='FALLO'
        assert 'meta' in node.navegacion.motivo.lower()
        def no_reintentar(*args):
            raise AssertionError('FALLO es terminal; no volver a elegir meta')
        node.navegacion.cancelar=no_reintentar
        node.ciclo_de_control()
        assert all(m.linear.x==m.linear.y==m.angular.z==0. for m in node.pub_cmd_vel.mensajes)
    finally:cerrar(node)


def test_margen_navegacion_separado_del_giro_de_m1():
    node=nodo(modo_navegacion='meta_externa',nav_margen_m=.075)
    try:
        assert node.valor('margen_giro_m')==.10
        assert node.huella['margin_m']==.075
        node.al_recibir_mapa(mapa_libre());preparar_lista(node)
        node.ciclo_de_control()
        assert node.navegacion.p_costos.margen_m==.075
    finally:cerrar(node)
