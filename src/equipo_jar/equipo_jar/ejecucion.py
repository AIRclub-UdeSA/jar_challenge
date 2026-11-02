"""Cómo corre cada nodo: que una excepción en un callback no lo mate.

El reglamento no deja reiniciar el software durante la corrida (§11.4): un nodo que se muere queda
muerto hasta el final. Sin misión el robot queda quieto (a los 60 s termina la corrida); sin
percepción no se encuentra ninguna caja más.

En rclpy, una excepción en cualquier callback (suscripción, timer, servicio o el done callback de
un future) sale por rclpy.spin(). Acá se loguea con su traceback y se vuelve a girar con el mismo
nodo, que conserva su estado. Cubre también los callbacks que se agreguen después. Lo que no
recupera es el mensaje que estaba procesando ese callback: por eso los que importan (el /map, que
llega una sola vez, o la imagen) se manejan además en el propio nodo.
"""
import traceback

import rclpy
from rclpy.executors import ExternalShutdownException


def correr(nodo):
    """rclpy.spin(nodo) hasta el apagado, sin que una excepción en un callback lo termine.

    Durante el apagado, rclpy corta el spin con ExternalShutdownException o con RCLError: eso no
    es un error de la corrida y se sale en silencio. Ctrl+C también sale.
    """
    while rclpy.ok():
        try:
            rclpy.spin(nodo)
        except (KeyboardInterrupt, ExternalShutdownException):
            return
        except Exception:  # a propósito: cualquier error
            if not rclpy.ok():
                return
            nodo.get_logger().error(
                f'Error en un callback, el nodo sigue:\n{traceback.format_exc()}',
                throttle_duration_sec=5.0)


def main_de(clase_nodo):
    """El main() de un nodo: arranca rclpy, lo corre con correr() y apaga ordenado."""
    rclpy.init()
    nodo = clase_nodo()
    try:
        correr(nodo)
    finally:
        nodo.destroy_node()
        rclpy.try_shutdown()
