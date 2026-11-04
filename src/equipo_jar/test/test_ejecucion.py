"""ejecucion.correr: una excepción en un callback no termina el nodo (el reglamento no deja
reiniciar el software durante la corrida, §11.4)."""
import threading

import rclpy
from rclpy.node import Node

from equipo_jar.ejecucion import correr


def apagar():
    """Apaga desde otro hilo, como la señal de apagado de la corrida. En Humble, apagar adentro de
    un callback se cuelga: espera al executor global, que está corriendo ese mismo callback."""
    threading.Thread(target=rclpy.try_shutdown, daemon=True).start()


def test_una_excepcion_en_un_callback_no_termina_el_nodo():
    rclpy.init()
    nodo = Node('prueba_ejecucion')
    llamadas = []

    def falla_la_primera_vez():
        llamadas.append(1)
        if len(llamadas) == 1:
            raise RuntimeError('falla una vez')
        if len(llamadas) == 2:
            apagar()  # la segunda vez anda: termina la prueba

    nodo.create_timer(0.01, falla_la_primera_vez)
    nodo.create_timer(5.0, apagar)  # por si correr() no vuelve a girar: que no cuelgue
    try:
        correr(nodo)  # no lanza la excepción
    finally:
        nodo.destroy_node()
        rclpy.try_shutdown()
    assert len(llamadas) >= 2


def test_el_apagado_termina_correr_sin_error():
    rclpy.init()
    nodo = Node('prueba_apagado')
    nodo.create_timer(0.01, apagar)
    try:
        correr(nodo)  # vuelve solo, sin lanzar nada
    finally:
        nodo.destroy_node()
        rclpy.try_shutdown()
