"""Todos los nodos del equipo en un solo proceso.

Cada nodo sigue siendo un nodo de ROS aparte (mismo nombre, mismos topics y parámetros), pero
corren todos en el mismo proceso y el mismo executor. Ahorra CPU de dos formas:

- Un proceso Python por nodo cuesta memoria, un participante DDS y su tráfico de descubrimiento.
- Con use_sim_time (simulador), cada nodo procesa por su cuenta cada mensaje de /clock, que
  Gazebo publica a cientos de Hz: unos 20 % de CPU por nodo sin hacer nada. Acá hay una sola
  suscripción a /clock que mueve los relojes de todos los nodos.

Un error en el callback de un nodo se loguea y no tumba a los demás.

    principal --evasor true|false   (el resto llega como --ros-args, igual que a cualquier nodo)
"""
import argparse
import sys
import traceback

import rclpy
from geometry_msgs.msg import Twist
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.parameter import Parameter
from rclpy.time_source import TimeSource

from equipo_jar.campo_verosimilitud import CampoVerosimilitud
from equipo_jar.detector_obstaculos import DetectorObstaculos
from equipo_jar.detector_scan import DetectorScanColor
from equipo_jar.evasor import Evasor
from equipo_jar.gestor_victimas import GestorVictimas
from equipo_jar.localizador import Localizador
from equipo_jar.nodo_equipo import NodoEquipo


def compartir_reloj(nodos):
    """Con use_sim_time, deja una única suscripción a /clock (en el primer nodo) que mueve
    los relojes de todos. Devuelve el TimeSource compartido, o None con reloj real."""
    if not nodos[0].get_parameter('use_sim_time').value:
        return None
    # Apagar el TimeSource propio de cada nodo destruye su suscripción a /clock...
    for nodo in nodos:
        nodo.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, False)])
    # ...y este los reemplaza a todos.
    reloj = TimeSource(node=nodos[0])
    reloj.ros_time_is_active = True
    for nodo in nodos:
        reloj.attach_clock(nodo.get_clock())
    return reloj


def main():
    argumentos = rclpy.utilities.remove_ros_args(sys.argv)[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument('--evasor', default='true', help='false para teleoperar')
    opciones = ap.parse_args(argumentos)

    rclpy.init()
    nodo_equipo = NodoEquipo()
    nodos = [nodo_equipo, DetectorScanColor(), Localizador(), CampoVerosimilitud(),
             DetectorObstaculos(), GestorVictimas()]
    if opciones.evasor.lower() in ('true', '1'):
        nodos.append(Evasor())
    reloj = compartir_reloj(nodos)  # noqa: F841 (tiene que seguir vivo)

    executor = SingleThreadedExecutor()
    for nodo in nodos:
        executor.add_node(nodo)
    try:
        while rclpy.ok():
            try:
                executor.spin_once(timeout_sec=0.5)
            except (KeyboardInterrupt, ExternalShutdownException):
                raise
            except Exception:  # noqa: BLE001
                # Que un nodo con un error no deje al robot sin el resto del software.
                nodo_equipo.get_logger().error(f'Error en un callback:\n{traceback.format_exc()}')
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            if rclpy.ok():
                nodo_equipo.pub_cmd_vel.publish(Twist())  # dejar el robot frenado
            for nodo in nodos:
                executor.remove_node(nodo)
                nodo.destroy_node()
        except KeyboardInterrupt:
            pass   # un segundo Ctrl-C mientras se cierra: no hay nada más que limpiar
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
