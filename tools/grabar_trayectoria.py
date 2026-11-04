#!/usr/bin/env python3
"""Graba la trayectoria del robot (/ground_truth/odom) en JSONL, una pose por segundo.

    python3 tools/grabar_trayectoria.py --salida trayectoria.jsonl [--intervalo 1.0]

Termina con SIGINT/SIGTERM (el trap de correr_simulador.sh lo mata al cerrar
la corrida). Cada línea: {"t": s desde el arranque, "x": m, "y": m}.
Solo simulador: usa la pose real de Gazebo, en el marco del mundo (= `map` en los
mundos de práctica), para que el recorrido calce con el mapa y las víctimas.
`/odom` no sirve: arranca en cero donde sea que largue el robot y se desvía.
Solo necesita rclpy (corre donde haya ROS).
"""
import argparse
import json
import sys
import time

import rclpy
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data


class Grabador(Node):
    def __init__(self, salida, intervalo):
        super().__init__('grabador_trayectoria')
        self.archivo = open(salida, 'w')
        self.t0 = time.monotonic()
        self.ultimo = -intervalo
        self.intervalo = intervalo
        self.create_subscription(Odometry, '/ground_truth/odom', self.al_odom,
                                 qos_profile_sensor_data)

    def al_odom(self, m):
        ahora = time.monotonic() - self.t0
        if ahora - self.ultimo >= self.intervalo:
            self.ultimo = ahora
            p = m.pose.pose.position
            self.archivo.write(json.dumps(
                {'t': round(ahora, 1), 'x': round(p.x, 3), 'y': round(p.y, 3)}) + '\n')
            self.archivo.flush()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--salida', required=True)
    ap.add_argument('--intervalo', type=float, default=1.0)
    args = ap.parse_args()

    rclpy.init()
    grabador = Grabador(args.salida, args.intervalo)
    try:
        rclpy.spin(grabador)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    grabador.archivo.close()
    grabador.destroy_node()
    if rclpy.ok():   # con SIGINT/SIGTERM, rclpy ya cerró el contexto
        rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
