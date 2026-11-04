#!/usr/bin/env python3
"""Mide el error de la localización contra la pose real del simulador, una vez por segundo.

    python3 scripts/medir_localizacion.py --salida localizacion.jsonl

Compara TF map -> base_footprint (lo que cree nuestro software) con /ground_truth/odom (dónde
está de verdad el robot en Gazebo). Cada línea: {"t", "real": [x, y, yaw], "estimada": [...],
"error_m", "error_deg"}.

Es una herramienta de prueba, NO parte del software del equipo: el ground truth solo existe en
el simulador y nuestro código (src/) no lo puede usar nunca (§7.1). Por eso vive en scripts/.
Termina con Ctrl+C o SIGTERM (probar.sh lo corta al final de la corrida).
"""
import argparse
import json
import math
import signal
import sys

import rclpy
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener


def yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class MedidorLocalizacion(Node):
    def __init__(self, salida, intervalo):
        super().__init__('medidor_localizacion',
                         parameter_overrides=[Parameter('use_sim_time', value=True)])
        self.archivo = open(salida, 'w')
        self.real = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(Odometry, '/ground_truth/odom', self.al_real,
                                 qos_profile_sensor_data)
        self.create_timer(intervalo, self.medir)

    def al_real(self, msg):
        self.real = msg.pose.pose

    def medir(self):
        if self.real is None:
            return
        try:
            tf = self.tf_buffer.lookup_transform('map', 'base_footprint', Time())
        except TransformException:
            return  # AMCL todavía no se localizó
        r, e = self.real, tf.transform
        real = (r.position.x, r.position.y, yaw(r.orientation))
        estimada = (e.translation.x, e.translation.y, yaw(e.rotation))
        error_m = math.hypot(estimada[0] - real[0], estimada[1] - real[1])
        error_deg = math.degrees(math.remainder(estimada[2] - real[2], 2 * math.pi))
        self.archivo.write(json.dumps({
            't': round(self.get_clock().now().nanoseconds * 1e-9, 1),
            'real': [round(v, 3) for v in real],
            'estimada': [round(v, 3) for v in estimada],
            'error_m': round(error_m, 3),
            'error_deg': round(error_deg, 2),
        }) + '\n')
        self.archivo.flush()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--salida', required=True)
    ap.add_argument('--intervalo', type=float, default=1.0, help='segundos de simulación')
    args = ap.parse_args()

    def al_terminar(*_):
        raise KeyboardInterrupt  # SIGTERM cierra igual que Ctrl+C

    signal.signal(signal.SIGTERM, al_terminar)
    rclpy.init()
    nodo = MedidorLocalizacion(args.salida, args.intervalo)
    try:
        rclpy.spin(nodo)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        nodo.archivo.close()
        nodo.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
