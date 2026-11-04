#!/usr/bin/env python3
"""Mide qué tan pegado queda el lidar a las paredes de /map: sirve para evaluar AMCL sin ground truth.

    python3 scripts/medir_ajuste_scan.py --salida ajuste.jsonl

Pensado para reproducir un bag del robot real (que no tiene /ground_truth/odom) con AMCL
corriendo encima. Por cada scan proyecta los puntos al mapa con dos poses y mide la distancia de
cada punto a la pared más cercana:
- `amcl`: TF map -> laser_link, lo que cree nuestro software.
- `odom`: TF odom -> laser_link tomando odom como si fuera map, o sea la odometría sola desde el
  origen (sirve si el robot arrancó en el origen, como en el reglamento §1).
Si la pose es buena, los puntos caen sobre las paredes (distancia ~0).

Cada línea: {"t", "amcl": {"mediana_m", "dentro_10cm"}, "odom": {...}}. Al terminar imprime un
resumen. Es una herramienta de prueba, NO parte del software del equipo.
Termina con Ctrl+C o SIGTERM.
"""
import argparse
import json
import math
import signal
import sys

import cv2
import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformException, TransformListener

QOS_MAPA = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                      durability=DurabilityPolicy.TRANSIENT_LOCAL)


def yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class MedidorAjuste(Node):
    def __init__(self, salida, frame_mapa, frame_odom):
        super().__init__('medidor_ajuste_scan',
                         parameter_overrides=[Parameter('use_sim_time', value=True)])
        self.archivo = open(salida, 'w')
        self.frames = {'amcl': frame_mapa, 'odom': frame_odom}
        self.distancias = None  # distancia (m) de cada celda a la pared más cercana
        self.info = None
        self.historia = {'amcl': [], 'odom': []}
        self.tf_buffer = Buffer()
        # spin_thread: TF se recibe en su propio hilo, así esperar una transformación no lo bloquea.
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)
        self.pendientes = []  # scans cuya TF todavía no llegó (AMCL publica un poco después)
        self.create_subscription(OccupancyGrid, '/map', self.al_recibir_mapa, QOS_MAPA)
        self.create_subscription(LaserScan, '/scan', self.pendientes.append, qos_profile_sensor_data)
        self.create_timer(0.05, self.procesar_pendientes)

    def al_recibir_mapa(self, mapa):
        grilla = np.array(mapa.data, dtype=np.int16).reshape(mapa.info.height, mapa.info.width)
        libre = np.where(grilla >= 65, 0, 255).astype(np.uint8)  # 0 = pared
        self.distancias = cv2.distanceTransform(libre, cv2.DIST_L2, 5) * mapa.info.resolution
        self.info = mapa.info

    def procesar_pendientes(self):
        """Procesa los scans de hace más de 0,5 s: para entonces ya llegaron sus TF."""
        ahora = self.get_clock().now().nanoseconds * 1e-9
        while self.pendientes:
            s = self.pendientes[0].header.stamp
            if ahora - (s.sec + s.nanosec * 1e-9) < 0.5:
                break
            self.medir(self.pendientes.pop(0))

    def medir(self, scan):
        if self.distancias is None:
            return
        rangos = np.array(scan.ranges)
        angulos = scan.angle_min + np.arange(len(rangos)) * scan.angle_increment
        validos = np.isfinite(rangos) & (rangos > scan.range_min) & (rangos < scan.range_max)
        xs, ys = rangos[validos] * np.cos(angulos[validos]), rangos[validos] * np.sin(angulos[validos])
        fila = {'t': scan.header.stamp.sec + scan.header.stamp.nanosec * 1e-9}
        for nombre, frame in self.frames.items():
            try:
                tf = self.tf_buffer.lookup_transform(
                    frame, scan.header.frame_id, scan.header.stamp)
            except TransformException:
                continue
            t = tf.transform.translation
            th = yaw(tf.transform.rotation)
            mx = t.x + xs * math.cos(th) - ys * math.sin(th)
            my = t.y + xs * math.sin(th) + ys * math.cos(th)
            o = self.info.origin.position
            col = ((mx - o.x) / self.info.resolution).astype(int)
            row = ((my - o.y) / self.info.resolution).astype(int)
            adentro = (col >= 0) & (col < self.info.width) & (row >= 0) & (row < self.info.height)
            d = np.full(len(mx), 1.0)  # fuera del mapa cuenta como lejos
            d[adentro] = self.distancias[row[adentro], col[adentro]]
            fila[nombre] = {'mediana_m': round(float(np.median(d)), 3),
                            'dentro_10cm': round(float(np.mean(d < 0.10)), 3),
                            'pose': [round(t.x, 3), round(t.y, 3), round(math.degrees(th), 1)]}
            self.historia[nombre].append(fila[nombre])
        if len(fila) > 1:
            self.archivo.write(json.dumps(fila) + '\n')
            self.archivo.flush()

    def resumen(self):
        for nombre, h in self.historia.items():
            if not h:
                print(f'{nombre}: sin datos')
                continue
            med = np.array([f['mediana_m'] for f in h])
            dentro = np.array([f['dentro_10cm'] for f in h])
            print(f'{nombre}: {len(h)} scans | distancia mediana a la pared: '
                  f'p50 {np.median(med):.3f} m, p90 {np.percentile(med, 90):.3f} m, '
                  f'peor {med.max():.3f} m | puntos a <10 cm de una pared: '
                  f'{100 * dentro.mean():.0f}% (peor scan {100 * dentro.min():.0f}%)')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--salida', default='ajuste_scan.jsonl')
    parser.add_argument('--frame-mapa', default='map')
    parser.add_argument('--frame-odom', default='odom')
    args = parser.parse_args()
    rclpy.init()
    nodo = MedidorAjuste(args.salida, args.frame_mapa, args.frame_odom)
    signal.signal(signal.SIGTERM, lambda *_: rclpy.try_shutdown())
    try:
        rclpy.spin(nodo)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        nodo.resumen()
        nodo.archivo.close()
        nodo.destroy_node()
        rclpy.try_shutdown()
    sys.exit(0)


if __name__ == '__main__':
    main()
