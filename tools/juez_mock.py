#!/usr/bin/env python3
"""Juez simulado, mínimo. NO es el juez oficial.

Sirve para dos cosas:
- que el CI compruebe lo que pide §9.2 (el launch espera el mapa, los reportes
  tienen el formato oficial);
- que los equipos prueben en su máquina el mismo protocolo de la corrida.

Hace lo que hace el juez en §5, sin puntuar:
1. espera `--retardo` segundos (el "launch del equipo ya está corriendo y espera");
2. publica el mapa en /map (reliable + transient local) y arranca el cronómetro;
3. escucha /victimas, /done y /cmd_vel;
4. termina con el primer /done, o a los `--duracion` segundos del mapa.

Falla (exit 1) si el equipo, ANTES de que exista el mapa:
- movió el robot (cmd_vel distinto de cero) o publicó reportes;
o si en cualquier momento publicó un reporte mal formado (§7.2).

No sabe dónde están las víctimas ni puntúa: eso lo hace el juez oficial con el
mapa de competencia, que no está en este repositorio.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import rclpy
import yaml
from geometry_msgs.msg import PointStamped, Twist
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
try:
    from rclpy.event_handler import SubscriptionEventCallbacks
except ImportError:      # ROS 2 Humble
    from rclpy.qos_event import SubscriptionEventCallbacks
from std_msgs.msg import Empty

QOS_MAPA = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)
UMBRAL_MOVIMIENTO = 1e-3


def sanear_json(obj):
    """Reemplaza flotantes no finitos (NaN/inf) por None para que el JSON sea válido.

    Un reporte mal formado puede traer x/y no finitos (§7.2): json.dumps los
    escribiría como NaN, que no es JSON válido y rompe JSON.parse en Pages.
    """
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: sanear_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanear_json(v) for v in obj]
    return obj


def leer_pgm(ruta):
    """Lee un PGM binario (P5). Devuelve (ancho, alto, bytes de píxeles)."""
    datos = Path(ruta).read_bytes()
    partes, pos = [], 0
    while len(partes) < 4:  # magic, ancho, alto, maxval
        while datos[pos:pos + 1].isspace():
            pos += 1
        if datos[pos:pos + 1] == b'#':
            pos = datos.index(b'\n', pos)
            continue
        ini = pos
        while not datos[pos:pos + 1].isspace():
            pos += 1
        partes.append(datos[ini:pos])
    pos += 1  # un único whitespace tras maxval
    if partes[0] != b'P5':
        raise ValueError(f'{ruta}: solo se soporta PGM binario (P5)')
    ancho, alto, maxval = int(partes[1]), int(partes[2]), int(partes[3])
    if maxval > 255:
        raise ValueError(f'{ruta}: solo se soporta 8 bits')
    return ancho, alto, datos[pos:pos + ancho * alto]


def cargar_mapa(ruta_yaml):
    """Convierte un mapa de map_server (yaml + pgm) en OccupancyGrid."""
    ruta_yaml = Path(ruta_yaml)
    meta = yaml.safe_load(ruta_yaml.read_text())
    ancho, alto, px = leer_pgm(ruta_yaml.parent / meta['image'])
    negate = int(meta.get('negate', 0))
    occ_t = float(meta.get('occupied_thresh', 0.65))
    libre_t = float(meta.get('free_thresh', 0.25))

    celdas = []
    for fila in range(alto - 1, -1, -1):  # la imagen tiene y hacia abajo, el mapa hacia arriba
        for v in px[fila * ancho:(fila + 1) * ancho]:
            p = (v / 255.0) if negate else ((255 - v) / 255.0)
            celdas.append(100 if p > occ_t else 0 if p < libre_t else -1)

    mapa = OccupancyGrid()
    mapa.header.frame_id = 'map'
    mapa.info.resolution = float(meta['resolution'])
    mapa.info.width, mapa.info.height = ancho, alto
    mapa.info.origin.position.x = float(meta['origin'][0])
    mapa.info.origin.position.y = float(meta['origin'][1])
    mapa.info.origin.orientation.w = 1.0
    mapa.data = celdas
    return mapa


def mapa_sintetico():
    """Un cuarto vacío de 5 x 5 m con paredes. Sirve cuando no hay simulador instalado."""
    n = 100
    mapa = OccupancyGrid()
    mapa.header.frame_id = 'map'
    mapa.info.resolution = 0.05
    mapa.info.width = mapa.info.height = n
    mapa.info.origin.position.x = mapa.info.origin.position.y = -2.5
    mapa.info.origin.orientation.w = 1.0
    mapa.data = [100 if f in (0, n - 1) or c in (0, n - 1) else 0
                 for f in range(n) for c in range(n)]
    return mapa


class JuezMock(Node):
    def __init__(self, args):
        super().__init__('juez_mock')
        self.args = args
        self.mapa = cargar_mapa(args.mapa) if args.mapa else mapa_sintetico()
        self.t_inicio = time.monotonic()
        self.t_mapa = None
        self.listo_para_contar = args.esperar_topic is None
        self.terminado = None  # motivo

        self.previo = {'cmd_vel_no_nulo': 0, 'reportes': 0}
        self.reportes = []
        self.mal_formados = []
        self.done = False

        self.pub_mapa = self.create_publisher(OccupancyGrid, '/map', QOS_MAPA)
        self.qos_incompatible = []
        self.create_subscription(Twist, '/cmd_vel', self.al_cmd_vel, 10,
                                 event_callbacks=self._aviso_qos('/cmd_vel'))
        # /victimas y /done: cola profunda. Son pocos mensajes (el reglamento cuenta a lo sumo N reportes),
        # pero si el juez está ocupado no se puede perder el más viejo: los que cuentan son los primeros.
        self.create_subscription(PointStamped, '/victimas', self.al_reporte, 1000,
                                 event_callbacks=self._aviso_qos('/victimas'))
        self.create_subscription(Empty, '/done', self.al_done, 1000,
                                 event_callbacks=self._aviso_qos('/done'))
        if args.esperar_topic:
            self.create_subscription(
                LaserScan, args.esperar_topic, self.al_sensor,
                rclpy.qos.qos_profile_sensor_data)
        self.create_timer(0.1, self.ciclo)

    def _aviso_qos(self, topic):
        def cb(_info):
            if topic not in self.qos_incompatible:
                self.qos_incompatible.append(topic)
                self.get_logger().error(
                    f'Un publicador de {topic} usa una QoS incompatible (best effort): el juez NO '
                    'recibe sus mensajes. Tiene que ser reliable (docs/QOS.md).')
                self.al_qos_incompatible(topic)
        return SubscriptionEventCallbacks(incompatible_qos=cb)

    def al_qos_incompatible(self, topic):
        """Gancho para las subclases (el juez completo lo anota en el registro)."""

    def al_sensor(self, _):
        if not self.listo_para_contar:
            self.listo_para_contar = True
            self.t_inicio = time.monotonic()
            self.get_logger().info('Robot listo: empieza la espera previa al mapa')

    def al_cmd_vel(self, msg):
        if self.t_mapa is None:
            mueve = any(abs(v) > UMBRAL_MOVIMIENTO for v in (
                msg.linear.x, msg.linear.y, msg.linear.z,
                msg.angular.x, msg.angular.y, msg.angular.z))
            self.previo['cmd_vel_no_nulo'] += int(mueve)

    def al_reporte(self, msg):
        t = None if self.t_mapa is None else round(time.monotonic() - self.t_mapa, 2)
        if t is None:
            self.previo['reportes'] += 1
        finito = math.isfinite(msg.point.x) and math.isfinite(msg.point.y)
        if msg.header.frame_id != 'map' or not finito:
            self.mal_formados.append({'t': t, 'frame_id': msg.header.frame_id})
        self.reportes.append({'t': t, 'x': msg.point.x, 'y': msg.point.y})

    def al_done(self, _):
        if self.t_mapa is not None and not self.done:
            self.done = True
            self.get_logger().info('/done recibido')

    def ciclo(self):
        ahora = time.monotonic()
        if self.t_mapa is None:
            if self.listo_para_contar and ahora - self.t_inicio >= self.args.retardo:
                self.mapa.header.stamp = self.get_clock().now().to_msg()
                self.pub_mapa.publish(self.mapa)
                self.t_mapa = ahora
                self.get_logger().info('Mapa publicado en /map: empieza la corrida')
            return
        if self.done:
            self.terminado = 'done'
        elif ahora - self.t_mapa >= self.args.duracion:
            self.terminado = 'tiempo'

    def resultado(self):
        violaciones = []
        if self.previo['cmd_vel_no_nulo']:
            violaciones.append(
                f"movió el robot antes del mapa ({self.previo['cmd_vel_no_nulo']} mensajes "
                'en /cmd_vel distintos de cero)')
        if self.previo['reportes']:
            violaciones.append(f"publicó {self.previo['reportes']} reporte(s) antes del mapa")
        for topic in self.qos_incompatible:
            violaciones.append(f'QoS incompatible en {topic}: el juez no recibe sus mensajes')
        if self.mal_formados:
            violaciones.append(f'{len(self.mal_formados)} reporte(s) mal formado(s) (§7.2)')
        return {
            'termino_por': self.terminado,
            'mapa_publicado': self.t_mapa is not None,
            'reportes': self.reportes,
            'reportes_mal_formados': self.mal_formados,
            'antes_del_mapa': self.previo,
            'violaciones': violaciones,
        }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--mapa', help='yaml de map_server (por defecto, un cuarto vacío sintético)')
    ap.add_argument('--retardo', type=float, default=8.0, help='segundos de espera antes de publicar /map')
    ap.add_argument('--duracion', type=float, default=60.0, help='segundos de corrida desde el mapa')
    ap.add_argument('--esperar-topic', help='contar --retardo recién cuando llegue un mensaje de este topic (p. ej. /scan)')
    ap.add_argument('--salida', help='archivo JSON con el resultado')
    args = ap.parse_args()

    rclpy.init()
    juez = JuezMock(args)
    try:
        while rclpy.ok() and juez.terminado is None:
            rclpy.spin_once(juez, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    res = juez.resultado()
    res = sanear_json(res)
    if args.salida:
        Path(args.salida).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    juez.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    for v in res['violaciones']:
        print(f'::error::juez_mock: {v}')
    return 1 if res['violaciones'] or not res['mapa_publicado'] else 0


if __name__ == '__main__':
    sys.exit(main())
