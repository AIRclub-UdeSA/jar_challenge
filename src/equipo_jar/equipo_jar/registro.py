"""Run log for the organizers: every node's log (/rosout) in one text file.

The organizers keep `log.txt` after each run and share it with all teams, so we can tune
params between runs. Rules (their message of 2026-11-01): write only `logs_jar/log.txt` at
the root of the workspace where the delivery is built, or in `$JAR_LOG_DIR` if it is set;
up to 25 MB; a failing log must never break the software. Nothing here is secret: the
code and params are already public in the fork.
"""
import math
import os
from pathlib import Path
import time

from geometry_msgs.msg import PointStamped, PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid
from rcl_interfaces.msg import Log
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Empty

from equipo_jar.comun import QOS_MAPA, correr, parametros_en_vivo

NIVELES = {10: 'DEBUG', 20: 'INFO', 30: 'WARN', 40: 'ERROR', 50: 'FATAL'}


def carpeta_de_logs(entorno=os.environ, prefijo=None):
    """`$JAR_LOG_DIR`, or `<workspace>/logs_jar`; None if neither can be found.

    The workspace is the folder that holds `install/`, where colcon put this package
    (`<ws>/install/equipo_jar` or, merged, `<ws>/install`).
    """
    if entorno.get('JAR_LOG_DIR'):
        return Path(entorno['JAR_LOG_DIR'])
    if prefijo is None:
        from ament_index_python.packages import get_package_prefix
        prefijo = get_package_prefix('equipo_jar')
    ruta = Path(prefijo).resolve()
    for carpeta in (ruta, *ruta.parents):
        if carpeta.name == 'install':
            return carpeta.parent / 'logs_jar'
    return None


class Registro(Node):
    def __init__(self):
        super().__init__('registro')
        self.p = parametros_en_vivo(self, {
            'max_bytes': 20_000_000,       # stop writing here (organizers keep up to 25 MB)
            'nivel_minimo': 20,            # 20 INFO, 30 WARN
            'repetidas_s': 5.0,            # same node and text within this: counted, not written
            'pose_cada_s': 1.0,            # AMCL pose line every this many seconds
        })
        self.archivo = None
        self.escritos = 0
        self.t0 = None
        self.ultima = {}                   # (node, text) -> [time written, skipped count]
        self.ultima_pose = 0.0
        self.abrir()
        self.create_subscription(Log, '/rosout', self.al_log, QoSProfile(
            depth=1000, reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(OccupancyGrid, '/map', self.al_mapa, QOS_MAPA)
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.al_pose, QOS_MAPA)
        self.create_subscription(PointStamped, '/victimas', self.al_reporte, 10)
        self.create_subscription(Empty, '/done', self.al_done, 10)
        self.create_timer(1.0, self.vaciar)

    def abrir(self):
        try:
            carpeta = carpeta_de_logs()
            if carpeta is None:
                self.get_logger().warning('Run log off: no JAR_LOG_DIR and no workspace found')
                return
            carpeta.mkdir(parents=True, exist_ok=True)
            ruta = carpeta / 'log.txt'
            # One file for all runs: start over when the last ones already used half of it.
            modo = 'w' if ruta.exists() and ruta.stat().st_size > self.p['max_bytes'] / 2 else 'a'
            self.archivo = open(ruta, modo, encoding='utf-8', errors='replace')
            self.escritos = self.archivo.tell()
            self.escribir(f'===== run start {time.strftime("%Y-%m-%d %H:%M:%S")} =====')
            self.volcar_params()
            self.get_logger().info(f'Run log in {ruta}')
        except Exception as error:  # never break the run because of the log
            self.cerrar(f'Run log off: {error}')

    def volcar_params(self):
        """The params files in use: between runs only params change."""
        try:
            from ament_index_python.packages import get_package_share_directory
            carpeta = Path(get_package_share_directory('equipo_jar')) / 'params'
            for ruta in sorted(carpeta.glob('*.yaml')):
                self.escribir(f'----- params/{ruta.name} -----')
                for linea in ruta.read_text(encoding='utf-8').splitlines():
                    self.escribir(linea)
            self.escribir('----- end of params -----')
        except Exception as error:
            self.escribir(f'params not dumped: {error}')

    def segundos(self):
        if self.t0 is None:
            return '   -   '
        return f'{self.get_clock().now().nanoseconds / 1e9 - self.t0:7.1f}'

    def escribir(self, texto):
        if self.archivo is None:
            return
        linea = f'{time.strftime("%H:%M:%S")} {self.segundos()} {texto}\n'
        try:
            if self.escritos + len(linea) > self.p['max_bytes']:
                self.archivo.write('===== log full: stopped writing =====\n')
                self.cerrar()
                return
            self.archivo.write(linea)
            self.escritos += len(linea.encode('utf-8', errors='replace'))
        except Exception as error:
            self.cerrar(f'Run log off: {error}')

    def cerrar(self, motivo=None):
        archivo, self.archivo = self.archivo, None
        if motivo:
            self.get_logger().warning(motivo)
        try:
            if archivo is not None:
                archivo.close()
        except Exception:
            pass

    def vaciar(self):
        try:
            if self.archivo is not None:
                self.archivo.flush()
        except Exception as error:
            self.cerrar(f'Run log off: {error}')

    def al_log(self, msg):
        if self.archivo is None or msg.level < self.p['nivel_minimo'] or msg.name == 'registro':
            return
        ahora = time.monotonic()
        clave = (msg.name, msg.msg)
        previa = self.ultima.get(clave)
        if previa is not None and ahora - previa[0] < self.p['repetidas_s']:
            previa[1] += 1
            return
        repetidas = f' [+{previa[1]} same]' if previa is not None and previa[1] else ''
        if len(self.ultima) > 5000:        # bound memory over a long run
            self.ultima.clear()
        self.ultima[clave] = [ahora, 0]
        nivel = NIVELES.get(msg.level, str(msg.level))
        self.escribir(f'{nivel:5} [{msg.name}] {msg.msg}{repetidas}')

    def al_mapa(self, msg):
        if self.t0 is None:
            self.t0 = self.get_clock().now().nanoseconds / 1e9
            info = msg.info
            self.escribir(f'MAP {info.width}x{info.height} res {info.resolution:.3f} '
                          f'origin ({info.origin.position.x:.2f}, {info.origin.position.y:.2f})')

    def al_pose(self, msg):
        ahora = time.monotonic()
        if ahora - self.ultima_pose < self.p['pose_cada_s']:
            return
        self.ultima_pose = ahora
        p = msg.pose.pose
        yaw = 2.0 * math.atan2(p.orientation.z, p.orientation.w)
        cov = msg.pose.covariance
        self.escribir(f'POSE {p.position.x:.3f} {p.position.y:.3f} {yaw:.3f} '
                      f'sd_xy {math.sqrt(max(cov[0], 0.0)):.3f} {math.sqrt(max(cov[7], 0.0)):.3f}')

    def al_reporte(self, msg):
        self.escribir(f'REPORT {msg.point.x:.3f} {msg.point.y:.3f} frame {msg.header.frame_id}')

    def al_done(self, _msg):
        self.escribir('DONE')
        self.vaciar()

    def destroy_node(self):
        self.escribir('===== run end =====')
        self.cerrar()
        super().destroy_node()


def main(args=None):
    correr(Registro, args)
