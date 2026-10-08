#!/usr/bin/env python3
"""Prueba de punta a punta del juez, sin simulador: un equipo falso contra el juez real.

    python3 tools/tests/e2e_juez.py            (con ROS 2 sourceado; ~40 s)

Levanta tools/juez.py como proceso aparte, hace de equipo (espera /map, publica reportes y
/done en momentos conocidos) y compara el resultado del juez con lo que dice el reglamento.
"""
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import rclpy
from geometry_msgs.msg import PointStamped, Twist
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from std_msgs.msg import Empty
from std_srvs.srv import SetBool, Trigger

RAIZ = Path(__file__).resolve().parents[2]
JUEZ = RAIZ / 'tools' / 'juez.py'
V = [(1.0, 1.0), (4.0, 1.0)]           # dos víctimas, separadas 3 m
QOS_MAPA = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                      durability=DurabilityPolicy.TRANSIENT_LOCAL)


class Equipo(Node):
    def __init__(self):
        super().__init__('equipo_falso')
        self.mapa = False
        self.create_subscription(OccupancyGrid, '/map', lambda m: setattr(self, 'mapa', True), QOS_MAPA)
        self.pub = self.create_publisher(PointStamped, '/victimas', 10)
        self.done = self.create_publisher(Empty, '/done', 10)
        self.cmd = self.create_publisher(Twist, '/cmd_vel', 10)

    def esperar_mapa(self, segundos=30):
        fin = time.monotonic() + segundos
        while not self.mapa and time.monotonic() < fin:
            rclpy.spin_once(self, timeout_sec=0.1)
        return self.mapa

    def dormir(self, s):
        fin = time.monotonic() + s
        while time.monotonic() < fin:
            rclpy.spin_once(self, timeout_sec=0.05)

    def reportar(self, x, y, frame='map'):
        m = PointStamped()
        m.header.frame_id = frame
        m.point.x, m.point.y = float(x), float(y)
        self.pub.publish(m)
        self.dormir(0.3)


def correr_juez(extra, salida):
    victimas = Path(tempfile.mkdtemp()) / 'victimas.json'
    victimas.write_text(json.dumps(V))
    return subprocess.Popen(
        [sys.executable, str(JUEZ), '--victimas-json', str(victimas), '--salida', str(salida)] + extra,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def llamar(e, servicio, tipo=Trigger, **campos):
    cli = e.create_client(tipo, servicio)
    assert cli.wait_for_service(timeout_sec=10), servicio
    req = tipo.Request()
    for k, v in campos.items():
        setattr(req, k, v)
    fut = cli.call_async(req)
    fin = time.monotonic() + 3
    while not fut.done() and time.monotonic() < fin:
        rclpy.spin_once(e, timeout_sec=0.05)
    assert fut.result().success, fut.result().message


def caso(nombre, extra, accion, esperado, registro=False):
    if '--espera-veredicto' not in extra:
        extra = extra + ['--espera-veredicto', '0']     # los casos sin veredicto no esperan
    salida = Path(tempfile.mkdtemp()) / 'juez.json'
    log = Path(tempfile.mkdtemp()) / 'eventos.jsonl'
    if registro:
        extra = extra + ['--equipo', 'los_osos', '--registro', str(log)]
    juez = correr_juez(extra, salida)
    equipo = Equipo()
    time.sleep(1.0)                      # que el juez levante y el descubrimiento se asiente
    accion(equipo)
    juez.wait(timeout=60)
    res = json.loads(salida.read_text())
    equipo.destroy_node()

    def obtenido(k):
        return res['puntaje'][k] if k in res['puntaje'] else res.get(k)
    ok = all((v(obtenido(k)) if callable(v) else obtenido(k) == v) for k, v in esperado.items())
    if registro:
        eventos = [json.loads(linea)['evento'] for linea in log.read_text().splitlines()]
        ok = ok and {'mapa_publicado', 'reporte', 'done'} <= set(eventos) and res['equipo'] == 'los_osos'
    print(f"{'OK   ' if ok else 'FALLA'} {nombre}: {res['resumen']}  (fin: {res['termino_por']})")
    if not ok:
        print('      esperado:', esperado, '\n      obtenido:', {k: obtenido(k) for k in esperado})
    return ok


def perfecta(e):
    assert e.esperar_mapa()
    e.reportar(1.10, 1.0)               # 0,10 m -> 100
    e.reportar(4.0, 1.20)               # 0,20 m -> 100
    e.done.publish(Empty())


def repetida_y_malformada(e):
    assert e.esperar_mapa()
    e.reportar(1.30, 1.0)               # 0,30 m -> 50
    e.reportar(1.05, 1.0)               # repite V1, 0,05 m -> 100 (gasta el 2.º reporte)
    e.reportar(4.0, 1.0)                # tercer mensaje: ya pasó el límite N=2 -> se ignora
    e.done.publish(Empty())


def malformada(e):
    assert e.esperar_mapa()
    e.reportar(1.0, 1.0, frame='odom')  # gasta un reporte y no puntúa
    e.reportar(4.0, 1.0)                # 100
    e.reportar(1.0, 1.0)                # ignorado: ya van 2 mensajes
    e.done.publish(Empty())


def quieta(e):
    assert e.esperar_mapa()
    e.dormir(6)                         # no publica /odom: el juez la ve inmóvil


def manual(e):
    cli = e.create_client(Trigger, '/juez/iniciar')
    assert cli.wait_for_service(timeout_sec=10)
    e.dormir(3)
    assert not e.mapa, 'el mapa no debía salir antes de largar'
    cli.call_async(Trigger.Request())
    assert e.esperar_mapa(10), 'el mapa no salió al llamar al servicio'
    e.reportar(1.0, 1.0)
    e.done.publish(Empty())


def con_colision_a_mano(e):
    cli = e.create_client(Trigger, '/juez/colision')
    assert cli.wait_for_service(timeout_sec=10)
    assert e.esperar_mapa()
    e.reportar(1.0, 1.0)                 # 100
    for _ in range(2):                   # un juez marca dos colisiones
        fut = cli.call_async(Trigger.Request())
        fin = time.monotonic() + 3
        while not fut.done() and time.monotonic() < fin:
            rclpy.spin_once(e, timeout_sec=0.05)
        assert fut.result().success
    e.done.publish(Empty())


def veredicto(dentro):
    def accion(e):
        cli = e.create_client(SetBool, '/juez/regreso')
        assert cli.wait_for_service(timeout_sec=10)
        assert e.esperar_mapa()
        e.reportar(1.0, 1.0)             # 100
        e.done.publish(Empty())
        e.dormir(1.0)
        req = SetBool.Request()
        req.data = dentro
        fut = cli.call_async(req)
        fin = time.monotonic() + 3
        while not fut.done() and time.monotonic() < fin:
            rclpy.spin_once(e, timeout_sec=0.05)
        assert fut.result().success
    return accion


def con_servicio(servicio):
    def accion(e):
        assert e.esperar_mapa()
        e.reportar(1.0, 1.0)             # 100
        llamar(e, servicio)
        e.dormir(0.5)
    return accion


def mueve_despues_del_done(e):
    assert e.esperar_mapa()
    e.reportar(1.0, 1.0)
    e.done.publish(Empty())
    e.dormir(0.5)
    m = Twist()
    m.linear.x = 0.2
    for _ in range(5):                   # el robot no debería moverse más
        e.cmd.publish(m)
        e.dormir(0.1)


def rafaga(e):
    """5 reportes publicados de golpe, sin pausa: el juez tiene que recibirlos todos y en orden."""
    assert e.esperar_mapa()
    puntos = [(1.0, 1.0), (4.0, 1.0), (9.0, 9.0), (9.0, 9.0), (9.0, 9.0)]   # los 2 primeros puntúan
    for x, y in puntos:
        m = PointStamped()
        m.header.frame_id = 'map'
        m.point.x, m.point.y = x, y
        e.pub.publish(m)                 # sin dormir entre uno y otro
    e.dormir(0.5)
    e.done.publish(Empty())


def publica_victimas_best_effort(e):
    assert e.esperar_mapa()
    pub = e.create_publisher(PointStamped, '/victimas', qos_profile_sensor_data)   # best effort
    e.dormir(1.0)                        # que el juez detecte la incompatibilidad
    m = PointStamped()
    m.header.frame_id = 'map'
    m.point.x, m.point.y = 1.0, 1.0
    for _ in range(3):
        pub.publish(m)
        e.dormir(0.2)
    e.done.publish(Empty())


def caso_auditor():
    """El auditor de interfaces marca la QoS incorrecta del equipo y deja pasar la correcta."""
    from sensor_msgs.msg import LaserScan
    auditor = RAIZ / 'tools' / 'auditor_interfaces.py'
    base = Path(tempfile.mkdtemp()) / 'base.txt'
    subprocess.run([sys.executable, str(auditor), '--capturar', str(base)], check=True,
                   capture_output=True)

    def auditar(nodo_equipo):
        fin = time.monotonic() + 3
        while time.monotonic() < fin:
            rclpy.spin_once(nodo_equipo, timeout_sec=0.05)
        return subprocess.run([sys.executable, str(auditor), '--auditar', str(base)],
                              capture_output=True, text=True)

    malo = Node('equipo_malo')
    malo.create_publisher(PointStamped, '/victimas', qos_profile_sensor_data)    # best effort: mal
    malo.create_subscription(LaserScan, '/scan', lambda m: None, 10)             # reliable: mal
    r = auditar(malo)
    ok_malo = (r.returncode != 0 and '/victimas' in r.stdout and 'best effort' in r.stdout
               and '/scan' in r.stdout and 'reliable' in r.stdout)
    malo.destroy_node()

    subprocess.run([sys.executable, str(auditor), '--capturar', str(base)], check=True,
                   capture_output=True)
    bueno = Node('equipo_bueno')
    bueno.create_publisher(PointStamped, '/victimas', 10)                        # por defecto: bien
    bueno.create_subscription(LaserScan, '/scan', lambda m: None, qos_profile_sensor_data)
    r2 = auditar(bueno)
    ok_bueno = r2.returncode == 0
    bueno.destroy_node()
    ok = ok_malo and ok_bueno
    print(f"{'OK   ' if ok else 'FALLA'} el auditor marca la QoS incorrecta y acepta la correcta")
    if not ok:
        print(r.stdout, r2.stdout)
    return ok


def main():
    os.environ.setdefault('ROS_DOMAIN_ID', '95')
    os.environ.setdefault('ROS_LOCALHOST_ONLY', '1')
    rclpy.init()
    r = [
        caso('corrida perfecta', ['--retardo', '1'], perfecta,
             {'total': 400, 'puntos_victimas': 200, 'bono_rapido': True, 'termino_por': 'done',
              'tiempo_ultimo_reporte_que_puntua_s': lambda t: t is not None and t > 0,
              'tiempo_total_s': lambda t: t is not None and t > 0}, registro=True),
        caso('repite una víctima y se pasa del límite', ['--retardo', '1'], repetida_y_malformada,
             {'total': 100, 'puntos_victimas': 100, 'bono_rapido': False, 'reportes_ignorados': 1}),
        caso('reporte mal formado gasta un reporte', ['--retardo', '1'], malformada,
             {'total': 100, 'puntos_victimas': 100, 'bono_rapido': False, 'reportes_ignorados': 1}),
        caso('robot inmóvil termina la corrida', ['--retardo', '1', '--inmovil', '3'], quieta,
             {'total': 0, 'termino_por': 'inmovil'}),
        caso('colisiones registradas por un juez', ['--retardo', '1'], con_colision_a_mano,
             {'total': 0, 'puntos_victimas': 100, 'penalizacion_colisiones': 100}),
        caso('un juez confirma que el robot volvió a la largada', ['--retardo', '1', '--espera-veredicto', '20'],
             veredicto(True), {'total': 200, 'puntos_victimas': 100, 'bono_regreso': True}),
        caso('un juez dice que el robot no volvió', ['--retardo', '1', '--espera-veredicto', '20'],
             veredicto(False), {'total': 100, 'bono_regreso': False}),
        caso('sin veredicto no hay bono de regreso', ['--retardo', '1', '--espera-veredicto', '2'],
             lambda e: (e.esperar_mapa(), e.reportar(1.0, 1.0), e.done.publish(Empty())),
             {'total': 100, 'bono_regreso': False}),
        caso('un juez marca la salida del laberinto: penaliza y termina', ['--retardo', '1'],
             con_servicio('/juez/salida'), {'total': 50, 'puntos_victimas': 100, 'penalizacion_salida': 50, 'termino_por': 'salida'}),
        caso('un vuelco anula la corrida', ['--retardo', '1'], con_servicio('/juez/vuelco'),
             {'total': 0, 'anulada': True, 'puntos_victimas': 100, 'termino_por': 'vuelco'}),
        caso('el juez principal termina la corrida', ['--retardo', '1'], con_servicio('/juez/terminar'),
             {'total': 100, 'termino_por': 'juez'}),
        caso('el movimiento después del /done queda anotado', ['--retardo', '1', '--espera-veredicto', '3'],
             mueve_despues_del_done, {'total': 100, 'movimiento_despues_de_done': lambda n: n >= 1}),
        caso('ráfaga de 5 reportes seguidos: no se pierde ninguno y se respeta el orden', ['--retardo', '1'],
             rafaga, {'total': 400, 'puntos_victimas': 200, 'reportes_contados': 2,
                      'reportes_ignorados': 3}),
        caso('QoS incompatible en /victimas: el juez lo detecta y lo avisa', ['--retardo', '1'],
             publica_victimas_best_effort,
             {'total': 0, 'violaciones': lambda v: any('QoS incompatible en /victimas' in x for x in v)}),
        caso_auditor(),
        caso('largada manual por servicio', ['--manual'], manual,
             {'total': 100, 'puntos_victimas': 100, 'termino_por': 'done'}),
    ]
    rclpy.shutdown()
    print(f"\n{sum(r)}/{len(r)} casos en verde")
    return 0 if all(r) else 1


if __name__ == '__main__':
    sys.exit(main())
