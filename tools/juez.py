#!/usr/bin/env python3
"""Juez de práctica del Challenge JAR 2026: publica el mapa, escucha los eventos y suma puntos.

Extiende a juez_mock.py (el protocolo del mapa y la validación de formato) con lo que hace un
juez de verdad (reglamento §5, §7.2, §8, §11):

- Víctimas: se leen de un mundo de práctica (`--mundo`) o de un JSON (`--victimas-json`).
- Puntaje en vivo: cada reporte se puntúa al llegar y se publica en /juez/puntaje (JSON).
- Fin de corrida: /done, 10 minutos desde el mapa, robot inmóvil 60 s (velocidad de /odom) o
  salida del laberinto.
- Disparador del mapa: automático (`--retardo`) o a mano (`--manual`): se llama al servicio
  /juez/iniciar (std_srvs/Trigger) cuando el juez principal da la largada. Una colisión que ve
  un juez se registra con el servicio /juez/colision.
- Colisiones, salida del laberinto, vuelco y regreso a la largada los decide una persona, en el
  simulador y en el evento: el software del juez no puede saber dónde está el robot real. Con
  `--ground-truth` (solo simulador) el regreso se puede medir con la pose real, para practicar.
  Después del /done, un juez mira si el robot quedó dentro del círculo de la largada y lo
  registra con /juez/regreso (std_srvs/SetBool: true = adentro). El juez espera ese veredicto
  hasta `--espera-veredicto` segundos antes de cerrar la corrida; sin veredicto, no hay bono.
  Lo mismo con la salida del laberinto (/juez/salida), el vuelco (/juez/vuelco, anula la corrida)
  y cualquier otro motivo por el que el juez principal la termina (/juez/terminar).
- Registro: cada evento se guarda en un archivo (`--registro`) apenas ocurre, así que si el juez se
  cae a mitad de una corrida no se pierde lo anterior.
- Después del /done el robot no debería volver a moverse: el juez lo anota, no lo sanciona.

El puntaje sale de puntaje.py. Nada de esto puntúa con datos de la competencia: las posiciones
reales de esa arena viven en el repositorio privado del juez.
"""
import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import rclpy
from nav_msgs.msg import Odometry
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String
from std_srvs.srv import SetBool, Trigger

sys.path.insert(0, str(Path(__file__).resolve().parent))
import puntaje  # noqa: E402
from juez_mock import UMBRAL_MOVIMIENTO, JuezMock, sanear_json  # noqa: E402
from mundos import resolver_mundo, victimas_de_mundo  # noqa: E402


class Juez(JuezMock):
    def __init__(self, args, victimas):
        super().__init__(args)
        self.victimas = victimas
        self.eventos = []
        self.validez = []           # un bool por reporte, en el mismo orden que self.reportes
        self.colisiones = 0
        self.salio = False
        self.pose_largada = None
        self.pose_actual = None
        self.dist_done = None
        self.veredicto = None          # True/False cuando un juez confirma el regreso
        self.t_done = None
        self.t_fin = None              # segundos desde /map hasta que terminó la corrida
        self.anulada = False
        self.cmd_vel_tras_done = 0
        self._mapa_registrado = False
        self._registro = open(args.registro, 'a', buffering=1) if args.registro else None
        self.t_ultimo_movimiento = time.monotonic()
        self.puntaje_en_vivo = None

        self.pub_puntaje = self.create_publisher(String, '/juez/puntaje', 10)
        self.create_subscription(Odometry, '/odom', self.al_odom, qos_profile_sensor_data)
        if args.ground_truth:
            self.create_subscription(Odometry, '/ground_truth/odom', self.al_gt,
                                     qos_profile_sensor_data)
        # Un juez humano registra una colisión a mano (robot físico: no hay ground truth, §8.1).
        # Usar esto O --ground-truth, no los dos: se contarían dos veces.
        self.create_service(Trigger, '/juez/colision', self.al_colision_manual)
        self.create_service(SetBool, '/juez/regreso', self.al_veredicto_regreso)
        self.create_service(Trigger, '/juez/salida', self.al_salida_manual)
        self.create_service(Trigger, '/juez/vuelco', self.al_vuelco)
        self.create_service(Trigger, '/juez/terminar', self.al_terminar_manual)
        if args.manual:
            self.args.retardo = float('inf')
            self.create_service(Trigger, '/juez/iniciar', self.al_iniciar)
            self.get_logger().info('Modo manual: llamá al servicio /juez/iniciar para largar')
        self.registrar('juez_listo', victimas=len(victimas))

    # ---- eventos -------------------------------------------------------------------
    def t(self):
        return None if self.t_mapa is None else time.monotonic() - self.t_mapa

    def registrar(self, tipo, **datos):
        t = self.t()
        evento = {'t': None if t is None else round(t, 2), 'evento': tipo, **datos}
        if self.args.equipo:
            evento['equipo'] = self.args.equipo
        self.eventos.append(evento)
        if self._registro:
            self._registro.write(json.dumps(sanear_json(evento)) + '\n')

    def fin(self, motivo):
        """Cierra la corrida por `motivo` (y fija el tiempo total, que no incluye la espera del veredicto)."""
        if self.terminado is None:
            self.terminado = motivo
        if self.t_fin is None:
            self.t_fin = self.t()

    def al_iniciar(self, _req, resp):
        if self.t_mapa is None:
            self.args.retardo = 0.0
            self.t_inicio = time.monotonic() - 1.0
            resp.success, resp.message = True, 'largada'
        else:
            resp.success, resp.message = False, 'el mapa ya fue publicado'
        return resp

    def al_colision_manual(self, _req, resp):
        if self.t_mapa is None or self.terminado:
            resp.success, resp.message = False, 'no hay corrida en curso'
            return resp
        self.colisiones += 1
        self.registrar('colision_manual')
        self.publicar_puntaje('colisión (juez)')
        resp.success, resp.message = True, f'colisiones: {self.colisiones}'
        return resp

    def al_veredicto_regreso(self, req, resp):
        if not self.done:
            resp.success, resp.message = False, 'todavía no hubo /done'
            return resp
        self.veredicto = bool(req.data)
        self.registrar('regreso_veredicto', dentro_del_circulo=self.veredicto)
        self.publicar_puntaje('regreso (juez)')
        resp.success, resp.message = True, 'registrado'
        return resp

    def al_qos_incompatible(self, topic):
        self.registrar('qos_incompatible', topic=topic)

    def _hay_corrida(self, resp):
        if self.t_mapa is None or self.terminado or self.done:
            resp.success, resp.message = False, 'no hay corrida en curso'
            return False
        return True

    def al_salida_manual(self, _req, resp):
        if self._hay_corrida(resp):
            self.salio = True
            self.registrar('salida_del_laberinto', por='juez')
            self.publicar_puntaje('salida (juez)')
            self.fin('salida')
            resp.success, resp.message = True, 'salida registrada'
        return resp

    def al_vuelco(self, _req, resp):
        if self._hay_corrida(resp):
            self.anulada = True
            self.registrar('vuelco')
            self.publicar_puntaje('vuelco (juez)')
            self.fin('vuelco')
            resp.success, resp.message = True, 'corrida anulada'
        return resp

    def al_terminar_manual(self, _req, resp):
        if self._hay_corrida(resp):
            self.registrar('terminada_por_el_juez')
            self.fin('juez')
            resp.success, resp.message = True, 'corrida terminada'
        return resp

    def al_cmd_vel(self, msg):
        super().al_cmd_vel(msg)
        mueve = any(abs(v) > UMBRAL_MOVIMIENTO for v in (
            msg.linear.x, msg.linear.y, msg.linear.z, msg.angular.x, msg.angular.y, msg.angular.z))
        if self.done and mueve:
            self.cmd_vel_tras_done += 1
            if self.cmd_vel_tras_done == 1:
                self.registrar('movimiento_despues_de_done')

    def al_odom(self, m):
        v = math.hypot(m.twist.twist.linear.x, m.twist.twist.linear.y)
        w = abs(m.twist.twist.angular.z)
        if v > self.args.umbral_lineal or w > self.args.umbral_angular:
            self.t_ultimo_movimiento = time.monotonic()

    def al_gt(self, m):
        p = m.pose.pose
        yaw = math.atan2(2 * (p.orientation.w * p.orientation.z + p.orientation.x * p.orientation.y),
                         1 - 2 * (p.orientation.y ** 2 + p.orientation.z ** 2))
        self.pose_actual = (p.position.x, p.position.y, yaw)

    def al_reporte(self, msg):
        super().al_reporte(msg)
        valido = (msg.header.frame_id == 'map'
                  and math.isfinite(msg.point.x) and math.isfinite(msg.point.y))
        self.validez.append(valido)
        self.registrar('reporte', x=round(msg.point.x, 3), y=round(msg.point.y, 3), valido=valido)
        self.publicar_puntaje(f'reporte #{len(self.reportes)}')

    def al_done(self, msg):
        estaba = self.done
        super().al_done(msg)
        if self.done and not estaba:
            if self.pose_largada and self.pose_actual:
                self.dist_done = math.dist(self.pose_actual[:2], self.pose_largada[:2])
            self.t_fin = self.t()
            self.registrar('done', distancia_a_largada=None if self.dist_done is None
                           else round(self.dist_done, 3))

    def ciclo(self):
        ya_terminado = self.terminado
        super().ciclo()
        if ya_terminado:
            self.terminado = ya_terminado        # el motivo que ya se decidió no se pisa
        if self.t_mapa is None:
            return
        if not self._mapa_registrado:
            self._mapa_registrado = True
            self.registrar('mapa_publicado')
        if self.terminado == 'tiempo' and self.t_fin is None:
            self.t_fin = self.t()
        if self.terminado == 'done' and not self.args.ground_truth:
            # Robot físico: el /done cerró la corrida, pero falta el veredicto del juez sobre el
            # regreso. Se espera un rato antes de cerrar el resultado.
            if self.t_done is None:
                self.t_done = time.monotonic()
            if self.veredicto is None and time.monotonic() - self.t_done < self.args.espera_veredicto:
                self.terminado = None
                return
        if self.terminado:
            return
        if self.pose_largada is None and self.pose_actual is not None:
            self.pose_largada = self.pose_actual     # el robot no se movió antes del mapa
            self.registrar('largada_pose', x=round(self.pose_largada[0], 3),
                           y=round(self.pose_largada[1], 3))
        # La inmovilidad se cuenta desde la publicación del mapa (§11.3).
        self.t_ultimo_movimiento = max(self.t_ultimo_movimiento, self.t_mapa)
        if time.monotonic() - self.t_ultimo_movimiento >= self.args.inmovil:
            self.registrar('robot_inmovil', segundos=self.args.inmovil)
            self.fin('inmovil')

    # ---- puntaje -------------------------------------------------------------------
    def corrida(self):
        reps = [puntaje.Reporte(t=-1.0 if r['t'] is None else r['t'], x=r['x'], y=r['y'], valido=ok)
                for r, ok in zip(self.reportes, self.validez)]
        return puntaje.Corrida(
            victimas=self.victimas, reportes=reps, colisiones=self.colisiones,
            salio_del_laberinto=self.salio, hizo_done=self.done, anulada=self.anulada,
            regreso_confirmado=self.veredicto, distancia_a_largada_en_done=self.dist_done)

    def puntuar(self):
        return puntaje.puntuar(self.corrida(), self.args.pen_colision, self.args.pen_salida)

    def publicar_puntaje(self, motivo):
        res = self.puntuar()
        self.puntaje_en_vivo = res
        msg = String()
        msg.data = json.dumps({'motivo': motivo, 'total': res.total, 'resumen': res.resumen()})
        self.pub_puntaje.publish(msg)
        self.get_logger().info(f'[{motivo}] {res.resumen()}')

    def resultado(self):
        base = super().resultado()
        res = self.puntuar()
        base.update({
            'puntaje': asdict(res),
            'resumen': res.resumen(),
            'victimas': self.victimas,
            'eventos': self.eventos,
            'equipo': self.args.equipo,
            'duracion_s': None if self.t() is None else round(self.t(), 1),
            'tiempo_total_s': None if self.t_fin is None else round(self.t_fin, 2),
            'tiempo_ultimo_reporte_que_puntua_s': (
                None if res.tiempo_ultimo_reporte_que_puntua is None
                else round(res.tiempo_ultimo_reporte_que_puntua, 2)),
            'movimiento_despues_de_done': self.cmd_vel_tras_done,
            'medidas_con_ground_truth': self.args.ground_truth,
        })
        return base


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--mapa', help='yaml de map_server (por defecto, un cuarto vacío sintético)')
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--mundo', help='mundo de práctica (nombre o ruta .world) del que salen las víctimas')
    g.add_argument('--victimas-json', help='JSON con [[x, y], ...]')
    ap.add_argument('--retardo', type=float, default=5.0, help='segundos hasta publicar /map')
    ap.add_argument('--manual', action='store_true', help='largar con el servicio /juez/iniciar')
    ap.add_argument('--duracion', type=float, default=600.0, help='segundos de corrida (10 min)')
    ap.add_argument('--inmovil', type=float, default=60.0, help='segundos quieto que terminan la corrida')
    ap.add_argument('--umbral-lineal', type=float, default=0.02, help='m/s (POR CONFIRMAR)')
    ap.add_argument('--umbral-angular', type=float, default=0.05, help='rad/s (POR CONFIRMAR)')
    ap.add_argument('--espera-veredicto', type=float, default=60.0,
                    help='robot físico: segundos que espera, tras el /done, el veredicto de /juez/regreso')
    ap.add_argument('--pen-colision', type=int, default=puntaje.PENALIZACION_COLISION)
    ap.add_argument('--pen-salida', type=int, default=puntaje.PENALIZACION_SALIDA)
    ap.add_argument('--ground-truth', action='store_true',
                    help='solo simulador: mide el regreso a la largada con la pose real')
    ap.add_argument('--esperar-topic', help='contar --retardo desde el primer mensaje de este topic')
    ap.add_argument('--equipo', help='nombre del equipo (queda en el resultado y en el registro)')
    ap.add_argument('--registro', help='archivo donde se guarda cada evento apenas ocurre (JSON por línea)')
    ap.add_argument('--salida', help='archivo JSON con el resultado')
    args = ap.parse_args()

    if args.victimas_json:
        victimas = [tuple(v) for v in json.loads(Path(args.victimas_json).read_text())]
    else:
        victimas = victimas_de_mundo(resolver_mundo(args.mundo))

    # El equipo lanza su software con victimas:=N; N tiene que ser la cantidad de este mundo.
    print(f'Este mundo tiene {len(victimas)} víctimas: lanzá tu software con victimas:={len(victimas)}',
          flush=True)
    rclpy.init()
    juez = Juez(args, victimas)
    try:
        while rclpy.ok() and juez.terminado is None:
            rclpy.spin_once(juez, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    res = juez.resultado()
    res = sanear_json(res)
    if args.salida:
        Path(args.salida).write_text(json.dumps(res, indent=2))
    print('\n=== FIN DE LA CORRIDA:', res['termino_por'], '===')
    print(res['resumen'])
    juez.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    for v in res['violaciones']:
        print(f'::error::juez: {v}')
    return 1 if res['violaciones'] or not res['mapa_publicado'] else 0


if __name__ == '__main__':
    sys.exit(main())
