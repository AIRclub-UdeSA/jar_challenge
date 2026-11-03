"""Reduce temporalmente el costo alrededor de obstáculos cuando Nav2 no progresa."""

import math
import time

from rcl_interfaces.srv import GetParameters, SetParameters
from rclpy.parameter import Parameter


PARAMETROS_INFLACION = {
    'inflacion_adaptativa': True,
    # true: el radio normal es el inflation_radius que Nav2 cargó de nav2_exploracion.yaml
    # (se lee del global_costmap al arrancar); inflacion_normal_m queda sólo de respaldo.
    'inflacion_desde_nav2': True,
    'inflacion_normal_m': 0.27,
    'inflacion_minima_m': 0.23,
    'paso_inflacion_m': 0.02,
    'espera_radio_nav2_s': 15.0,
    'tiempo_pre_atasco_s': 4.0,
    'intervalo_inflacion_s': 3.0,
    'distancia_restaurar_inflacion_m': 0.40,
}


class InflacionAdaptativa:
    """Histéresis de movimiento y actualizaciones asíncronas de ambos costmaps.

    Usa tiempo ROS para medir progreso y tiempo real para los servicios. Sólo
    modifica inflation_layer.inflation_radius: conserva footprint y obstáculos.
    """

    def __init__(self, nodo, cfg):
        self.nodo, self.cfg = nodo, cfg
        c = cfg
        if (not all(math.isfinite(c[k]) and c[k] > 0 for k in PARAMETROS_INFLACION
                    if k not in ('inflacion_adaptativa', 'inflacion_desde_nav2'))
                or c['inflacion_minima_m'] > c['inflacion_normal_m']
                or not 0 < c['tiempo_pre_atasco_s'] < c['tiempo_atascado']
                or c['distancia_restaurar_inflacion_m'] <= c['distancia_atascado']):
            raise ValueError('Parámetros de inflación adaptativa inválidos')
        # Con inflacion_desde_nav2, hasta leer el radio de Nav2 no se envía nada:
        # así nunca se pisa el inflation_radius de nav2_exploracion.yaml con otro valor.
        self.desde_nav2 = bool(c['inflacion_adaptativa'] and c.get('inflacion_desde_nav2', False))
        self.objetivo = None if self.desde_nav2 else c['inflacion_normal_m']
        self.lectura = None
        self.ultimo_t = None
        self.ultima_reduccion = -math.inf
        self.origen_reduccion = None
        self._reiniciar_medicion()
        self.costmaps = []
        if c['inflacion_adaptativa']:
            if self.desde_nav2:
                self.lectura = dict(cli=nodo.create_client(
                    GetParameters, '/global_costmap/global_costmap/get_parameters'),
                    futuro=None, t_envio=0.0, t_inicio=None)
            for nombre in ('global_costmap', 'local_costmap'):
                cli = nodo.create_client(SetParameters, f'/{nombre}/{nombre}/set_parameters')
                self.costmaps.append(dict(nombre=nombre, cli=cli, aplicado=None,
                    futuro=None, enviado=None, t_envio=0.0, reintentar=0.0))

    def _reiniciar_medicion(self):
        self.ancla = None
        self.t_avance = None
        self.meta = None
        self.mejor_progreso = None

    def reducir(self, pose, motivo):
        if not self.cfg['inflacion_adaptativa']:
            return
        if self.objetivo is None:
            return                      # todavía no se leyó el radio de Nav2
        radio = max(self.cfg['inflacion_minima_m'], round(self.objetivo - self.cfg['paso_inflacion_m'], 6))
        if radio >= self.objetivo:
            return
        self.objetivo = radio
        self.ultima_reduccion = self.nodo.get_clock().now().nanoseconds * 1e-9
        self.origen_reduccion = tuple(pose[:2]) if pose is not None else None
        self.nodo.get_logger().warn(f'[inflación] {motivo}: solicito radio {radio:.2f} m en ambos costmaps')
        self.sincronizar()

    def restaurar(self, motivo):
        if self.objetivo is not None and self.objetivo != self.cfg['inflacion_normal_m']:
            self.objetivo = self.cfg['inflacion_normal_m']
            self.nodo.get_logger().info(
                f'[inflación] {motivo}: restauro radio {self.objetivo:.2f} m')
        self.origen_reduccion = None
        self.ultima_reduccion = -math.inf
        self._reiniciar_medicion()
        self.sincronizar()

    def paso(self, pose, navegando, progreso=None, meta=None, restaurar=False):
        """`progreso` crece al acercarse por el camino; `pose` es odometría estable."""
        if not self.cfg['inflacion_adaptativa']:
            return
        t = self.nodo.get_clock().now().nanoseconds * 1e-9
        if restaurar or (self.ultimo_t is not None and t < self.ultimo_t):
            self.restaurar('navegación detenida o reinicio del reloj')
        elif not navegando or pose is None or not all(math.isfinite(v) for v in pose):
            # Mirar una víctima, girar o esperar una acción no es un atasco.
            self._reiniciar_medicion()
        else:
            if progreso is not None and not math.isfinite(progreso):
                progreso = None
            if self.ancla is None:
                self.ancla, self.t_avance = tuple(pose[:2]), t
            if meta != self.meta or self.mejor_progreso is None:
                self.meta, self.mejor_progreso = meta, progreso
            if progreso is None:
                avanza = math.dist(pose[:2], self.ancla) >= self.cfg['distancia_atascado']
            else:
                avanza = progreso >= self.mejor_progreso + self.cfg['distancia_atascado']
            if avanza:
                self.ancla, self.t_avance = tuple(pose[:2]), t
                self.mejor_progreso = progreso
            estancado = t - self.t_avance >= self.cfg['tiempo_pre_atasco_s']
            if estancado and t - self.ultima_reduccion >= self.cfg['intervalo_inflacion_s']:
                self.reducir(pose, f'{t - self.t_avance:.1f} s sin progreso')
            elif (not estancado and self.origen_reduccion is not None
                  and t - self.ultima_reduccion >= self.cfg['intervalo_inflacion_s']
                  and math.dist(pose[:2], self.origen_reduccion) >= self.cfg['distancia_restaurar_inflacion_m']):
                self.restaurar('el robot volvió a avanzar')
        self.ultimo_t = t
        self.sincronizar()

    def leer_radio_nav2(self):
        """Toma como radio normal el inflation_radius del global_costmap (el de
        nav2_exploracion.yaml). Si Nav2 no responde en espera_radio_nav2_s, usa
        inflacion_normal_m de exploracion.yaml y lo avisa."""
        lectura = self.lectura
        if lectura is None or self.objetivo is not None:
            return
        ahora = time.monotonic()
        if lectura['t_inicio'] is None:
            lectura['t_inicio'] = ahora
        futuro = lectura['futuro']
        if futuro is not None and futuro.done():
            lectura['futuro'] = None
            try:
                valores = futuro.result().values
                radio = float(valores[0].double_value) if len(valores) == 1 else float('nan')
            except Exception:  # noqa: BLE001  se reintenta
                radio = float('nan')
            if math.isfinite(radio) and radio > 0:
                self._fijar_normal(radio, 'leído de Nav2 (nav2_exploracion.yaml)')
                return
        elif futuro is not None and ahora - lectura['t_envio'] >= 2.0:
            lectura['cli'].remove_pending_request(futuro)
            lectura['futuro'] = None
        if ahora - lectura['t_inicio'] >= self.cfg.get('espera_radio_nav2_s', 15.0):
            self.nodo.get_logger().warn('[inflación] Nav2 no informó su inflation_radius: uso '
                                        'inflacion_normal_m de exploracion.yaml')
            self._fijar_normal(self.cfg['inflacion_normal_m'], 'de respaldo (exploracion.yaml)')
            return
        if lectura['futuro'] is None and lectura['cli'].service_is_ready():
            try:
                lectura['futuro'] = lectura['cli'].call_async(
                    GetParameters.Request(names=['inflation_layer.inflation_radius']))
                lectura['t_envio'] = ahora
            except Exception:  # noqa: BLE001
                lectura['futuro'] = None

    def _fijar_normal(self, radio, origen):
        self.cfg['inflacion_normal_m'] = radio
        if self.cfg['inflacion_minima_m'] > radio:
            self.nodo.get_logger().warn(
                f'[inflación] inflacion_minima_m ({self.cfg["inflacion_minima_m"]:.2f}) es mayor que el radio '
                f'de Nav2 ({radio:.2f}): no se reduce por debajo de {radio:.2f}')
            self.cfg['inflacion_minima_m'] = radio
        self.objetivo = radio
        self.lectura = None
        self.nodo.get_logger().info(
            f'[inflación] radio normal {radio:.2f} m {origen}; mínimo {self.cfg["inflacion_minima_m"]:.2f} m')

    def sincronizar(self):
        """Espera ACK antes de dar un radio por aplicado; reintenta sin bloquear ROS."""
        self.leer_radio_nav2()
        if self.objetivo is None:
            return
        ahora = time.monotonic()
        for estado in self.costmaps:
            cli, futuro = estado['cli'], estado['futuro']
            if futuro is not None:
                error = None
                if futuro.done():
                    try:
                        respuesta = futuro.result()
                        if len(respuesta.results) != 1 or not respuesta.results[0].successful:
                            error = '; '.join(r.reason for r in respuesta.results) or 'parámetro rechazado'
                        else:
                            estado['aplicado'] = estado['enviado']
                            self.nodo.get_logger().info(
                                f'[inflación] {estado["nombre"]}: radio aplicado {estado["aplicado"]:.2f} m')
                    except Exception as exc:  # noqa: BLE001  un servicio no detiene la navegación
                        error = str(exc)
                elif ahora - estado['t_envio'] >= 2.0:
                    cli.remove_pending_request(futuro)
                    futuro.cancel()
                    error = 'sin respuesta en 2 s'
                else:
                    continue
                estado['futuro'] = None
                if error is not None:
                    # Un timeout no permite saber si el servidor llegó a aplicar el cambio.
                    estado['aplicado'] = None
                    estado['reintentar'] = ahora + 2.0
                    self.nodo.get_logger().warn(
                        f'[inflación] {estado["nombre"]}: {error}; volveré a solicitar el radio')
            if (estado['aplicado'] == self.objetivo or ahora < estado['reintentar']
                    or not cli.service_is_ready()):
                continue
            solicitud = SetParameters.Request()
            solicitud.parameters = [Parameter('inflation_layer.inflation_radius',
                Parameter.Type.DOUBLE, float(self.objetivo)).to_parameter_msg()]
            try:
                estado['futuro'] = cli.call_async(solicitud)
                estado['enviado'], estado['t_envio'] = self.objetivo, ahora
            except Exception as exc:  # noqa: BLE001
                estado['reintentar'] = ahora + 2.0
                self.nodo.get_logger().warn(f'[inflación] {estado["nombre"]}: no pude solicitar el cambio: {exc}')
