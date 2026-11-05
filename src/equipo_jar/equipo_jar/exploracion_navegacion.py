import math
import time
import numpy as np
from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import NavigateThroughPoses, NavigateToPose, Spin
from nav2_msgs.srv import ClearEntireCostmap
from lifecycle_msgs.msg import State, Transition
from lifecycle_msgs.srv import ChangeState, GetState
from rclpy.task import Future
from equipo_jar.planificador import segmento_transitable
from equipo_jar.exploracion_estados import (
    LOCALIZANDO,
    ACTIVANDO_NAV2,
    ELIGIENDO,
    NAVEGANDO,
    COMPORTAMIENTO,
    OBSERVANDO_OBJETO,
    ENVIAR_PENDIENTE,
    REGRESAR,
    VOLVIENDO,
)
from equipo_jar.exploracion_geometria import yaw_a_quat, ang_norm, bfs


class NavegacionExploracion:

    NODOS_NAV2 = ('controller_server', 'smoother_server', 'planner_server', 'behavior_server',
                  'bt_navigator', 'waypoint_follower', 'velocity_smoother')

    def revisar_ciclo_de_vida(self, nodos, activar=True):
        """Con la PC cargada, a veces se pierde la respuesta de un nodo al
        lifecycle_manager ("failed to send response to /amcl/change_state") y
        el manager queda esperando para siempre: el nodo queda configurado pero
        inactivo. Acá se pregunta el estado y se lo termina de activar."""
        ahora = time.monotonic()
        for n in nodos:
            if n not in self.lc:
                self.lc[n] = (self.create_client(GetState, f'/{n}/get_state'),
                              self.create_client(ChangeState, f'/{n}/change_state'))
            get_cli, _ = self.lc[n]
            pendiente = self.consultas_lc.get(n)
            if pendiente is not None and not pendiente[0].done() and ahora - pendiente[1] < 4.0:
                continue
            if get_cli.service_is_ready():
                try:
                    futuro = get_cli.call_async(GetState.Request())
                except Exception as error:  # noqa: BLE001
                    self.estados_lc.pop(n, None)
                    self.get_logger().warn(f'No pude consultar {n}: {error}', throttle_duration_sec=5.0)
                    continue
                self.consultas_lc[n] = (futuro, ahora)
                futuro.add_done_callback(lambda f, n=n: self._estado_ciclo_de_vida(n, f, activar))

    def _estado_ciclo_de_vida(self, nodo, fut, activar=True):
        consulta = self.consultas_lc.get(nodo)
        if consulta is not None and consulta[0] is not fut:
            return
        try:
            estado = fut.result().current_state.id
        except Exception:  # noqa: BLE001
            self.estados_lc.pop(nodo, None)
            return
        ahora = time.monotonic()
        self.estados_lc[nodo] = (estado, ahora)
        if not activar:
            return
        # 1 = unconfigured, 2 = inactive (3 = active, el resto son transitorios).
        transicion = {1: Transition.TRANSITION_CONFIGURE, 2: Transition.TRANSITION_ACTIVATE}.get(estado)
        if transicion is None:
            return
        pendiente = self.transiciones_lc.get(nodo)
        if pendiente is not None and not pendiente[0].done() and ahora - pendiente[1] < 8.0:
            return
        self.get_logger().warn(
            f"{nodo} quedó {'sin configurar' if estado == 1 else 'inactivo'} (se perdió una "
            f"respuesta del lifecycle_manager): lo {'configuro' if estado == 1 else 'activo'} yo")
        req = ChangeState.Request()
        req.transition.id = transicion
        try:
            self.transiciones_lc[nodo] = (self.lc[nodo][1].call_async(req), ahora)
        except Exception as error:  # noqa: BLE001
            self.get_logger().warn(f'No pude activar {nodo}: {error}', throttle_duration_sec=5.0)

    def nav2_activo(self):
        """Un action server puede existir antes de estar activo: no alcanza con descubrirlo."""
        ahora = time.monotonic()
        return all(n in self.estados_lc and self.estados_lc[n][0] == State.PRIMARY_STATE_ACTIVE
                   and ahora - self.estados_lc[n][1] <= 10.0 for n in self.NODOS_NAV2)

    def recuperar_nav2(self, siguiente=None, giro=None):
        """Recuperación de infraestructura: esperar quieto, sin marcar metas como obstáculos."""
        previo = self.estado
        cancelacion = self.cancelar_actual()
        self.goal_actual = None
        self.pub_cmd.publish(Twist())
        self.escape = None
        self.giro_pendiente = None
        if self.objeto_observado is not None:
            self.reanudar_nav2 = OBSERVANDO_OBJETO
            self.objeto_observado.update(cancelacion=cancelacion, fase='frenando', desde=None,
                                         quieto_desde=None, giros=0)
            giro = None  # Recalcular hacia el objeto al recuperar los servidores.
        elif self.motivo_regreso is not None or previo in (VOLVIENDO, REGRESAR):
            self.reanudar_nav2 = REGRESAR
        else:
            self.reanudar_nav2 = siguiente or ELIGIENDO
        self.giro_reanudar_nav2 = giro
        self.estado = ACTIVANDO_NAV2
        self.estados_lc.clear()
        self.t_ult_revision_lc = time.monotonic()
        self.revisar_ciclo_de_vida(self.NODOS_NAV2)
        self.get_logger().warn('Nav2 no está operativo: espero su activación; no es un atasco físico')

    def paso_activacion_nav2(self, ahora):
        self.pub_cmd.publish(Twist())
        if ahora - self.t_ult_revision_lc >= 1.0:
            self.t_ult_revision_lc = ahora
            self.revisar_ciclo_de_vida(self.NODOS_NAV2)
        if (not self.nav2_activo() or not self.nav.server_is_ready()
                or not self.nav_ruta.server_is_ready() or not self.spin_cli.server_is_ready()):
            return
        o = self.objeto_observado
        if o is not None:
            o.update(inicio=ahora, quieto_desde=None, desde=None)
        self.limite_observacion(100.0)
        siguiente = self.reanudar_nav2
        giro = self.giro_reanudar_nav2
        self.giro_reanudar_nav2 = None
        self.estado = siguiente
        self.get_logger().info('Nav2 activo: reanudo la misión sin maniobras de escape')
        if giro is not None:
            self.girar(giro, siguiente)

    def dividir_tramo(self, pasos):
        """Un sector por acción: hasta cinco guías y la entrada del próximo portal."""
        if not pasos:
            return pasos, []

        def acotar(tramo):
            if len(tramo) <= 5:
                return tramo
            # Última barrera: aunque un llamador entregue el trazado denso,
            # sólo enviar entrada, mitad por distancia recorrida y salida.
            largos = np.cumsum([0.] + [math.dist(a[:2], b[:2]) for a, b in zip(tramo, tramo[1:])])
            medio = 1 + int(np.argmin(np.abs(largos[1:-1] - largos[-1] / 2)))
            return [tramo[0], tramo[medio], tramo[-1]]

        if self.sectorizador is None:
            return acotar(pasos), []
        regiones = self.sectorizador.regiones

        def region(punto):
            f, c = self.grilla.a_celda(*punto[:2])
            return int(regiones[f, c]) if self.grilla.dentro(f, c) else -1

        actual = region(pasos[0])
        for i in range(1, len(pasos)):
            if region(pasos[i]) != actual:
                # El cruce pertenece también al comienzo del tramo siguiente.
                # Mantenerlo en la cola permite continuar desde esa entrada,
                # sin recuperar puntos de sectores ya recorridos.
                if self.cfg.get('guias_portal_y_medio', False):
                    # Portal + medio de esta zona; el portal siguiente abre la próxima acción.
                    return acotar(pasos[:i]), pasos[i:]
                resto = pasos[i:] if i < len(pasos) - 1 else []
                return acotar(pasos[:i]) + [pasos[i]], resto
        return acotar(pasos), []

    def enviar_goal(self, x, y, yaw, ganancia, d_m, etiqueta='vista', pasos=None):
        heredado = self.heredado_vivo()
        if heredado and not pasos:
            pasos = [(x, y, yaw)]          # mismo servidor que el goal en marcha: lo reemplaza sin frenar
        pasos, continuacion = self.dividir_tramo(pasos)
        ruta_continua = bool(pasos)
        cliente = self.nav_ruta if ruta_continua else self.nav
        if not cliente.server_is_ready():
            accion = 'navigate_through_poses' if ruta_continua else 'navigate_to_pose'
            self.get_logger().warn(f'Nav2 ({accion}) todavía no está listo')
            return False
        self.goal_id += 1
        gid = self.goal_id
        pose = self.pose_robot()
        distancias_guias = []
        d_tramo = 0.0
        anterior = pose[:2] if pose is not None else ((pasos[0][:2]) if pasos else (x, y))
        for punto in pasos or []:
            d_tramo += math.dist(anterior, punto[:2])
            distancias_guias.append(d_tramo)
            anterior = punto[:2]
        if not pasos:
            d_tramo = d_m
        if ruta_continua:
            goal = NavigateThroughPoses.Goal()
            goal.behavior_tree = self.bt_portales
            goal.poses = []
            for px, py, rumbo in pasos:
                punto = PoseStamped()
                punto.header.frame_id = self.grilla.frame
                punto.header.stamp = self.get_clock().now().to_msg()
                punto.pose.position.x, punto.pose.position.y = px, py
                punto.pose.orientation.z, punto.pose.orientation.w = yaw_a_quat(rumbo)
                goal.poses.append(punto)
        else:
            goal = NavigateToPose.Goal()
            goal.pose = PoseStamped()
            goal.pose.header.frame_id = self.grilla.frame
            goal.pose.header.stamp = self.get_clock().now().to_msg()
            goal.pose.pose.position.x, goal.pose.pose.position.y = x, y
            goal.pose.pose.orientation.z, goal.pose.pose.orientation.w = yaw_a_quat(yaw)
        self.goal_actual = dict(x=x, y=y, yaw=yaw, ganancia0=max(ganancia, 1.0), t0=time.monotonic(),
                                p_inicio=pose, continuacion=continuacion,
                                meta_tramo=pasos[-1] if pasos else (x, y, yaw),
                                distancias_guias=distancias_guias, mejor_avance=None,
                                t_progreso=self.get_clock().now().nanoseconds * 1e-9,
                                t_max=20.0 + 2.5 * d_tramo / max(0.05, self.cfg['velocidad_estimada']),
                                etiqueta=etiqueta, pasos=pasos,
                                pasos_restantes=len(pasos) if pasos else 0)
        self.hist.clear()
        if heredado:
            self.goal_heredado = None      # Nav2 lo reemplaza por este (preempción, sin frenar)
        if ruta_continua:
            futuro = cliente.send_goal_async(
                goal, feedback_callback=lambda mensaje, gid=gid: self.cb_feedback_ruta(mensaje, gid))
        else:
            futuro = cliente.send_goal_async(goal)
        self.solicitud_goal = futuro
        futuro.add_done_callback(lambda f, gid=gid: self.cb_goal_aceptado(f, gid))
        local = self.goal_actual['meta_tramo']
        self.get_logger().info(
            f'[{etiqueta}] goal #{gid}: tramo hasta ({local[0]:.2f}, {local[1]:.2f}), '
            f'destino final ({x:.2f}, {y:.2f}); '
            f'pasos={len(pasos) if pasos else 1} yaw={math.degrees(yaw):.0f}° gan={ganancia:.0f} '
            f'dist tramo≈{d_tramo:.1f} m; guías pendientes={len(continuacion)} | cobertura {100 * self.cobertura():.1f}% | '
            f't={self.tiempo_transcurrido():.0f}s')
        if pasos:
            self.get_logger().info('Guías Nav2 entrada/medio/salida: ' +
                                   ' -> '.join(f'({px:.2f}, {py:.2f})' for px, py, _ in pasos))
        return True

    def continuar_tramo(self, ga):
        pasos = ga.get('continuacion', [])
        if not pasos:
            return False
        pose = self.pose_robot()
        distancia = math.dist(pose[:2], pasos[0][:2]) if pose is not None else 0.0
        distancia += sum(math.dist(a[:2], b[:2]) for a, b in zip(pasos, pasos[1:]))
        if pose is not None:
            proxima = next((p for p in pasos if math.dist(p[:2], pose[:2]) >= 0.25), pasos[-1])
            self.dir_ruta = self.direccion_de_ruta(pose, proxima[0], proxima[1])
        return self.enviar_goal(ga['x'], ga['y'], ga['yaw'], ga['ganancia0'], distancia,
                                etiqueta=ga['etiqueta'], pasos=pasos)

    def anticipar_siguiente_tramo(self, pose):
        """Actualiza la misma acción cerca de la salida para atravesarla sin pausa."""
        ga = self.goal_actual
        if (ga is None or not ga.get('continuacion') or self.goal_handle is None
                or self.solicitud_goal is not None or ga['pasos_restantes'] > 2):
            return False
        meta = ga['meta_tramo']
        if math.dist(pose[:2], meta[:2]) > self.cfg['anticipacion_portal_m']:
            return False
        inicio = self.grilla.a_celda(*pose[:2])[1::-1]
        destino = self.grilla.a_celda(*meta[:2])[1::-1]
        if not segmento_transitable(self.grilla.transitable, inicio, destino):
            return False
        return self.continuar_tramo(ga)

    def cb_feedback_ruta(self, mensaje, gid):
        if gid == self.goal_id and self.goal_actual is not None:
            self.goal_actual['pasos_restantes'] = min(
                self.goal_actual['pasos_restantes'], mensaje.feedback.number_of_poses_remaining)
            restante = getattr(mensaje.feedback, 'distance_remaining', None)
            if restante is not None and math.isfinite(restante) and restante > 0:
                self.goal_actual['restante_nav2'] = float(restante)

    def siguiente_guia(self, ga):
        pasos = ga.get('pasos') if ga else None
        if not pasos:
            return ga['x'], ga['y']
        indice = max(0, min(len(pasos) - 1, len(pasos) - ga['pasos_restantes']))
        return pasos[indice][:2]

    def direccion_de_ruta(self, pose, gx, gy, pasos=10):
        """Rumbo (marco map) del primer tramo (~0.5 m) del camino más corto hacia (gx, gy)."""
        g = self.grilla
        dist = bfs(g.transitable, *g.a_celda(gx, gy))
        if dist is None:
            return None
        f, c = g.a_celda(pose[0], pose[1])
        if not g.dentro(f, c) or dist[f, c] < 0:
            return math.atan2(gy - pose[1], gx - pose[0])
        for _ in range(pasos):
            mejor = None
            for df in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nf, nc = f + df, c + dc
                    if (df and dc and g.dentro(nf, nc)
                            and not (g.transitable[f, nc] and g.transitable[nf, c])):
                        continue
                    if (df or dc) and g.dentro(nf, nc) and 0 <= dist[nf, nc] < dist[f, c] and \
                            (mejor is None or dist[nf, nc] < dist[mejor]):
                        mejor = (nf, nc)
            if mejor is None:
                break
            f, c = mejor
        wx, wy = g.a_mundo(f, c)
        if math.hypot(wx - pose[0], wy - pose[1]) < 0.05:
            return math.atan2(gy - pose[1], gx - pose[0])
        return math.atan2(wy - pose[1], wx - pose[0])

    def encarar_y_enviar(self, x, y, yaw, ganancia, d_m, etiqueta, estado_destino, pasos=None):
        """Si el camino arranca "para atrás" (>100° del rumbo del robot), primero gira en el
        lugar para encararlo y recién después manda el goal."""
        pose = self.pose_robot()
        if pose is None:
            return False
        if pasos is None and pose is not None:
            fila, col = self.grilla.a_celda(*pose[:2])
            if not self.grilla.dentro(fila, col) or not self.grilla.transitable[fila, col]:
                self.get_logger().warn('El robot está dentro del margen de un obstáculo: me separo antes de calcular otra ruta')
                self.cancelar_actual()
                self.goal_actual = None
                self.n_atascos += 1
                self.dir_ruta = None
                self.iniciar_escape(REGRESAR if estado_destino == VOLVIENDO else ELIGIENDO)
                return True
            pasos, d_guias = self.ruta_a_destino(pose, x, y, yaw)
            if pasos:
                d_m = d_guias
            else:
                # El trazador propio (portales con margen) no encontró guías, pero el
                # destino sí es alcanzable (lo eligió el BFS): que planifique Nav2.
                # Antes se lo descartaba y, en zonas angostas, el robot se quedaba
                # quieto descartando un destino tras otro.
                self.get_logger().info(f'[{etiqueta}] sin guías por portales hasta ({x:.2f}, {y:.2f}): '
                                       'le paso el destino directo a Nav2')
                pasos = None
        primero = next((p for p in pasos or [] if math.dist(p[:2], pose[:2]) >= 0.25),
                       pasos[-1] if pasos else (x, y))
        self.dir_ruta = self.direccion_de_ruta(pose, primero[0], primero[1]) if pose else None
        if pose is not None and self.dir_ruta is not None and self.spin_cli.server_is_ready():
            giro = ang_norm(self.dir_ruta - pose[2])
            sin_lugar = False
            if abs(giro) > math.radians(getattr(self, 'cfg', {}).get('giro_previo_deg', 135.0)):
                try:
                    holgura = self.holgura_giro()
                except Exception:  # noqa: BLE001
                    holgura = None
                # Sin lugar para girar (pasillo alrededor de una caja), el giro previo
                # terminaba en "Muy cerca de una pared para girar" + desencaje: el X3 es
                # omnidireccional, mejor dejar que Nav2 arranque sin girar.
                sin_lugar = holgura is not None and holgura < getattr(self, 'cfg', {}).get('holgura_minima_giro_previo_m', 0.03)
                if sin_lugar:
                    self.get_logger().info(
                        f'[{etiqueta}] la ruta arranca a {math.degrees(giro):.0f}° pero no hay lugar para girar '
                        f'(holgura {100 * holgura:.0f} cm): no giro, Nav2 arranca como está')
            if not sin_lugar and abs(giro) > math.radians(getattr(self, 'cfg', {}).get('giro_previo_deg', 135.0)):
                self.get_logger().info(
                    f'[{etiqueta}] la ruta arranca a {math.degrees(giro):.0f}° de donde miro: giro primero')
                self.pendiente = (x, y, yaw, ganancia, d_m, etiqueta, estado_destino, pasos)
                self.girar(giro, ENVIAR_PENDIENTE)
                return True
        if self.enviar_goal(x, y, yaw, ganancia, d_m, etiqueta=etiqueta, pasos=pasos):
            self.estado = estado_destino
            return True
        return False

    def cb_goal_aceptado(self, fut, gid):
        handle = fut.result()
        if gid != self.goal_id:
            if handle.accepted:
                handle.cancel_goal_async()
            return
        self.solicitud_goal = None
        if not handle.accepted:
            if self.goal_handle is not None:
                self.goal_handle.cancel_goal_async()
            self.get_logger().warn(f'goal #{gid} rechazado por Nav2')
            self.recuperar_nav2()
            return
        self.goal_handle = handle
        handle.get_result_async().add_done_callback(
            lambda f, gid=gid: gid == self.goal_id and self.resultado_goal(f.result().status))

    def resultado_goal(self, status):
        self.goal_handle = None
        ok = status == GoalStatus.STATUS_SUCCEEDED
        ga = self.goal_actual
        o = getattr(self, 'objeto_observado', None)
        if (self.estado == OBSERVANDO_OBJETO and o is not None
                and o['fase'] == 'acercando'):
            self.goal_actual = None
            if ok:
                o.update(fase='frenando', quieto_desde=None, desde=None, imagenes=0)
                self.pub_cmd.publish(Twist())
            else:
                self.terminar_observacion_objeto(False, 'Nav2 no alcanzó la posición de observación')
            return
        if (ok and self.estado in (NAVEGANDO, VOLVIENDO)
                and ga is not None and ga.get('continuacion')):
            if not self.continuar_tramo(ga):
                self.goal_actual = None
                self.estado = REGRESAR if self.estado == VOLVIENDO else ELIGIENDO
            return
        if self.estado == VOLVIENDO:
            self.goal_actual = None
            if ok:
                self.ruta_regreso.clear()
                if self.sectorizador is None:
                    self.iniciar_aproximacion()
                else:
                    pose = self.pose_robot()
                    if pose is not None and math.hypot(
                            pose[0] - self.largada[0], pose[1] - self.largada[1]) <= 0.6:
                        self.iniciar_aproximacion()
                    else:
                        self.reintentar_regreso()
            else:
                self.reintentar_regreso()
            return
        if self.estado != NAVEGANDO:
            return
        if ok:
            self.al_llegar()
        else:
            self.fallo_goal()

    def se_movio_desde(self, ga, minimo=0.10):
        pose = self.pose_robot()
        p0 = ga.get('p_inicio') if ga else None
        return pose is None or p0 is None or math.hypot(pose[0] - p0[0], pose[1] - p0[1]) >= minimo

    def fallo_goal(self):
        ga = self.goal_actual
        if ga is not None:
            if ga.get('etiqueta', '').startswith('portal sector'):
                self.portales_fallidos.add(self.grilla.a_celda(*self.siguiente_guia(ga))[1::-1])
                self.ruta_sector.clear()
                self.get_logger().warn('Ruta por portales falló: recalculo desde la posición actual')
            else:
                self.lista_negra.append((ga['x'], ga['y']))
                self.get_logger().warn(f"goal ({ga['x']:.2f}, {ga['y']:.2f}) falló: lista negra")
        self.goal_actual = None
        self.estado = ELIGIENDO
        if ga is not None and self.registrar_fallo_localizacion(ELIGIENDO):
            return
        if ga is not None and not self.se_movio_desde(ga):
            # Nav2 rechazó el goal sin mover al robot: casi seguro está trabado
            # contra algo (DWB: "No valid trajectories"). Desencajar ya.
            inflacion = getattr(self, 'inflacion', None)
            if inflacion is not None:
                inflacion.reducir(self.lookup(self.cfg['odom_frame'], self.cfg['base_frame']),
                                  'Nav2 falló sin desplazamiento')
            self.n_atascos += 1
            self.get_logger().warn('Nav2 falló sin mover al robot -> me desencajo')
            self.limpiar_costmaps()
            self.iniciar_escape(ELIGIENDO)

    def al_llegar(self, ga=None):
        ga = ga or self.goal_actual
        self.goal_actual = None
        self.estado = ELIGIENDO
        if ga is None:
            return
        if ga.get('etiqueta', '').startswith('portal sector'):
            self.ruta_sector.clear()
            pose = self.pose_robot()
            if (pose is not None and self.sectorizador is not None
                    and not self.en_sector(pose)):
                self.portales_fallidos.add(self.grilla.a_celda(ga['x'], ga['y'])[1::-1])
                self.get_logger().warn('Nav2 terminó fuera del sector: descarto esa entrada y busco otra más interior')
                self.registrar_fallo_localizacion(ELIGIENDO)
            else:
                self.n_atascos = 0
                self.fallos_localizacion.clear()
            return
        self.n_atascos = 0
        self.fallos_localizacion.clear()
        self.lista_negra.append((ga['x'], ga['y']))   # visitado: no volver a elegirlo
        pose = self.pose_robot()
        giro = ang_norm(ga['yaw'] - pose[2]) if pose is not None else 0.0
        if ga.get('etiqueta', '').startswith('confirmar'):
            # Quedarse mirando la caja: el detector necesita varias imágenes de cerca.
            tolerancia = math.radians(self.cfg['tolerancia_mirar_caja_deg'])
            self.espera_mirar_hasta = (time.monotonic() + self.cfg['tiempo_mirar_caja_s']
                                       + (abs(giro) / 0.8 if abs(giro) > tolerancia else 0.0))
            if pose is not None and abs(giro) > tolerancia:   # dentro de ±25° la caja ya está en cuadro
                self.girar(giro)
            return
        if self.cfg['girar_al_llegar'] and pose is not None:
            if abs(giro) > math.radians(self.cfg['tolerancia_giro_llegada_deg']):
                self.girar(giro)

    def holgura_giro(self):
        """Metros libres alrededor del círculo que barre el chasis al girar en el lugar
        (esquinas a 0.21 m del centro). Negativo: alguna esquina tocaría algo."""
        puntos = self.puntos_scan()
        if puntos is None or puntos[0].size == 0:
            return None
        radio = math.hypot(self.ATRAS, self.MEDIO_ANCHO) + 0.01
        return float(np.min(np.hypot(*puntos))) - radio

    def girar(self, angulo, siguiente=ELIGIENDO):
        if not self.spin_cli.server_is_ready():
            self.recuperar_nav2(siguiente=siguiente, giro=angulo)
            return
        # Pegado a una pared, Spin de Nav2 aborta ("Collision Ahead") y el robot no
        # gira: primero separarse ~10 cm hacia donde hay lugar y después girar.
        holgura = self.holgura_giro() if self.estado != LOCALIZANDO else None
        if holgura is not None and holgura < 0.02 and self.giro_pendiente is None:
            self.giro_pendiente = (angulo, siguiente)
            self.get_logger().info(f'Muy cerca de una pared para girar (holgura {100 * holgura:.0f} cm): '
                                   'me separo antes')
            self.iniciar_escape(siguiente)
            self.escape['dist'] = 0.10
            return
        g = Spin.Goal()
        g.target_yaw = float(angulo)
        g.time_allowance = Duration(sec=10)
        self.comportamiento(self.spin_cli, g, siguiente)

    def comportamiento(self, cliente, goal, siguiente):
        """Corre una acción de behavior_server (Spin) y al terminar (bien o mal) pasa a `siguiente`."""
        if getattr(self, 'goal_heredado', None) is not None:
            self.cancelar_actual()
        self.estado = COMPORTAMIENTO
        self.goal_id += 1
        gid = self.goal_id
        self.hist.clear()
        self.hist_global.clear()

        def fin(_f=None, gid=gid):
            if gid == self.goal_id and self.estado == COMPORTAMIENTO:
                self.goal_handle = None
                if siguiente == OBSERVANDO_OBJETO and self.objeto_observado is not None:
                    self.objeto_observado['giro_resultado'] = _f.result().status
                self.estado = siguiente

        def aceptado(f, gid=gid):
            h = f.result()
            if gid != self.goal_id:
                if h.accepted:
                    h.cancel_goal_async()
                return
            self.solicitud_goal = None
            if not h.accepted:
                self.get_logger().warn('Acción de giro rechazada: verifico la activación de Nav2')
                self.recuperar_nav2(siguiente=siguiente, giro=goal.target_yaw)
                return
            self.goal_handle = h
            h.get_result_async().add_done_callback(fin)

        self.solicitud_goal = cliente.send_goal_async(goal)
        self.solicitud_goal.add_done_callback(aceptado)

    def soltar_sin_frenar(self):
        """Al salir de una zona: en vez de cancelar (Nav2 frena en seco y recién después
        arranca hacia el destino nuevo), deja andar el goal actual de NavigateThroughPoses
        hasta que el próximo goal lo reemplace (preempción: el controlador sigue con la
        ruta nueva sin detenerse). Devuelve False si no se puede (otro tipo de acción)."""
        ga = self.goal_actual
        if (not self.cfg.get('cambio_sin_frenar', True) or ga is None or not ga.get('pasos')
                or (self.goal_handle is None and self.solicitud_goal is None)):
            return False
        self.goal_id += 1     # su resultado ya no cuenta
        self.goal_heredado = dict(handle=self.goal_handle, solicitud=self.solicitud_goal,
                                  t=time.monotonic())
        self.goal_handle = None
        self.solicitud_goal = None
        return True

    def heredado_vivo(self):
        h = getattr(self, 'goal_heredado', None)
        if h is None:
            return False
        handle = h['handle']
        if handle is not None and handle.status in (GoalStatus.STATUS_SUCCEEDED, GoalStatus.STATUS_CANCELED,
                                                    GoalStatus.STATUS_ABORTED):
            self.goal_heredado = None
            return False
        return True

    def cancelar_actual(self):
        """Espera el resultado final de todas las acciones que aún puedan mover el robot."""
        self.goal_id += 1     # cualquier resultado pendiente queda obsoleto
        anterior = self.cancelacion_en_curso
        handle, solicitud = self.goal_handle, self.solicitud_goal
        if getattr(self, 'goal_heredado', None) is not None:
            if handle is None and solicitud is None:
                handle, solicitud = self.goal_heredado['handle'], self.goal_heredado['solicitud']
            self.goal_heredado = None
        self.solicitud_goal = None
        self.goal_handle = None
        if handle is None and solicitud is None:
            return anterior if anterior is not None and not anterior.done() else None

        barrera = Future()
        self.cancelacion_en_curso = barrera
        # El centinela impide completar la barrera si un callback se ejecuta
        # inmediatamente, antes de registrar las otras acciones pendientes.
        pendientes = 1

        def completar():
            nonlocal pendientes
            pendientes -= 1
            if pendientes == 0 and not barrera.done():
                barrera.set_result(None)

        def fallo(error):
            # Sin resultado definitivo no sabemos si dejó de publicar velocidad.
            # Mantener pendiente la barrera conserva detenida la recuperación.
            self.get_logger().warn(f'No pude confirmar el fin de una acción cancelada: {error}; mantengo la espera')

        def resultado_final(futuro):
            try:
                resultado = futuro.result()
                if resultado.status not in (GoalStatus.STATUS_SUCCEEDED,
                                             GoalStatus.STATUS_CANCELED,
                                             GoalStatus.STATUS_ABORTED):
                    fallo(f'estado final desconocido {resultado.status}')
                    return
            except Exception as error:  # noqa: BLE001
                fallo(error)
                return
            completar()

        def cancelar_handle(h):
            nonlocal pendientes
            pendientes += 1
            try:
                h.get_result_async().add_done_callback(resultado_final)
            except Exception as error:  # noqa: BLE001
                fallo(error)
            try:
                h.cancel_goal_async()
            except Exception as error:  # noqa: BLE001
                fallo(error)

        def cancelar_al_aceptar(futuro):
            try:
                h = futuro.result()
                if h.accepted:
                    cancelar_handle(h)
            except Exception as error:  # noqa: BLE001
                fallo(error)
                return
            completar()

        if anterior is not None and not anterior.done():
            pendientes += 1

            def fin_anterior(futuro):
                try:
                    futuro.result()
                except Exception as error:  # noqa: BLE001
                    fallo(error)
                    return
                completar()

            try:
                anterior.add_done_callback(fin_anterior)
            except Exception as error:  # noqa: BLE001
                fallo(error)
        if handle is not None:
            cancelar_handle(handle)
        if solicitud is not None:
            pendientes += 1
            try:
                solicitud.add_done_callback(cancelar_al_aceptar)
            except Exception as error:  # noqa: BLE001
                fallo(error)
        completar()  # liberar el centinela después de registrar todas las esperas
        return barrera

    def limpiar_costmaps(self):
        for cli in self.clear_cli:
            if cli.service_is_ready():
                cli.call_async(ClearEntireCostmap.Request())
