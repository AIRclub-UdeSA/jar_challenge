import math
import time
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty
from std_srvs.srv import Trigger
from equipo_jar.exploracion_estados import (
    DESENCAJANDO,
    RELOCALIZANDO,
    REGRESAR,
    VOLVIENDO,
    APROXIMANDO,
    TERMINADO,
    FINALIZANDO,
)
from equipo_jar.exploracion_geometria import ang_norm, bfs


class RegresoExploracion:

    def tiempo_transcurrido(self):
        return time.monotonic() - self.t_mapa if self.t_mapa else 0.0

    def velocidad_media(self):
        v = self.recorrido / self.t_moviendo if self.t_moviendo > 20.0 else self.cfg['velocidad_estimada']
        return min(0.50, max(0.12, v))

    def campo_casa(self):
        """Distancias por camino a la largada (se recalcula cada 5 s por obstáculos nuevos)."""
        ahora = time.monotonic()
        if self.dist_casa is None or ahora - self.t_dist_casa > 5.0:
            f, c = self.grilla.a_celda(self.largada[0], self.largada[1])
            self.dist_casa = bfs(self.grilla.transitable, f, c)
            self.t_dist_casa = ahora
        return self.dist_casa

    def segundos_para_volver(self, x, y):
        d = self.campo_casa()
        g = self.grilla
        f, c = g.a_celda(x, y)
        if d is None or not g.dentro(f, c) or d[f, c] < 0:
            metros = 1.4 * math.hypot(x - self.largada[0], y - self.largada[1])
        else:
            metros = d[f, c] * g.res
        return 1.3 * metros / self.velocidad_media() + 5.0

    def tiempo_restante(self):
        return self.cfg['duracion_corrida_s'] - self.tiempo_transcurrido()

    def pedir_regreso(self, motivo):
        if self.motivo_regreso is not None or self.estado in (
                VOLVIENDO, APROXIMANDO, TERMINADO, REGRESAR, DESENCAJANDO, RELOCALIZANDO):
            return
        self.motivo_regreso = motivo
        self.get_logger().info(f'Vuelvo a la largada: {motivo}')
        cancelacion = self.cancelar_actual()
        self.cancelacion_cobertura = (
            cancelacion, time.monotonic() + 2.0) if cancelacion is not None else None
        self.goal_actual = None
        if self.objeto_observado is not None:
            self.recordar_inspeccion(self.objeto_observado, False)
            self.limite_observacion(100.0)
        self.objeto_observado = None
        self.ruta_sector.clear()
        self.ruta_regreso.clear()
        self.estado = REGRESAR

    def volver_a_largada(self):
        if self.cancelacion_cobertura is not None:
            cancelacion, limite = self.cancelacion_cobertura
            if not cancelacion.done() and time.monotonic() < limite:
                return
            self.cancelacion_cobertura = None
        if not self.cfg['volver_al_inicio']:
            self.terminar('sin regreso (volver_al_inicio=false)')
            return
        self.fijar_largada(forzar=True)
        pose = self.pose_robot()
        if (self.cfg['modo_exploracion'] == 'sectores_nbv'
                and self.sectorizador is None and pose is not None):
            self.preparar_sectores(pose)
        x, y, yaw = self.largada
        d = math.hypot(x - pose[0], y - pose[1]) if pose else 3.0
        self.estado = VOLVIENDO
        if d < self.cfg['tolerancia_llegada']:
            self.iniciar_aproximacion()
            return
        if self.regreso_directo:
            if pose is None or not self.enviar_goal(x, y, yaw, 1.0, d, etiqueta='regreso directo'):
                self.estado = REGRESAR
            else:
                self.estado = VOLVIENDO
            self.publicar_debug([])
            return
        if self.sectorizador is not None:
            if pose is None:
                self.estado = REGRESAR
                return
            g = self.grilla
            if not self.ruta_regreso:
                fila, col = g.a_celda(pose[0], pose[1])
                fila_casa, col_casa = g.a_celda(x, y)
                def planificar(seguro):
                    ruta = self.sectorizador.ruta_hacia(
                        seguro, (col, fila), (col_casa, fila_casa),
                        self.cfg['paso_maximo_portal_m'])
                    if ruta:
                        ultima, _ = ruta[-1]
                        wx, wy = g.a_mundo(ultima[1], ultima[0])
                        if math.hypot(wx - x, wy - y) <= 0.35:
                            return ruta
                    return []
                (self.ruta_regreso, pasos, d, self.seguro_ruta_regreso,
                 margen_usado) = self.seleccionar_ruta_portales(pose, planificar, self.largada)
                if not self.ruta_regreso:
                    self.get_logger().warn('Sin ruta transitable por portales hasta la largada')
                    self.reintentar_regreso()
                    return
                self.get_logger().info(
                    f'Regreso por {len(self.ruta_regreso)} puntos calculados, '
                    f'{sum(portal for _, portal in self.ruta_regreso)} portal(es), '
                    f'margen de paradas {margen_usado:.2f} m')
            else:
                pasos, d = self.guias_de_ruta(
                    self.ruta_regreso, pose, self.largada, seguro=self.seguro_ruta_regreso)
            if not pasos:
                self.reintentar_regreso()
                return
            x, y, yaw = pasos[-1]
            etiqueta = 'regreso portales'
        else:
            etiqueta = 'regreso'
            pasos = None
        if not self.encarar_y_enviar(x, y, yaw, 1.0, d, etiqueta, VOLVIENDO, pasos):
            self.estado = REGRESAR   # Nav2 no está: se reintenta en el próximo ciclo
        self.publicar_debug([])

    def reintentar_regreso(self):
        self.ruta_regreso.clear()
        if self.registrar_fallo_localizacion(REGRESAR):
            return
        self.reintentos_regreso += 1
        if self.reintentos_regreso > self.cfg['max_reintentos_regreso']:
            if not self.regreso_directo:
                # Por portales no hubo caso: probar con el destino directo a Nav2.
                self.regreso_directo = True
                self.reintentos_regreso = 0
                self.get_logger().warn('El regreso por portales falló varias veces: pruebo con la '
                                       'largada como destino directo de Nav2')
            elif self.tiempo_restante() > 10.0:
                # Mientras quede tiempo se sigue intentando: /done antes de llegar
                # pierde el bono de regreso sin ganar nada.
                self.regreso_directo = False
                self.reintentos_regreso = 0
                self.get_logger().warn(f'Regreso sigue fallando; quedan {self.tiempo_restante():.0f} s: '
                                       'vuelvo a intentar')
            else:
                self.get_logger().error('No pude volver a la largada y se acaba el tiempo: termino donde estoy')
                self.terminar('regreso fallido')
                return
        self.get_logger().warn(f'Regreso falló, reintento {self.reintentos_regreso}')
        self.limpiar_costmaps()
        self.n_atascos += 1
        self.iniciar_escape(REGRESAR)   # despegarse antes de reintentar

    def iniciar_aproximacion(self):
        self.estado = APROXIMANDO
        self.t_aprox = time.monotonic()

    def paso_aproximacion(self):
        """Ajuste fino holonómico hasta < tolerancia_final de la largada (bono ≤ 0.25 m)."""
        t = time.monotonic() - self.t_aprox
        if t < 1.2:          # dejar que el velocity_smoother de Nav2 se calle
            return
        pose = self.pose_robot()
        if pose is None:
            return
        dx, dy = self.largada[0] - pose[0], self.largada[1] - pose[1]
        d = math.hypot(dx, dy)
        if d < self.cfg['tolerancia_final']:
            self.pub_cmd.publish(Twist())
            self.terminar(f'en la largada (error {d:.2f} m)')
            return
        if t > 9.0 or d > 0.6:
            self.pub_cmd.publish(Twist())
            self.terminar(f'aproximación incompleta (error {d:.2f} m)')
            return
        puntos = self.puntos_scan()
        direccion = ang_norm(math.atan2(dy, dx) - pose[2])
        if puntos is None or self.espacio_libre(puntos, direccion) < 0.05:
            self.pub_cmd.publish(Twist())
            return
        v = min(self.cfg['velocidad_aproximacion'], 0.8 * d + 0.03)
        c, s = math.cos(-pose[2]), math.sin(-pose[2])
        cmd = Twist()
        cmd.linear.x = v * (c * dx - s * dy) / d
        cmd.linear.y = v * (s * dx + c * dy) / d
        self.pub_cmd.publish(cmd)

    def terminar(self, detalle):
        if self.estado in (TERMINADO, FINALIZANDO):
            return
        self.cancelar_actual()
        self.pub_cmd.publish(Twist())
        inflacion = getattr(self, 'inflacion', None)
        if inflacion is not None:
            inflacion.restaurar('fin de la corrida')
        if self.cfg['publicar_done'] and self.reportes < int(self.cfg['cantidad_victimas']):
            self.estado = FINALIZANDO
            self.publicar_visto()
            espera = min(self.cfg['espera_reportes_finales_s'], max(
                0.0, self.cfg['duracion_corrida_s'] - self.tiempo_transcurrido() - 0.3))
            self.cierre_reportes = dict(detalle=detalle, limite=time.monotonic() + espera,
                                       futuro=None, listo=None)
            self.get_logger().info(
                f'Antes de /done quedan {int(self.cfg["cantidad_victimas"]) - self.reportes} '
                'reportes: solicito las mejores detecciones y estimaciones disponibles')
            self.paso_reportes_finales()
            return
        self.finalizar_done(detalle)

    def paso_reportes_finales(self):
        """Espera asíncrona y acotada; el robot permanece detenido."""
        cierre, ahora = self.cierre_reportes, time.monotonic()
        self.pub_cmd.publish(Twist())
        if cierre['futuro'] is None and ahora < cierre['limite'] and self.reportes_finales_cli.service_is_ready():
            cierre['futuro'] = self.reportes_finales_cli.call_async(Trigger.Request())
        futuro = cierre['futuro']
        if futuro is not None and futuro.done() and cierre['listo'] is None:
            try:
                respuesta = futuro.result()
                self.get_logger().info(f'Cierre de reportes: {respuesta.message}')
            except Exception as exc:
                self.get_logger().warn(f'No se completó el cierre de reportes: {exc}')
            # Deja que /victimas llegue a los suscriptores antes de emitir /done.
            cierre['listo'] = ahora + 0.15
        if ahora >= cierre['limite'] or (cierre['listo'] is not None and ahora >= cierre['listo']):
            if cierre['listo'] is None:
                self.get_logger().warn('Detector sin respuesta al cierre: continúo con /done por el plazo')
            self.finalizar_done(cierre['detalle'])

    def finalizar_done(self, detalle):
        self.estado = TERMINADO
        self.get_logger().info(
            f'FIN: {self.motivo_regreso or "-"} | {detalle} | cobertura {100 * self.cobertura():.1f}% '
            f'| reportes {self.reportes} | t={self.tiempo_transcurrido():.0f}s')
        if self.cfg['publicar_done']:
            self.pub_done.publish(Empty())
            self.get_logger().info('/done publicado')
