import math
import time
import numpy as np
from geometry_msgs.msg import Twist
from nav2_msgs.msg import SpeedLimit
from rclpy.time import Time
from equipo_jar.exploracion_estados import (
    ESPERANDO_MAPA,
    LOCALIZANDO,
    ACTIVANDO_NAV2,
    ELIGIENDO,
    NAVEGANDO,
    DESENCAJANDO,
    RELOCALIZANDO,
    REGRESAR,
    VOLVIENDO,
    APROXIMANDO,
    TERMINADO,
    FINALIZANDO,
)
from equipo_jar.exploracion_geometria import ang_norm


class MovimientoExploracion:

    ADELANTE, ATRAS, MEDIO_ANCHO = 0.119, 0.181, 0.100

    ESQUINAS = np.array([[ADELANTE, MEDIO_ANCHO], [ADELANTE, -MEDIO_ANCHO],
                         [-ATRAS, -MEDIO_ANCHO], [-ATRAS, MEDIO_ANCHO]])

    def distancia_minima_chasis(self):
        puntos = self.puntos_scan()
        if puntos is None or puntos[0].size == 0:
            return None
        x, y = puntos
        dx = np.maximum(np.maximum(-self.ATRAS - x, x - self.ADELANTE), 0.0)
        dy = np.maximum(np.abs(y) - self.MEDIO_ANCHO, 0.0)
        return float(np.min(np.hypot(dx, dy)))

    def libre_hacia_donde_va(self):
        """Metros libres que tiene el chasis en la dirección en que se está moviendo
        (None si está casi quieto o sin láser). Una pared al costado no cuenta."""
        vel = getattr(self, 'vel_base', None)
        if vel is None or math.hypot(*vel) < 0.05:
            return None
        puntos = self.puntos_scan()
        if puntos is None or puntos[0].size == 0:
            return None
        return self.espacio_libre(puntos, math.atan2(vel[1], vel[0]))

    def publicar_limite_velocidad(self):
        """Velocidad máxima salvo que vaya HACIA un obstáculo: frena según los metros
        libres en la dirección de avance. Pasar al lado de una pared no frena."""
        limite = 100.0
        libre = self.libre_hacia_donde_va()
        if libre is not None:
            if libre < self.cfg['distancia_muy_lenta']:
                limite = float(self.cfg['porcentaje_muy_lenta'])
            elif libre < self.cfg['distancia_lenta']:
                limite = float(self.cfg['porcentaje_lenta'])
        if self.error_loc is not None and self.error_loc > self.cfg['error_loc_malo']:
            limite = min(limite, float(self.cfg['porcentaje_loc_mala']))
        if limite != self.ultimo_limite or int(time.monotonic()) % 2 == 0:
            m = SpeedLimit()
            m.header.stamp = self.get_clock().now().to_msg()
            m.percentage = True
            m.speed_limit = limite
            self.pub_vel.publish(m)
            if limite != self.ultimo_limite and limite < 100.0:
                self.get_logger().debug(f'Límite de velocidad {limite:.0f}%')
            self.ultimo_limite = limite

    def actualizar_inflacion(self):
        """Aviso temprano; el anti-atascos completo sigue disponible si no alcanza."""
        forzar_normal = self.estado in (ESPERANDO_MAPA, LOCALIZANDO, ACTIVANDO_NAV2,
            RELOCALIZANDO, APROXIMANDO, FINALIZANDO, TERMINADO)
        navegando = self.estado in (NAVEGANDO, VOLVIENDO) and self.goal_handle is not None
        pose = self.lookup(self.cfg['odom_frame'], self.cfg['base_frame']) if navegando else None
        ga = self.goal_actual or {}
        restante = ga.get('restante_nav2')
        self.inflacion.paso(pose, navegando, progreso=-restante if restante is not None else None,
            meta=self.goal_id, restaurar=forzar_normal or self.loc_mala_desde is not None)

    def detectar_atasco(self, pose):
        t = self.get_clock().now().nanoseconds * 1e-9      # sim time en Gazebo
        T = self.cfg['tiempo_atascado']
        ga = self.goal_actual
        if ga is not None and ga.get('pasos'):
            pasos = ga['pasos']
            indice = max(0, min(len(pasos) - 1, len(pasos) - ga['pasos_restantes']))
            avance = ga['distancias_guias'][indice] - math.dist(pose[:2], pasos[indice][:2])
            if ga.get('restante_nav2') is not None:
                # Con pocas guías el camino de Nav2 puede alejarse de la recta (rodear
                # una pared): el progreso se mide con lo que le falta según Nav2.
                avance = -ga['restante_nav2']
            mejor = ga['mejor_avance']
            if mejor is None or avance >= mejor + self.cfg['distancia_atascado']:
                ga['mejor_avance'], ga['t_progreso'] = avance, t
            elif t - ga['t_progreso'] >= T:
                self.get_logger().warn('Oscilación: hay movimiento pero no progreso hacia la próxima guía del tramo')
                return True
        self.hist.append((t, pose[0], pose[1], pose[2]))
        while self.hist and t - self.hist[0][0] > T + 2.0:
            self.hist.popleft()
        if not self.hist or t - self.hist[0][0] < T:
            return False
        ventana = [h for h in self.hist if t - h[0] <= T]
        if len(ventana) < 3:
            return False
        # Solo cuenta la traslación: girar en el lugar 6 s navegando tampoco es avanzar.
        desp = max(math.hypot(h[1] - pose[0], h[2] - pose[1]) for h in ventana)
        return desp < self.cfg['distancia_atascado']

    def vigilancia_global(self, pose):
        """Atasco aunque se cambie de goal seguido: `tiempo_atascado_global` s sin trasladarse 12 cm."""
        t = self.get_clock().now().nanoseconds * 1e-9
        self.hist_global.append((t, pose[0], pose[1]))
        T = self.cfg['tiempo_atascado_global']
        while self.hist_global and t - self.hist_global[0][0] > T + 2.0:
            self.hist_global.popleft()
        if t - self.hist_global[0][0] < T:
            return False
        ventana = [h for h in self.hist_global if t - h[0] <= T]
        return max(math.hypot(h[1] - pose[0], h[2] - pose[1]) for h in ventana) < self.cfg['distancia_atascado']

    def marcar_paso_estrecho(self, pose):
        """Si se atascó en un lugar angosto, las próximas rutas no vuelven a pasar por ahí
        (mientras exista otro camino). En la corrida de las 14:59 intentó 3 veces el
        mismo paso de ~30 cm junto a la caja roja y perdió 150 s."""
        g = self.grilla
        f, c = g.a_celda(*pose[:2])
        if not g.dentro(f, c):
            return
        holgura = float(self.holgura_m()[f, c])
        if holgura >= self.cfg.get('holgura_paso_estrecho_m', 0.25):
            return
        k = max(1, int(round(self.cfg.get('radio_paso_fallido_m', 0.15) / g.res)))
        for df in range(-k, k + 1):
            for dc in range(-k, k + 1):
                if df * df + dc * dc <= k * k and g.dentro(f + df, c + dc):
                    self.portales_fallidos.add((c + dc, f + df))
        self.ruta_sector.clear()
        self.get_logger().warn(
            f'Paso angosto en ({pose[0]:.2f}, {pose[1]:.2f}) (holgura {100 * holgura:.0f} cm): '
            'las próximas rutas lo evitan si hay otro camino')

    def resolver_atasco(self, pose):
        self.n_atascos += 1
        volviendo = self.estado == VOLVIENDO
        ga = self.goal_actual
        if self.registrar_fallo_localizacion(REGRESAR if volviendo else ELIGIENDO):
            return
        inflacion = getattr(self, 'inflacion', None)
        if inflacion is not None:
            inflacion.reducir(self.lookup(self.cfg['odom_frame'], self.cfg['base_frame']), 'atasco confirmado')
        self.get_logger().warn(
            f"ATASCADO (sin avanzar) en ({pose[0]:.2f}, {pose[1]:.2f}) "
            f'(#{self.n_atascos} seguido) -> limpio costmaps y '
            + ('recalculo el regreso' if volviendo else 'cambio de punto de vista'))
        self.cancelar_actual()
        self.marcar_paso_estrecho(pose)
        if volviendo:
            self.ruta_regreso.clear()
        if ga is not None and not volviendo:
            if ga.get('etiqueta', '').startswith('portal sector'):
                self.portales_fallidos.add(self.grilla.a_celda(*self.siguiente_guia(ga))[1::-1])
                self.ruta_sector.clear()
            else:
                self.lista_negra.append((ga['x'], ga['y']))
        self.goal_actual = None
        self.limpiar_costmaps()
        # Desencajar con el láser: si el robot quedó pegado a algo, Nav2 (DWB,
        # Spin, BackUp) se niega a moverlo ("Collision Ahead"); acá se lo mueve
        # despacio hacia donde el LiDAR ve más espacio libre.
        self.iniciar_escape(REGRESAR if volviendo else ELIGIENDO)

    def puntos_scan(self, alcance=3.0):
        """Puntos del último /scan en el marco base_footprint (sin los que caen sobre el robot)."""
        s = self.scan
        if s is None or not s.header.frame_id:
            return None
        edad = (self.get_clock().now().nanoseconds - Time.from_msg(s.header.stamp).nanoseconds) / 1e9
        if edad < 0.0 or edad > self.cfg['timeout_scan_s']:
            return None
        tf = self.lookup(self.cfg['base_frame'], s.header.frame_id)
        if tf is None:
            return None
        r = np.asarray(s.ranges, dtype=np.float64)
        validos = np.isfinite(r) & (r >= s.range_min) & (r <= s.range_max)
        if not np.any(validos | np.isposinf(r)):
            return None
        a = s.angle_min + np.arange(r.size) * s.angle_increment
        ok = validos & (r > max(s.range_min, 0.02)) & (r < alcance)
        xl, yl = r[ok] * np.cos(a[ok]), r[ok] * np.sin(a[ok])
        c, sn = math.cos(tf[2]), math.sin(tf[2])
        x, y = tf[0] + c * xl - sn * yl, tf[1] + sn * xl + c * yl
        fuera = ~((x > -0.19) & (x < 0.13) & (np.abs(y) < 0.11))
        return x[fuera], y[fuera]

    def espacio_libre(self, puntos, ang):
        """Metros que el chasis puede avanzar en la dirección `ang` (marco del robot)."""
        x, y = puntos
        d0, d1 = math.cos(ang), math.sin(ang)
        frente = float(np.max(self.ESQUINAS @ np.array([d0, d1])))
        # El chasis no está centrado en base_footprint (sobresale 0.18 m atrás y
        # 0.12 m adelante): el "tubo" que barre al moverse es asimétrico.
        lat_esq = -self.ESQUINAS[:, 0] * d1 + self.ESQUINAS[:, 1] * d0
        lat_min, lat_max = float(lat_esq.min()) - 0.01, float(lat_esq.max()) + 0.01
        proy = x * d0 + y * d1
        lat = -x * d1 + y * d0
        en = (proy > 0.0) & (lat > lat_min) & (lat < lat_max)
        return min(0.6, (float(proy[en].min()) if en.any() else 1.0) - frente)

    def mejor_escape(self):
        puntos = self.puntos_scan()
        if puntos is None:
            return None
        # Preferir, entre las direcciones libres, la que va hacia la ruta deseada
        # (antes elegía "la más libre" y a veces se metía más adentro de un rincón).
        pose = self.pose_robot()
        pref = ang_norm(self.dir_ruta - pose[2]) if (self.dir_ruta is not None and pose) else None
        mejor, mejor_puntaje = None, -1e9
        for ang in np.radians(np.arange(0, 360, 15)):
            libre = self.espacio_libre(puntos, float(ang))
            puntaje = min(libre, 0.6)
            if pref is not None and libre > 0.10:
                puntaje += 0.25 * math.cos(float(ang) - pref)
            if puntaje > mejor_puntaje + 1e-3:
                mejor, mejor_puntaje = (libre, float(ang)), puntaje
        return mejor

    def iniciar_escape(self, siguiente):
        if getattr(self, 'goal_heredado', None) is not None:
            self.cancelar_actual()
        self.estado = DESENCAJANDO
        # Si se traba varias veces seguidas en el mismo lugar, se aleja más.
        dist = self.cfg['distancia_escape'] * min(3, max(1, self.n_atascos))
        self.escape = dict(t0=time.monotonic(), siguiente=siguiente, ang=None, p0=self.pose_robot(),
                           dist=dist)
        self.hist.clear()
        self.hist_global.clear()

    def paso_escape(self):
        e = self.escape
        t = time.monotonic() - e['t0']
        if t < 1.0:          # que el velocity_smoother de Nav2 termine de frenar
            return
        if e['ang'] is None:
            mejor = self.mejor_escape()
            if mejor is None or mejor[0] < 0.03:
                self.fin_escape('sin espacio libre alrededor')
                return
            e['ang'] = mejor[1]
            self.get_logger().info(
                f"Desencajando hacia {math.degrees(mejor[1]):.0f}° (libre {mejor[0]:.2f} m)")
        pose = self.pose_robot()
        movido = math.hypot(pose[0] - e['p0'][0], pose[1] - e['p0'][1]) if pose and e['p0'] else 0.0
        puntos = self.puntos_scan()
        libre = self.espacio_libre(puntos, e['ang']) if puntos is not None else 0.0
        v = self.cfg['velocidad_escape']
        if movido >= e['dist'] or libre < 0.03 or t > 1.0 + 3.0 * e['dist'] / v:
            self.fin_escape(f'movido {movido:.2f} m')
            return
        cmd = Twist()
        cmd.linear.x = v * math.cos(e['ang'])
        cmd.linear.y = v * math.sin(e['ang'])
        self.pub_cmd.publish(cmd)

    def fin_escape(self, detalle):
        self.pub_cmd.publish(Twist())
        self.get_logger().info(f'Fin del desencaje: {detalle}')
        self.estado = self.escape['siguiente']
        self.escape = None
        self.hist.clear()
        if self.giro_pendiente is not None:
            angulo, siguiente = self.giro_pendiente
            self.girar(angulo, siguiente)       # una sola vez: giro_pendiente sigue puesto
            self.giro_pendiente = None
