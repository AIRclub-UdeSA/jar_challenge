"""Lógica del supervisor de movimiento (ADR-0007), sin ROS.

El supervisor es el único que publica /cmd_vel. Recibe pedidos de velocidad de
varias fuentes internas, elige el de mayor prioridad que siga vigente, lo
limita, lo frena si el rectángulo del robot, trasladándose, choca algo cerca y lo
suaviza con
límites de aceleración. Los tiempos son segundos del reloj de ROS (simulado en el
simulador, de pared en el robot).
"""
import math
from dataclasses import dataclass
from enum import Enum

import numpy as np


class Estado(Enum):
    ESPERANDO_MAPA = 'esperando_mapa'
    ACTIVO = 'activo'
    DETENIDO = 'detenido'


@dataclass
class Config:
    vel_lineal_max: float = 0.30      # m/s, en x
    vel_lateral_max: float = 0.20     # m/s, en y
    vel_angular_max: float = 1.00     # rad/s
    acel_lineal_max: float = 1.00     # m/s²
    acel_angular_max: float = 3.00    # rad/s²
    vencimiento_s: float = 0.30       # un pedido sin renovar deja de valer
    medio_largo_m: float = 0.16       # el robot es un rectángulo (x adelante, y izquierda)
    medio_ancho_m: float = 0.11
    margen_m: float = 0.02            # se suma a cada lado del rectángulo
    distancia_freno_m: float = 0.06   # hueco libre por debajo del cual no avanza
    distancia_lenta_m: float = 0.25   # entre freno y lenta, escala la velocidad
    giro_anticipado_s: float = 0.3    # cuánto giro adelante se chequea contra el LiDAR
    giro_max_chequeo_rad: float = 0.2
    # Giros: sólo contacto. Con el margen de traslación (2 cm) no dejaba girar en los lugares
    # angostos y la misión se quedaba 20 s esperando para orientarse.
    margen_giro_m: float = 0.005
    # Patinamiento: la odometría dice que gira y el giróscopo de la IMU dice que no (el robot
    # trabado contra algo). Se cortan los giros un rato: si no, la odometría se va y AMCL se
    # pierde (en práctica: 1.3 rad de error con el robot quieto).
    diferencia_patina_rad_s: float = 0.3
    tiempo_patina_s: float = 0.4
    giro_cortado_s: float = 2.0


class Supervisor:
    """Elige, limita y frena el comando de velocidad.

    `prioridades`: nombres de las fuentes, de mayor a menor prioridad.
    """

    def __init__(self, config, prioridades):
        self.config = config
        self.prioridades = list(prioridades)
        self.estado = Estado.ESPERANDO_MAPA
        self.pedidos = {}                 # fuente -> ((vx, vy, wz), t)
        self.ultimo = (0.0, 0.0, 0.0)
        self.t_ultimo = None
        self.puntos = np.zeros((0, 2))    # obstáculos en el marco del robot (x adelante, y izquierda)
        self.giro_odom = self.giro_imu = None
        self.patina_desde = None
        self.sin_giro_hasta = -math.inf
        # Diagnóstico del último ciclo: qué se pidió, cuánto hueco había y cuánto se frenó.
        self.diag = {'pedido': (0.0, 0.0, 0.0), 'hueco': math.inf, 'escala': 1.0}

    # Entradas -----------------------------------------------------------------
    def mapa_recibido(self):
        if self.estado == Estado.ESPERANDO_MAPA:
            self.estado = Estado.ACTIVO

    def detener(self):
        self.estado = Estado.DETENIDO

    def pedir(self, fuente, vx, vy, wz, t):
        if fuente not in self.prioridades:
            raise ValueError(f'fuente desconocida: {fuente}')
        self.pedidos[fuente] = ((vx, vy, wz), t)

    def actualizar_obstaculos(self, puntos):
        """`puntos`: arreglo N×2 en el marco base del robot, sólo valores finitos."""
        self.puntos = np.asarray(puntos, dtype=float).reshape(-1, 2)

    def actualizar_giros(self, giro_odom=None, giro_imu=None):
        """Velocidad angular que dice la odometría (ruedas) y la que mide la IMU (rad/s)."""
        if giro_odom is not None:
            self.giro_odom = giro_odom
        if giro_imu is not None:
            self.giro_imu = giro_imu

    def _patinando(self, t):
        """¿Las ruedas giran el robot y el robot no gira? Sostenido `tiempo_patina_s`."""
        c = self.config
        if self.giro_odom is None or self.giro_imu is None:
            return False
        if abs(self.giro_odom - self.giro_imu) > c.diferencia_patina_rad_s:
            if self.patina_desde is None:
                self.patina_desde = t
            if t - self.patina_desde >= c.tiempo_patina_s:
                self.sin_giro_hasta = t + c.giro_cortado_s
                self.patina_desde = None
                return True
        else:
            self.patina_desde = None
        return False

    # Salida -------------------------------------------------------------------
    def comando(self, t):
        """Comando a publicar en el instante `t`."""
        if self.estado != Estado.ACTIVO:
            self.ultimo, self.t_ultimo = (0.0, 0.0, 0.0), t
            return self.ultimo
        deseado = self._elegir(t)
        self.diag['pedido'] = deseado
        deseado = self._limitar(*deseado)
        deseado = self._frenar(*deseado)
        self.diag['patina'] = self._patinando(t) or t < self.sin_giro_hasta
        if self.diag['patina']:
            deseado = (deseado[0], deseado[1], 0.0)
        salida = self._suavizar(deseado, t)
        self.ultimo, self.t_ultimo = salida, t
        return salida

    def fuente_activa(self, t):
        for fuente in self.prioridades:
            if fuente in self.pedidos and t - self.pedidos[fuente][1] <= self.config.vencimiento_s:
                return fuente
        return None

    def _elegir(self, t):
        fuente = self.fuente_activa(t)
        return self.pedidos[fuente][0] if fuente else (0.0, 0.0, 0.0)

    def _limitar(self, vx, vy, wz):
        c = self.config
        return (_recortar(vx, c.vel_lineal_max), _recortar(vy, c.vel_lateral_max),
                _recortar(wz, c.vel_angular_max))

    def hueco_libre(self, vx, vy):
        """Distancia libre en la dirección de traslación (inf si no hay nada adelante)."""
        c = self.config
        return hueco_en_direccion(self.puntos, vx, vy, c.medio_largo_m + c.margen_m,
                                  c.medio_ancho_m + c.margen_m, c.margen_m)

    def _frenar(self, vx, vy, wz):
        c = self.config
        libre = self.hueco_libre(vx, vy)
        self.diag['hueco'] = libre
        if libre <= c.distancia_freno_m:
            escala = 0.0
        elif libre < c.distancia_lenta_m:
            escala = (libre - c.distancia_freno_m) / (c.distancia_lenta_m - c.distancia_freno_m)
        else:
            escala = 1.0
        self.diag['escala'] = escala
        # Giros: un giro en el lugar pegado a una pared no se completa, las ruedas patinan, la
        # odometría cree que giró y AMCL se pierde (en práctica: 4 m de error). Se frena el giro si
        # el rectángulo, rotando, toca un punto del LiDAR; la traslación sigue para alejarse.
        self.diag['giro_frenado'] = False
        if abs(wz) > 1e-3:
            angulo = math.copysign(min(c.giro_max_chequeo_rad, abs(wz) * c.giro_anticipado_s), wz)
            if not giro_libre(self.puntos, angulo, c.medio_largo_m + c.margen_giro_m,
                              c.medio_ancho_m + c.margen_giro_m):
                wz = 0.0
                self.diag['giro_frenado'] = True
        return vx * escala, vy * escala, wz

    def _suavizar(self, deseado, t):
        if self.t_ultimo is None:
            dt = 0.0
        else:
            dt = max(0.0, t - self.t_ultimo)
        c = self.config
        vx = _acercar(self.ultimo[0], deseado[0], c.acel_lineal_max * dt)
        vy = _acercar(self.ultimo[1], deseado[1], c.acel_lineal_max * dt)
        wz = _acercar(self.ultimo[2], deseado[2], c.acel_angular_max * dt)
        # Frenar nunca se demora: si el deseado es más chico en módulo, se toma directo.
        vx = deseado[0] if abs(deseado[0]) < abs(vx) and _mismo_signo(vx, deseado[0]) else vx
        vy = deseado[1] if abs(deseado[1]) < abs(vy) and _mismo_signo(vy, deseado[1]) else vy
        return vx, vy, wz


def _recortar(v, maximo):
    return max(-maximo, min(maximo, v))


def _acercar(actual, objetivo, paso):
    if objetivo > actual:
        return min(objetivo, actual + paso)
    return max(objetivo, actual - paso)


def _mismo_signo(a, b):
    return a * b >= 0.0


def scan_a_puntos(ranges, angle_min, angle_increment, range_min, range_max,
                  tx=0.0, ty=0.0, yaw=0.0):
    """Puntos finitos del scan, pasados al marco del robot con la transformada (tx, ty, yaw)."""
    r = np.asarray(ranges, dtype=float)
    ang = angle_min + angle_increment * np.arange(len(r))
    validos = np.isfinite(r) & (r >= range_min) & (r <= range_max)
    r, ang = r[validos], ang[validos] + yaw
    return np.column_stack((tx + r * np.cos(ang), ty + r * np.sin(ang)))


def hueco_en_direccion(puntos, vx, vy, medio_largo, medio_ancho, margen=0.0):
    """Cuánto puede trasladarse hacia (vx, vy) el rectángulo del robot antes de tocar un
    punto (inf si nada). `medio_largo` y `medio_ancho` ya incluyen `margen`.

    Con un círculo, algo al costado del robot frenaba el avance aunque el robot (más angosto
    que largo) pasara: así quedaba trabado al lado de las cajas.

    Un punto en el margen de un eje (fuera del cuerpo real en ese eje) sólo cuenta si el
    movimiento se acerca a él por ese eje: una pared a 1 cm del costado no frena el avance
    paralelo a ella. Un punto ya adentro frena sólo si el movimiento lo mete más por la cara
    por la que entró (el eje en el que está más cerca del borde). Antes frenaba cualquier
    dirección con componente hacia él, y con roces en las esquinas de un hueco justo el
    robot quedaba sin ninguna dirección libre (regresión 3, en tres laberintos de práctica).
    """
    rapidez = math.hypot(vx, vy)
    puntos = np.asarray(puntos, dtype=float).reshape(-1, 2)
    if rapidez < 1e-6 or len(puntos) == 0:
        return math.inf
    d = np.array([vx, vy]) / rapidez
    medios = np.array([medio_largo, medio_ancho])
    hacia = puntos * d                                   # > 0: se acerca por ese eje
    fuera_del_cuerpo = np.abs(puntos) > medios - margen
    puntos = puntos[~np.any(fuera_del_cuerpo & (hacia <= 0.0), axis=1)]
    if len(puntos) == 0:
        return math.inf
    # Para cada eje, el intervalo de avance s en el que el punto queda dentro del rectángulo.
    entra, sale = np.full(len(puntos), -np.inf), np.full(len(puntos), np.inf)
    for eje, medio in ((0, medio_largo), (1, medio_ancho)):
        p, v = puntos[:, eje], d[eje]
        if abs(v) < 1e-9:
            afuera = np.abs(p) > medio
            entra[afuera], sale[afuera] = np.inf, -np.inf
        else:
            a, b = (p - medio) / v, (p + medio) / v
            entra = np.maximum(entra, np.minimum(a, b))
            sale = np.minimum(sale, np.maximum(a, b))
    choca = (entra <= sale) & (sale > 0.0)
    adelante = choca & (entra >= 0.0)
    eje_cara = np.argmin(medios - np.abs(puntos), axis=1)
    mete_mas = (puntos * d)[np.arange(len(puntos)), eje_cara] > 0.0
    adentro = choca & (entra < 0.0) & mete_mas
    if np.any(adentro):
        return 0.0
    if not np.any(adelante):
        return math.inf
    return float(np.min(entra[adelante]))


def direccion_mas_libre(puntos, medio_largo, medio_ancho, preferida=None, direcciones=16,
                        tope=0.5, margen=0.0):
    """Dirección unitaria (x, y) de traslación con más hueco libre, en el marco del robot.

    Más allá de `tope` metros todas valen lo mismo y desempata la cercanía a `preferida`.
    """
    mejor, puntaje_mejor = None, -math.inf
    for k in range(direcciones):
        a = 2.0 * math.pi * k / direcciones
        d = (math.cos(a), math.sin(a))
        puntaje = min(hueco_en_direccion(puntos, d[0], d[1], medio_largo, medio_ancho, margen),
                      tope)
        if preferida is not None:
            puntaje += 0.05 * (d[0] * preferida[0] + d[1] * preferida[1])
        if puntaje > puntaje_mejor:
            mejor, puntaje_mejor = d, puntaje
    return mejor


def giro_para_salir(puntos, medio_largo, medio_ancho, margen, margen_giro, angulo=0.2,
                    hueco_suficiente=0.10, direcciones=16):
    """Sentido de giro (+1, -1) que abre lugar para trasladarse, o 0 si no hace falta girar
    (ya hay una dirección con `hueco_suficiente`) o ningún giro lo mejora.

    Regresión 3: el robot quedó cruzado en diagonal en un hueco de 43 cm entre una
    caja y la pared, con dos esquinas opuestas a 1 cm. Ninguna traslación tenía lugar y el
    giro sí, pero el escape sólo trasladaba. `medio_largo` y `medio_ancho` ya incluyen
    `margen`; el giro se mide como lo frena el supervisor (cuerpo + `margen_giro`).
    """
    p = np.asarray(puntos, dtype=float).reshape(-1, 2)
    if len(p) == 0:
        return 0

    def mejor_hueco(q):
        return max(hueco_en_direccion(q, math.cos(a), math.sin(a), medio_largo, medio_ancho,
                                      margen)
                   for a in np.linspace(0.0, 2.0 * math.pi, direcciones, endpoint=False))

    actual = mejor_hueco(p)
    if actual >= hueco_suficiente:
        return 0
    largo_giro = medio_largo - margen + margen_giro
    ancho_giro = medio_ancho - margen + margen_giro
    sentido, mejor = 0, actual
    for signo in (1, -1):
        t = signo * angulo
        if not giro_libre(p, t, largo_giro, ancho_giro):
            continue
        c, s = math.cos(t), math.sin(t)
        girados = np.column_stack((c * p[:, 0] + s * p[:, 1], -s * p[:, 0] + c * p[:, 1]))
        hueco = mejor_hueco(girados)
        if hueco > mejor + 1e-3:
            sentido, mejor = signo, hueco
    return sentido


def giro_libre(puntos, angulo, medio_largo, medio_ancho):
    """¿El rectángulo, girando `angulo` en el lugar, queda sin puntos del LiDAR adentro?

    Un punto que ya está adentro también frena: es un contacto, y es justo lo que traba el
    giro (ignorarlo dejó al robot girando contra una pared en práctica).
    """
    p = np.asarray(puntos, dtype=float).reshape(-1, 2)
    if len(p) == 0:
        return True
    for t in np.linspace(0.0, angulo, 4)[1:]:
        c, s = math.cos(t), math.sin(t)
        x = c * p[:, 0] + s * p[:, 1]                  # el punto visto desde el robot girado
        y = -s * p[:, 0] + c * p[:, 1]
        if np.any((np.abs(x) <= medio_largo) & (np.abs(y) <= medio_ancho)):
            return False
    return True


def matriz_de_cuaternion(x, y, z, w):
    """Matriz de rotación 3×3 del cuaternión (x, y, z, w)."""
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
