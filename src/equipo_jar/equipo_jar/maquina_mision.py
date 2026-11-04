"""Máquina de estados de la misión (ADR-0008), sin ROS.

    ESPERANDO_MAPA → LOCALIZANDO → EXPLORANDO ⇄ ORIENTANDO
                                       ↓
                                  REGRESANDO → AJUSTANDO → TERMINADO

En cada ciclo el nodo arma unas `Entradas` y la máquina devuelve las
`Acciones` que hay que ejecutar. El tiempo `t` son segundos desde /map.

Paso angosto: si el robot queda trabado frente a un hueco angosto (Nav2 tiene plan pero no
avanza) con algo cerca, la misión cancela el goal y pasa ella, despacio y deslizándose con el
mecanum: sigue el plan de Nav2 y se centra con los puntos crudos del LiDAR (paso_angosto.py).
Al dejar atrás el hueco le devuelve el goal a Nav2; si no avanza, lo descarta y escapa.

Escape: si Nav2 falla (o el robot queda trabado) con un obstáculo muy cerca,
antes de pedir otro goal el robot se traslada hacia donde hay más lugar libre
(o, si no se sabe, en línea recta lejos del punto más cercano del LiDAR). Pasa
cuando un obstáculo que no está en el mapa aparece tarde y el robot ya quedó
metido en la zona prohibida del costmap: ahí Nav2
no planifica ("Starting point in lethal space") y girar en el lugar choca.

Cruzar huecos (r9, `cruzar_huecos`): Nav2 decide si un hueco se pasa con el costmap, que lo
cierra (ver huecos.py); el LiDAR lo mide bien. Trabado, con Nav2 sin camino o a punto de tapar
un hueco imposible, si hay un hueco medido pasable cerca con el robot de un lado y el goal del
otro, la maniobra de paso angosto lo cruza con un plan recto propio (antes, centro, después)
en vez del de Nav2. Si no hay ninguno medido y el goal se iba a descartar, primero va a mirar
el cuello del camino en el mapa estático (`buscar_cuello`, del nodo) y recién ahí decide.
"""
import dataclasses
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np

from equipo_jar import paso_angosto
from equipo_jar.politica_reportes import PoliticaReportes

Pose = Tuple[float, float, float]


class Estado(Enum):
    ESPERANDO_MAPA = 'esperando_mapa'
    LOCALIZANDO = 'localizando'
    EXPLORANDO = 'explorando'
    ORIENTANDO = 'orientando'
    REGRESANDO = 'regresando'
    AJUSTANDO = 'ajustando'
    TERMINADO = 'terminado'


@dataclass
class Config:
    duracion_s: float = 600.0
    radio_descarte_m: float = 0.3            # un goal cerca de uno que falló no se repite
    vel_giro_localizacion: float = 0.6       # rad/s
    giro_localizacion_rad: float = 6.3       # una vuelta
    tiempo_max_localizacion_s: float = 15.0
    tolerancia_orientacion_rad: float = 0.15
    ganancia_giro: float = 1.5
    vel_giro_max: float = 0.8
    tiempo_max_orientacion_s: float = 8.0
    tiempo_giro_trabado_s: float = 1.5       # orientando sin que baje el error: se deja así
    tiempo_max_goal_s: float = 60.0
    tiempo_inmovil_s: float = 20.0           # el reglamento corta a los 60 s
    vel_giro_desatasco: float = 0.6
    tiempo_desatasco_s: float = 3.0
    vel_media_regreso: float = 0.15          # m/s, para estimar cuánto tarda volver
    factor_camino: float = 1.5               # largo del camino / distancia en línea recta
    margen_regreso_s: float = 30.0
    margen_final_s: float = 10.0
    intentos_regreso: int = 2
    distancia_max_ajuste_m: float = 0.5
    tolerancia_ajuste_m: float = 0.05
    ganancia_ajuste: float = 1.0
    vel_ajuste_max: float = 0.10
    tiempo_max_ajuste_s: float = 12.0
    distancia_escape_m: float = 0.30         # obstáculo a menos (desde el centro): escapar
    vel_escape: float = 0.08                 # m/s, alejándose del obstáculo (usa el lateral)
    tiempo_max_escape_s: float = 4.0
    vel_giro_escape: float = 0.3             # rad/s, si el robot quedó cruzado en un hueco
    # Huecos angostos (ADR-0008): por defecto no se pelean, se bloquean y se busca otro camino.
    # Según el ancho que mide el LiDAR a lo largo del plan:
    #   < ancho_imposible_m: no entra, se bloquea y se busca otro camino;
    #   < ancho_min_hueco_m: maniobra de paso angosto (paso_angosto.py), si está activada;
    #   el resto, Nav2. En los laberintos de práctica los cubos dejan pasos de ~0.39 m que son el único acceso a
    #   varias salas: bloquear todo lo menor a 0.45 dejaba medio laberinto sin explorar.
    usar_paso_angosto: bool = True
    ancho_imposible_m: float = 0.34          # robot 0.22 + márgenes
    ancho_min_hueco_m: float = 0.45
    # Más angosto que esto no es un hueco: es el plan (en `map`, corrido por el error de AMCL)
    # pisando una esquina que el LiDAR ve. En práctica se taparon "huecos" de 0.00-0.05 m que
    # cerraban la salida (laberinto_s1 y s2, 2026-10-28). 0 lo apaga.
    ancho_minimo_creible_m: float = 0.0
    no_repetir_tapon: bool = False           # no volver a tapar donde ya se tapó (se encerraba)
    # Volviendo, si se agotan los intentos y hay tapones puestos, se borran una vez y se
    # reintenta: un tapón propio puede ser la única salida (laberinto_s4, cruzó un hueco de
    # 0.35 m a un bolsillo cuya salida había tapado).
    liberar_tapones_al_volver: bool = False
    distancia_inicio_paso_m: float = 0.6     # la maniobra arranca con el angosto a esto
    tiempo_max_sin_goal_s: float = 20.0      # explorando sin ningún goal usable: se vuelve
    pausa_inspeccion_s: float = 1.5          # quieto mirando un candidato (la percepción vota)
    # En vez de la pausa fija, quieto mirando hasta que la percepción decida (víctima o
    # descartado), con este máximo: con 1 s se iba sin votar y había que volver.
    esperar_decision: bool = False
    espera_max_decision_s: float = 4.0
    tapon_hasta_m: float = 0.8               # tramo del plan, adelante, que se tapa
    mirar_desde_m: float = 0.15              # tramo del plan que se revisa antes de entrar
    mirar_hasta_m: float = 0.9
    tapon_antes_m: float = 0.15              # el tapón empieza esto antes del punto angosto
    tapon_despues_m: float = 0.30
    espera_tras_bloqueo_s: float = 2.0       # que el costmap global tome el tapón
    tapon_medio_ancho_m: float = 0.15
    tiempo_trabado_paso_s: float = 5.0       # quieto con un goal y algo cerca: hueco angosto
    distancia_paso_m: float = 0.40           # "algo cerca" para el paso angosto
    tiempo_max_paso_s: float = 40.0
    tiempo_sin_progreso_paso_s: float = 15.0  # sin avanzar 5 cm de plan en este tiempo: se rinde
    avance_paso_m: float = 0.7               # de plan dejado atrás para devolverle el goal a Nav2
    largo_plan_paso_m: float = 2.5           # cuánto del plan de Nav2 se guarda para la maniobra
    libre_fin_paso_m: float = 0.30           # para devolverle el goal a Nav2, nada más cerca
    intentos_paso: int = 3                   # maniobras por goal (el regreso es siempre el mismo)
    # Una maniobra que se rinde casi sin avanzar marca el lugar: ahí no se vuelve a intentar
    # (se tapa el hueco), aunque el goal sea otro. El explorador proponía el mismo candidato
    # desde otra pose, el contador por goal volvía a cero y el robot se quedaba en el lugar
    # hasta que el juez cortaba por inmovilidad.
    avance_fallido_m: float = 0.10
    radio_lugar_fallido_m: float = 0.40
    # Una maniobra que "cruza" sin avanzar (el plan de Nav2 era de 3 puntos y ya no quedaba nada)
    # descarta el goal: si no, le devolvía el goal a Nav2, que se volvía a trabar, y así hasta el
    # vigilante (laberinto_puertas). No marca el lugar: el hueco puede ser la única entrada.
    descartar_paso_sin_avance: bool = False
    # Recuperaciones del BT de Nav2 (spin, backup, wait) que se le dejan correr a un goal antes de
    # cancelarlo y tomarlo como una falla (0: sin límite), sólo explorando. Sin plan o con el
    # controlador agotado, el BT giraba 50-75 s por goal sin ir a ningún lado (laberinto_puertas,
    # 23 recuperaciones).
    max_recuperaciones_nav: int = 0
    # Red de seguridad: tanto tiempo sin trasladarse (explorando o volviendo) corta lo que esté
    # haciendo, descarta el goal y escapa. El juez corta a los 60 s.
    tiempo_sin_moverse_s: float = 30.0
    # Trabado, con Nav2 fallando o con un paso angosto que no avanzó a menos de esto del goal:
    # se lo da por alcanzado (a orientarse) en vez de descartarlo. En laberinto_puertas el robot
    # quedó a 0.22 m de una inspección, una maniobra de 4 puntos "no avanzó", se descartó el
    # goal y el candidato perdió su último intento. 0: como antes.
    radio_llegada_m: float = 0.0
    # Nav2 pide girar en el lugar y el supervisor no lo deja (el rectángulo, girando, toca algo)
    # con el robot quieto: tanto así y escapa, sin soltar el goal (Nav2 sigue cuando hay lugar).
    # Con robot_radius 0.12 Nav2 deja el centro a 0.12 de una esquina y girar necesita ~0.19:
    # en laberinto_puertas_cfg_r8_2_2137, 24 s frenado al lado de la caja de B hasta el
    # vigilante. Apagado: como antes.
    escapar_giro_frenado: bool = False
    tiempo_giro_frenado_s: float = 2.0
    espera_escape_giro_s: float = 5.0        # entre un escape de estos y el siguiente
    # Cruzar huecos medidos con el LiDAR (ver arriba). Apagado: como antes.
    cruzar_huecos: bool = False
    distancia_max_hueco_m: float = 1.2       # hueco medido a menos del robot
    antes_hueco_m: float = 0.40              # el plan propio arranca acá antes del centro
    despues_hueco_m: float = 0.50            # y termina acá del otro lado
    # Antes de descartar un goal sin camino: ir a medir el cuello (necesita cruzar_huecos).
    medir_antes_de_descartar: bool = False
    antes_cuello_m: float = 0.60             # dónde pararse a medir, antes del cuello
    espera_medicion_s: float = 2.0           # quieto mirando el cuello
    tiempo_max_medir_s: float = 40.0         # yendo a medir: más, se descarta el goal


@dataclass
class Entradas:
    t: Optional[float]                       # s desde /map; None antes del mapa
    pose: Optional[Pose] = None              # (x, y, yaw) en `map`
    yaw_odom: Optional[float] = None
    nav_libre: bool = True                   # no hay un goal de Nav2 en curso
    nav_resultado: Optional[str] = None      # 'ok' o 'falla' del último goal, una sola vez
    goal_exploracion: Optional[Pose] = None
    goal_es_inspeccion: bool = False         # el goal es para mirar de frente un candidato
    inspeccion_id: Optional[int] = None      # qué candidato (de /jar/candidatos)
    candidatos: list = field(default_factory=list)   # /jar/candidatos: [{'id', 'estado', ...}]
    exploracion_terminada: bool = False
    objetivos: list = field(default_factory=list)
    quieto_s: float = 0.0
    obstaculo: Optional[Tuple[float, float]] = None   # punto más cercano del LiDAR, marco del robot
    salida: Optional[Tuple[float, float]] = None      # dirección unitaria con más lugar libre
    giro_salida: int = 0                     # +1/-1: girar abre lugar (ninguna traslación tiene)
    plan: Optional[list] = None              # plan de Nav2 en `map`, [(x, y), ...]
    pose_odom: Optional[Pose] = None         # pose en `odom`: no salta cuando AMCL corrige
    puntos: Optional[object] = None          # puntos del LiDAR en el marco del robot (N×2)
    huecos: list = field(default_factory=list)   # pasables, en `map`: [{'a', 'b', 'ancho'}]
    giro_frenado_s: float = 0.0              # s seguidos con el giro frenado por el supervisor


@dataclass
class Acciones:
    cmd: Optional[Tuple[float, float, float]] = None   # pedido para /jar/cmd_vel_mision
    goal: Optional[Pose] = None
    cancelar: bool = False
    descartar: Optional[Tuple[float, float]] = None
    visitado: Optional[Pose] = None          # goal de exploración al que se llegó y se miró
    avisos: List[str] = field(default_factory=list)   # para el log del nodo
    bloquear: List[Tuple[float, float]] = field(default_factory=list)  # puntos en `map`
    reportes: List[Tuple[float, float]] = field(default_factory=list)
    liberar_tapones: bool = False            # borrar todos los tapones puestos
    done: bool = False


def envolver(a):
    return math.atan2(math.sin(a), math.cos(a))


def hueco_hacia(pose, goal, huecos, distancia_max):
    """El hueco pasable por el que conviene cruzar para ir de `pose` a `goal`: a menos de
    `distancia_max` del robot y con el robot de un lado y el goal del otro (el de menor camino
    robot → hueco → goal). Devuelve (centro, normal hacia el goal, ancho) o None."""
    mejor = None
    for h in huecos:
        a, b = np.asarray(h['a'], dtype=float), np.asarray(h['b'], dtype=float)
        centro = (a + b) / 2.0
        d = b - a
        largo = float(np.hypot(*d))
        if largo < 1e-6:
            continue
        n = np.array([-d[1], d[0]]) / largo
        lado_robot = float(np.dot(np.asarray(pose[:2]) - centro, n))
        lado_goal = float(np.dot(np.asarray(goal[:2]) - centro, n))
        cerca = float(np.hypot(*(np.asarray(pose[:2]) - centro)))
        if lado_robot * lado_goal >= 0.0 or cerca > distancia_max:
            continue
        if lado_robot > 0.0:
            n = -n
        costo = cerca + float(np.hypot(*(np.asarray(goal[:2]) - centro)))
        if mejor is None or costo < mejor[0]:
            mejor = (costo, (float(centro[0]), float(centro[1])), (float(n[0]), float(n[1])),
                     float(h['ancho']))
    return None if mejor is None else mejor[1:]


def plan_por_hueco(pose, centro, normal, antes, despues, paso=0.05):
    """Plan recto en `map`: del robot a `antes` m antes del centro, el centro y `despues` m
    del otro lado, un punto cada `paso` m."""
    puntos = [np.asarray(pose[:2], dtype=float),
              np.asarray(centro) - antes * np.asarray(normal),
              np.asarray(centro, dtype=float),
              np.asarray(centro) + despues * np.asarray(normal)]
    plan = [tuple(puntos[0])]
    for p0, p1 in zip(puntos[:-1], puntos[1:]):
        n = max(1, int(math.ceil(float(np.hypot(*(p1 - p0))) / paso)))
        plan += [tuple(p0 + (p1 - p0) * k / n) for k in range(1, n + 1)]
    return [(float(x), float(y)) for x, y in plan]


class Mision:
    def __init__(self, config, cupos, radio_union, config_paso=None):
        self.config = config
        self.config_paso = config_paso or paso_angosto.ConfigPaso()
        self.politica = PoliticaReportes(cupos, radio_union)
        self.estado = Estado.ESPERANDO_MAPA
        self.t_estado = 0.0
        self.largada: Optional[Pose] = None
        self.goal_actual: Optional[Pose] = None
        self.t_goal = 0.0
        self.yaw_objetivo = 0.0
        self.descartados = []     # goals que fallaron: no se vuelven a pedir
        self.goal_llegado: Optional[Pose] = None
        self.goal_inspeccion = False             # el goal actual es una inspección
        self.candidato_inspeccion = None         # id del candidato que se va a mirar
        self.mirando_hasta = None                # quieto mirando un candidato hasta este t
        self.giro_ref = None                     # (mejor error de orientación, cuándo)
        self.maniobra = None          # paso angosto en curso (dict)
        self.intentos_paso = {}       # goal -> maniobras ya intentadas
        self.tapados = []             # puntos angostos ya tapados (en `map`)
        self.hubo_tapon = False       # se puso al menos un tapón (y no se liberaron)
        self.tapones_liberados = False
        self.lugares_fallidos = []    # donde una maniobra no avanzó (en `map`)
        self.t_vigilante = -1e9       # último corte del vigilante (o fin de la localización)
        self.t_sin_goal = None
        self.t_bloqueo = -1e9
        self.giro = 0.0
        self.yaw_previo = None
        self.desatasco_hasta = None
        self.fin_desatasco = -1e9
        self.escape_hasta = None
        self.regreso_enviado = False
        self.intentos = 0
        self.motivo = ''
        # Ir a medir antes de descartar: el nodo pone `buscar_cuello(pose, goal)` → (punto,
        # dirección) del lugar más angosto del camino en el mapa estático, o None.
        self.buscar_cuello = None
        self.midiendo = None          # {'goal', 'fase', 'hasta', 't0'}
        self.t_escape_giro = -1e9
        self.goals_medidos = []       # goals para los que ya se fue a medir (no se repite)

    # Utilidades ---------------------------------------------------------------
    def _cambiar(self, estado, t, motivo=''):
        self.estado, self.t_estado = estado, t
        if motivo:
            self.motivo = motivo

    def tiempo_restante(self, t):
        return self.config.duracion_s - t

    def tiempo_regreso(self, pose):
        c = self.config
        if pose is None or self.largada is None:
            return c.margen_regreso_s
        d = math.hypot(pose[0] - self.largada[0], pose[1] - self.largada[1])
        return d * c.factor_camino / c.vel_media_regreso + c.margen_regreso_s

    def _enviar(self, a, goal, t):
        a.goal = goal
        self.goal_actual, self.t_goal = goal, t

    def _soltar_goal(self, a):
        if self.goal_actual is not None:
            a.cancelar = True
        self.goal_actual = None

    def _quieto_con_goal(self, e):
        """Segundos sin trasladarse desde que se mandó el goal actual: lo de antes (por ejemplo
        el giro de localización) no cuenta contra un goal recién mandado."""
        return min(e.quieto_s, e.t - self.t_goal)

    def _obstaculo_cerca(self, e):
        return (e.obstaculo is not None
                and math.hypot(*e.obstaculo) < self.config.distancia_escape_m)

    def _iniciar_escape(self, e):
        if self._obstaculo_cerca(e):
            self.escape_hasta = e.t + self.config.tiempo_max_escape_s

    def _escapando(self, e, a):
        """Si hay un escape en curso, pide alejarse del obstáculo y devuelve True."""
        if self.escape_hasta is None:
            return False
        if e.t >= self.escape_hasta or not self._obstaculo_cerca(e):
            self.escape_hasta = None
            a.cmd = (0.0, 0.0, 0.0)
            return False
        if e.giro_salida:
            # Cruzado en un hueco (maze_4): no hay traslación con lugar, pero girar la abre.
            a.cmd = (0.0, 0.0, e.giro_salida * self.config.vel_giro_escape)
            return True
        v = self.config.vel_escape
        if e.salida is not None:
            # En una esquina, alejarse del punto más cercano mete al robot contra la otra pared.
            a.cmd = (e.salida[0] * v, e.salida[1] * v, 0.0)
            return True
        ox, oy = e.obstaculo
        d = math.hypot(ox, oy)
        a.cmd = (-ox / d * v, -oy / d * v, 0.0) if d > 1e-6 else (-v, 0.0, 0.0)
        return True

    # Huecos angostos -----------------------------------------------------------
    def _hueco_angosto(self, e, a, goal):
        """Trabado frente a un hueco: lo cruza (paso angosto) o lo bloquea. True si actuó."""
        c = self.config
        ancho = None
        if e.plan and e.pose is not None and e.puntos is not None:
            ancho, _ = paso_angosto.ancho_en_plan(paso_angosto.a_marco_robot(e.plan, e.pose),
                                                  e.puntos, hasta=c.tapon_hasta_m)
        if ancho is not None and (ancho >= c.ancho_min_hueco_m or ancho < c.ancho_minimo_creible_m):
            return False                          # no es un hueco angosto: sigue lo normal
        fallido = e.pose is not None and self._lugar_fallido(e.pose)
        if c.usar_paso_angosto and (ancho is None or ancho >= c.ancho_imposible_m) and not fallido:
            return self._iniciar_paso(e, a, goal, ancho=ancho)
        return self._bloquear_hueco(e, a, goal, forzar=fallido)

    def _hueco_adelante(self, e, a, goal):
        """Antes de entrar: si el plan va a pasar por un hueco más angosto que
        `ancho_min_hueco_m` (medido con el LiDAR), lo tapa y pide otro camino. True si actuó."""
        c = self.config
        if (goal is None or not e.plan or e.pose is None
                or e.puntos is None or e.t - self.t_bloqueo < c.espera_tras_bloqueo_s):
            return False
        plan_robot = paso_angosto.a_marco_robot(e.plan, e.pose)
        ancho, donde = paso_angosto.ancho_en_plan(plan_robot, e.puntos, desde=c.mirar_desde_m,
                                                  hasta=c.mirar_hasta_m)
        if ancho is None or ancho >= c.ancho_min_hueco_m or ancho < c.ancho_minimo_creible_m:
            return False
        c_, s_ = math.cos(e.pose[2]), math.sin(e.pose[2])
        punto = (e.pose[0] + c_ * donde[0] - s_ * donde[1],
                 e.pose[1] + s_ * donde[0] + c_ * donde[1])
        if (c.usar_paso_angosto and ancho >= c.ancho_imposible_m
                and not self._lugar_fallido(punto)):
            # Entra, justo: la misión lo cruza cuando el angosto está cerca.
            if float(np.hypot(*donde)) > c.distancia_inicio_paso_m:
                return False
            return self._iniciar_paso(e, a, goal, ancho=ancho)
        if self._cruzar_por_hueco(e, a, goal):
            return True                           # hay uno medido que sí se pasa: por ahí
        if any(math.hypot(punto[0] - x, punto[1] - y) < 0.3 for x, y in self.tapados):
            return False                          # ya tapado: el plan nuevo todavía no llegó
        distancia = float(np.hypot(*donde))
        a.bloquear = paso_angosto.tapon(e.plan, e.pose,
                                        desde=max(0.0, distancia - c.tapon_antes_m),
                                        hasta=distancia + c.tapon_despues_m,
                                        medio_ancho=c.tapon_medio_ancho_m)
        self.tapados.append(punto)
        self.hubo_tapon = True
        self.t_bloqueo = e.t
        self._soltar_goal(a)
        if self.estado == Estado.EXPLORANDO:
            a.descartar = goal[:2]
            self.descartados.append(a.descartar)
        else:
            self.regreso_enviado = False
        a.avisos.append(f'hueco angosto de {ancho:.2f} m adelante en ({punto[0]:.2f}, '
                        f'{punto[1]:.2f}): bloqueado antes de entrar, se busca otro camino')
        return True

    def _bloquear_hueco(self, e, a, goal, forzar=False):
        """Si el plan pasa por un hueco más angosto que `ancho_min_hueco_m`, lo tapa en el
        costmap global (el planificador busca otro camino y el explorador no manda goals que
        sólo se alcanzan por ahí), descarta el goal y se aleja."""
        c = self.config
        if not e.plan or e.pose is None or e.puntos is None:
            return False
        plan_robot = paso_angosto.a_marco_robot(e.plan, e.pose)
        ancho, _ = paso_angosto.ancho_en_plan(plan_robot, e.puntos, hasta=c.tapon_hasta_m)
        limite = c.ancho_imposible_m if c.usar_paso_angosto else c.ancho_min_hueco_m
        if not forzar and (ancho is None or ancho >= limite or ancho < c.ancho_minimo_creible_m):
            return False
        if c.no_repetir_tapon:
            if any(math.hypot(e.pose[0] - x, e.pose[1] - y) < 0.3 for x, y in self.tapados):
                return False                      # ya se tapó acá: no encerrarse
            self.tapados.append(tuple(e.pose[:2]))
        a.bloquear = paso_angosto.tapon(e.plan, e.pose, c.tapon_hasta_m, c.tapon_medio_ancho_m)
        self.hubo_tapon = True
        self._soltar_goal(a)
        if self.estado == Estado.EXPLORANDO:
            a.descartar = goal[:2]
            self.descartados.append(a.descartar)
        else:
            self.regreso_enviado = False          # replanifica la vuelta por otro lado
        self._iniciar_escape(e)
        medida = '?' if ancho is None else f'{ancho:.2f}'
        a.avisos.append(f'hueco angosto de {medida} m frente a ({e.pose[0]:.2f}, '
                        f'{e.pose[1]:.2f}): bloqueado, se busca otro camino')
        return True

    # Paso angosto --------------------------------------------------------------
    def _iniciar_paso(self, e, a, goal, ancho=None):
        """Si se puede, la misión toma el control para cruzar el hueco. True si arrancó.

        Con `ancho` (medido antes de entrar) no hace falta tener algo cerca todavía.
        """
        c = self.config
        cerca = e.obstaculo is not None and math.hypot(*e.obstaculo) <= c.distancia_paso_m
        if (goal is None or self.intentos_paso.get(goal, 0) >= c.intentos_paso
                or not e.plan or e.pose is None or e.puntos is None
                or (ancho is None and not cerca) or self._lugar_fallido(e.pose)):
            return False
        plan_robot = paso_angosto.a_marco_robot(e.plan, e.pose)
        i0 = paso_angosto.indice_actual(plan_robot)
        tramos = np.hypot(*np.diff(np.asarray(e.plan, dtype=float)[i0:], axis=0).T)
        fin = i0 + 1 + int(np.searchsorted(np.cumsum(tramos), c.largo_plan_paso_m))
        plan = [tuple(p) for p in np.asarray(e.plan, dtype=float)[:fin + 1]]
        if len(plan) - i0 < 3:
            return False
        plan_map = plan
        self.intentos_paso[goal] = self.intentos_paso.get(goal, 0) + 1
        avance0 = paso_angosto.avance_en_plan(plan_robot)
        # El plan se sigue en `odom`: mientras cruza, una corrección de AMCL no lo descoloca.
        marco = 'map'
        if e.pose_odom is not None:
            en_robot = paso_angosto.a_marco_robot(plan, e.pose)
            co, so = math.cos(e.pose_odom[2]), math.sin(e.pose_odom[2])
            plan = [(e.pose_odom[0] + co * x - so * y, e.pose_odom[1] + so * x + co * y)
                    for x, y in en_robot]
            marco = 'odom'
        self.maniobra = {'plan': plan, 'marco': marco, 'goal': goal, 'plan_map': plan_map,
                         'hasta': e.t + c.tiempo_max_paso_s,
                         'avance0': avance0, 'mejor': 0.0, 't_mejor': e.t}
        self._soltar_goal(a)
        a.avisos.append(f'paso angosto {self.intentos_paso[goal]}: inicio en '
                        f'({e.pose[0]:.2f}, {e.pose[1]:.2f}) hacia ({plan[-1][0]:.2f}, '
                        f'{plan[-1][1]:.2f}), '
                        f'hueco de {"?" if ancho is None else f"{ancho:.2f}"} m, '
                        f'plan de {len(plan)} puntos en {marco}')
        return True

    def _escapar_giro_frenado(self, e, a):
        """Con `escapar_giro_frenado`: Nav2 no puede girar ahí; escapa sin soltar el goal."""
        c = self.config
        if (not c.escapar_giro_frenado or self.goal_actual is None
                or self.escape_hasta is not None
                or min(e.giro_frenado_s, e.quieto_s) < c.tiempo_giro_frenado_s
                or e.t - self.t_escape_giro < c.espera_escape_giro_s):
            return False
        self._iniciar_escape(e)
        if self.escape_hasta is None:
            return False                          # nada cerca: no hay de qué alejarse
        self.t_escape_giro = e.t
        a.avisos.append(f'giro frenado {e.giro_frenado_s:.0f} s: escapa y sigue con el goal')
        return True

    def _cerca_del_goal(self, e, goal):
        r = self.config.radio_llegada_m
        return (r > 0 and goal is not None and e.pose is not None
                and self.estado == Estado.EXPLORANDO
                and math.hypot(e.pose[0] - goal[0], e.pose[1] - goal[1]) < r)

    def _llegar(self, a, goal, t, motivo):
        """El goal se da por alcanzado aunque Nav2 no lo haya dicho: a orientarse."""
        self._soltar_goal(a)
        self.yaw_objetivo = goal[2]
        self.goal_llegado = goal
        self._cambiar(Estado.ORIENTANDO, t)
        a.avisos.append(f'goal ({goal[0]:.2f}, {goal[1]:.2f}) dado por alcanzado: {motivo}')
        return True

    def _cruzar_por_hueco(self, e, a, goal):
        """Con `cruzar_huecos`: si hay un hueco medido pasable entre el robot y el goal, arranca
        el paso angosto por él con un plan propio. True si arrancó."""
        c = self.config
        if not c.cruzar_huecos or goal is None or e.pose is None or not e.huecos:
            return False
        h = hueco_hacia(e.pose, goal, e.huecos, c.distancia_max_hueco_m)
        if h is None:
            return False
        centro, normal, ancho = h
        plan = plan_por_hueco(e.pose, centro, normal, c.antes_hueco_m, c.despues_hueco_m)
        if not self._iniciar_paso(dataclasses.replace(e, plan=plan), a, goal, ancho=ancho):
            return False
        a.avisos.append(f'cruza por el hueco medido de {ancho:.2f} m en '
                        f'({centro[0]:.2f}, {centro[1]:.2f})')
        return True

    def _ir_a_medir(self, e, a, goal):
        """Con `medir_antes_de_descartar`: en vez de descartar un goal sin camino, va a mirar el
        cuello del camino en el mapa estático. True si salió a medir."""
        c = self.config
        if (not (c.cruzar_huecos and c.medir_antes_de_descartar) or goal is None
                or e.pose is None or self.buscar_cuello is None
                or any(math.hypot(goal[0] - x, goal[1] - y) < 0.05
                       for x, y in self.goals_medidos)):
            return False
        cuello = self.buscar_cuello(e.pose, goal)
        if cuello is None:
            return False
        (px, py), (dx, dy) = cuello
        self.goals_medidos.append(goal[:2])
        destino = (px - c.antes_cuello_m * dx, py - c.antes_cuello_m * dy, math.atan2(dy, dx))
        self._soltar_goal(a)
        self._enviar(a, destino, e.t)
        self.midiendo = {'goal': goal, 'fase': 'yendo', 't0': e.t, 'hasta': None}
        a.avisos.append(f'sin camino a ({goal[0]:.2f}, {goal[1]:.2f}): va a medir el cuello en '
                        f'({px:.2f}, {py:.2f}) desde ({destino[0]:.2f}, {destino[1]:.2f})')
        return True

    def _midiendo(self, e, a):
        """Si salió a medir un cuello, lo maneja y devuelve True."""
        m = self.midiendo
        if m is None:
            return False
        c = self.config
        if m['fase'] == 'yendo':
            if e.nav_resultado == 'ok':
                self.goal_actual = None
                m['fase'], m['hasta'] = 'mirando', e.t + c.espera_medicion_s
            elif e.nav_resultado == 'falla' or e.t - m['t0'] > c.tiempo_max_medir_s:
                self._soltar_goal(a)
                self.midiendo = None
                return self._descartar_medido(e, a, m['goal'], 'no llegó a medir el cuello')
            return True
        a.cmd = (0.0, 0.0, 0.0)
        if e.t < m['hasta']:
            return True
        self.midiendo = None
        if self._cruzar_por_hueco(e, a, m['goal']):
            return True
        return self._descartar_medido(e, a, m['goal'], 'medido: no hay hueco pasable')

    def _descartar_medido(self, e, a, goal, motivo):
        a.avisos.append(f'goal ({goal[0]:.2f}, {goal[1]:.2f}) descartado: {motivo}')
        if self.estado == Estado.EXPLORANDO:
            a.descartar = goal[:2]
            self.descartados.append(a.descartar)
        else:
            self.regreso_enviado = False
        self._iniciar_escape(e)
        return True

    def _pasando(self, e, a):
        """Si hay un paso angosto en curso, lo maneja y devuelve True."""
        if self.maniobra is None:
            return False
        c, p = self.config, self.maniobra
        pose = e.pose_odom if p['marco'] == 'odom' else e.pose
        if pose is None or e.puntos is None:
            a.cmd = (0.0, 0.0, 0.0)
            return True
        plan_robot = paso_angosto.a_marco_robot(p['plan'], pose)
        avance = paso_angosto.avance_en_plan(plan_robot) - p['avance0']
        restante = float(np.sum(np.hypot(*np.diff(plan_robot, axis=0).T))) - \
            paso_angosto.avance_en_plan(plan_robot)
        libre = e.obstaculo is None or math.hypot(*e.obstaculo) > c.libre_fin_paso_m
        if (avance >= c.avance_paso_m and libre) or restante < 0.15:
            self.maniobra = None
            a.cmd = (0.0, 0.0, 0.0)
            if avance < c.avance_fallido_m and self._cerca_del_goal(e, p['goal']):
                return self._llegar(a, p['goal'], e.t, 'paso angosto sin avance al lado')
            if c.descartar_paso_sin_avance and avance < c.avance_fallido_m:
                a.avisos.append(f'paso angosto: terminó sin avanzar ({avance:.2f} m de plan), '
                                f'se descarta el goal')
                if self.estado == Estado.EXPLORANDO:
                    a.descartar = p['goal'][:2]
                    self.descartados.append(a.descartar)
                else:
                    self.regreso_enviado = False
                self._iniciar_escape(e)
                return True
            a.avisos.append(f'paso angosto: cruzado ({avance:.2f} m de plan), vuelve Nav2')
            self._enviar(a, p['goal'], e.t)
            return True
        if avance > p['mejor'] + 0.05:
            p['mejor'], p['t_mejor'] = avance, e.t
        if e.t > p['hasta'] or e.t - p['t_mejor'] > c.tiempo_sin_progreso_paso_s:
            self.maniobra = None
            a.cmd = (0.0, 0.0, 0.0)
            a.avisos.append(f'paso angosto: abandonado tras {avance:.2f} m de plan')
            if self._cerca_del_goal(e, p['goal']):
                return self._llegar(a, p['goal'], e.t, 'paso angosto abandonado al lado')
            if p['mejor'] < c.avance_fallido_m and e.pose is not None:
                self._marcar_fallido(e, a, p['plan_map'])
            if self.estado == Estado.EXPLORANDO:
                a.descartar = p['goal'][:2]
                self.descartados.append(a.descartar)
            else:
                self.regreso_enviado = False
            self._iniciar_escape(e)
            return True
        a.cmd = paso_angosto.comando(plan_robot, e.puntos, self.config_paso)
        return True

    def _lugar_fallido(self, pose):
        r = self.config.radio_lugar_fallido_m
        return any(math.hypot(pose[0] - x, pose[1] - y) < r for x, y in self.lugares_fallidos)

    def _marcar_fallido(self, e, a, plan_map):
        """Por acá no se pasa: lo recuerda y tapa el plan adelante en el costmap global, para
        que el planificador y el explorador busquen otro camino."""
        c = self.config
        self.lugares_fallidos.append(tuple(e.pose[:2]))
        if plan_map:
            a.bloquear = paso_angosto.tapon(plan_map, e.pose, hasta=c.tapon_hasta_m,
                                            medio_ancho=c.tapon_medio_ancho_m,
                                            desde=c.tapon_antes_m)
            self.t_bloqueo = e.t
        a.avisos.append(f'paso angosto: sin avance en ({e.pose[0]:.2f}, {e.pose[1]:.2f}); '
                        f'no se vuelve a intentar ahí, hueco tapado')

    def _vigilar(self, e, a):
        """Red de seguridad contra la inmovilidad. True si actuó."""
        c = self.config
        # El giro de localización no cuenta: se mide desde que terminó.
        if (min(e.quieto_s, e.t - self.t_vigilante) < c.tiempo_sin_moverse_s
                or self.escape_hasta is not None):
            return False
        self.t_vigilante = e.t
        if self.maniobra is not None and e.pose is not None:
            self._marcar_fallido(e, a, self.maniobra['plan_map'])
        goal = self.maniobra['goal'] if self.maniobra is not None else self.goal_actual
        self.maniobra = None
        self.midiendo = None
        self._soltar_goal(a)
        if goal is not None:
            if self.estado == Estado.EXPLORANDO:
                a.descartar = goal[:2]
                self.descartados.append(a.descartar)
            else:
                self.regreso_enviado = False
        self.desatasco_hasta = None
        self._iniciar_escape(e)
        a.cmd = (0.0, 0.0, 0.0)
        a.avisos.append(f'{e.quieto_s:.0f} s sin moverse: se corta lo que estaba haciendo'
                        + (' y escapa' if self.escape_hasta is not None else ''))
        return True

    # Paso ---------------------------------------------------------------------
    def paso(self, e: Entradas) -> Acciones:
        a = Acciones()
        if self.estado == Estado.TERMINADO or e.t is None:
            return a
        if self.estado == Estado.ESPERANDO_MAPA:
            self._cambiar(Estado.LOCALIZANDO, e.t)
            self.yaw_previo = e.yaw_odom

        a.reportes.extend(self.politica.nuevos(e.objetivos))

        manejar = {
            Estado.LOCALIZANDO: self._localizando,
            Estado.EXPLORANDO: self._explorando,
            Estado.ORIENTANDO: self._orientando,
            Estado.REGRESANDO: self._regresando,
            Estado.AJUSTANDO: self._ajustando,
        }[self.estado]
        manejar(e, a)
        return a

    def _localizando(self, e, a):
        c = self.config
        if e.yaw_odom is not None:
            if self.yaw_previo is not None:
                self.giro += abs(envolver(e.yaw_odom - self.yaw_previo))
            self.yaw_previo = e.yaw_odom
        if self.giro >= c.giro_localizacion_rad or e.t - self.t_estado >= c.tiempo_max_localizacion_s:
            self.largada = e.pose if e.pose is not None else (0.0, 0.0, 0.0)
            self.t_vigilante = e.t
            a.cmd = (0.0, 0.0, 0.0)
            self._cambiar(Estado.EXPLORANDO, e.t)
        else:
            a.cmd = (0.0, 0.0, c.vel_giro_localizacion)

    def _explorando(self, e, a):
        c = self.config
        if self.politica.completo():
            return self._iniciar_regreso(e, a, 'todos los cupos reportados')
        if self.tiempo_restante(e.t) < self.tiempo_regreso(e.pose):
            return self._iniciar_regreso(e, a, 'reserva de tiempo para volver')
        if (self._vigilar(e, a) or self._escapando(e, a) or self._pasando(e, a)
                or self._midiendo(e, a)):
            return None                           # una maniobra en curso termina primero
        if e.exploracion_terminada and self.goal_actual is None and e.nav_libre:
            return self._iniciar_regreso(e, a, 'cobertura terminada')

        if self.goal_actual is not None:
            if self._escapar_giro_frenado(e, a):
                return None
            if self._hueco_adelante(e, a, self.goal_actual):
                return None
            if ((self._quieto_con_goal(e) > c.tiempo_trabado_paso_s
                 or e.nav_resultado == 'falla')
                    and e.nav_resultado != 'ok' and self._cerca_del_goal(e, self.goal_actual)):
                self._llegar(a, self.goal_actual, e.t, 'trabado al lado')
                return None
            if ((self._quieto_con_goal(e) > c.tiempo_trabado_paso_s
                 or e.nav_resultado == 'falla')
                    and e.nav_resultado != 'ok'
                    and (self._cruzar_por_hueco(e, a, self.goal_actual)
                         or self._hueco_angosto(e, a, self.goal_actual))):
                return None
            if e.nav_resultado == 'ok':
                self.yaw_objetivo = self.goal_actual[2]
                self.goal_llegado = self.goal_actual
                self.goal_actual = None
                self._cambiar(Estado.ORIENTANDO, e.t)
                return None
            if (e.nav_resultado == 'falla' or self._quieto_con_goal(e) > c.tiempo_inmovil_s
                    or e.t - self.t_goal > c.tiempo_max_goal_s):
                if self._ir_a_medir(e, a, self.goal_actual):
                    return None
                a.descartar = self.goal_actual[:2]
                self.descartados.append(a.descartar)
                if e.nav_resultado == 'falla':
                    self.goal_actual = None
                else:
                    self._soltar_goal(a)
                self._iniciar_escape(e)
                return None

        goal = e.goal_exploracion
        # Radio algo menor que el del explorador: los candidatos están cada 0.30 m y en el
        # borde (0.30 justo) uno los aceptaba y el otro no; la misión quedaba sin goal para
        # siempre y el explorador no se daba por terminado.
        if goal is not None and any(math.hypot(goal[0] - x, goal[1] - y)
                                    < 0.8 * c.radio_descarte_m for x, y in self.descartados):
            goal = None
        # Saliendo de una inspección apenas se decide, el último mensaje del explorador puede ser
        # todavía la inspección de ese mismo candidato: se espera el próximo.
        if (goal is not None and c.esperar_decision and e.goal_es_inspeccion
                and any(k.get('id') == e.inspeccion_id and k.get('estado') != 'sin_inspeccionar'
                        for k in e.candidatos)):
            goal = None
        if self.goal_actual is None and e.nav_libre and goal is not None:
            self._enviar(a, goal, e.t)
            self.goal_inspeccion = e.goal_es_inspeccion
            self.candidato_inspeccion = e.inspeccion_id if e.goal_es_inspeccion else None
            self.desatasco_hasta = None
            self.t_sin_goal = None
        elif self.goal_actual is None:
            # Red de seguridad: un rato largo sin ningún goal usable es lo mismo que terminar.
            if self.t_sin_goal is None:
                self.t_sin_goal = e.t
            elif e.t - self.t_sin_goal > c.tiempo_max_sin_goal_s:
                return self._iniciar_regreso(e, a, 'sin goals útiles')
            # Girar no cuenta como moverse: sin esta pausa, giraría sin parar.
            if (self.desatasco_hasta is None and e.quieto_s > c.tiempo_inmovil_s
                    and e.t - self.fin_desatasco > c.tiempo_inmovil_s):
                if self._obstaculo_cerca(e):          # girar ahí choca: mejor alejarse
                    self._iniciar_escape(e)
                    return None
                self.desatasco_hasta = e.t + c.tiempo_desatasco_s
            if self.desatasco_hasta is not None:
                if e.t < self.desatasco_hasta:
                    a.cmd = (0.0, 0.0, c.vel_giro_desatasco)
                else:
                    self.desatasco_hasta = None
                    self.fin_desatasco = e.t
        return None

    def _orientando(self, e, a):
        c = self.config
        error = None if e.pose is None else envolver(self.yaw_objetivo - e.pose[2])
        # Si el giro no avanza (el supervisor lo frena porque tocaría algo), no se espera.
        if error is not None:
            if self.giro_ref is None or abs(error) < self.giro_ref[0] - 0.05:
                self.giro_ref = (abs(error), e.t)
        trabado = self.giro_ref is not None and e.t - self.giro_ref[1] > c.tiempo_giro_trabado_s
        if (error is None or abs(error) < c.tolerancia_orientacion_rad or trabado
                or e.t - self.t_estado > c.tiempo_max_orientacion_s):
            a.cmd = (0.0, 0.0, 0.0)
            if self.goal_inspeccion:
                # Mirando de frente un candidato: quieto un rato para que la percepción decida.
                esperar = c.esperar_decision and self.candidato_inspeccion is not None
                if self.mirando_hasta is None:
                    self.mirando_hasta = e.t + (c.espera_max_decision_s if esperar
                                                else c.pausa_inspeccion_s)
                if e.t < self.mirando_hasta and not (
                        esperar and self._inspeccion_resuelta(e, self.candidato_inspeccion)):
                    return None
            self.mirando_hasta, self.goal_inspeccion, self.giro_ref = None, False, None
            # Avisa al explorador: ese lugar ya se visitó y no se vuelve a proponer, aunque
            # desde donde quedó el robot no se haya visto todo lo que el goal prometía.
            a.visitado, self.goal_llegado = self.goal_llegado, None
            return self._cambiar(Estado.EXPLORANDO, e.t)
        giro = max(-c.vel_giro_max, min(c.vel_giro_max, c.ganancia_giro * error))
        a.cmd = (0.0, 0.0, giro)
        return None

    def _inspeccion_resuelta(self, e, ident):
        """¿La percepción ya decidió el candidato `ident` (o lo borró)?"""
        cand = next((k for k in e.candidatos if k.get('id') == ident), None)
        return cand is None or cand.get('estado') != 'sin_inspeccionar'

    def _iniciar_regreso(self, e, a, motivo):
        self._soltar_goal(a)
        self.escape_hasta = None
        self.maniobra = None
        self.regreso_enviado = False
        self.intentos = 0
        self._cambiar(Estado.REGRESANDO, e.t, motivo)

    def _regresando(self, e, a):
        c = self.config
        if self.tiempo_restante(e.t) < c.margen_final_s:
            return self._cerrar(e, a, 'se acaba el tiempo')
        if self._vigilar(e, a) or self._escapando(e, a) or self._pasando(e, a):
            return None
        if self.regreso_enviado and self._escapar_giro_frenado(e, a):
            return None
        if (self.regreso_enviado and self.goal_actual is not None
                and self._hueco_adelante(e, a, self.goal_actual)):
            return None
        if (self.regreso_enviado and self.goal_actual is not None
                and (self._quieto_con_goal(e) > c.tiempo_trabado_paso_s
                     or e.nav_resultado == 'falla')
                and e.nav_resultado != 'ok'
                and (self._cruzar_por_hueco(e, a, self.goal_actual)
                     or self._hueco_angosto(e, a, self.goal_actual))):
            return None
        if not self.regreso_enviado:
            if e.nav_libre:
                self._enviar(a, self.largada, e.t)
                self.regreso_enviado = True
            return None
        if e.nav_resultado == 'ok':
            self.goal_actual = None
            return self._cambiar(Estado.AJUSTANDO, e.t)
        if e.nav_resultado == 'falla':
            self.goal_actual = None
            self.intentos += 1
            if self.intentos < c.intentos_regreso:
                if self._obstaculo_cerca(e):          # primero salir, después reintentar
                    self._iniciar_escape(e)
                    self.regreso_enviado = False
                else:
                    self._enviar(a, self.largada, e.t)
            elif e.pose is not None and self._distancia_a_largada(e.pose) < c.distancia_max_ajuste_m:
                self._cambiar(Estado.AJUSTANDO, e.t)
            elif c.liberar_tapones_al_volver and self.hubo_tapon and not self.tapones_liberados:
                a.liberar_tapones = True
                self.tapones_liberados, self.hubo_tapon = True, False
                self.tapados = []
                self.intentos = 0
                self.regreso_enviado = False
                self.t_bloqueo = e.t                  # espera a que el costmap los suelte
                a.avisos.append('no se pudo volver: se liberan los tapones y se reintenta')
            else:
                return self._cerrar(e, a, 'no se pudo volver')
        return None

    def _distancia_a_largada(self, pose):
        return math.hypot(pose[0] - self.largada[0], pose[1] - self.largada[1])

    def _ajustando(self, e, a):
        c = self.config
        if (e.pose is None or self.tiempo_restante(e.t) < c.margen_final_s
                or e.t - self.t_estado > c.tiempo_max_ajuste_s
                or self._distancia_a_largada(e.pose) < c.tolerancia_ajuste_m):
            return self._cerrar(e, a, 'en la largada')
        dx, dy = self.largada[0] - e.pose[0], self.largada[1] - e.pose[1]
        cy, sy = math.cos(e.pose[2]), math.sin(e.pose[2])
        vx, vy = c.ganancia_ajuste * (dx * cy + dy * sy), c.ganancia_ajuste * (-dx * sy + dy * cy)
        rapidez = math.hypot(vx, vy)
        if rapidez > c.vel_ajuste_max:
            vx, vy = vx * c.vel_ajuste_max / rapidez, vy * c.vel_ajuste_max / rapidez
        a.cmd = (vx, vy, 0.0)
        return None

    def _cerrar(self, e, a, motivo):
        self._soltar_goal(a)
        a.reportes.extend(self.politica.sobrantes(e.objetivos))
        a.cmd = (0.0, 0.0, 0.0)
        a.done = True
        self._cambiar(Estado.TERMINADO, e.t, motivo)
