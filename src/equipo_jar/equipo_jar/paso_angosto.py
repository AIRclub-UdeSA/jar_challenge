"""Maniobra de paso angosto (ADR-0008), sin ROS.

Cuando Nav2 se traba frente a un hueco angosto (una caja cerca de una pared), la misión toma
el control y pasa despacio usando el mecanum:

- la dirección general sale del plan de Nav2 (hacia dónde queda el hueco);
- el centrado fino sale de los puntos crudos del LiDAR: el robot se corre de costado hasta
  quedar a igual distancia de los dos lados. El plan está en `map` y AMCL se equivoca unos
  centímetros, así que el plan solo puede quedar corrido respecto del hueco real; el LiDAR no;
- si la dirección deseada choca (por ejemplo, la esquina del robot contra la esquina de la
  caja), prueba direcciones cada vez más abiertas hacia los costados y toma la más cercana que
  tenga lugar, medido como lo mide el supervisor: si no, el supervisor frena todo, también el
  lateral que lo sacaría;
- el cuerpo se alinea con el eje del plan (de frente o de culata, lo que quede más cerca),
  pero sólo gira si el rectángulo girado no toca ningún punto del LiDAR.

Todo en el marco del robot: x adelante, y a la izquierda.
"""
import math
from dataclasses import dataclass

import numpy as np

from jar_supervisor.logica import hueco_en_direccion


@dataclass
class ConfigPaso:
    medio_largo_m: float = 0.16
    medio_ancho_m: float = 0.11
    margen_giro_m: float = 0.01        # para girar en el lugar
    adelanto_m: float = 0.25           # punto del plan hacia el que se avanza
    vel_m_s: float = 0.08
    ganancia_centrado: float = 1.0     # 1/s: m de descentrado -> m/s de lateral
    vel_centrado_max: float = 0.05
    ganancia_giro: float = 1.0
    vel_giro_max: float = 0.4
    paso_giro_rad: float = 0.15        # cuánto se chequea hacia adelante al girar
    ventana_atras_m: float = 0.20      # puntos a los costados que cuentan para centrarse
    ventana_adelante_m: float = 0.35
    alcance_lateral_m: float = 0.40    # más lejos que esto de costado: no es un hueco
    radio_rulo_m: float = 0.20         # el plan que vuelve a pasar a esto del robot: rulo
    margen_supervisor_m: float = 0.02  # = margen_m del supervisor
    # Lugar libre que tiene que tener la dirección elegida: más que `distancia_freno_m` del
    # supervisor (0.06), que por debajo frena a cero. Con 0.05 la maniobra pedía una dirección
    # que el supervisor no dejaba pasar y el robot quedaba quieto.
    hueco_min_m: float = 0.08
    abanico_rad: float = math.radians(15.0)
    pasos_abanico: int = 12            # hasta ±180°: si adelante no hay lugar, retrocede


def a_marco_robot(puntos_map, pose):
    """Puntos (x, y) en `map` al marco del robot en `pose` = (x, y, yaw)."""
    p = np.asarray(puntos_map, dtype=float).reshape(-1, 2) - np.array(pose[:2])
    c, s = math.cos(pose[2]), math.sin(pose[2])
    return np.column_stack((c * p[:, 0] + s * p[:, 1], -s * p[:, 0] + c * p[:, 1]))


def indice_actual(plan_robot, radio_rulo=0.20):
    """Dónde está el robot en el plan: el ÚLTIMO punto del plan que pasa cerca de él.

    Un planificador diferencial puede salir para el otro lado y volver para dar la vuelta
    (un rulo); el mecanum no lo necesita: si el plan vuelve a pasar por donde está el robot,
    se sigue desde ahí. Si nada pasa a menos de `radio_rulo`, el punto más cercano.
    """
    plan = np.asarray(plan_robot, dtype=float).reshape(-1, 2)
    d = np.hypot(plan[:, 0], plan[:, 1])
    i = int(np.argmin(d))
    # Sólo es un rulo si entre medio el plan se alejó del robot (más del doble del radio).
    for j in np.flatnonzero(d < radio_rulo)[::-1]:
        if j > i and d[i:j].max() > 2.0 * radio_rulo:
            return int(j)
    return i


def objetivo_en_plan(plan_robot, adelanto, radio_rulo=0.20):
    """(índice actual en el plan, punto `adelanto` m más adelante, dirección ahí)."""
    plan = np.asarray(plan_robot, dtype=float).reshape(-1, 2)
    i0 = indice_actual(plan, radio_rulo)
    tramos = np.hypot(*np.diff(plan[i0:], axis=0).T) if len(plan) - i0 > 1 else np.zeros(0)
    acumulado = np.concatenate(([0.0], np.cumsum(tramos)))
    k = i0 + int(min(np.searchsorted(acumulado, adelanto), len(acumulado) - 1))
    a, b = plan[max(k - 1, 0)], plan[min(k + 1, len(plan) - 1)]
    direccion = math.atan2(b[1] - a[1], b[0] - a[0]) if np.any(a != b) else 0.0
    return i0, plan[k], direccion


def giro_libre(puntos, angulo, medio_largo, medio_ancho):
    """¿El rectángulo girado `angulo` en el lugar queda sin puntos del LiDAR adentro?"""
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


def descentrado(puntos, direccion, c):
    """Cuánto le sobra a la izquierda menos a la derecha (m), en el hueco que se cruza.

    Positivo: hay más lugar a la izquierda de la dirección de avance. None si no hay paredes
    a los dos costados (no es un hueco).
    """
    p = np.asarray(puntos, dtype=float).reshape(-1, 2)
    if len(p) == 0:
        return None
    d = np.array([math.cos(direccion), math.sin(direccion)])
    n = np.array([-d[1], d[0]])                        # izquierda de la dirección de avance
    s, lado = p @ d, p @ n
    cerca = ((s > -c.ventana_atras_m) & (s < c.ventana_adelante_m)
             & (np.abs(lado) < c.alcance_lateral_m))
    izq, der = lado[cerca & (lado > 0)], -lado[cerca & (lado < 0)]
    if len(izq) == 0 or len(der) == 0:
        return None
    return (float(izq.min()) - float(der.min())) / 2.0


def direccion_con_lugar(vx, vy, puntos, c):
    """La traslación (vx, vy), o la más cercana en ángulo que el supervisor deje pasar."""
    rapidez = math.hypot(vx, vy)
    if rapidez < 1e-6:
        return 0.0, 0.0
    base = math.atan2(vy, vx)
    largo, ancho = c.medio_largo_m + c.margen_supervisor_m, c.medio_ancho_m + c.margen_supervisor_m
    for k in range(c.pasos_abanico + 1):
        for signo in ((1,) if k == 0 else (1, -1)):
            a = base + signo * k * c.abanico_rad
            hueco = hueco_en_direccion(puntos, math.cos(a), math.sin(a), largo, ancho,
                                       c.margen_supervisor_m)
            if hueco >= c.hueco_min_m:
                return rapidez * math.cos(a), rapidez * math.sin(a)
    return 0.0, 0.0


def comando(plan_robot, puntos, c=ConfigPaso()):
    """(vx, vy, wz) para avanzar por el plan, centrado en el hueco con el LiDAR."""
    _, objetivo, direccion = objetivo_en_plan(plan_robot, c.adelanto_m, c.radio_rulo_m)
    distancia = math.hypot(*objetivo)
    if distancia < 1e-3:
        return 0.0, 0.0, 0.0
    ux, uy = objetivo[0] / distancia, objetivo[1] / distancia
    vx, vy = ux * c.vel_m_s, uy * c.vel_m_s
    sobra = descentrado(puntos, math.atan2(uy, ux), c)
    if sobra is not None:
        lateral = max(-c.vel_centrado_max, min(c.vel_centrado_max, c.ganancia_centrado * sobra))
        vx, vy = vx - uy * lateral, vy + ux * lateral   # hacia la izquierda de la dirección
    vx, vy = direccion_con_lugar(vx, vy, puntos, c)
    # Alinea el cuerpo con el EJE del plan: el rectángulo es simétrico y el mecanum anda igual
    # para atrás, así que nunca gira más de 90° (girar de más en un hueco lo traba).
    error = math.atan2(math.sin(direccion), math.cos(direccion))
    if abs(error) > math.pi / 2:
        error -= math.copysign(math.pi, error)
    wz = max(-c.vel_giro_max, min(c.vel_giro_max, c.ganancia_giro * error))
    if abs(wz) > 1e-3 and not (giro_libre(puntos, math.copysign(c.paso_giro_rad, wz),
                                          c.medio_largo_m + c.margen_giro_m,
                                          c.medio_ancho_m + c.margen_giro_m)
                               and sigue_pasando(puntos, math.copysign(c.paso_giro_rad, wz),
                                                 vx, vy, c)):
        wz = 0.0
    return vx, vy, wz


def sigue_pasando(puntos, angulo, vx, vy, c):
    """¿Después de girar `angulo` el robot todavía puede trasladarse hacia (vx, vy)?

    en una práctica: en un hueco de 0.43 m entre una caja y la pared, la maniobra giró para alinearse
    con el plan, que doblaba a la salida; el giro no chocaba, pero el robot quedó cruzado en
    diagonal (0.39 m de ancho) y ya no podía trasladarse hacia ningún lado. Con mecanum no
    hace falta girar para cruzar: se gira afuera.
    """
    if math.hypot(vx, vy) < 1e-6:
        return True
    p = np.asarray(puntos, dtype=float).reshape(-1, 2)
    co, si = math.cos(angulo), math.sin(angulo)
    girados = np.column_stack((co * p[:, 0] + si * p[:, 1], -si * p[:, 0] + co * p[:, 1]))
    dx, dy = co * vx + si * vy, -si * vx + co * vy      # la misma dirección en el marco girado
    largo, ancho = c.medio_largo_m + c.margen_supervisor_m, c.medio_ancho_m + c.margen_supervisor_m
    return hueco_en_direccion(girados, dx, dy, largo, ancho,
                              c.margen_supervisor_m) >= c.hueco_min_m


def avance_en_plan(plan_robot, radio_rulo=0.20):
    """Metros del plan que ya quedaron atrás del robot (hasta su índice actual)."""
    plan = np.asarray(plan_robot, dtype=float).reshape(-1, 2)
    i0 = indice_actual(plan, radio_rulo)
    return float(np.sum(np.hypot(*np.diff(plan[:i0 + 1], axis=0).T))) if i0 > 0 else 0.0


def ancho_en_plan(plan_robot, puntos, desde=0.0, hasta=0.8, franja=0.05, alcance=0.6):
    """Ancho libre más chico, perpendicular al plan, entre `desde` y `hasta` m adelante.

    Para cada punto del plan mide, con los puntos del LiDAR que caen a menos de `franja` a lo
    largo del plan, cuánto lugar hay a cada lado; el ancho es la suma. Es el hueco por el que
    el plan quiere pasar (una caja en el medio con lugar al costado no cuenta como angosto,
    porque el plan pasa por el costado). Devuelve (ancho, punto del plan donde es mínimo) o
    (None, None) si en ningún punto hay obstáculos a los dos lados.
    """
    plan = np.asarray(plan_robot, dtype=float).reshape(-1, 2)
    p = np.asarray(puntos, dtype=float).reshape(-1, 2)
    if len(plan) < 2 or len(p) == 0:
        return None, None
    i0 = indice_actual(plan)
    acumulado = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(plan[i0:], axis=0).T))))
    mejor, donde = None, None
    for k, s in enumerate(acumulado):
        if s < desde or s > hasta:
            continue
        i = i0 + k
        a, b = plan[max(i - 1, 0)], plan[min(i + 1, len(plan) - 1)]
        t = b - a
        if not np.any(t):
            continue
        t = t / np.hypot(*t)
        n = np.array([-t[1], t[0]])
        rel = p - plan[i]
        cerca = (np.abs(rel @ t) < franja) & (np.abs(rel @ n) < alcance)
        lado = rel[cerca] @ n
        izq, der = lado[lado > 0], -lado[lado < 0]
        if len(izq) and len(der):
            ancho = float(izq.min() + der.min())
            if mejor is None or ancho < mejor:
                mejor, donde = ancho, plan[i]
    return mejor, donde


def tapon(plan_map, pose, hasta=0.8, medio_ancho=0.15, paso=0.05, desde=0.0):
    """Puntos (en `map`) que tapan el plan entre `desde` y `hasta` m adelante del robot: una
    franja de ±`medio_ancho` alrededor del plan, para marcar el hueco como ocupado."""
    plan = np.asarray(plan_map, dtype=float).reshape(-1, 2)
    i0 = indice_actual(a_marco_robot(plan, pose))
    puntos, recorrido = [], 0.0
    for i in range(i0, len(plan) - 1):
        t = plan[i + 1] - plan[i]
        largo = float(np.hypot(*t))
        if largo < 1e-9:
            continue
        recorrido += largo
        if recorrido > hasta:
            break
        if recorrido < desde:
            continue
        n = np.array([-t[1], t[0]]) / largo
        for d in np.arange(-medio_ancho, medio_ancho + 1e-9, paso):
            puntos.append(tuple(plan[i + 1] + d * n))
    return puntos
