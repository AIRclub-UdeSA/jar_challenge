"""Lógica pura del equipo Donatello, sin ROS, para poder testearla aparte.

Basado en los controladores de donatello-jar (control_obstaculos.py,
elegir_direccion.py y detector_rojo.py), adaptado a la interfaz oficial.
"""
import math

import cv2
import numpy as np


def normalizar(angulo):
    """Lleva un ángulo a (-pi, pi]."""
    return math.atan2(math.sin(angulo), math.cos(angulo))


def yaw_de_cuaternion(q):
    """Yaw (rad) de un geometry_msgs/Quaternion (o algo con x, y, z, w)."""
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def lecturas_validas(ranges, angle_min, angle_increment, offset, range_min, range_max):
    """Devuelve una lista de (angulo_robot, distancia) con las lecturas válidas.

    `offset` corrige el montaje del LiDAR (en el Rosmaster está girado pi).
    """
    lecturas = []
    for i, d in enumerate(ranges):
        if d is None or not math.isfinite(d) or d < range_min or d > range_max:
            continue
        lecturas.append((normalizar(angle_min + i * angle_increment + offset), d))
    return lecturas


def minimo_en_sector(lecturas, desde, hasta, por_defecto=float('inf')):
    """Distancia mínima entre los ángulos `desde` y `hasta` (rad, marco del robot)."""
    valores = [d for a, d in lecturas if desde <= a <= hasta]
    return min(valores) if valores else por_defecto


def distancia_en_rumbo(lecturas, rumbo, ventana):
    """Mediana de las distancias a +-ventana del rumbo dado, o None."""
    valores = sorted(d for a, d in lecturas if abs(normalizar(a - rumbo)) <= ventana)
    if not valores:
        return None
    return valores[len(valores) // 2]


def rumbo_de_pixel(u, cx, fx):
    """Ángulo horizontal (rad, positivo a la izquierda) de la columna u de la imagen."""
    return math.atan2(cx - u, fx)


def proyectar(x, y, yaw, distancia, rumbo):
    """Punto a `distancia` en dirección `rumbo` relativa al robot, en el marco del mapa."""
    return (x + distancia * math.cos(yaw + rumbo), y + distancia * math.sin(yaw + rumbo))


def mascara_roja(imagen_rgb, s_min, v_min, h_bajo_max, h_alto_min):
    """Máscara binaria del rojo (el rojo ocupa los dos extremos del tono en HSV)."""
    hsv = cv2.cvtColor(imagen_rgb, cv2.COLOR_RGB2HSV)
    m1 = cv2.inRange(hsv, np.array([0, s_min, v_min]), np.array([h_bajo_max, 255, 255]))
    m2 = cv2.inRange(hsv, np.array([h_alto_min, s_min, v_min]), np.array([179, 255, 255]))
    return m1 | m2


def blobs(mascara, area_min):
    """Lista de (u_centro, area) de las manchas con área >= area_min, la mayor primero."""
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    salida = []
    for c in contornos:
        area = cv2.contourArea(c)
        if area < area_min:
            continue
        m = cv2.moments(c)
        if m['m00'] <= 0:
            continue
        salida.append((m['m10'] / m['m00'], area))
    salida.sort(key=lambda b: -b[1])
    return salida


class Agrupador:
    """Junta observaciones cercanas y decide cuándo una víctima está confirmada.

    Cada grupo guarda el promedio de sus observaciones. Un grupo se confirma al
    juntar `observaciones_min`, y un grupo ya reportado no se vuelve a reportar
    (cada mensaje en /victimas gasta un reporte).
    """

    def __init__(self, radio, observaciones_min):
        self.radio = radio
        self.observaciones_min = observaciones_min
        self.grupos = []  # dicts: sx, sy, n, reportado

    def agregar(self, x, y):
        mejor, dmin = None, self.radio
        for g in self.grupos:
            d = math.hypot(g['sx'] / g['n'] - x, g['sy'] / g['n'] - y)
            if d <= dmin:
                mejor, dmin = g, d
        if mejor is None:
            mejor = {'sx': 0.0, 'sy': 0.0, 'n': 0, 'reportado': False}
            self.grupos.append(mejor)
        mejor['sx'] += x
        mejor['sy'] += y
        mejor['n'] += 1
        return mejor

    def listos(self):
        """Grupos confirmados y todavía sin reportar, como (grupo, x, y)."""
        return [(g, g['sx'] / g['n'], g['sy'] / g['n']) for g in self.grupos
                if not g['reportado'] and g['n'] >= self.observaciones_min]

    def cantidad_reportados(self):
        return sum(1 for g in self.grupos if g['reportado'])
