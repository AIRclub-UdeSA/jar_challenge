"""Medición y corrección de escala de la odometría, sin dependencias de ROS."""

import math

from jar_localizacion.filtro import normalizar_angulo


def indice_rayo(angle_min, angle_increment, cantidad, angulo):
    """Índice del rayo del scan más cercano a `angulo` (frame del lidar)."""
    diferencia = normalizar_angulo(angulo - angle_min)
    if diferencia < 0.0:
        diferencia += 2.0 * math.pi
    return int(round(diferencia / angle_increment)) % cantidad


def indice_rayo_hacia_atras(angle_min, angle_increment, cantidad, yaw_lidar):
    """Rayo que apunta hacia atrás del robot, con el lidar girado `yaw_lidar`."""
    return indice_rayo(angle_min, angle_increment, cantidad, math.pi - yaw_lidar)


def paso_en_frame_robot(anterior, actual):
    """(avance, costado, giro) entre dos poses de odom, en el frame del robot."""
    x0, y0, theta0 = anterior
    x1, y1, theta1 = actual
    dx, dy = x1 - x0, y1 - y0
    c, s = math.cos(theta0), math.sin(theta0)
    return dx * c + dy * s, -dx * s + dy * c, normalizar_angulo(theta1 - theta0)


def aplicar_paso(pose, avance, costado, giro, escala_avance=1.0, escala_giro=1.0):
    """Mueve `pose` con el paso escalado, en la dirección de la propia pose."""
    x, y, theta = pose
    avance *= escala_avance
    costado *= escala_avance
    c, s = math.cos(theta), math.sin(theta)
    return (x + avance * c - costado * s, y + avance * s + costado * c,
            normalizar_angulo(theta + giro * escala_giro))
