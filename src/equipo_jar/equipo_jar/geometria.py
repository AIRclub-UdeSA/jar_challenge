"""Poses en el plano (x, y, yaw), sin ROS. Las usan percepción y el supervisor de localización.

Una pose `a` también se puede leer como una transformación: "parado en `a`, lo que veo en
(px, py) está en el mundo en componer(a, (px, py, 0))". Con eso se arman las cadenas de TF a mano:

    pose del robot en el mapa = componer(map -> odom, odom -> base_footprint)

que es exactamente lo que hace tf2 cuando le pedís map -> base_footprint.
"""
import math

from equipo_jar.evasion import normalizar_angulo


def componer(a, b):
    """a ∘ b: la pose `b`, expresada en el marco de `a`, pasada al marco donde está `a`."""
    ax, ay, ayaw = a
    bx, by, byaw = b
    c, s = math.cos(ayaw), math.sin(ayaw)
    return (ax + c * bx - s * by, ay + s * bx + c * by, normalizar_angulo(ayaw + byaw))


def inversa(a):
    """La transformación que deshace `a`: componer(a, inversa(a)) = (0, 0, 0)."""
    ax, ay, ayaw = a
    c, s = math.cos(ayaw), math.sin(ayaw)
    return (-c * ax - s * ay, s * ax - c * ay, normalizar_angulo(-ayaw))


def de_robot_a_mapa(pose_robot, px, py):
    """Un punto (px, py) visto desde el robot (x adelante, y a la izquierda), en el mapa."""
    x, y, _ = componer(pose_robot, (px, py, 0.0))
    return x, y


def diferencia(a, b):
    """(distancia en m, diferencia de ángulo en rad, en valor absoluto) entre dos poses."""
    return math.hypot(a[0] - b[0], a[1] - b[1]), abs(normalizar_angulo(a[2] - b[2]))
