"""Lógica para recorrer el laberinto sin chocar, sin ROS.

Está separada del nodo a propósito: son funciones puras, así se prueban con
`pytest` sin levantar el simulador (ver test/test_evasion.py).

Es el mismo problema del workshop de la semana 03 (evasión de obstáculos):
https://airclub-udesa.github.io/jar_site/workshops/semana-03-evasion-obstaculos/
"""
import math


def normalizar_angulo(angulo):
    """Lleva un ángulo en radianes al rango [-pi, pi]."""
    return math.atan2(math.sin(angulo), math.cos(angulo))


def distancia_en_sector(rangos, angulo_min, incremento, frente_deg, centro_deg, ancho_deg,
                        rango_min=0.0, rango_max=math.inf):
    """Distancia más corta que ve el lidar dentro de un sector, o inf si no ve nada.

    rangos, angulo_min, incremento: los campos `ranges`, `angle_min` y `angle_increment` de un
        `sensor_msgs/LaserScan`.
    frente_deg: ángulo del /scan que corresponde al frente del robot. Depende de cómo está
        montado el lidar: en el robot simulado es 180 (el 0 del /scan mira hacia atrás).
        Comprobalo poniendo un obstáculo conocido adelante y mirando `ros2 topic echo /scan`.
    centro_deg, ancho_deg: el sector, medido desde el frente. Positivo es hacia la izquierda:
        0 es el frente, 90 la izquierda pura y -90 la derecha pura.
    """
    frente = math.radians(frente_deg)
    centro = math.radians(centro_deg)
    mitad = math.radians(ancho_deg) / 2.0
    mas_cerca = math.inf
    for i, r in enumerate(rangos):
        # El lidar devuelve inf o nan cuando un rayo no vuelve, y los rangos fuera de
        # [range_min, range_max] no son confiables: esos rayos no cuentan.
        if not math.isfinite(r) or r < rango_min or r > rango_max:
            continue
        desde_el_frente = normalizar_angulo(angulo_min + i * incremento - frente)
        if abs(normalizar_angulo(desde_el_frente - centro)) <= mitad:
            mas_cerca = min(mas_cerca, r)
    return mas_cerca


def giro_para_seguir_pared(distancia_pared, velocidad_distancia, objetivo, ganancia,
                           amortiguacion, giro_max, pared_perdida):
    """Velocidad angular (rad/s) para avanzar manteniendo la pared de la derecha a `objetivo`.

    - Si la pared está más lejos que el objetivo, se gira un poco a la derecha (negativo) para
      acercarse; si está más cerca, un poco a la izquierda (positivo). Cuanto más error, más
      giro, hasta `giro_max`.
    - `velocidad_distancia` es cuánto cambia la distancia a la pared por segundo (m/s). Sumarla
      con `amortiguacion` frena el giro cuando el robot ya se está acercando o alejando: sin
      eso corrige de más y avanza en zigzag. Es un control PD (proporcional y derivativo).
    - Si la pared está más lejos que `pared_perdida` (se terminó, o hay una esquina), se gira a
      la derecha con `giro_max` para volver a encontrarla.
    """
    if distancia_pared > pared_perdida:
        return -giro_max
    giro = -(ganancia * (distancia_pared - objetivo) + amortiguacion * velocidad_distancia)
    return max(-giro_max, min(giro_max, giro))


def sentido_de_giro(hay_lugar_a_la_derecha):
    """Hacia dónde girar en el lugar cuando hay algo adelante: -1 derecha, +1 izquierda.

    Regla de la mano derecha: si por la derecha hay lugar (no hay pared cerca), se dobla por ahí;
    si hay pared, se gira a la izquierda. Sin esto, en una esquina abierta a la derecha el robot
    giraría hacia el lado cerrado y volvería por donde vino.
    """
    return -1 if hay_lugar_a_la_derecha else 1
