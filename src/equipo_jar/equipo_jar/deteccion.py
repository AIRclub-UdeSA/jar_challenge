"""Detectar una caja roja con la cámara y ubicarla con el lidar. Funciones puras, sin ROS.

Tres pasos, cada uno una función:
1. `buscar_caja_roja`: filtra el rojo en HSV y devuelve en qué columna de la imagen está.
2. `angulo_de_columna`: pasa esa columna a un ángulo respecto del frente del robot.
3. `distancia_a_la_caja`: qué distancia ve el lidar justo en esa dirección.
4. `distancia_por_altura`: una segunda estimación de la distancia, con el tamaño de la caja en la
   imagen. Si no coincide con la del lidar, no se reporta (ver `coinciden`).

Con el ángulo y la distancia, `punto_en_el_plano` da la posición de la caja. Es la misma idea del
workshop de la semana 04, simplificada: en vez de proyectar todo el scan sobre la imagen, se
mira el scan solo en la dirección donde la cámara vio el rojo.
https://airclub-udesa.github.io/jar_site/workshops/semana-04-deteccion-color/
"""
import math

import cv2
import numpy as np

from equipo_jar.evasion import normalizar_angulo


def mascara_rojo(imagen_bgr, saturacion_min, valor_min):
    """Máscara binaria (255 = rojo) de una imagen BGR.

    En HSV el color es un ángulo (H, de 0 a 179 en OpenCV) y el rojo queda partido en los dos
    extremos del círculo: por eso hacen falta dos rangos de H. Saturación y valor mínimos
    descartan grises y rosas pálidos, que tienen un tono parecido.
    """
    hsv = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2HSV)
    cerca_de_cero = cv2.inRange(hsv, (0, saturacion_min, valor_min), (10, 255, 255))
    cerca_de_180 = cv2.inRange(hsv, (170, saturacion_min, valor_min), (180, 255, 255))
    return cv2.bitwise_or(cerca_de_cero, cerca_de_180)


def buscar_caja_roja(imagen_bgr, saturacion_min, valor_min, area_minima_px, margen_borde_px):
    """(columna del centro, ancho en píxeles, alto en píxeles) de la mancha roja más grande, o None.

    Devuelve None si no hay ninguna, si es más chica que `area_minima_px` (ruido, o una caja tan
    lejos que no se puede ubicar) o si toca el borde de la imagen: una caja cortada por el borde
    tiene el centro mal.
    """
    mascara = mascara_rojo(imagen_bgr, saturacion_min, valor_min)
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return None
    mayor = max(contornos, key=cv2.contourArea)
    if cv2.contourArea(mayor) < area_minima_px:
        return None
    x, _, ancho, alto = cv2.boundingRect(mayor)
    if x < margen_borde_px or x + ancho > imagen_bgr.shape[1] - margen_borde_px:
        return None
    return x + ancho / 2.0, ancho, alto


def angulo_de_columna(columna, fx, cx):
    """Ángulo (rad) de una columna de la imagen respecto del eje de la cámara.

    Positivo hacia la izquierda del robot, que es el sentido de giro de ROS. `fx` y `cx` salen
    del topic de la cámara `camera_info` (matriz K: fx = k[0], cx = k[2]).
    """
    return math.atan2(cx - columna, fx)


def distancia_a_la_caja(rangos, angulo_min, incremento, frente_deg, centro_deg, ancho_deg,
                        rango_min=0.0, rango_max=math.inf):
    """Distancia a la cara de la caja según el lidar, o None si no hay rayos válidos.

    Mira los rayos dentro del sector donde la cámara vio la caja y devuelve la **mediana**, no
    el más corto: el borde de una pared que asoma en el sector, o un rayo suelto, darían una
    distancia mucho menor que la de la caja.
    centro_deg, ancho_deg: el sector medido desde el frente del robot (positivo, a la izquierda).
    frente_deg: ángulo del /scan que corresponde al frente (ver evasion.distancia_en_sector).
    """
    frente = math.radians(frente_deg)
    centro = math.radians(centro_deg)
    mitad = math.radians(ancho_deg) / 2.0
    dentro = []
    for i, r in enumerate(rangos):
        if not math.isfinite(r) or r < rango_min or r > rango_max:
            continue
        desde_el_frente = normalizar_angulo(angulo_min + i * incremento - frente)
        if abs(normalizar_angulo(desde_el_frente - centro)) <= mitad:
            dentro.append(r)
    return float(np.median(dentro)) if dentro else None


def distancia_por_altura(alto_px, fy, alto_caja_m):
    """Distancia a la caja (m) según cuántos píxeles de alto ocupa en la imagen (modelo pinhole).

    Sirve para contrastar con el lidar: el alto no lo tapa una pared que asoma de costado, que es
    justo lo que engaña al lidar cuando la caja se ve doblando una esquina.
    """
    return fy * alto_caja_m / alto_px


def coinciden(distancia_lidar, distancia_imagen, tolerancia_relativa):
    """True si las dos estimaciones difieren menos que `tolerancia_relativa` de la de la imagen."""
    return abs(distancia_lidar - distancia_imagen) <= tolerancia_relativa * distancia_imagen


def punto_en_el_plano(x_robot, y_robot, yaw_robot, angulo, distancia):
    """Posición (x, y) de un punto que está a `distancia` en la dirección `angulo` del robot."""
    direccion = yaw_robot + angulo
    return (x_robot + distancia * math.cos(direccion),
            y_robot + distancia * math.sin(direccion))


def yaw_de_cuaternion(x, y, z, w):
    """Ángulo de giro (yaw) en radianes de un cuaternión de orientación."""
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def cerca_de_alguno(punto, puntos, radio_m):
    """True si `punto` está a menos de `radio_m` de alguno de `puntos`."""
    return any(math.dist(punto, otro) < radio_m for otro in puntos)


def promedio(puntos):
    """Centro de una lista de puntos (x, y)."""
    return (float(np.mean([p[0] for p in puntos])), float(np.mean([p[1] for p in puntos])))
