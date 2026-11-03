"""Segmentación de rojo en HSV con OpenCV."""

import cv2
import numpy as np

PARAMETROS_ROJO = {
    # El rojo queda partido en ambos extremos del tono (0-179 en OpenCV).
    'hue_rojo_bajo_1': 0.0,
    'hue_rojo_alto_1': 10.0,
    'hue_rojo_bajo_2': 170.0,
    'hue_rojo_alto_2': 180.0,
    # Sin este piso, grises o rosas pálidos también entrarían como rojo.
    'saturacion_min': 120.0,
    'valor_min': 80.0,
}


def mascara_rojo(imagen_hsv, hue_rojo_bajo_1, hue_rojo_alto_1, hue_rojo_bajo_2,
                 hue_rojo_alto_2, saturacion_min, valor_min):
    """Máscara uint8 con 255 donde el pixel es rojo."""
    def rango(hue_bajo, hue_alto):
        bajo = np.array([hue_bajo, saturacion_min, valor_min], dtype=np.uint8)
        alto = np.array([hue_alto, 255, 255], dtype=np.uint8)
        return cv2.inRange(imagen_hsv, bajo, alto)

    return cv2.bitwise_or(rango(hue_rojo_bajo_1, hue_rojo_alto_1),
                          rango(hue_rojo_bajo_2, hue_rojo_alto_2))


def area_mayor_contorno(mascara) -> float:
    """Área en píxeles del contorno más grande; 0.0 si no hay ninguno."""
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return 0.0
    return max(cv2.contourArea(c) for c in contornos)
