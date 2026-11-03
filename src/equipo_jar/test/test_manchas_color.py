"""Manchas de color (no rojo) para ubicar objetos con cámara + LiDAR."""
import numpy as np

from equipo_jar.deteccion_victimas import segmentar_colores

CFG = {'saturacion_minima': 120, 'brillo_minimo': 50, 'area_minima_px': 40,
       'tono_rojo_bajo_max': 10, 'tono_rojo_alto_min': 165,
       'tono_naranja_max': 20, 'tono_amarillo_max': 35, 'tono_verde_max': 85,
       'tono_cian_max': 95, 'tono_azul_max': 135, 'tono_violeta_max': 150,
       'ancho_maximo_mancha': 0.6}


def test_caja_azul_y_verde_con_su_color_y_rojo_ignorado():
    img = np.zeros((240, 320, 3), np.uint8)
    img[80:150, 20:80] = (255, 0, 0)      # azul (BGR)
    img[80:150, 120:170] = (0, 200, 0)    # verde
    img[80:150, 220:280] = (0, 0, 255)    # rojo: lo maneja el detector de víctimas
    colores = sorted(m[5] for m in segmentar_colores(img, CFG))
    assert colores == ['azul', 'verde']


def test_piso_o_pared_de_color_no_es_objeto():
    img = np.zeros((240, 320, 3), np.uint8)
    img[180:240, :] = (0, 128, 255)       # piso naranja que toca el borde inferior
    img[20:100, 10:300] = (255, 0, 0)     # pared azul de frente: casi todo el ancho
    assert segmentar_colores(img, CFG) == []
