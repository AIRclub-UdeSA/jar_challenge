import math

import numpy as np
import pytest

from equipo_jar.deteccion import (
    FiltroColor, FiltroForma, Mancha, angulo_de_columna, apoyada_en_el_piso, buscar_caja_roja,
    buscar_cajas_rojas, cerca_de_alguno, coinciden, distancia_a_la_caja, distancia_por_altura,
    filas_esperadas, frente_del_lidar, hay_algo_delante, hay_rojo_en_rayos, mascara_rojo,
    mediana_del_tramo, profundidad, promedio, punto_en_el_plano, rayos_en_la_imagen,
    tramo_del_lidar, yaw_de_cuaternion)

# La cámara del robot (real_robot_contract.yaml del simulador): 320 x 240, a 0,109 m del piso.
ANCHO, ALTO = 320, 240
FX = FY = 271.8
CX, CY = 158.7, 120.1
ALTURA_CAMARA = 0.109
ALTO_CAJA, LADO_CAJA = 0.30, 0.25

COLOR = FiltroColor()
FORMA = FiltroForma()

# BGR. Los de la caja son los del simulador (diffuse 0.8 0.05 0.05); el resto, distractores.
ROJO = (13, 13, 204)
AZUL = (204, 13, 13)
AMARILLO = (0, 210, 220)
VERDE = (20, 200, 20)
NARANJA = (0, 140, 255)     # H ~ 16
MAGENTA = (200, 0, 220)     # H ~ 152
ROSA_PALIDO = (190, 180, 230)  # H del rojo, pero sin saturación
GRIS = (128, 128, 128)


def imagen_vacia():
    return np.full((ALTO, ANCHO, 3), GRIS, dtype=np.uint8)


def dibujar_caja(img, prof, columna=CX, color=ROJO, lado=LADO_CAJA, alto=ALTO_CAJA,
                 altura_base=0.0):
    """Dibuja una caja apoyada (a `altura_base` del piso) a `prof` m, como la vería la cámara."""
    ancho_px = FX * lado / prof
    arriba = CY - FY * (altura_base + alto - ALTURA_CAMARA) / prof
    abajo = CY - FY * (altura_base - ALTURA_CAMARA) / prof
    x0, x1 = int(round(columna - ancho_px / 2)), int(round(columna + ancho_px / 2))
    img[max(0, int(round(arriba))):int(round(abajo)), max(0, x0):x1] = color
    return img


def buscar(img):
    return buscar_caja_roja(img, COLOR, FORMA)


# --- Color: distractores -------------------------------------------------------------------

def test_encuentra_la_caja_roja():
    mancha, motivo = buscar(dibujar_caja(imagen_vacia(), 1.0))
    assert motivo is None
    assert mancha.columna == pytest.approx(CX, abs=1.0)
    assert mancha.ancho == pytest.approx(FX * LADO_CAJA, abs=2)
    assert mancha.alto == pytest.approx(FY * ALTO_CAJA, abs=2)


@pytest.mark.parametrize('color', [AZUL, AMARILLO, VERDE, NARANJA, MAGENTA, ROSA_PALIDO])
def test_ignora_cajas_de_otro_color(color):
    mancha, _ = buscar(dibujar_caja(imagen_vacia(), 1.0, color=color))
    assert mancha is None


def test_elige_la_roja_entre_distractores():
    img = imagen_vacia()
    dibujar_caja(img, 0.8, columna=60, color=AZUL)      # más cerca y más grande, pero azul
    dibujar_caja(img, 0.9, columna=250, color=AMARILLO)
    dibujar_caja(img, 1.6, columna=150, color=ROJO)
    mancha, _ = buscar(img)
    assert mancha is not None and mancha.columna == pytest.approx(150, abs=2)


def test_devuelve_todas_las_cajas_de_la_imagen():
    # Una caja cerca (quizás ya confirmada) no tiene que tapar a otra más lejos.
    img = imagen_vacia()
    dibujar_caja(img, 0.8, columna=80)
    dibujar_caja(img, 1.8, columna=240)
    cajas, motivo = buscar_cajas_rojas(img, COLOR, FORMA)
    assert motivo is None
    assert [c.columna for c in cajas] == [pytest.approx(80, abs=2), pytest.approx(240, abs=2)]


def test_todas_las_cajas_saltea_las_descartadas():
    img = imagen_vacia()
    dibujar_caja(img, 0.6, columna=5)       # la más grande, cortada por el costado
    dibujar_caja(img, 1.5, columna=200)
    cajas, motivo = buscar_cajas_rojas(img, COLOR, FORMA)
    assert motivo == 'cortada por un costado de la imagen'
    assert len(cajas) == 1 and cajas[0].columna == pytest.approx(200, abs=2)
    assert buscar_cajas_rojas(imagen_vacia(), COLOR, FORMA) == ([], 'no hay rojo')


def test_la_limpieza_borra_bordes_finos_de_otros_colores():
    # Un hilo rojizo de 2 px alrededor de una caja amarilla (lo que hace una cámara real).
    img = dibujar_caja(imagen_vacia(), 0.8, color=AMARILLO)
    img[60:200, 100:102] = ROJO
    img[60:200, 216:218] = ROJO
    assert mascara_rojo(img, COLOR).sum() == 0
    assert buscar(img)[0] is None


def test_rangos_de_hue_ajustables():
    # Con hue_max_bajo más alto, el naranja pasa a contar como rojo: el parámetro funciona.
    naranja = dibujar_caja(imagen_vacia(), 1.0, color=NARANJA)
    permisivo = FiltroColor(hue_max_bajo=20)
    assert buscar_caja_roja(naranja, permisivo, FORMA)[0] is not None


# --- Forma ---------------------------------------------------------------------------------

def test_ignora_manchas_chicas():
    img = imagen_vacia()
    img[60:72, 100:112] = ROJO  # 12 x 12 px: menos que el área mínima
    assert buscar(img) == (None, 'chica')


def test_ignora_una_caja_cortada_por_los_costados():  # su centro estaría corrido
    assert buscar(dibujar_caja(imagen_vacia(), 1.0, columna=20))[0] is None
    assert buscar(dibujar_caja(imagen_vacia(), 1.0, columna=300))[0] is None


def test_acepta_una_caja_cortada_arriba_y_la_valida_con_la_base():
    # A 0,35 m la tapa queda fuera de la imagen (pasa siempre que el robot pasa cerca): la
    # columna sigue bien, así que se usa, y se valida solo con la fila de la base.
    mancha, motivo = buscar(dibujar_caja(imagen_vacia(), 0.35))
    assert motivo is None and mancha.cortada_arriba and not mancha.cortada_abajo
    assert mancha.columna == pytest.approx(CX, abs=1.0)
    assert apoyada_en_el_piso(mancha, 0.35, FY, CY, ALTURA_CAMARA, ALTO_CAJA, 0.2)
    # Un rojo cortado arriba pero que no llega al piso (un cartel grande, cerca) no pasa.
    cartel, _ = buscar(dibujar_caja(imagen_vacia(), 0.35, altura_base=0.06))
    assert cartel is not None and cartel.cortada_arriba
    assert not apoyada_en_el_piso(cartel, 0.35, FY, CY, ALTURA_CAMARA, ALTO_CAJA, 0.2)


def test_ignora_una_caja_tapada_a_medias():
    # Una pared tapa la mitad izquierda: la mancha queda angosta y su centro, corrido.
    img = dibujar_caja(imagen_vacia(), 1.0)
    img[:, :int(CX)] = GRIS
    mancha, motivo = buscar(img)
    assert mancha is None and 'proporción' in motivo


def test_ignora_manchas_que_no_son_rectangulos():
    # Una "L" roja: tiene el área, pero llena muy poco su rectángulo.
    img = imagen_vacia()
    img[40:200, 100:120] = ROJO
    img[180:200, 100:240] = ROJO
    mancha, motivo = buscar(img)
    assert mancha is None and 'llena' in motivo


def test_vista_de_costado_tiene_proporcion_de_caja():
    # De costado (45°) se ven dos caras: ~0,35 m de ancho para 0,30 m de alto.
    mancha, motivo = buscar(dibujar_caja(imagen_vacia(), 1.0, lado=0.354))
    assert motivo is None and mancha.proporcion == pytest.approx(1.18, abs=0.05)


# --- Lidar ---------------------------------------------------------------------------------

def test_angulo_de_columna():
    # En el centro de la imagen, de frente; a la izquierda del centro, ángulo positivo.
    assert angulo_de_columna(150.0, fx=300.0, cx=150.0) == pytest.approx(0.0)
    assert angulo_de_columna(0.0, fx=300.0, cx=150.0) == pytest.approx(math.atan2(150, 300))
    assert angulo_de_columna(300.0, fx=300.0, cx=150.0) < 0


INCREMENTO = math.radians(1.0)
ANGULO_MIN = -math.pi


def lidar(**rayos):
    """Scan libre (3 m) con algunos rayos distintos: g<rayo>=<distancia>. Frente en el rayo 0."""
    rangos = [3.0] * 360
    for rayo, d in rayos.items():
        rangos[int(rayo[1:])] = d
    return rangos


def tramo(rangos, desde_deg, hasta_deg):
    return tramo_del_lidar(rangos, ANGULO_MIN, INCREMENTO, 180.0, desde_deg, hasta_deg)


def a_la_caja(rangos, centro_deg=0.0, ancho_deg=6.0):
    return distancia_a_la_caja(rangos, ANGULO_MIN, INCREMENTO, 180.0, centro_deg, ancho_deg)


def test_distancia_a_la_caja_es_la_mediana_del_tramo():
    # Caja a 1,2 m, pero el borde de una pared asoma en un rayo del sector a 0,5 m: no cuenta.
    rangos = lidar(**{f'g{r}': 1.2 for r in (358, 359, 0, 1, 2)}, g3=0.5)
    assert a_la_caja(rangos, ancho_deg=10.0) == pytest.approx(1.2)


def test_distancia_a_la_caja_en_otra_direccion():
    # El sector se mide desde el frente: 20 grados a la izquierda es el rayo 20.
    rangos = lidar(**{f'g{r}': 1.0 for r in range(16, 25)})
    assert a_la_caja(rangos, centro_deg=20.0) == pytest.approx(1.0)


def test_distancia_a_la_caja_sin_rayos_validos():
    assert a_la_caja([math.inf] * 360) is None


def test_un_tramo_parejo_es_de_una_caja_y_uno_partido_no():
    parejo = tramo(lidar(**{f'g{r}': 1.2 for r in range(-4, 5)}), -4, 4)
    assert mediana_del_tramo(parejo, 0.08) == (pytest.approx(1.2), pytest.approx(1.0))
    # Mitad caja (1,2 m) y mitad la pared de atrás (1,8 m): un borde. Pocos cerca de la mediana.
    partido = tramo(lidar(**{f'g{r}': (1.2 if r < 0 else 1.8) for r in range(-4, 5)}), -4, 4)
    _, fraccion = mediana_del_tramo(partido, 0.08)
    assert fraccion < 0.6


def test_detecta_una_pared_que_tapa_un_costado():
    # Caja a 1,5 m en [-5°, 5°]; justo a la izquierda (6° a 9°) una esquina de pared a 0,8 m.
    rangos = lidar(**{f'g{r}': 1.5 for r in range(-5, 6)}, **{f'g{r}': 0.8 for r in range(6, 10)})
    assert hay_algo_delante(tramo(rangos, 5.5, 9.5), 1.5, 0.15)
    assert not hay_algo_delante(tramo(rangos, -9.5, -5.5), 1.5, 0.15)  # del otro lado, nada


# --- Geometría de la cámara ----------------------------------------------------------------

def test_distancia_por_altura():
    # Con fy = 270 px, una caja de 0,30 m que ocupa 81 px de alto está a 1 m.
    assert distancia_por_altura(81, fy=270.0, alto_caja_m=0.30) == pytest.approx(1.0)


def test_la_altura_da_profundidad_no_distancia():
    # Caja a 1,2 m de rango pero 30° al costado: en la imagen se ve como a 1,04 m.
    prof = profundidad(1.2, math.radians(30))
    assert prof == pytest.approx(1.039, abs=1e-3)
    alto_px = FY * ALTO_CAJA / prof
    assert coinciden(prof, distancia_por_altura(alto_px, FY, ALTO_CAJA), 0.05)
    assert not coinciden(1.2, distancia_por_altura(alto_px, FY, ALTO_CAJA), 0.10)


def test_el_lidar_y_la_imagen_tienen_que_coincidir():
    assert coinciden(1.0, 1.1, tolerancia_relativa=0.25)
    # El lidar ve la esquina de una pared a 1 m, pero la caja está a 1,65 m: no coinciden.
    assert not coinciden(1.0, 1.65, tolerancia_relativa=0.25)


def test_una_caja_en_el_piso_esta_donde_tiene_que_estar():
    mancha, _ = buscar(dibujar_caja(imagen_vacia(), 1.2))
    assert apoyada_en_el_piso(mancha, 1.2, FY, CY, ALTURA_CAMARA, ALTO_CAJA, 0.2)


def test_un_rojo_a_otra_altura_no_es_una_caja():
    # Del mismo tamaño que una caja a 1,2 m, pero flotando a 0,15 m del piso (un cartel).
    mancha, _ = buscar(dibujar_caja(imagen_vacia(), 1.2, altura_base=0.15))
    assert mancha is not None
    assert not apoyada_en_el_piso(mancha, 1.2, FY, CY, ALTURA_CAMARA, ALTO_CAJA, 0.2)
    # Y una caja real a 2 m, si el lidar dice 1,2 m (ve una pared delante), tampoco.
    lejana, _ = buscar(dibujar_caja(imagen_vacia(), 2.0))
    assert not apoyada_en_el_piso(lejana, 1.2, FY, CY, ALTURA_CAMARA, ALTO_CAJA, 0.2)


def test_filas_esperadas():
    arriba, abajo = filas_esperadas(1.0, FY, CY, ALTURA_CAMARA, ALTO_CAJA)
    assert abajo - arriba == pytest.approx(FY * ALTO_CAJA)
    assert abajo > CY > arriba  # la cámara está entre el piso y la tapa de la caja


def test_mancha():
    m = Mancha(x0=10, y0=20, ancho=40, alto=50, area=1800.0)
    assert (m.x1, m.y1, m.columna) == (50, 70, 30.0)
    assert m.relleno == pytest.approx(0.9)
    assert m.proporcion == pytest.approx(0.8)


# --- Utilidades ----------------------------------------------------------------------------

def test_punto_en_el_plano():
    # Robot en (1, 2) mirando a +y, caja a 2 m justo al frente.
    x, y = punto_en_el_plano(1.0, 2.0, math.pi / 2, 0.0, 2.0)
    assert (x, y) == pytest.approx((1.0, 4.0))
    # La misma caja vista 90 grados a la izquierda del robot queda hacia -x.
    x, y = punto_en_el_plano(1.0, 2.0, math.pi / 2, math.pi / 2, 2.0)
    assert (x, y) == pytest.approx((-1.0, 2.0))


def test_yaw_de_cuaternion():
    # Giro de 90 grados alrededor de z.
    s = math.sin(math.pi / 4)
    assert yaw_de_cuaternion(0.0, 0.0, s, s) == pytest.approx(math.pi / 2)


def test_duplicados_y_promedio():
    assert cerca_de_alguno((1.0, 1.0), [(1.2, 1.0)], 0.5)
    assert not cerca_de_alguno((1.0, 1.0), [(3.0, 1.0)], 0.5)
    assert not cerca_de_alguno((1.0, 1.0), [], 0.5)
    assert promedio([(0.0, 0.0), (2.0, 4.0)]) == pytest.approx((1.0, 2.0))


def cuaternion(roll=0.0, pitch=0.0, yaw=0.0):
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy)


def test_frente_del_lidar():
    # El del simulador: laser_link con rpy 0 0 pi (al revés): el frente es el rayo de 180°.
    frente, derecho = frente_del_lidar(*cuaternion(yaw=math.pi))
    assert derecho and abs(frente) == pytest.approx(180.0)
    assert frente_del_lidar(*cuaternion()) == (pytest.approx(0.0), True)
    # Girado 90° a la izquierda: el frente del robot queda a la derecha del lidar.
    assert frente_del_lidar(*cuaternion(yaw=math.pi / 2)) == (pytest.approx(-90.0), True)
    # Dado vuelta: los ángulos del /scan van al revés y no se puede usar.
    assert not frente_del_lidar(*cuaternion(roll=math.pi))[1]


# --- Qué miró la cámara --------------------------------------------------------------------

def scan_con(angulos_deg, distancia=1.5, n=1080, frente_deg=0.0):
    """Un /scan de n rayos en 360° que solo pega en los ángulos pedidos (desde el frente)."""
    incremento = 2 * math.pi / n
    rangos = [math.inf] * n
    for a in angulos_deg:
        rangos[int(round((math.radians(a + frente_deg) + math.pi) / incremento)) % n] = distancia
    return rangos, -math.pi, incremento


def rayos(rangos, angulo_min, incremento, frente_deg=0.0, giro_deg=0.0):
    return rayos_en_la_imagen(rangos, angulo_min, incremento, frente_deg, giro_deg, FX, CX,
                              ANCHO, 5, 0.35, 2.5)


def test_solo_los_rayos_que_caen_adentro_de_la_imagen():
    # La cámara ve ±30° (320 px, fx 271,8): 40° y lo de atrás quedan afuera.
    angulos, distancias = rayos(*scan_con([0.0, 20.0, -25.0, 40.0, 180.0]))
    assert sorted(np.degrees(angulos).round()) == [-25.0, 0.0, 20.0]
    assert list(distancias) == pytest.approx([1.5, 1.5, 1.5])


def test_rayos_con_el_lidar_al_reves_y_corregidos_por_el_giro():
    # El lidar del simulador está montado al revés: el frente es el rayo de 180°.
    angulos, _ = rayos(*scan_con([10.0], frente_deg=180.0), frente_deg=180.0)
    assert np.degrees(angulos) == pytest.approx([10.0], abs=0.2)
    # Si entre el rayo y la imagen el robot giró 4° a la izquierda, en la imagen está a 6°.
    angulos, _ = rayos(*scan_con([10.0], frente_deg=180.0), frente_deg=180.0, giro_deg=4.0)
    assert np.degrees(angulos) == pytest.approx([6.0], abs=0.2)


@pytest.mark.parametrize('distancia', [0.2, 3.0, math.inf])
def test_rayos_fuera_del_rango_no_cuentan(distancia):
    angulos, _ = rayos(*scan_con([0.0], distancia=distancia))
    assert len(angulos) == 0


def rojo_en(img, columnas, prof=1.5, altura_camara=ALTURA_CAMARA, margen_px=8):
    angulos = [angulo_de_columna(c, FX, CX) for c in columnas]
    return list(hay_rojo_en_rayos(mascara_rojo(img, COLOR), angulos, [prof] * len(angulos),
                                  FX, CX, FY, CY, altura_camara, ALTO_CAJA, margen_px, 0.25))


def test_hay_rojo_donde_esta_la_caja_roja_y_no_al_lado():
    img = dibujar_caja(imagen_vacia(), 1.5, columna=100)
    assert rojo_en(img, [100, 250]) == [True, False]


def test_una_caja_azul_no_tiene_rojo():
    assert rojo_en(dibujar_caja(imagen_vacia(), 1.5, color=AZUL), [CX]) == [False]


def test_el_margen_de_columnas_agarra_el_borde_de_la_caja():
    img = dibujar_caja(imagen_vacia(), 1.5, columna=100)  # 45 px de ancho: hasta la columna 123
    assert rojo_en(img, [128], margen_px=8) == [True]
    assert rojo_en(img, [128], margen_px=2) == [False]


def test_un_rojo_fuera_de_la_altura_de_una_caja_no_cuenta():
    # Un cartel rojo arriba en la pared: donde estaría una caja a 1,5 m no hay rojo.
    img = imagen_vacia()
    img[0:20, 140:180] = ROJO
    assert rojo_en(img, [160]) == [False]
    # Sin la altura de la cámara se mira la columna entera: ahí sí cuenta.
    assert rojo_en(img, [160], altura_camara=None) == [True]


def test_la_mascara_ya_calculada_da_lo_mismo():
    img = dibujar_caja(imagen_vacia(), 1.0)
    con, _ = buscar_cajas_rojas(img, COLOR, FORMA, mascara_rojo(img, COLOR))
    sin, _ = buscar_cajas_rojas(img, COLOR, FORMA)
    assert con == sin and len(con) == 1
