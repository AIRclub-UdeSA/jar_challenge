import cv2
import numpy as np

from equipo_jar.core import segments

def mascara_rojo(imagen_hsv, hue_bajo_1, hue_alto_1, hue_bajo_2, hue_alto_2, saturacion_min, valor_min):
    """Máscara binaria (255 = rojo). """
    bajo1 = np.array([hue_bajo_1, saturacion_min, valor_min])
    alto1 = np.array([hue_alto_1, 255, 255])
    bajo2 = np.array([hue_bajo_2, saturacion_min, valor_min])
    alto2 = np.array([hue_alto_2, 255, 255])
    mascara1 = cv2.inRange(imagen_hsv, bajo1, alto1)
    mascara2 = cv2.inRange(imagen_hsv, bajo2, alto2)
    return cv2.bitwise_or(mascara1, mascara2)

def puntos_laser_a_pixeles(xyz, rotacion, traslacion, fx, fy, cx, cy):
    """
    Convertir cada punto del lidar en el pixel (u, v) donde caería si la cámara lo estuviera mirando.

    Parámetros: xyz son los puntos del lidar (N, 3) en su frame. Rotacion es una matriz 3x3 y traslacion un vector de 3 (la tf laser_link -> cam_1_color_optical_frame. 
    La matriz sale de su cuaternión con scipy: Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()); fx, fy, cx, cy son los intrínsecos de la cámara (de camera_info).
    """
    punto_camara = xyz @ rotacion.T + traslacion
    x_cam = punto_camara[:, 0]
    y_cam = punto_camara[:, 1]
    z_cam = punto_camara[:, 2]

    adelante = z_cam > 0.01

    #proyeccion pinhole
    with np.errstate(divide='ignore', invalid='ignore'):
        u = fx * (x_cam/z_cam) + cx
        v = fy * (y_cam / z_cam) + cy
    
    return u,v,adelante

class ProcesadorVision:
    def __init__(self, hue_rojo_bajo_1, hue_rojo_alto_1, hue_rojo_bajo_2, hue_rojo_alto_2, saturacion_min, valor_min, mitad_caja):
        # Umbrales HSV y tamaño de la caja: los valores están solo en params.yaml (sección nodo_equipo), comentados; se pueden retocar después del
        # cierre de entregas. Elegidos con test_color_lighting.py (fotos reales del bag con luz alterada)
        self.hue_rojo_bajo_1 = hue_rojo_bajo_1
        self.hue_rojo_alto_1 = hue_rojo_alto_1
        self.hue_rojo_bajo_2 = hue_rojo_bajo_2
        self.hue_rojo_alto_2 = hue_rojo_alto_2
        self.saturacion_min = saturacion_min
        self.valor_min = valor_min
        self.mitad_caja = mitad_caja # m: de la cara que ve el láser al centro, que es lo que mide el juez

        # Contra el paralaje en los bordes de la caja
        self.erosion_px = 2 # píxeles que se le sacan a la silueta roja de cada lado
        self.salto_max = 0.05 # m: un salto mayor entre rayos rojos vecinos corta el tramo
        self.rayos_min = 5 # tramos más cortos se descartan (una caja a 3 m tiene ~14 rayos por cara)

    def extraer_coordenadas_cajas(self, imagen_bgr, rangos, angulos, rotacion, traslacion, fx, fy, cx, cy):
        """
        Toma la imagen y el láser, devuelve una lista de puntos (x_local, y_local) indicando dónde hay cajas rojas.
        Quedan en el frame del lidar (laser_link), que está girado 180° respecto del robot: falta llevarlos a map con TF.

        Paralaje: un rayo que pasa rozando la caja pega en la pared, pero en la foto cae 1-2 px adentro del borde rojo
        (cámara y lidar en lugares distintos, bordes difusos) y daría una caja falsa metida en la pared. 
        
        Dos filtros: se erosiona la silueta roja y los rayos rojos se cortan en tramos por saltos de distancia (una cara es una tira continua, un intruso, un rayo
        suelto que salta decenas de cm). 
        
        En escenas sintéticas, las escenas con detecciones falsas bajan de 58 % a 4 % con pared detrás y de 48 % a 0 % en el borde de la foto (test_detector.py).
        """
        xyz = np.column_stack([rangos * np.cos(angulos), rangos * np.sin(angulos), np.zeros_like(rangos)]) # Polares -> Cartesianas

        # 1. Armar máscara roja
        img_hsv = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2HSV)
        red_mask = mascara_rojo(
            img_hsv, self.hue_rojo_bajo_1, self.hue_rojo_alto_1, 
            self.hue_rojo_bajo_2, self.hue_rojo_alto_2, 
            self.saturacion_min, self.valor_min
        )
        lado = 2 * self.erosion_px + 1
        red_mask = cv2.erode(red_mask, np.ones((lado, lado), np.uint8))  # el borde es donde foto y lidar no coinciden

        # 2. Proyectar el láser en la imagen
        u, v, adelante = puntos_laser_a_pixeles(xyz, rotacion, traslacion, fx, fy, cx, cy)

        # 3. Filtrar los rayos válidos que caen dentro de la foto
        alto, ancho = imagen_bgr.shape[:2]
        validos = np.isfinite(rangos) & (rangos > 0.1) # Filtro básico de validez
        dentro_de_imagen = adelante & validos & (u >= 0) & (u < ancho) & (v >= 0) & (v < alto)

        # 4. Encontrar qué rayos pegaron en píxeles rojos
        idx_img = np.where(dentro_de_imagen)[0]
        v_val = v[idx_img].astype(int)
        u_val = u[idx_img].astype(int)
        
        # Filtramos de esos índices, cuáles cayeron en la máscara > 0
        es_rojo_mask = red_mask[v_val, u_val] > 0
        idx_rojos = idx_img[es_rojo_mask] # Índices finales de los rayos del láser que tocaron la caja

        # Tramos de rayos rojos con distancia continua: los no rojos y los saltos cortan, los tramos cortos se descartan
        r = np.full(len(rangos), np.nan)
        r[idx_rojos] = rangos[idx_rojos]
        corte = int(np.argmax(np.isnan(r))) # el scan arranca en un rayo no rojo: una caja justo en la costura no queda partida en dos
        tramos = segments(np.roll(r, -corte), self.salto_max, self.rayos_min)
        idx_rojos = (np.concatenate(tramos).astype(int) + corte) % len(r) if tramos else np.array([], int)

        # 5. CONVERSIÓN GEOMÉTRICA
        # El láser rebota en la cara de la caja y el juez mide desde el centro: se estira cada punto
        # media caja más a lo largo de su rayo, (r + mitad_caja) / r.
        r = rangos[idx_rojos]
        centros = xyz[idx_rojos, :2] * ((r + self.mitad_caja) / r)[:, None]

        return [tuple(p) for p in centros.tolist()]