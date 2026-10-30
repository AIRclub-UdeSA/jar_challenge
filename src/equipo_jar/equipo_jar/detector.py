import cv2
import numpy as np

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
    def __init__(self):
        # Guardamos los umbrales HSV directamente acá
        self.hue_rojo_bajo_1 = 0.0
        self.hue_rojo_alto_1 = 10.0
        self.hue_rojo_bajo_2 = 170.0
        self.hue_rojo_alto_2 = 180.0
        self.saturacion_min = 120.0
        self.valor_min = 80.0

    def extraer_coordenadas_cajas(self, imagen_bgr, rangos, angulos, rotacion, traslacion, fx, fy, cx, cy):
        """
        Toma la imagen y el láser, devuelve una lista de puntos (x_local, y_local) indicando dónde hay cajas rojas.
        Quedan en el frame del lidar (laser_link), que está girado 180° respecto del robot: falta llevarlos a map con TF.
        """
        xyz = np.column_stack([rangos * np.cos(angulos), rangos * np.sin(angulos), np.zeros_like(rangos)]) # Polares -> Cartesianas

        # 1. Armar máscara roja
        img_hsv = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2HSV)
        red_mask = mascara_rojo(
            img_hsv, self.hue_rojo_bajo_1, self.hue_rojo_alto_1, 
            self.hue_rojo_bajo_2, self.hue_rojo_alto_2, 
            self.saturacion_min, self.valor_min
        )

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

        # 5. CONVERSIÓN GEOMÉTRICA
        # El láser rebota en la cara de la caja y el juez mide desde el centro: se estira cada punto
        # 12.5 cm más a lo largo de su rayo, (r + 0.125) / r.
        r = rangos[idx_rojos]
        centros = xyz[idx_rojos, :2] * ((r + 0.125) / r)[:, None]

        return [tuple(p) for p in centros.tolist()]