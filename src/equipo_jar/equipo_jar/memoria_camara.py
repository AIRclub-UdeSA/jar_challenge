"""Qué partes del laberinto miró la cámara y si había rojo. Sin ROS.

Para qué: nodo_mision va a inspeccionar de cerca cada objeto que el lidar ve y que no está en
/map. Muchos no son cajas rojas: las distractoras (nunca son rojas, reglamento §6.3), los señuelos
(§6.2) y pedazos de pared que en el /map faltan o están corridos. En 21 corridas de simulación,
82 de 120 inspecciones fueron a objetos así; las de pedazos de pared tardaron 15 s de media y
causaron 11 de las 18 trabas. Si la cámara ya miró el objeto y no había rojo, no hace falta ir.

Cómo: percepción, con cada imagen, toma los rayos del lidar que caen adentro de la imagen
(deteccion.rayos_en_la_imagen) y marca la celda del /map donde pega cada uno: "vista sin rojo", o
"vista con rojo" si en la imagen había rojo donde estaría una caja apoyada ahí
(deteccion.hay_rojo_en_rayos). La grilla se publica en /equipo/vistas_camara (OccupancyGrid con
la misma geometría que /map) y nodo_mision la consulta con `veredicto` para cada objeto.

Valores de la grilla:
- 0: la cámara nunca vio esa celda;
- 1 a 99: cuántas imágenes la vieron sin rojo (se satura en 99);
- 100: alguna imagen vio rojo ahí. Queda así para siempre: si hay que equivocarse, que sea
  inspeccionando de más.
"""
import cv2
import numpy as np

ROJO = 100           # valor de una celda donde alguna imagen vio rojo
MAX_VECES = 99       # tope de imágenes sin rojo que se cuentan por celda

# Lo que se sabe de un objeto (veredicto)
ROJO_VISTO = 'rojo visto'
SIN_ROJO = 'visto sin rojo'
SIN_VER = 'la cámara no lo vio'


class MemoriaCamara:
    """La grilla de lo que miró la cámara, del tamaño del /map (filas = y, columnas = x)."""

    def __init__(self, alto, ancho):
        self.alto = alto
        self.ancho = ancho
        self.veces = np.zeros((alto, ancho), dtype=np.uint8)
        self.rojo = np.zeros((alto, ancho), dtype=bool)

    def anotar(self, filas, columnas, con_rojo):
        """Una imagen: las celdas donde pegaron los rayos que se vieron y si había rojo.

        Cada celda suma a lo sumo una vez por imagen, aunque le peguen varios rayos: lo que se
        cuenta es cuántas veces se miró, no cuántos rayos le tocaron.
        """
        filas = np.asarray(filas, dtype=int)
        columnas = np.asarray(columnas, dtype=int)
        con_rojo = np.asarray(con_rojo, dtype=bool)
        self.rojo[filas[con_rojo], columnas[con_rojo]] = True
        sin_rojo = np.unique(filas[~con_rojo] * self.ancho + columnas[~con_rojo])
        f, c = np.divmod(sin_rojo, self.ancho)
        self.veces[f, c] = np.minimum(self.veces[f, c].astype(int) + 1, MAX_VECES)

    def datos(self):
        """La grilla con los valores de arriba, como int8 en el orden de OccupancyGrid.data."""
        grilla = self.veces.astype(np.int8)
        grilla[self.rojo] = ROJO
        return grilla.ravel()


def veredicto(vistas, objeto, veces_min, celdas_min):
    """Qué vio la cámara de un objeto: ROJO_VISTO, SIN_ROJO o SIN_VER.

    vistas: la grilla de /equipo/vistas_camara (o un pedazo), valores de 0 a 100.
    objeto: máscara booleana del mismo tamaño con las celdas del objeto (donde pegó el lidar).
    Se agranda una celda para cada lado: percepción redondea las posiciones a celdas distinto
    que nodo_mision (hasta media celda de diferencia).

    - ROJO_VISTO si alguna celda del objeto vio rojo alguna vez;
    - SIN_ROJO si al menos `celdas_min` celdas del objeto se vieron `veces_min` imágenes o más
      (y ninguna con rojo);
    - SIN_VER en otro caso: todavía no se sabe, hay que ir a mirarlo.
    """
    objeto = np.asarray(objeto, dtype=bool)
    if not objeto.any():
        return SIN_VER
    cerca = cv2.dilate(objeto.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    valores = np.asarray(vistas)[cerca].astype(int)
    if (valores >= ROJO).any():
        return ROJO_VISTO
    if int(((valores >= veces_min) & (valores < ROJO)).sum()) >= celdas_min:
        return SIN_ROJO
    return SIN_VER
