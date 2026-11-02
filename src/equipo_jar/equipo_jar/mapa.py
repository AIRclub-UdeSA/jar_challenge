"""Consultas al mapa (/map) que usa percepción, sin ROS.

Las cajas NO están en el mapa (son obstáculos que el mapa no tiene), pero el mapa igual sirve
para descartar ubicaciones imposibles:
- una caja no puede estar adentro de una pared ni afuera del laberinto;
- el robot no puede ver una caja a través de una pared.
Si pasa alguna de las dos, la pose del robot está mal (AMCL perdido) o el rojo no es una caja.

No se guarda nada del mapa en el código: se arma con el /map que llega en la corrida.
"""
import math

import numpy as np

DESCONOCIDA = -1


class GrillaMapa:
    """Un nav_msgs/OccupancyGrid como matriz, con consultas en metros (marco `map`)."""

    def __init__(self, ancho, alto, resolucion, origen_x, origen_y, origen_yaw, datos,
                 umbral_ocupado=50):
        self.ancho = ancho
        self.alto = alto
        self.resolucion = resolucion
        self.origen = (origen_x, origen_y, origen_yaw)
        # Filas = y, columnas = x, como viene en OccupancyGrid.data.
        self.celdas = np.asarray(datos, dtype=np.int16).reshape(alto, ancho)
        self.umbral_ocupado = umbral_ocupado

    @classmethod
    def desde_mensaje(cls, msg, umbral_ocupado=50):
        info = msg.info
        q = info.origin.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        return cls(info.width, info.height, info.resolution, info.origin.position.x,
                   info.origin.position.y, yaw, msg.data, umbral_ocupado)

    def celda(self, x, y):
        """(fila, columna) de un punto del mapa, o None si cae afuera de la grilla."""
        ox, oy, oyaw = self.origen
        dx, dy = x - ox, y - oy
        c, s = math.cos(oyaw), math.sin(oyaw)
        lx, ly = c * dx + s * dy, -s * dx + c * dy
        col, fila = int(math.floor(lx / self.resolucion)), int(math.floor(ly / self.resolucion))
        if 0 <= fila < self.alto and 0 <= col < self.ancho:
            return fila, col
        return None

    def celdas_de(self, xs, ys):
        """Como `celda`, para muchos puntos a la vez: (filas, columnas, adentro) como arrays.

        `adentro` dice qué puntos caen en la grilla; filas y columnas de los de afuera no sirven.
        """
        ox, oy, oyaw = self.origen
        dx, dy = np.asarray(xs, dtype=float) - ox, np.asarray(ys, dtype=float) - oy
        c, s = math.cos(oyaw), math.sin(oyaw)
        lx, ly = c * dx + s * dy, -s * dx + c * dy
        columnas = np.floor(lx / self.resolucion).astype(int)
        filas = np.floor(ly / self.resolucion).astype(int)
        adentro = (filas >= 0) & (filas < self.alto) & (columnas >= 0) & (columnas < self.ancho)
        return filas, columnas, adentro

    def ocupada(self, x, y):
        celda = self.celda(x, y)
        return celda is not None and self.celdas[celda] >= self.umbral_ocupado

    def libre(self, x, y):
        celda = self.celda(x, y)
        return celda is not None and 0 <= self.celdas[celda] < self.umbral_ocupado

    def hay_libre_cerca(self, x, y, radio_m):
        """¿Hay alguna celda libre a menos de `radio_m` de (x, y)?

        El centro estimado de una caja pegada a una pared puede caer, por unos cm de error, en
        la celda de la pared: por eso no se exige que esa celda justa esté libre.
        """
        pasos = int(math.ceil(radio_m / self.resolucion))
        for i in range(-pasos, pasos + 1):
            for j in range(-pasos, pasos + 1):
                dx, dy = i * self.resolucion, j * self.resolucion
                if math.hypot(dx, dy) <= radio_m + 1e-9 and self.libre(x + dx, y + dy):
                    return True
        return False

    def linea_de_vista(self, desde, hasta, ignorar_inicio_m, ignorar_fin_m):
        """¿El segmento `desde` -> `hasta` no cruza paredes del mapa?

        Se ignoran los primeros y los últimos centímetros: con unos cm de error en la pose, el
        robot pegado a una pared o una caja contra una pared "tocan" la pared en el mapa.
        """
        largo = math.dist(desde, hasta)
        paso = self.resolucion / 2.0  # medio casillero: no saltea paredes finas en diagonal
        n = int(largo / paso)
        for k in range(n + 1):
            recorrido = k * paso
            if recorrido < ignorar_inicio_m or recorrido > largo - ignorar_fin_m:
                continue
            f = recorrido / largo if largo > 0 else 0.0
            x = desde[0] + f * (hasta[0] - desde[0])
            y = desde[1] + f * (hasta[1] - desde[1])
            if self.ocupada(x, y):
                return False
        return True
