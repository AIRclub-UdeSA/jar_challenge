"""Seguimiento de candidatos a víctima, sin ROS (se testea en test/test_seguimiento.py).

detector_obstaculos publica, en cada scan, los objetos rojos no mapeados que ve en ese
instante: con ruido, repetidos y alguno falso. Acá se acumulan en "tracks" (un candidato
por caja) y un track se confirma cuando juntó suficientes observaciones. Cada reporte en
/victimas gasta uno de los N, así que se reporta una sola vez y solo lo confirmado.
"""
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class Track:
    x: float
    y: float
    observaciones: int = 1
    ultima_vez: float = 0.0      # s, reloj del nodo
    confirmado: bool = False


def cara_a_centro(x_obj: float, y_obj: float, x_robot: float, y_robot: float,
                  medio_lado: float) -> Tuple[float, float]:
    """El lidar ve la cara de la caja que mira al robot: el centro está `medio_lado` más
    atrás, sobre la recta robot -> objeto."""
    dx, dy = x_obj - x_robot, y_obj - y_robot
    d = math.hypot(dx, dy)
    if d < 1e-6:
        return x_obj, y_obj
    return x_obj + dx / d * medio_lado, y_obj + dy / d * medio_lado


class Seguidor:
    def __init__(self, radio_asociacion: float, observaciones_minimas: int,
                 tiempo_olvido: float):
        self.radio_asociacion = radio_asociacion
        self.observaciones_minimas = observaciones_minimas
        self.tiempo_olvido = tiempo_olvido
        self.tracks: List[Track] = []

    def mas_cercano(self, x: float, y: float) -> Optional[Track]:
        mejor, d_mejor = None, self.radio_asociacion
        for track in self.tracks:
            d = math.hypot(track.x - x, track.y - y)
            if d <= d_mejor:
                mejor, d_mejor = track, d
        return mejor

    def observar(self, x: float, y: float, t: float) -> Optional[Track]:
        """Suma una observación. Devuelve el track solo en el momento en que se confirma
        (una única vez por track); si no, None."""
        track = self.mas_cercano(x, y)
        if track is None:
            self.tracks.append(Track(x, y, 1, t))
            track = self.tracks[-1]
        else:
            # Promedio incremental: después de n observaciones, (x, y) es la media de todas.
            track.observaciones += 1
            track.x += (x - track.x) / track.observaciones
            track.y += (y - track.y) / track.observaciones
            track.ultima_vez = t
        if not track.confirmado and track.observaciones >= self.observaciones_minimas:
            track.confirmado = True
            return track
        return None

    def olvidar(self, t: float):
        """Descarta los candidatos sin confirmar que hace rato no se ven (detecciones
        sueltas, ruido). Los confirmados quedan para no volver a reportar la misma caja."""
        self.tracks = [tr for tr in self.tracks
                       if tr.confirmado or t - tr.ultima_vez <= self.tiempo_olvido]
