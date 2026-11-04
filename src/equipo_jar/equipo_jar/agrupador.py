"""Junta las vistas sueltas de una caja en una sola posición confirmada, sin ROS.

Antes, percepción pedía N vistas *seguidas* y una sola vista mala (un cuadro movido, un rayo del
lidar en el borde) borraba todo y había que empezar de nuevo. Ahora cada vista se suma al grupo
de vistas que está cerca (como `juntar_victimas` del workshop de la semana 10):

- las vistas buenas de una caja caen todas juntas y su grupo crece;
- una vista mala cae lejos, arma un grupo propio que no llega al mínimo, y se olvida;
- el centro de un grupo es la **mediana**, que unos pocos puntos malos no mueven.

Un grupo se confirma cuando junta `confirmaciones` vistas y están apretadas (dispersión chica).
Una caja confirmada se publica una sola vez: lo que se vea cerca de ella después no cuenta.
"""
import math

import numpy as np


class Grupo:
    def __init__(self, punto, t):
        self.puntos = [punto]
        self.t_ultimo = t

    def centro(self):
        m = np.median(np.array(self.puntos), axis=0)
        return float(m[0]), float(m[1])

    def dispersion(self):
        """Mediana de las distancias de cada vista al centro (m)."""
        c = self.centro()
        return float(np.median([math.dist(p, c) for p in self.puntos]))


class AgrupadorCajas:
    def __init__(self, confirmaciones, radio_grupo_m, dispersion_max_m, olvido_s,
                 radio_duplicado_m):
        self.confirmaciones = confirmaciones
        self.radio_grupo_m = radio_grupo_m
        self.dispersion_max_m = dispersion_max_m
        self.olvido_s = olvido_s
        self.radio_duplicado_m = radio_duplicado_m
        self.grupos = []
        self.confirmadas = []

    def ya_confirmada(self, punto):
        return any(math.dist(punto, c) < self.radio_duplicado_m for c in self.confirmadas)

    def agregar(self, punto, t):
        """Suma una vista (x, y) en el instante `t` (s). Devuelve el centro si confirmó una caja."""
        self.olvidar_viejos(t)
        if self.ya_confirmada(punto):
            return None
        grupo = self.grupo_cercano(punto)
        if grupo is None:
            grupo = Grupo(punto, t)
            self.grupos.append(grupo)
        else:
            grupo.puntos.append(punto)
            grupo.t_ultimo = t
        if (len(grupo.puntos) < self.confirmaciones
                or grupo.dispersion() > self.dispersion_max_m):
            return None
        centro = grupo.centro()
        self.confirmadas.append(centro)
        # El grupo confirmado y cualquier otro pegado a él son la misma caja.
        self.grupos = [g for g in self.grupos if g is not grupo
                       and math.dist(g.centro(), centro) >= self.radio_duplicado_m]
        return centro

    def grupo_cercano(self, punto):
        mejor, d_mejor = None, self.radio_grupo_m
        for g in self.grupos:
            d = math.dist(g.centro(), punto)
            if d < d_mejor:
                mejor, d_mejor = g, d
        return mejor

    def olvidar_viejos(self, t):
        """Los grupos sin vistas nuevas hace más de `olvido_s` se borran: si eran una caja, al
        volver a verla se arma otro; si eran un error, no se juntan con otro error de más tarde."""
        self.grupos = [g for g in self.grupos if t - g.t_ultimo <= self.olvido_s]

    def olvidar_pendientes(self):
        """Borra los grupos sin confirmar (por ejemplo, si cambió de dónde sale la pose)."""
        self.grupos = []
