"""Watch AMCL with the LiDAR and say when to move it to a better pose (no ROS here).

While the robot is still, localizacion scores the last sweep at AMCL's pose (pose_inicial's
score: share of points on a wall of /map, 0..1; 0.8-0.9 when it is right). A low score
means AMCL is lost or the robot sees things missing from the map (boxes, people). Only
then it searches nearby poses. If one scores clearly better, twice in a row at the same
place, AMCL is moved there. A real robot that slips or gets bumped is the case this is for;
in the simulator AMCL has not been lost since alpha4 0.01 (notes/hans-die/NOTES.md).
"""
import math


class Vigia:
    """`p`: live params vigia_puntaje_bajo, vigia_puntaje_bueno, vigia_mejora_minima,
    vigia_distancia_minima_m, vigia_coincidencia_m, vigia_confirmaciones."""

    def __init__(self, p):
        self.p = p
        self.candidata = None       # (x, y, yaw) seen better last time
        self.vistas = 0
        self.reubicaciones = 0

    def hace_falta_buscar(self, puntaje_amcl):
        return puntaje_amcl < self.p['vigia_puntaje_bajo']

    def revisar(self, pose_amcl, puntaje_amcl, mejor):
        """mejor: (x, y, yaw, score) of the nearby search. Returns a pose to move AMCL to, or None."""
        p = self.p
        x, y, yaw, puntaje = mejor
        lejos = math.hypot(x - pose_amcl[0], y - pose_amcl[1]) >= p['vigia_distancia_minima_m'] or \
            abs(math.atan2(math.sin(yaw - pose_amcl[2]), math.cos(yaw - pose_amcl[2]))) >= 0.1
        mejor_claro = puntaje >= p['vigia_puntaje_bueno'] and \
            puntaje >= puntaje_amcl + p['vigia_mejora_minima']
        if not (lejos and mejor_claro):
            self.candidata, self.vistas = None, 0
            return None
        if self.candidata is not None and math.hypot(
                x - self.candidata[0], y - self.candidata[1]) <= p['vigia_coincidencia_m']:
            self.vistas += 1
        else:
            self.vistas = 1
        self.candidata = (x, y, yaw)
        if self.vistas >= p['vigia_confirmaciones']:
            self.candidata, self.vistas = None, 0
            self.reubicaciones += 1
            return (x, y, yaw)
        return None

    def descartar(self):
        """AMCL moved (the robot drove): an old candidate no longer applies."""
        self.candidata, self.vistas = None, 0
