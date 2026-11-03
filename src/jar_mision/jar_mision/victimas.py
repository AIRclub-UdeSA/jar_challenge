"""Agrupamiento de detecciones de cajas rojas, sin dependencias de ROS."""

import math

import numpy as np


def centro_desde_cara(robot, deteccion, mitad_caja):
    """
    Corre la detección de la cara visible media caja hacia adentro.

    Devuelve None si la detección coincide con la posición del robot.
    """
    dx = deteccion[0] - robot[0]
    dy = deteccion[1] - robot[1]
    d = math.hypot(dx, dy)
    if d < 1e-6:
        return None
    return deteccion[0] + mitad_caja * dx / d, deteccion[1] + mitad_caja * dy / d


class AgrupadorVictimas:
    """
    Junta detecciones cercanas y decide cuándo reportar cada grupo.

    La mediana del grupo ignora las pocas detecciones que caen lejos de la
    caja, que con un promedio la desplazarían.
    """

    def __init__(self, victimas=2, detecciones_minimas=30, radio_grupo=0.30,
                 inmediato=True):
        self.victimas = victimas
        self.minimas = detecciones_minimas
        self.radio = radio_grupo
        # Si es False, nada se reporta hasta llamar a cerrar().
        self.inmediato = inmediato
        self.grupos = []
        self.publicados = 0
        self.reportes = []

    def agregar(self, x, y):
        """Suma el punto a su grupo; devuelve la posición a reportar o None."""
        for grupo in self.grupos:
            centro = np.median(np.array(grupo['puntos']), axis=0)
            if math.hypot(x - centro[0], y - centro[1]) < self.radio:
                grupo['puntos'].append((x, y))
                return self.listo(grupo) if self.inmediato else None
        grupo = {'puntos': [(x, y)], 'publicado': False}
        self.grupos.append(grupo)
        return self.listo(grupo) if self.inmediato else None

    def cerrar(self, minimas_final, separacion):
        """
        Elige los reportes que quedan: primero los grupos con más detecciones.

        Usa los grupos con detecciones_minimas y, si sobran reportes, los que
        tienen al menos minimas_final: un reporte errado no resta puntos. Saltea
        grupos a menos de `separacion` de un reporte, que serían la misma caja.
        """
        elegidos = []
        candidatos = sorted((g for g in self.grupos if not g['publicado']),
                            key=lambda g: len(g['puntos']), reverse=True)
        for minimas in (self.minimas, minimas_final):
            for grupo in candidatos:
                if self.publicados >= self.victimas:
                    return elegidos
                if grupo['publicado'] or len(grupo['puntos']) < minimas:
                    continue
                x, y = (float(v) for v in np.median(np.array(grupo['puntos']), axis=0))
                if any(math.hypot(x - rx, y - ry) < separacion for rx, ry in self.reportes):
                    continue
                grupo['publicado'] = True
                self.publicados += 1
                self.reportes.append((x, y))
                elegidos.append((x, y))
        return elegidos

    def listo(self, grupo):
        """Marca el grupo como publicado si alcanzó las detecciones mínimas."""
        if (grupo['publicado'] or self.publicados >= self.victimas
                or len(grupo['puntos']) < self.minimas):
            return None
        grupo['publicado'] = True
        self.publicados += 1
        x, y = np.median(np.array(grupo['puntos']), axis=0)
        self.reportes.append((float(x), float(y)))
        return float(x), float(y)
