"""Política de reportes (ADR-0008, requisitos RQ04, RQ14, RQ16, RQ18), sin ROS.

Hay como máximo N reportes. Un reporte errado no resta, pero gasta un cupo.
Por eso:
- se reporta cada objetivo confirmado una sola vez, apenas se confirma;
- nunca dos reportes a menos de `radio_union` entre sí;
- al cierre, los cupos que sobran se gastan en los mejores candidatos sin confirmar.

Un objetivo es un dict con al menos `x`, `y`, `n` (observaciones) y `confirmado`.
"""
import math


class PoliticaReportes:
    def __init__(self, cupos, radio_union):
        self.cupos = int(cupos)
        self.radio_union = float(radio_union)
        self.hechos = []          # (x, y) de cada reporte emitido, en orden

    @property
    def restantes(self):
        return max(0, self.cupos - len(self.hechos))

    def completo(self):
        return self.restantes == 0

    def _lejos_de_lo_reportado(self, obj):
        return all(math.hypot(obj['x'] - x, obj['y'] - y) >= self.radio_union
                   for x, y in self.hechos)

    def _elegir(self, candidatos):
        elegidos = []
        for obj in sorted(candidatos, key=lambda o: -o['n']):
            if self.restantes == 0:
                break
            if not (math.isfinite(obj['x']) and math.isfinite(obj['y'])):
                continue
            if self._lejos_de_lo_reportado(obj):
                self.hechos.append((obj['x'], obj['y']))
                elegidos.append((obj['x'], obj['y']))
        return elegidos

    def nuevos(self, objetivos):
        """Confirmados que todavía no se reportaron. Los marca como reportados."""
        return self._elegir([o for o in objetivos if o.get('confirmado')])

    def sobrantes(self, objetivos):
        """Al cierre: gasta los cupos que quedan en candidatos sin confirmar."""
        return self._elegir([o for o in objetivos if not o.get('confirmado') and o['n'] > 0])
