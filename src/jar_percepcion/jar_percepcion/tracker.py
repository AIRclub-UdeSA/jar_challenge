"""Tracker de objetivos en `map` (ADR-0006), sin ROS.

Une las detecciones cercanas en un mismo objetivo. Un objetivo queda confirmado
cuando tiene suficientes observaciones, poca dispersión y se vio desde lugares
distintos; una vez confirmado no se desconfirma (la misión ya lo puede haber
reportado).
"""
import math
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class ConfigTracker:
    radio_union_m: float = 0.35
    observaciones_min: int = 5
    dispersion_max_m: float = 0.10
    separacion_vistas_min_m: float = 0.30
    observaciones_max: int = 60      # para la media se usan las últimas


@dataclass
class Objetivo:
    id: int
    obs: List[Tuple[float, float]] = field(default_factory=list)
    vistas: List[Tuple[float, float]] = field(default_factory=list)
    n: int = 0
    confirmado: bool = False

    @property
    def x(self):
        return sum(p[0] for p in self.obs) / len(self.obs)

    @property
    def y(self):
        return sum(p[1] for p in self.obs) / len(self.obs)

    def dispersion(self):
        mx, my = self.x, self.y
        return math.sqrt(sum((px - mx) ** 2 + (py - my) ** 2 for px, py in self.obs) / len(self.obs))

    def separacion_vistas(self):
        mejor = 0.0
        for i, a in enumerate(self.vistas):
            for b in self.vistas[i + 1:]:
                mejor = max(mejor, math.hypot(a[0] - b[0], a[1] - b[1]))
        return mejor


class Tracker:
    def __init__(self, config):
        self.config = config
        self.objetivos: List[Objetivo] = []
        self._proximo_id = 1

    def agregar(self, x, y, desde):
        """Suma una detección en (x, y), vista desde la posición `desde` del robot."""
        if not (math.isfinite(x) and math.isfinite(y)):
            return None
        c = self.config
        cercano, d_min = None, None
        for obj in self.objetivos:
            d = math.hypot(obj.x - x, obj.y - y)
            if d < c.radio_union_m and (d_min is None or d < d_min):
                cercano, d_min = obj, d
        if cercano is None:
            cercano = Objetivo(self._proximo_id)
            self._proximo_id += 1
            self.objetivos.append(cercano)
        cercano.obs.append((x, y))
        del cercano.obs[:-c.observaciones_max]
        cercano.n += 1
        if not cercano.vistas or min(math.hypot(desde[0] - v[0], desde[1] - v[1])
                                     for v in cercano.vistas) > 0.05:
            cercano.vistas.append(tuple(desde))
            del cercano.vistas[:-30]
        if (not cercano.confirmado and cercano.n >= c.observaciones_min
                and cercano.dispersion() <= c.dispersion_max_m
                and cercano.separacion_vistas() >= c.separacion_vistas_min_m):
            cercano.confirmado = True
        return cercano

    def resumen(self):
        return [{'id': o.id, 'x': round(o.x, 3), 'y': round(o.y, 3), 'n': o.n,
                 'dispersion': round(o.dispersion(), 3), 'confirmado': o.confirmado}
                for o in self.objetivos]
