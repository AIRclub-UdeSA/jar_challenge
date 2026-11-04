"""Tipos compartidos por la navegación."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float


@dataclass(frozen=True)
class Comando:
    """Velocidad en el marco del robot (base_footprint)."""
    vx: float
    vy: float
    wz: float

    @staticmethod
    def cero():
        return Comando(0.0, 0.0, 0.0)

    def es_cero(self):
        return self.vx == 0.0 and self.vy == 0.0 and self.wz == 0.0
