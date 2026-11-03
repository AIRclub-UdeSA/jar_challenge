"""Planner watch: notice when planner_server plans from a stale robot pose.

Seen in the official simulator (a practice maze, 2026-11-02): planner_server's TF view froze, so
every new path started where the robot had been seconds before, while other nodes saw the
right pose. MPPI followed those paths until the robot was ~1.5 m from their start, then
rejected every path ("Resulting plan has 0 poses") until the run ended. NavFn and Smac 2D
both showed it. Restarting planner_server (cleanup + configure) builds a new TF buffer.

Pure logic: localizacion.py feeds each /plan start and the robot pose it sees in TF.
"""
import math


class VigiaPlanner:
    """Counts consecutive paths that start far from the robot.

    `p`: dict with vigia_planner_distancia_m (<= 0: off), vigia_planner_veces,
    vigia_planner_espera_s (minimum time between restarts).
    """

    def __init__(self, p):
        self.p = p
        self.seguidos = 0
        self.ultimo_reinicio = None
        self.reinicios = 0

    def con_plan(self, ahora, inicio_xy, robot_xy):
        """One path. Returns the distance (m) if planner_server must restart now, else None."""
        limite = self.p['vigia_planner_distancia_m']
        if limite <= 0:
            return None
        distancia = math.dist(inicio_xy, robot_xy)
        if not math.isfinite(distancia) or distancia <= limite:
            self.seguidos = 0
            return None
        self.seguidos += 1
        if self.seguidos < self.p['vigia_planner_veces']:
            return None
        if (self.ultimo_reinicio is not None
                and ahora - self.ultimo_reinicio < self.p['vigia_planner_espera_s']):
            return None
        self.seguidos = 0
        self.ultimo_reinicio = ahora
        self.reinicios += 1
        return distancia
