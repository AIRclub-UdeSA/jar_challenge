"""When AMCL's heading needs a fix after a turn, and when it is fixed again (no ROS here).

On the real robot /odom counts turns ~25 % too much. AMCL fixes the heading with the
LiDAR, but only while the robot moves: when it stops after a turn the error stays,
just when the camera looks. So after a turn of at least `giro_minimo_rad` the heading is
"in doubt" until AMCL has done `actualizaciones_giro` updates with scans taken after the
turn ended. Meanwhile the node asks AMCL for no-motion updates
(`/request_nomotion_update`): AMCL then uses the next scan even if the robot barely moved.

A turn is a run of /odom samples with |wz| >= `girando_rads`, and its size is the signed
yaw change over that run. Small corrections while driving (wobble, S-curves) stay below
`girando_rads` or cancel out, and the odometry error they add cancels too.
Checked on the week-10 bag (notes/hans-die/NOTES.md).
"""
import math


def diferencia_angulo(a, b):
    """a - b in (-pi, pi]."""
    return math.atan2(math.sin(a - b), math.cos(a - b))


class CorreccionGiro:
    """Feed it /odom and AMCL's updates; read `en_duda` and `pedir_actualizacion()`.

    `p` is the node's live parameter dict: actualizaciones_giro (0 = off),
    giro_minimo_rad, girando_rads, max_correccion_s, max_duda_s.
    Times are message stamps in seconds (same clock for /odom and /amcl_pose).
    """

    def __init__(self, p):
        self.p = p
        self.yaw_previo = None
        self.girado = 0.0          # signed rad of the current turn (0 when not turning)
        self.en_duda = False       # heading not trusted: turned and not fixed yet
        self.inicio_duda = None    # stamp when the doubt started
        self.fin_giro = None       # stamp when the last turn ended (None: still turning)
        self.actualizaciones = 0   # AMCL updates with scans after fin_giro
        self.t = None
        self.correcciones = 0      # finished fixes, for the log
        self.vencidas = 0          # fixes ended by max_correccion_s or max_duda_s
        self.ultima = None         # (updates, seconds) of the last finished fix

    def con_odom(self, t, yaw, wz):
        self.t = t
        gira = abs(wz) >= self.p['girando_rads']
        if gira and self.yaw_previo is not None:
            self.girado += diferencia_angulo(yaw, self.yaw_previo)
        elif not gira:
            self.girado = 0.0
        self.yaw_previo = yaw
        if self.p['actualizaciones_giro'] <= 0:
            self.en_duda = False
            self.inicio_duda = self.fin_giro = None
            self.actualizaciones = 0
            return
        if gira and abs(self.girado) >= self.p['giro_minimo_rad']:
            # A real turn (new, or still going): wait for scans after it.
            if not self.en_duda:
                self.en_duda = True
                self.inicio_duda = t
            self.fin_giro = None
            self.actualizaciones = 0
        elif self.en_duda and self.fin_giro is None and not gira:
            self.fin_giro = t
        if self.en_duda and (
                (self.fin_giro is not None and t - self.fin_giro > self.p['max_correccion_s'])
                or t - self.inicio_duda > self.p['max_duda_s']):
            self.vencidas += 1
            self._terminar()

    def con_actualizacion_amcl(self, t_scan):
        """AMCL published a pose for the scan taken at t_scan."""
        if self.en_duda and self.fin_giro is not None and t_scan > self.fin_giro:
            self.actualizaciones += 1
            if self.actualizaciones >= self.p['actualizaciones_giro']:
                self._terminar()

    def pedir_actualizacion(self):
        """True: ask AMCL for a no-motion update now (the turn ended, not fixed yet)."""
        return self.en_duda and self.fin_giro is not None

    def _terminar(self):
        self.ultima = (self.actualizaciones, (self.t or 0.0) - (self.fin_giro or self.t or 0.0))
        self.correcciones += 1
        self.en_duda = False
        self.girado = 0.0          # a long turn that timed out needs another full turn
        self.inicio_duda = None
        self.fin_giro = None
        self.actualizaciones = 0
