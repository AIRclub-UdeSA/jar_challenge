"""Odometry with the IMU's heading instead of the wheels' (no ROS here).

On the real robot /odom counts turns (and distance) ~25 % too much, and AMCL keeps the
heading error after each turn. The IMU measures turns well. This class builds a second
odometry, `odom_fused`: it keeps the wheels' steps in the robot frame but turns them with
the IMU heading. AMCL uses it as its odom frame (`odom_frame_id: odom_fused`), so the
tree is map -> odom_fused -> odom -> base_footprint and the robot's own odom is untouched.

For each /odom message it gives T(odom_fused -> odom) at the same stamp:
T(odom_fused -> base) * inverse(T(odom -> base)).

Heading source (`fuente_rumbo`):
- `giroscopo`: angular_velocity.z integrated. On the real robot its sign is inverted, so
  by default (`signo_giroscopo: 0`) the sign is taken from the correlation with the
  wheels' turn rate; until then the wheels' heading is used.
- `orientacion`: the yaw of the IMU's orientation (may use a magnetometer).
- `ruedas`: the wheels' heading, as /odom (off: the transform stays the identity).
Each /odom waits for the IMU sample of its stamp; if none comes within `imu_vieja_s`,
that step uses the wheels' heading.

Distance (`escala_avance`): the wheels' steps are multiplied by a scale. 0 = learned: the
IMU's turn over the wheels' turn while turning (bag: 0.80; sim: 1.0). The same wheel
constant makes /odom count turns and distance too much (bag: turns 0.80, forward 0.78,
sideways 0.79), so the turn ratio fixes distance too. Used after `escala_giro_minimo_rad` of
wheel turning, kept within [escala_minima, escala_maxima]. A number > 0 fixes it (1.0: off).
Checked on the week-10 bag (notes/hans-die/NOTES.md).
"""
import math
from collections import deque


def diferencia_angulo(a, b):
    """a - b in (-pi, pi]."""
    return math.atan2(math.sin(a - b), math.cos(a - b))


class RumboImu:
    """Feed it /imu/data and /odom; each call returns the transforms to publish.

    `p` is the node's live parameter dict: fuente_rumbo, signo_giroscopo, imu_vieja_s,
    signo_wz_minimo_rads, signo_giro_minimo_rad, signo_correlacion_minima, escala_avance,
    escala_giro_minimo_rad, escala_minima, escala_maxima.
    Times are header stamps in seconds. A transform is (stamp, x, y, yaw) of
    odom_fused -> odom.
    """

    def __init__(self, p):
        self.p = p
        # IMU samples: (t, gyro integral, orientation integral), both unwrapped.
        self.muestras = deque(maxlen=500)
        self.yaw_imu_previo = 0.0
        # Gyro sign from the wheels: sums over samples where the wheels turn.
        self.sxy = self.sxx = self.syy = 0.0
        self.giro_signo = 0.0        # rad the wheels turned while the sums were collected
        self.signo_auto = 0          # 0: not decided yet
        self.odom_wz = None          # (t, wz) of the last /odom, to compare with the gyro
        # Last processed /odom: stamp, pose in odom, pose in odom_fused, IMU integrals there.
        self.t = None
        self.odom = None
        self.fusionada = None
        self.integrales = None
        self.pendientes = deque()    # /odom messages waiting for the IMU sample of their stamp
        self.pasos_ruedas = 0        # turning steps that used the wheels' heading, for the log
        # Turning, wheels vs IMU (rad), over steps where both measured it: the distance scale.
        self.giro_ruedas = self.giro_imu = 0.0

    # --- inputs ---

    def con_imu(self, t, gz, yaw):
        """An /imu/data sample: stamp, angular_velocity.z, orientation yaw."""
        if self.muestras:
            t0, ig, io = self.muestras[-1]
            dt = t - t0
            if dt <= 0.0:
                return []
            if dt <= self.p['imu_vieja_s']:     # after a gap the wheels cover it
                # The sample's rate over the interval that ends at its stamp: on the bag this
                # beats the trapezoid rule (heading at stops 1.8° vs 3.1° median).
                ig += gz * dt
                io += diferencia_angulo(yaw, self.yaw_imu_previo)
            self.muestras.append((t, ig, io))
        else:
            self.muestras.append((t, 0.0, 0.0))
        self.yaw_imu_previo = yaw
        self._aprender_signo(t, gz)
        return self._vaciar(None)

    def con_odom(self, t, x, y, yaw, wz):
        """An /odom message: stamp, pose in odom, twist.angular.z."""
        self.odom_wz = (t, wz)
        self.pendientes.append((t, x, y, yaw))
        return self._vaciar(t)

    def _vaciar(self, t_odom):
        """Processes the /odom messages the IMU already covers, or that cannot wait any longer.

        On the robot /imu/data has the same stamps as /odom and comes within ~1 ms, before
        or after it: waiting for it beats guessing the turn. A message waits at most
        `imu_vieja_s` behind the newest /odom; then (or with the IMU silent) its step uses
        the wheels' heading.
        """
        salida = []
        vieja = self.p['imu_vieja_s']
        while self.pendientes:
            t = self.pendientes[0][0]
            ultima = self.muestras[-1][0] if self.muestras else None
            cubierta = ultima is not None and t <= ultima
            sin_imu = (self.p['fuente_rumbo'] == 'ruedas' or ultima is None or t > ultima + vieja
                       or (t_odom is not None and t_odom - t > vieja))
            if not (cubierta or sin_imu):
                break
            salida += self._procesar(*self.pendientes.popleft(), usar_imu=cubierta)
        return salida

    # --- state ---

    @property
    def signo(self):
        """Gyro sign in use: the param if set, else the learned one (0: unknown)."""
        fijo = self.p['signo_giroscopo']
        return int(math.copysign(1, fijo)) if fijo else self.signo_auto

    @property
    def escala(self):
        """Scale for the wheels' steps: the param if > 0, else the learned one (1.0 until known)."""
        fija = self.p['escala_avance']
        if fija > 0.0:
            return fija
        if self.giro_ruedas < self.p['escala_giro_minimo_rad']:
            return 1.0
        return min(max(self.giro_imu / self.giro_ruedas, self.p['escala_minima']),
                   self.p['escala_maxima'])

    def _aprender_signo(self, t, gz):
        if self.signo_auto or self.odom_wz is None:
            return
        t_odom, wz = self.odom_wz
        if abs(t - t_odom) > self.p['imu_vieja_s'] or abs(wz) < self.p['signo_wz_minimo_rads']:
            return
        self.sxy += gz * wz
        self.sxx += gz * gz
        self.syy += wz * wz
        if len(self.muestras) > 1:
            self.giro_signo += abs(wz) * (t - self.muestras[-2][0])
        if self.giro_signo < self.p['signo_giro_minimo_rad'] or self.sxx <= 0.0:
            return
        correlacion = self.sxy / math.sqrt(self.sxx * self.syy)
        if abs(correlacion) >= self.p['signo_correlacion_minima']:
            self.signo_auto = 1 if correlacion > 0 else -1

    def _integrales_en(self, t):
        """(gyro, orientation) integrals at t, or None if the IMU does not cover t."""
        if not self.muestras or not self.muestras[0][0] <= t <= self.muestras[-1][0]:
            return None
        vieja = self.p['imu_vieja_s']
        if t == self.muestras[-1][0]:
            return self.muestras[-1][1:]
        for k in range(len(self.muestras) - 1, 0, -1):
            t0, ig0, io0 = self.muestras[k - 1]
            if t0 <= t:
                t1, ig1, io1 = self.muestras[k]
                if t1 - t0 > vieja:
                    return None
                f = (t - t0) / (t1 - t0)
                return ig0 + f * (ig1 - ig0), io0 + f * (io1 - io0)
        return None

    def _procesar(self, t, x, y, yaw, usar_imu):
        integrales = self._integrales_en(t) if usar_imu else None
        if self.t is None:
            self.t, self.odom, self.fusionada = t, (x, y, yaw), (x, y, yaw)
            self.integrales = integrales
            return [self._transformada()]
        if t <= self.t:
            return []
        x0, y0, yaw0 = self.odom
        d_ruedas = diferencia_angulo(yaw, yaw0)
        d_yaw = d_ruedas
        fuente = self.p['fuente_rumbo']
        if fuente != 'ruedas':
            d_imu = None
            if integrales is not None and self.integrales is not None:
                if fuente == 'orientacion':
                    d_imu = integrales[1] - self.integrales[1]
                elif self.signo:
                    d_imu = self.signo * (integrales[0] - self.integrales[0])
            if d_imu is None:
                if abs(d_ruedas) > 1e-3:        # for the log: still steps don't matter
                    self.pasos_ruedas += 1
            else:
                d_yaw = d_imu
                if abs(d_ruedas) >= self.p['signo_wz_minimo_rads'] * (t - self.t):
                    self.giro_ruedas += abs(d_ruedas)
                    self.giro_imu += abs(d_imu)
        fx, fy, fyaw = self.fusionada
        # The wheels' step, turned from odom into odom_fused at the mean heading difference.
        giro = diferencia_angulo(fyaw, yaw0) + 0.5 * (d_yaw - d_ruedas)
        c, s = math.cos(giro), math.sin(giro)
        k = self.escala
        dx, dy = k * (x - x0), k * (y - y0)
        self.fusionada = (fx + c * dx - s * dy, fy + s * dx + c * dy, fyaw + d_yaw)
        self.t, self.odom, self.integrales = t, (x, y, yaw), integrales
        return [self._transformada()]

    def _transformada(self):
        """(stamp, x, y, yaw) of odom_fused -> odom = T(fused -> base) * inverse(T(odom -> base))."""
        x, y, yaw = self.odom
        fx, fy, fyaw = self.fusionada
        giro = diferencia_angulo(fyaw, yaw)
        c, s = math.cos(giro), math.sin(giro)
        return (self.t, fx - (c * x - s * y), fy - (s * x + c * y), giro)
