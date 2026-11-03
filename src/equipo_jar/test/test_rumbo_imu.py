"""Odometry with the IMU heading: the transform odom_fused -> odom it publishes."""
import math

import pytest

from equipo_jar.rumbo_imu import RumboImu

P = {'fuente_rumbo': 'giroscopo', 'signo_giroscopo': 0, 'imu_vieja_s': 0.5,
     'signo_wz_minimo_rads': 0.2, 'signo_giro_minimo_rad': 0.5, 'signo_correlacion_minima': 0.7,
     'escala_avance': 0.0, 'escala_giro_minimo_rad': 1.0, 'escala_minima': 0.7,
     'escala_maxima': 1.1}


def componer(a, b):
    """Pose a * pose b, poses as (x, y, yaw)."""
    c, s = math.cos(a[2]), math.sin(a[2])
    return (a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], a[2] + b[2])


class Robot:
    """Simulated robot: the wheels count turns and distance with `escala`, the gyro has `signo`."""

    def __init__(self, p, escala=1.25, signo=-1, con_imu=True):
        self.r = RumboImu(dict(P, **p))
        self.escala, self.signo, self.con_imu = escala, signo, con_imu
        self.t = 0.0
        self.real = [0.0, 0.0, 0.0]
        self.odom = [0.0, 0.0, 0.0]
        self.yaw_imu = 0.0
        self.tf = None
        self.paso(0.0, 0.0, 0.0)
        self.paso(0.0, 0.0, 0.0)

    def paso(self, vx, vy, wz, dt=0.1, imu_primero=False):
        """Real body velocity for dt; /imu/data and /odom with the same stamp, as on the robot."""
        self.t += dt
        for pose, k in ((self.real, 1.0), (self.odom, self.escala)):
            c, s = math.cos(pose[2] + 0.5 * k * wz * dt), math.sin(pose[2] + 0.5 * k * wz * dt)
            pose[0] += k * (c * vx - s * vy) * dt
            pose[1] += k * (s * vx + c * vy) * dt
            pose[2] += k * wz * dt
        self.yaw_imu += wz * dt
        salida = []
        imu = lambda: self.r.con_imu(self.t, self.signo * wz, self.yaw_imu) if self.con_imu else []
        odom = lambda: self.r.con_odom(self.t, *self.odom, self.escala * wz)
        salida += imu() + odom() if imu_primero else odom() + imu()
        for tr in salida:
            assert tr[0] <= self.t + 1e-9
            self.tf = tr
        return salida

    def fusionada(self):
        """Pose of base in odom_fused from the last transform: T(fused->odom) * T(odom->base)."""
        return componer(self.tf[1:], self.odom)


def girar(robot, angulo, wz=1.0, **kw):
    for _ in range(round(abs(angulo) / (wz * 0.1))):
        robot.paso(0.0, 0.0, math.copysign(wz, angulo), **kw)
    robot.paso(0.0, 0.0, 0.0, **kw)    # stop


def avanzar(robot, metros, v=0.3, **kw):
    for _ in range(round(metros / (v * 0.1))):
        robot.paso(v, 0.0, 0.0, **kw)


def test_wheels_source_publishes_the_identity():
    robot = Robot({'fuente_rumbo': 'ruedas'})
    girar(robot, 2.0)
    avanzar(robot, 1.0)
    assert robot.tf[1:] == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)


def test_gyro_sign_is_learned_and_heading_follows_the_imu():
    robot = Robot({})
    girar(robot, 1.0)                        # learns the sign during this turn
    assert robot.r.signo == -1
    yaw_antes = robot.fusionada()[2]
    real_antes = robot.real[2]
    girar(robot, math.pi / 2)
    avanzar(robot, 1.0)
    # After the sign is known, turns follow the IMU, not the wheels' 25 % excess.
    assert robot.fusionada()[2] - yaw_antes == pytest.approx(robot.real[2] - real_antes, abs=1e-6)


def test_fixed_sign_trajectory_is_wheel_steps_turned_with_the_imu():
    robot = Robot({'signo_giroscopo': -1, 'escala_avance': 1.0}, escala=1.25)
    girar(robot, math.pi / 2)
    avanzar(robot, 1.0)
    x, y, yaw = robot.fusionada()
    assert yaw == pytest.approx(robot.real[2], abs=1e-6)
    # Distance keeps the wheels' scale (1.25 m), heading is the real one.
    assert x == pytest.approx(1.25 * robot.real[0], abs=1e-6)
    assert y == pytest.approx(1.25 * robot.real[1], abs=1e-6)


def test_odom_waits_for_the_imu_sample_of_its_stamp():
    robot = Robot({'signo_giroscopo': -1})
    salida = robot.paso(0.0, 0.0, 1.0)             # /odom first: held until the IMU arrives
    assert len(salida) == 1 and salida[0][0] == pytest.approx(robot.t)
    salida = robot.paso(0.0, 0.0, 0.0, imu_primero=True)
    assert len(salida) == 1 and salida[0][0] == pytest.approx(robot.t)
    assert robot.fusionada()[2] == pytest.approx(0.1, abs=1e-6)


def test_without_imu_the_wheels_heading_is_used():
    robot = Robot({'signo_giroscopo': -1}, con_imu=False)
    girar(robot, 1.0)
    assert robot.tf[1:] == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)
    assert robot.r.pasos_ruedas > 0


def test_stale_imu_falls_back_to_the_wheels_and_back():
    robot = Robot({'signo_giroscopo': -1}, escala=1.25)
    girar(robot, 0.5)
    robot.con_imu = False
    girar(robot, 1.0)                              # IMU silent: wheels, 1.25 rad
    robot.con_imu = True
    girar(robot, 0.5)
    assert robot.fusionada()[2] == pytest.approx(0.5 + 1.25 + 0.5, abs=0.15)


def test_orientation_source_uses_the_imu_yaw():
    robot = Robot({'fuente_rumbo': 'orientacion'}, signo=-1)
    girar(robot, 3.5)                              # past pi: the yaw wraps, the heading does not
    assert robot.fusionada()[2] == pytest.approx(3.5, abs=1e-6)


def test_sign_is_not_learned_without_turning():
    robot = Robot({})
    avanzar(robot, 1.0)
    assert robot.r.signo == 0


def test_distance_scale_is_learned_from_the_turns():
    robot = Robot({}, escala=1.25)
    girar(robot, 0.6)                              # sign learned, not enough turning yet
    assert robot.r.escala == 1.0
    girar(robot, math.pi / 2)
    assert robot.r.escala == pytest.approx(0.8, abs=0.01)
    x0, y0, _ = robot.fusionada()
    rx, ry = robot.real[0], robot.real[1]
    avanzar(robot, 1.0)
    x, y, _ = robot.fusionada()
    # Distance now follows the real one, not the wheels' 25 % excess.
    assert math.hypot(x - x0, y - y0) == pytest.approx(
        math.hypot(robot.real[0] - rx, robot.real[1] - ry), rel=0.02)


def test_exact_odometry_learns_scale_one():
    robot = Robot({}, escala=1.0)
    girar(robot, 2.0)
    girar(robot, -2.0)
    assert robot.r.escala == pytest.approx(1.0, abs=0.01)


def test_learned_scale_is_clamped():
    robot = Robot({}, escala=2.0)
    girar(robot, 2.0)
    assert robot.r.escala == 0.7
