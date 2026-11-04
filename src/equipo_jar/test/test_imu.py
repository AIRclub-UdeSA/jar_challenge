import math

import pytest

from equipo_jar.imu import IntegradorGiro, fila_z_de_rotacion


def cuaternion_de_rpy(roll, pitch, yaw):
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy)


def test_sin_rotacion_el_z_queda_igual():
    assert fila_z_de_rotacion(0.0, 0.0, 0.0, 1.0) == pytest.approx((0.0, 0.0, 1.0))


def test_imu_montado_como_en_el_urdf_invierte_el_giro():
    # rpy del IMU en rosmaster_x3.urdf.xacro: dado vuelta (pitch π) y girado (yaw π/2).
    fila = fila_z_de_rotacion(*cuaternion_de_rpy(0.0, 3.1415, 1.5707))
    assert fila == pytest.approx((0.0, 0.0, -1.0), abs=1e-3)


def test_integra_un_giro_constante():
    integ = IntegradorGiro()
    for i in range(11):                  # 1 s a 10 Hz y 0,5 rad/s
        integ.agregar(i * 0.1, 0.5)
    assert integ.yaw == pytest.approx(0.5)
    assert integ.velocidad == 0.5


def test_no_normaliza_para_poder_restar_vueltas_completas():
    integ = IntegradorGiro()
    for i in range(101):                 # 10 s a 1 rad/s: más de una vuelta
        integ.agregar(i * 0.1, 1.0)
    assert integ.yaw == pytest.approx(10.0)


def test_un_corte_del_imu_no_se_integra():
    integ = IntegradorGiro(dt_max=0.5)
    integ.agregar(0.0, 1.0)
    integ.agregar(3.0, 1.0)              # 3 s sin datos: no se sabe qué pasó
    integ.agregar(3.1, 1.0)
    assert integ.yaw == pytest.approx(0.1)


def test_tiempos_repetidos_o_hacia_atras_no_suman():
    integ = IntegradorGiro()
    integ.agregar(1.0, 1.0)
    integ.agregar(1.0, 1.0)
    integ.agregar(0.5, 1.0)
    assert integ.yaw == 0.0
