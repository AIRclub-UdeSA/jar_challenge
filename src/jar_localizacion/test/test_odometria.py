"""Pruebas de medición y corrección de odometría sin ejecutar ROS."""

import math

from jar_localizacion.odometria import (aplicar_paso, indice_rayo_hacia_atras,
                                        paso_en_frame_robot)


def test_rayo_hacia_atras_con_lidar_girado():
    """Con el lidar girado 180°, atrás del robot es el 0 del lidar."""
    n = 360
    inc = 2 * math.pi / n
    assert indice_rayo_hacia_atras(-math.pi, inc, n, math.pi) == 180
    assert indice_rayo_hacia_atras(0.0, inc, n, math.pi) == 0
    assert indice_rayo_hacia_atras(0.0, inc, n, 0.0) == 180


def test_paso_en_frame_robot():
    """Mirando hacia +y, moverse en +y es avanzar y en -x es ir a la izquierda."""
    avance, costado, giro = paso_en_frame_robot((0, 0, math.pi / 2), (-0.5, 1.0, math.pi))
    assert math.isclose(avance, 1.0) and math.isclose(costado, 0.5)
    assert math.isclose(giro, math.pi / 2)


def test_aplicar_paso_con_escalas():
    x, y, theta = aplicar_paso((1.0, 1.0, math.pi / 2), 1.0, 0.0, 0.2, 0.5, 2.0)
    assert math.isclose(x, 1.0, abs_tol=1e-9) and math.isclose(y, 1.5)
    assert math.isclose(theta, math.pi / 2 + 0.4)


def test_paso_ida_y_vuelta_sin_escalas():
    """Reaplicar los pasos de odom sin escalas reproduce la odometría."""
    poses = [(0, 0, 0), (1, 0, 0.5), (1.5, 0.7, 2.0), (0.2, 1.0, -3.0)]
    corregida = poses[0]
    for a, b in zip(poses, poses[1:]):
        corregida = aplicar_paso(corregida, *paso_en_frame_robot(a, b))
    for u, v in zip(corregida, poses[-1]):
        assert math.isclose(u, v, abs_tol=1e-9)
