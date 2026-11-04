import math

import pytest

from equipo_jar.deteccion import yaw_de_cuaternion
from equipo_jar.localizacion import covarianza_pose_inicial, cuaternion_de_yaw


@pytest.mark.parametrize('yaw', [0.0, 0.5, -1.2, math.pi / 2, 3.0])
def test_cuaternion_de_yaw_ida_y_vuelta(yaw):
    assert yaw_de_cuaternion(*cuaternion_de_yaw(yaw)) == pytest.approx(yaw)


def test_cuaternion_de_yaw_es_unitario():
    x, y, z, w = cuaternion_de_yaw(0.7)
    assert x == y == 0.0
    assert z * z + w * w == pytest.approx(1.0)


def test_covarianza_pose_inicial():
    cov = covarianza_pose_inicial(0.2, 0.1)
    assert len(cov) == 36
    assert cov[0] == pytest.approx(0.04)   # varianza de x
    assert cov[7] == pytest.approx(0.04)   # varianza de y
    assert cov[35] == pytest.approx(0.01)  # varianza de yaw
    assert sum(cov) == pytest.approx(0.09)  # el resto en cero
