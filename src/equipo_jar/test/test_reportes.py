import math

import pytest
from builtin_interfaces.msg import Time

from equipo_jar.reportes import armar_reporte, vale_el_bono


def test_reporte_bien_formado():
    msg = armar_reporte(1.5, -2.0, Time(sec=3, nanosec=0))
    assert msg.header.frame_id == 'map'
    assert msg.point.x == pytest.approx(1.5)
    assert msg.point.y == pytest.approx(-2.0)


@pytest.mark.parametrize('x, y', [(math.nan, 0.0), (0.0, math.inf)])
def test_reporte_rechaza_valores_no_finitos(x, y):
    with pytest.raises(ValueError):
        armar_reporte(x, y, Time())


def test_bono_conviene_con_candidatos_probables():
    assert vale_el_bono([0.9, 0.9, 0.9])          # 146 de bono esperado contra 30
    assert not vale_el_bono([0.9, 0.9, 0.3])      # 49 contra 90
    assert vale_el_bono([0.99])
    assert vale_el_bono([0.5])                    # un solo lugar: 100 contra 50
    assert not vale_el_bono([0.3])                # 60 contra 70
