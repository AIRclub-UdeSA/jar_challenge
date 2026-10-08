import math

import pytest
from builtin_interfaces.msg import Time

from equipo_jar.reportes import armar_reporte


def test_reporte_bien_formado():
    msg = armar_reporte(1.5, -2.0, Time(sec=3, nanosec=0))
    assert msg.header.frame_id == 'map'
    assert msg.point.x == pytest.approx(1.5)
    assert msg.point.y == pytest.approx(-2.0)


@pytest.mark.parametrize('x, y', [(math.nan, 0.0), (0.0, math.inf)])
def test_reporte_rechaza_valores_no_finitos(x, y):
    with pytest.raises(ValueError):
        armar_reporte(x, y, Time())
