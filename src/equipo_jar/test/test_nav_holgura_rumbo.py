import math
import numpy as np
from equipo_jar.navegacion.grilla import Grilla
from equipo_jar.navegacion.orientado import planificar_orientado
from equipo_jar.navegacion.tipos import Pose2D
from equipo_jar.navegacion.costos import ParametrosCosto


def test_prefiere_rumbo_con_mas_holgura_en_corredor_sin_bajar_margen():
    occupied = np.zeros((60, 100), bool)
    occupied[20,:] = True
    occupied[32,:] = True
    g = Grilla(.05, 0., 0., 0., 100, 60, occupied)
    hull = dict(corners_base_m=[[-.19,-.12],[.13,-.12],[.13,.12],[-.19,.12]],
                radius_m=.225, margin_m=.04, heading_uncertainty_rad=.1)
    route = planificar_orientado(g, occupied, Pose2D(.7,1.325,math.pi/4),
        (4.2,1.325), hull, plazo_s=5.,
        parametros_costos=ParametrosCosto(.12,.225,.04,.25,5.))
    assert route is not None
    # Girar el cuerpo deja más espacio lateral que mantener 45 grados.
    assert abs(math.atan2(math.sin(route.orientaciones[-1]),
                          math.cos(route.orientaciones[-1]))) < .4
    assert all(1.05 < y < 1.6 for _,y in route.puntos)
    assert hull['margin_m'] == .04
