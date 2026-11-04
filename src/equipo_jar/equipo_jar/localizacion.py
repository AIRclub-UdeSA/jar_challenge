"""Cuentas de la localización, sin ROS (se prueban en test/test_localizacion.py).

La localización la hace AMCL (nav2_amcl, workshop de la semana 06): compara el lidar con /map y
corrige la deriva de la odometría publicando la transformación map -> odom. Acá solo está lo que
necesita nuestro código alrededor de AMCL: armar la pose inicial con su incertidumbre.
"""
import math


def cuaternion_de_yaw(yaw):
    """Cuaternión (x, y, z, w) de una rotación de `yaw` radianes alrededor del eje z."""
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


def covarianza_pose_inicial(sigma_xy_m, sigma_yaw_rad):
    """Covarianza 6x6 (aplanada, 36 valores) de una pose en el plano.

    El orden de ROS es (x, y, z, giro en x, giro en y, giro en z): solo se completan las varianzas
    de x, y (índices 0 y 7) y yaw (índice 35). Es la gaussiana de la que AMCL saca sus partículas
    iniciales: cuanto más grande, más lejos del origen puede arrancar el robot y seguir
    encontrándose, pero más le cuesta converger.
    """
    cov = [0.0] * 36
    cov[0] = sigma_xy_m ** 2
    cov[7] = sigma_xy_m ** 2
    cov[35] = sigma_yaw_rad ** 2
    return cov
