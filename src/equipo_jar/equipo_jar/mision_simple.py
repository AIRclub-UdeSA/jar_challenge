"""Global camera-view selection over Nav2's live costmap; no second SE2 search."""
import copy
import math
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
from equipo_jar.navegacion.planificador import costo_desde
from equipo_jar.mision import ParametrosMision, clave_vista, elegir_vista
from equipo_jar.coordinador_mision import filtro_huella_estatica
from equipo_jar.navegacion.guardia import huella_convexa_x3


def huella_nav2():
    """Same conservative chassis/wheel envelope already used by M2."""
    return dict(huella_convexa_x3(-.19,.13,.07,.08,.0845,.033,.04),
                heading_uncertainty_rad=0.)


def siguiente_vista(grilla, costos, pose, cobertura, intentadas, extrinseca,
                    sugeridas=(), paso_m=.30, extra=None, largada=None, restante_s=600.):
    """Compare global visual gain; a local batch must not hide another room.

    Only actual RGB observations credit coverage. Nav2 owns route execution.
    """
    start=largada if largada is not None else pose
    _, ida=costo_desde(grilla,costos,pose[:2])
    _, regreso=costo_desde(grilla,costos,start[:2])
    occupied=grilla.ocupada if extra is None else grilla.ocupada|extra
    cov=copy.copy(cobertura)
    if hasattr(cov,'grilla'):
        cov.grilla=replace(grilla,ocupada=occupied)
    proxy=SimpleNamespace(
        ganancia=lambda p:cov.ganancia(p,extra),
        cotas_ganancia=(cov.cotas_ganancia if hasattr(cov,'cotas_ganancia')
                        else lambda poses:np.full(len(poses),math.inf)))
    excluded=set(intentadas)|{clave_vista((*pose[:2],grilla.origen_yaw+k*math.pi/4))
                              for k in range(8)}
    check=filtro_huella_estatica(grilla,occupied,huella_nav2())
    return elegir_vista(grilla,costos,pose,start,proxy,excluded,restante_s,
        ParametrosMision(separacion_vistas_m=paso_m),sugeridas,extrinseca,
        distancias=(ida,regreso),filtro_pose=check)
