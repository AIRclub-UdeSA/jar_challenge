"""Mensajes RViz: sectores, observaciones LiDAR y trayectoria real."""
from copy import deepcopy
import colorsys
import math

import numpy as np
from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray


def marcadores_sectores(mapa, regiones):
    """Geometría en coordenadas locales, transformada por el origen del mapa."""
    regiones = np.asarray(regiones)
    salida = MarkerArray()
    limpiar = Marker()
    limpiar.header = mapa.header
    limpiar.action = Marker.DELETEALL
    salida.markers.append(limpiar)
    for region in sorted(int(r) for r in np.unique(regiones) if r >= 0):
        ys, xs = np.where(regiones == region)
        marker = Marker()
        marker.header = mapa.header
        marker.ns, marker.id = 'sectores_bcd', region
        marker.type, marker.action = Marker.CUBE_LIST, Marker.ADD
        marker.pose = mapa.info.origin
        marker.scale.x = marker.scale.y = float(mapa.info.resolution)
        marker.scale.z = 0.01
        rgb = colorsys.hsv_to_rgb((region * 0.61803398875) % 1, 0.65, 1.0)
        marker.color.r, marker.color.g, marker.color.b = rgb
        marker.color.a = 0.55
        marker.points = [Point(x=float((x + 0.5)*mapa.info.resolution),
                               y=float((y + 0.5)*mapa.info.resolution), z=0.02)
                         for x, y in zip(xs, ys)]
        salida.markers.append(marker)
        # Etiqueta sobre una celda del sector, incluso en regiones cóncavas.
        i = int(np.argmin((xs-xs.mean())**2 + (ys-ys.mean())**2))
        q = mapa.info.origin.orientation
        yaw = math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
        x, y = (xs[i]+0.5)*mapa.info.resolution, (ys[i]+0.5)*mapa.info.resolution
        etiqueta = Marker()
        etiqueta.header = mapa.header
        etiqueta.ns, etiqueta.id = 'etiquetas_bcd', region
        etiqueta.type = Marker.TEXT_VIEW_FACING
        etiqueta.pose.position.x = float(mapa.info.origin.position.x + math.cos(yaw)*x - math.sin(yaw)*y)
        etiqueta.pose.position.y = float(mapa.info.origin.position.y + math.sin(yaw)*x + math.cos(yaw)*y)
        etiqueta.pose.position.z = mapa.info.origin.position.z + 0.15
        etiqueta.pose.orientation.w = 1.0
        etiqueta.scale.z = 0.18
        etiqueta.color.r = etiqueta.color.g = etiqueta.color.b = etiqueta.color.a = 1.0
        etiqueta.text = f'S{region}'
        salida.markers.append(etiqueta)
    return salida


def mapa_observado(mapa, observado):
    """RViz Map: sólo lo observado tiene valor; lo pendiente es transparente."""
    salida = OccupancyGrid()
    salida.header = deepcopy(mapa.header)
    salida.info = mapa.info
    salida.data = np.where(observado, 100, -1).astype(np.int8).ravel().tolist()
    return salida
