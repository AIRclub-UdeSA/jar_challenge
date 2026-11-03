"""Formato oficial de los reportes de víctimas."""
import math

from geometry_msgs.msg import PointStamped

FRAME_REPORTES = 'map'


def armar_reporte(x, y, ahora):
    """Devuelve el PointStamped de un reporte, o lanza ValueError.

    x, y: posición estimada del centro de la caja en metros, marco `map`.
    ahora: builtin_interfaces/Time (por ejemplo `nodo.get_clock().now().to_msg()`).
    El juez no usa header.stamp para puntuar, pero igual se completa.
    """
    if not (math.isfinite(x) and math.isfinite(y)):
        raise ValueError(f'reporte con valores no finitos: x={x}, y={y}')
    msg = PointStamped()
    msg.header.frame_id = FRAME_REPORTES
    msg.header.stamp = ahora
    msg.point.x = float(x)
    msg.point.y = float(y)
    msg.point.z = 0.0
    return msg
