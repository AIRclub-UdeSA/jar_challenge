"""Formato oficial de los reportes de víctimas (reglamento §7.2).

Un mensaje en /victimas es un reporte y gasta uno de los reportes disponibles,
puntúe o no. Por eso conviene pasar siempre por acá: construye el mensaje con
el formato que espera el juez y se niega a armar uno mal formado.
"""
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


def vale_el_bono(aciertos, bono=200.0, por_victima=100.0):
    """Si conviene gastar ya los reportes que quedan para ganar el bono de tiempo (§8).

    aciertos: P(acierto) de cada candidato, uno por reporte que queda. Un reporte errado
    solo gasta su lugar, que vale por_victima si su víctima aparece después. Conviene si
    el bono esperado, bono * P(todos aciertan), supera lo que cuestan los errados.
    """
    return bono * math.prod(aciertos) > por_victima * sum(1.0 - q for q in aciertos)
