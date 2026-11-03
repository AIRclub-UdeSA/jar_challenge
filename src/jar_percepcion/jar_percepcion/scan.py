"""Construcción de LaserScan filtrados."""

from sensor_msgs.msg import LaserScan


def copiar_scan(msg: LaserScan, rangos) -> LaserScan:
    """Copia del scan con otros rangos (inf = sin retorno)."""
    salida = LaserScan()
    salida.header = msg.header
    salida.angle_min = msg.angle_min
    salida.angle_max = msg.angle_max
    salida.angle_increment = msg.angle_increment
    salida.time_increment = msg.time_increment
    salida.scan_time = msg.scan_time
    salida.range_min = msg.range_min
    salida.range_max = msg.range_max
    salida.ranges = [float(r) for r in rangos]
    return salida
