#!/usr/bin/env python3
"""Víctimas de un mundo de práctica, sin ROS.

    python3 tools/mundos.py <mundo.world>   -> imprime N (cantidad de víctimas)

`juez.py` importa de acá para no duplicar la lógica. Sin dependencias de ROS:
solo stdlib, así lo usan el CI, los scripts y los tests sin ROS instalado.
"""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def victimas_de_mundo(ruta):
    """Posiciones (x, y) de las cajas rojas de un .world (marco del mundo = `map`)."""
    raiz = ET.parse(ruta).getroot()
    victimas = []
    for modelo in raiz.iter('model'):
        pose_modelo = modelo.find('pose')
        ox = oy = 0.0
        if pose_modelo is not None and pose_modelo.text:
            ox, oy = (float(v) for v in pose_modelo.text.split()[:2])
        for link in modelo.findall('link'):
            if link.get('name', '').startswith('cuadrado_rojo'):
                pose = link.find('pose')
                x, y = (float(v) for v in pose.text.split()[:2])
                victimas.append((x + ox, y + oy))
    if not victimas:
        raise ValueError(f'{ruta}: no encontré cajas rojas (links cuadrado_rojo_*)')
    return victimas


def resolver_mundo(mundo):
    """Acepta una ruta o el nombre de un mundo de práctica (p. ej. maze_1_6x5)."""
    p = Path(mundo)
    if p.exists():
        return p
    from ament_index_python.packages import get_package_share_directory
    base = Path(get_package_share_directory('yahboom_rosmaster_gazebo') + '/worlds')
    for nombre in (f'{mundo}_victimas.world', f'{mundo}.world', mundo):
        if (base / nombre).exists():
            return base / nombre
    raise FileNotFoundError(f'no encuentro el mundo {mundo}')


if __name__ == '__main__':
    print(len(victimas_de_mundo(resolver_mundo(sys.argv[1]))))
