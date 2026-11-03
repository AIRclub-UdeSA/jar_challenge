"""Lectura de mapas de map_server (yaml + pgm) sin dependencias de ROS."""

from pathlib import Path

import numpy as np
import yaml


def leer_pgm(ruta):
    """Lee un PGM binario (P5) de 8 bits como array (alto, ancho), fila 0 arriba."""
    datos = Path(ruta).read_bytes()
    partes, pos = [], 0
    while len(partes) < 4:  # magic, ancho, alto, maxval
        while pos < len(datos) and datos[pos:pos + 1].isspace():
            pos += 1
        if datos[pos:pos + 1] == b'#':
            pos = datos.find(b'\n', pos)
            if pos < 0:
                raise ValueError(f'{ruta}: encabezado PGM incompleto')
            continue
        inicio = pos
        while pos < len(datos) and not datos[pos:pos + 1].isspace():
            pos += 1
        if pos == inicio:
            raise ValueError(f'{ruta}: encabezado PGM incompleto')
        partes.append(datos[inicio:pos])
    pos += 1  # un único whitespace después de maxval
    if partes[0] != b'P5':
        raise ValueError(f'{ruta}: solo se soporta PGM binario (P5)')
    ancho, alto, maxval = (int(p) for p in partes[1:])
    if maxval != 255:
        raise ValueError(f'{ruta}: solo se soporta PGM de 8 bits (maxval 255)')
    pixeles = np.frombuffer(datos, dtype=np.uint8, count=ancho * alto, offset=pos)
    return pixeles.reshape(alto, ancho)


def cargar_mapa(ruta_yaml):
    """
    Convierte un mapa de map_server en (grilla, resolución, origen).

    La grilla usa la convención de OccupancyGrid: fila 0 abajo, 100 ocupado,
    0 libre y -1 desconocido, con los umbrales del modo trinary de map_server.
    """
    ruta_yaml = Path(ruta_yaml)
    meta = yaml.safe_load(ruta_yaml.read_text())
    pixeles = leer_pgm(ruta_yaml.parent / meta['image'])[::-1].astype(float)
    if int(meta.get('negate', 0)):
        ocupacion = pixeles / 255.0
    else:
        ocupacion = (255.0 - pixeles) / 255.0
    grilla = np.where(ocupacion > float(meta.get('occupied_thresh', 0.65)), 100,
                      np.where(ocupacion < float(meta.get('free_thresh', 0.25)), 0, -1))
    origen = [float(v) for v in meta['origin'][:2]]
    return grilla.astype(np.int8), float(meta['resolution']), origen
