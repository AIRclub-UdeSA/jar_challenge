#!/usr/bin/env python3
"""Valida que params/ solo tenga valores generales (§9.1, §10).

Reglas (esquema propuesto, POR CONFIRMAR en el reglamento):
- formato de parámetros de ROS 2 (secciones ros__parameters, con cualquier anidado, como Nav2);
- valores: bool, int, float, str, o listas cortas de escalares;
- prohibido: coordenadas (listas de 2+ pares de números), nombres que sugieran
  posiciones del mapa (waypoint, víctima, coordenada...) y strings que sean
  rutas a mapas o mundos.
- permitido: una pose inicial (`initial_pose`, `set_initial_pose`, `largada`...).
  El robot arranca cerca del origen y el equipo puede usarlo como referencia (§1).

Es una barrera contra el error honesto y el atajo obvio. No reemplaza la
revisión del juez técnico (§9.2).
"""
import re
import sys
from pathlib import Path

import yaml

RAIZ = Path(__file__).resolve().parent.parent
CARPETAS_PARAMS = list((RAIZ / 'src').glob('*/params'))

# Solo nombres de parámetros (hojas), no de nodos: Nav2 tiene un nodo
# `waypoint_follower` y parámetros legítimos como `xy_goal_tolerance`.
NOMBRES_SOSPECHOSOS = re.compile(
    r'(victim|coord|spawn|waypoints?$|posicion_)', re.I)
RUTA_DE_MAPA = re.compile(r'\.(pgm|png|world|sdf|yaml)$|maze|laberinto', re.I)
MAX_LARGO_LISTA = 12


def es_numero(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def revisar_valor(ruta, valor, errores):
    if isinstance(valor, dict):
        for k, v in valor.items():
            if not isinstance(v, dict) and NOMBRES_SOSPECHOSOS.search(str(k)):
                errores.append(f'{ruta}.{k}: el nombre sugiere una posición del mapa')
            revisar_valor(f'{ruta}.{k}', v, errores)
    elif isinstance(valor, list):
        # Las listas de nombres (plugins de Nav2, críticos) pueden ser largas;
        # las numéricas no, porque ahí es donde se esconden coordenadas.
        if len(valor) > MAX_LARGO_LISTA and any(es_numero(v) for v in valor):
            errores.append(f'{ruta}: lista numérica de {len(valor)} elementos (máx {MAX_LARGO_LISTA})')
        if any(isinstance(v, (list, dict)) for v in valor):
            errores.append(f'{ruta}: listas anidadas (¿coordenadas?) no permitidas')
        elif len(valor) >= 4 and all(es_numero(v) for v in valor):
            # Un rango de color HSV son 3 números; 4+ numéricos seguidos huele a
            # puntos (x, y, x, y...). Se acepta hasta 3, o se avisa.
            errores.append(
                f'{ruta}: lista de {len(valor)} números; si son coordenadas no van acá. '
                'Si es otra cosa (p. ej. un rango), dividila en parámetros con nombre')
    elif isinstance(valor, str):
        if RUTA_DE_MAPA.search(valor):
            errores.append(f'{ruta}: "{valor}" parece una ruta a un mapa o mundo')
    elif not (isinstance(valor, bool) or es_numero(valor) or valor is None):
        errores.append(f'{ruta}: tipo no permitido ({type(valor).__name__})')


def main():
    errores = []
    archivos = [f for c in CARPETAS_PARAMS for f in sorted(c.glob('*.y*ml'))]
    if not archivos:
        errores.append('no hay archivos .yaml en params/')
    for f in archivos:
        rel = f.relative_to(RAIZ)
        try:
            datos = yaml.safe_load(f.read_text())
        except yaml.YAMLError as e:
            errores.append(f'{rel}: YAML inválido ({e})')
            continue
        if not isinstance(datos, dict):
            errores.append(f'{rel}: tiene que ser un diccionario de nodos')
            continue
        if 'ros__parameters' not in yaml.dump(datos):
            errores.append(f'{rel}: no tiene ninguna sección ros__parameters')
        sub = []
        revisar_valor('', datos, sub)
        errores.extend(f'{rel}: {s.lstrip(".")}' for s in sub)

    for e in errores:
        print(f'::error::{e}')
    print('params: OK' if not errores else f'params: {len(errores)} error(es)')
    return 1 if errores else 0


if __name__ == '__main__':
    sys.exit(main())
