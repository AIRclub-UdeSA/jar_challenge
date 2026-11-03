"""Compara rutas y tiempos en un mapa real sin iniciar ROS ni Gazebo."""

import argparse
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
import yaml

from jar_exploracion.planificador import PlanificadorCompilado, a_estrella
from jar_exploracion.seguridad import espacio_transitable


def main():
    """Alterna tres repeticiones del planificador de referencia y del compilado."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mapa', type=Path)
    parser.add_argument('--inicio', nargs=2, type=float, required=True)
    parser.add_argument('--objetivo', nargs=2, type=float, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.mapa.read_text())
    imagen = np.asarray(Image.open(args.mapa.parent / config['image']).convert('L'))[::-1]
    ocupacion = imagen / 255 if config.get('negate', 0) else 1 - imagen / 255
    datos = np.full(imagen.shape, -1, dtype=int)
    datos[ocupacion < config['free_thresh']] = 0
    datos[ocupacion > config['occupied_thresh']] = 100
    datos = np.repeat(np.repeat(datos, 2, axis=0), 2, axis=1)
    resolucion = config['resolution'] / 2
    libres = espacio_transitable(datos, resolucion, 0.37, 0.24, 0.056)
    distancia = ndimage.distance_transform_edt(datos == 0) * resolucion
    costos = 100 * np.exp(-distancia**2 / (2 * 0.15**2))

    def celda(punto):
        return (math.floor((punto[1] - config['origin'][1]) / resolucion),
                math.floor((punto[0] - config['origin'][0]) / resolucion))

    inicio, objetivo = celda(args.inicio), celda(args.objetivo)
    comienzo = time.perf_counter()
    compilado = PlanificadorCompilado(libres, costos)
    construccion = time.perf_counter() - comienzo
    mediciones = []
    for repeticion in range(3):
        resultados = {}
        metodos = [('referencia', lambda: a_estrella(libres, inicio, objetivo, costos)),
                   ('compilado', lambda: compilado.buscar(inicio, objetivo))]
        for nombre, metodo in metodos[::1 if repeticion % 2 == 0 else -1]:
            comienzo = time.perf_counter()
            camino = metodo()
            duracion = time.perf_counter() - comienzo
            if camino is None:
                raise ValueError('Los extremos deben estar conectados y fuera de obstáculos')
            costo = sum(np.hypot(a[0] - b[0], a[1] - b[1]) * (1 + costos[b])
                        for a, b in zip(camino, camino[1:]))
            resultados[nombre] = {'segundos': duracion, 'costo': costo, 'puntos': len(camino)}
        assert abs(resultados['referencia']['costo'] - resultados['compilado']['costo']) < 1e-6
        mediciones.append(resultados)
    salida = json.dumps({'maquina': platform.machine(), 'procesador': platform.processor(),
                         'forma': list(datos.shape), 'construccion_grafo_s': construccion,
                         'mediciones': mediciones}, indent=2)
    if args.output:
        args.output.write_text(salida + '\n')
    print(salida)


if __name__ == '__main__':
    main()
