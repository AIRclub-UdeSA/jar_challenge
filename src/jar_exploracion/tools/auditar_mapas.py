"""Audita configuración y conectividad de mapas ROS sin iniciar Gazebo."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
import yaml

from jar_exploracion.planificador import PlanificadorCompilado, segmento_transitable
from jar_exploracion.seguridad import espacio_transitable


def auditar(ruta):
    """Comprueba rutas largas del componente principal para dos orientaciones."""
    config = yaml.safe_load(ruta.read_text())
    imagen = np.asarray(Image.open(ruta.parent / config['image']).convert('L'))[::-1]
    ocupacion = imagen / 255.0 if config.get('negate', 0) else 1 - imagen / 255.0
    datos = np.full(imagen.shape, -1, dtype=int)
    datos[ocupacion < config['free_thresh']] = 0
    datos[ocupacion > config['occupied_thresh']] = 100
    resultado = {'mapa': ruta.name, 'forma': list(datos.shape), 'orientaciones': []}
    datos = np.repeat(np.repeat(datos, 2, axis=0), 2, axis=1)
    resolucion = config['resolution'] / 2
    resultado['forma_planificacion'] = list(datos.shape)
    for yaw in (0., math.pi / 2):
        libres = espacio_transitable(datos, resolucion, 0.37, 0.24, 0.056, yaw)
        etiquetas, cantidad = ndimage.label(libres)
        rutas = 0
        if cantidad:
            planificador = PlanificadorCompilado(libres)
            tamanos = np.bincount(etiquetas.ravel())
            tamanos[0] = 0
            componente = np.argwhere(etiquetas == np.argmax(tamanos))
            for fraccion in (0.25, 0.5, 0.75):
                inicio = tuple(map(int, componente[0]))
                indice = min(len(componente) - 1, int(fraccion * len(componente)))
                fin = tuple(map(int, componente[indice]))
                camino = planificador.buscar(inicio, fin)
                if camino is None:
                    raise AssertionError(f'{ruta.name}: A* falló dentro del componente principal')
                if not all(segmento_transitable(libres, a, b)
                           for a, b in zip(camino, camino[1:])):
                    raise AssertionError(f'{ruta.name}: un segmento atraviesa un obstáculo')
                rutas += 1
        resultado['orientaciones'].append({
            'yaw': yaw, 'celdas_seguras': int(libres.sum()),
            'componentes': cantidad, 'rutas_comprobadas': rutas})
    return resultado


def main():
    """Guarda resultados reproducibles para todos los YAML de un directorio."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mapas', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    resultados = [auditar(ruta) for ruta in sorted(args.mapas.glob('*.yaml'))]
    texto = json.dumps(resultados, indent=2)
    if args.output:
        args.output.write_text(texto + '\n')
    print(texto)


if __name__ == '__main__':
    main()
