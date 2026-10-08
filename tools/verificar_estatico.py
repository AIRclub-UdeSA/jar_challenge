#!/usr/bin/env python3
"""Revisión estática del código del equipo (§9.2, §10).

Dos niveles, a propósito:

- ERROR (rompe el CI): evidencia directa de romper §7.1, por ejemplo leer el
  ground truth del simulador o hablar con Gazebo por línea de comandos.
- ADVERTENCIA (no rompe el CI): heurísticas de posible hardcodeo. Quedan
  marcadas en el resumen del job para que el juez técnico las confirme antes
  de aplicar cualquier sanción (§9.2). Una advertencia NO es una sanción: hay
  usos legítimos (por ejemplo un patrón de búsqueda definido en el marco del
  robot).
"""
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from interfaces import (  # noqa: E402
    PATRONES_PROHIBIDOS, TOPICS_INFRAESTRUCTURA, TOPICS_PERMITIDOS)

RAIZ = Path(__file__).resolve().parent.parent
CODIGO = RAIZ / 'src'
EXTENSIONES = {'.py', '.cpp', '.hpp', '.h', '.cc', '.launch', '.xml', '.yaml', '.yml', '.sh'}

# Literales de topic entre comillas: "/algo" o '/algo/otro'
RE_TOPIC = re.compile(r'''["'](/[A-Za-z_][A-Za-z0-9_/]*)["']''')

# 3 o más pares de decimales seguidos: [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]
RE_PARES = re.compile(
    r'[\[\(]\s*[\[\(]?\s*-?\d+\.\d+\s*,\s*-?\d+\.\d+\s*[\]\)]?'
    r'(\s*,\s*[\[\(]?\s*-?\d+\.\d+\s*,\s*-?\d+\.\d+\s*[\]\)]?){2,}')
RE_NOMBRES = re.compile(
    r'\b\w*(waypoint|victim|victima|checkpoint|ruta)s?\w*\s*=\s*[\[\(\{]', re.I)
RE_MUNDOS = re.compile(r'\b(maze_\d|laberinto_|cafe_world|willowgarage)\w*', re.I)


def recorrer():
    for p in sorted(CODIGO.rglob('*')):
        if p.is_file() and p.suffix in EXTENSIONES and '__pycache__' not in p.parts:
            if p.parts[-2] == 'test' or p.name.startswith('test_') or p.name == 'setup.py':
                continue
            yield p


def anotar(nivel, archivo, linea, msg):
    # Formato de anotaciones de GitHub Actions; en terminal se lee igual.
    print(f'::{nivel} file={archivo},line={linea}::{msg}')


def main():
    errores = advertencias = 0
    for p in recorrer():
        rel = p.relative_to(RAIZ)
        texto = p.read_text(errors='replace')
        lineas = texto.splitlines()

        for n, linea in enumerate(lineas, 1):
            codigo = linea.split('#', 1)[0] if p.suffix in {'.py', '.sh', '.yaml', '.yml'} else linea
            for patron, motivo in PATRONES_PROHIBIDOS:
                if re.search(patron, codigo):
                    anotar('error', rel, n, f'interfaz no permitida: {motivo}')
                    errores += 1
            for topic in RE_TOPIC.findall(codigo):
                if topic in TOPICS_PERMITIDOS or topic in TOPICS_INFRAESTRUCTURA:
                    continue
                anotar('notice', rel, n,
                       f'topic {topic} no es de §7.1: está bien si es interno de tu software, '
                       'pero no puede ser de la plataforma o del simulador')
            if RE_NOMBRES.search(codigo):
                anotar('warning', rel, n,
                       'posible hardcodeo: variable con nombre de waypoints/víctimas/ruta '
                       'inicializada con datos literales. Revisión del juez técnico')
                advertencias += 1
            if RE_MUNDOS.search(codigo):
                anotar('warning', rel, n,
                       'referencia a un mundo/mapa de práctica por nombre. Revisión del juez técnico')
                advertencias += 1

        for m in RE_PARES.finditer(texto):
            n = texto.count('\n', 0, m.start()) + 1
            anotar('warning', rel, n,
                   'posible hardcodeo: 3+ pares de coordenadas literales. Si son posiciones '
                   'del mapa de competencia están prohibidas (§10). Revisión del juez técnico')
            advertencias += 1

    resumen = (f'estático: {errores} error(es), {advertencias} advertencia(s) '
               'de hardcodeo para revisión del juez técnico')
    print(resumen)
    destino = os.environ.get('GITHUB_STEP_SUMMARY')
    if destino and (errores or advertencias):
        with open(destino, 'a') as f:
            f.write(f'### Revisión estática\n\n{resumen}\n\n'
                    'Las advertencias no rompen el CI: las confirma el juez técnico (§9.2).\n')
    return 1 if errores else 0


if __name__ == '__main__':
    sys.exit(main())
