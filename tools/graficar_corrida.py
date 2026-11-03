#!/usr/bin/env python3
"""Grafica una corrida: mapa + víctimas reales + reportes del equipo.

    python3 tools/graficar_corrida.py --mapa <mapa>.yaml --resultado resultado.json
                                       --salida grafico.png

Lee el YAML de map_server (+ su PGM) y el `resultado.json` de `tools/juez.py`.
Dibuja el mapa como RViz (libre blanco, pared negra, desconocido gris), cada
víctima como una estrella y cada reporte con una línea hasta la víctima que
puntuó (verde: 100, amarillo: 50, rojo: 0).
Las víctimas sin reporte quedan en gris.

Con `--trayectoria` dibuja además el recorrido de `grabar_trayectoria.py`.
No necesita ROS: solo matplotlib + yaml.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402


def leer_pgm(ruta):
    """Lee un PGM binario (P5). Devuelve (ancho, alto, lista de filas con bytes)."""
    datos = Path(ruta).read_bytes()
    partes, pos = [], 0
    while len(partes) < 4:  # magic, ancho, alto, maxval
        while datos[pos:pos + 1].isspace():
            pos += 1
        if datos[pos:pos + 1] == b'#':
            pos = datos.index(b'\n', pos)
            continue
        ini = pos
        while not datos[pos:pos + 1].isspace():
            pos += 1
        partes.append(datos[ini:pos])
    pos += 1
    if partes[0] != b'P5':
        raise ValueError(f'{ruta}: solo PGM binario (P5)')
    ancho, alto = int(partes[1]), int(partes[2])
    px = datos[pos:pos + ancho * alto]
    return ancho, alto, [px[f * ancho:(f + 1) * ancho] for f in range(alto)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--mapa', required=True, help='yaml de map_server')
    ap.add_argument('--resultado', required=True, help='resultado.json de juez.py')
    ap.add_argument('--trayectoria', help='trayectoria.jsonl de grabar_trayectoria.py (opcional)')
    ap.add_argument('--salida', required=True, help='PNG de salida')
    args = ap.parse_args()

    ruta_yaml = Path(args.mapa)
    meta = yaml.safe_load(ruta_yaml.read_text())
    res = float(meta['resolution'])
    ox, oy = float(meta['origin'][0]), float(meta['origin'][1])
    ancho, alto, filas = leer_pgm(ruta_yaml.parent / meta['image'])

    fig, ax = plt.subplots(figsize=(min(10, ancho * res + 2), min(8, alto * res + 2)))
    # En el PGM de map_server la fila 0 es la de arriba (y máxima): origin='upper'.
    ax.imshow([list(fila) for fila in filas], cmap='gray', vmin=0, vmax=255,
              extent=[ox, ox + ancho * res, oy, oy + alto * res], origin='upper')
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')

    d = json.loads(Path(args.resultado).read_text())
    victimas = [tuple(v) for v in d.get('victimas', [])]
    reportes = d.get('reportes', [])
    puntaje = d.get('puntaje', {})
    por_victima = {r['indice']: r for r in puntaje.get('por_victima', [])}
    colores = {100: 'green', 50: 'gold', 0: 'red'}
    def finito(v):
        return isinstance(v, (int, float))
    for i, (vx, vy) in enumerate(victimas):
        r = por_victima.get(i)
        if r and r.get('indice_reporte') is not None:
            rep = reportes[r['indice_reporte']]
            if not finito(rep.get('x')) or not finito(rep.get('y')):
                continue  # mal formado saneado a null: no se dibuja
            c = colores.get(r['puntos'], 'red')
            ax.plot([vx, rep['x']], [vy, rep['y']], color=c, linewidth=1.5)
            ax.plot(rep['x'], rep['y'], marker='o', color=c, markersize=7)
            ax.plot(vx, vy, marker='*', color=c, markersize=12)
        else:
            ax.plot(vx, vy, marker='*', color='gray', markersize=12)
    ax.set_title(d.get('resumen', 'corrida'))
    if args.trayectoria and Path(args.trayectoria).exists():
        xs, ys = [], []
        for linea in Path(args.trayectoria).read_text().splitlines():
            try:
                p = json.loads(linea)
                xs.append(p['x'])
                ys.append(p['y'])
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
        if xs:
            ax.plot(xs, ys, color='steelblue', linewidth=1.0, alpha=0.8)
            ax.plot(xs[0], ys[0], marker='o', color='steelblue', markersize=5)
    fig.tight_layout()
    fig.savefig(args.salida, dpi=100)
    print(f'gráfico: {args.salida}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
