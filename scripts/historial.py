#!/usr/bin/env python3
"""Resume una corrida de probar.sh, la suma al historial y muestra las últimas.

    python3 scripts/historial.py --carpeta pruebas/<corrida> --historial pruebas/historial.csv \
        --commit abc1234 --mapa maze_1_6x5 --spawn "0.2 -0.15 0.3" --nota "subí alpha1"
    python3 scripts/historial.py --historial pruebas/historial.csv   # solo mostrar

Lee resultado.json (del juez), localizacion.jsonl (de medir_localizacion.py) y equipo.log.
Solo usa la biblioteca estándar de Python.
"""
import argparse
import csv
import json
import statistics
from datetime import datetime
from pathlib import Path

COLUMNAS = ['fecha', 'commit', 'mapa', 'spawn', 'duracion_s', 'total', 'victimas',
            'distancias_m', 'bono_rapido', 'bono_regreso', 'colisiones', 'termino_por',
            'loc_convergencia_s', 'loc_error_medio_m', 'loc_error_max_m', 'loc_error_max_deg',
            'respaldo_odom', 'nota']
# La localización se considera convergida con menos de este error (posición y ángulo).
CONVERGIDO_M = 0.10
CONVERGIDO_DEG = 5.0


def resumir(carpeta):
    fila = {}
    resultado = carpeta / 'resultado.json'
    if resultado.exists():
        d = json.loads(resultado.read_text())
        p = d.get('puntaje', {})
        por_victima = p.get('por_victima', [])
        fila.update({
            'duracion_s': d.get('tiempo_total_s'),
            'total': p.get('total'),
            'victimas': f"{sum(1 for v in por_victima if v.get('puntos'))}/{len(por_victima)}",
            'distancias_m': ' '.join(
                '-' if v.get('distancia') is None else f"{v['distancia']:.2f}"
                for v in por_victima),
            'bono_rapido': 'sí' if p.get('bono_rapido') else 'no',
            'bono_regreso': 'sí' if p.get('bono_regreso') else 'no',
            'colisiones': p.get('penalizacion_colisiones', 0),
            'termino_por': d.get('termino_por'),
        })
    medidas = carpeta / 'localizacion.jsonl'
    if medidas.exists():
        filas = [json.loads(linea) for linea in medidas.read_text().splitlines() if linea]
        # Al arrancar AMCL todavía no convergió (parte de la pose inicial, no de la real): eso
        # se mide aparte como convergencia, y el error medio y máximo se calculan después.
        convergida = next((i for i, f in enumerate(filas) if f['error_m'] < CONVERGIDO_M
                           and abs(f['error_deg']) < CONVERGIDO_DEG), None)
        if convergida is None:
            fila['loc_convergencia_s'] = 'nunca'
        else:
            despues = filas[convergida:]
            errores = [f['error_m'] for f in despues]
            fila.update({
                'loc_convergencia_s': f"{despues[0]['t'] - filas[0]['t']:.0f}",
                'loc_error_medio_m': f'{statistics.mean(errores):.3f}',
                'loc_error_max_m': f'{max(errores):.3f}',
                'loc_error_max_deg': f"{max(abs(f['error_deg']) for f in despues):.1f}",
            })
    log = carpeta / 'equipo.log'
    if log.exists():
        fila['respaldo_odom'] = 'SÍ' if 'AMCL no se localizó' in log.read_text(
            errors='replace') else 'no'
    return fila


def mostrar(historial, ultimas):
    with historial.open() as f:
        filas = list(csv.DictReader(f))[-ultimas:]
    cols = ['fecha', 'commit', 'mapa', 'spawn', 'total', 'victimas', 'distancias_m',
            'bono_regreso', 'loc_convergencia_s', 'loc_error_medio_m', 'loc_error_max_m', 'nota']
    anchos = {c: max(len(c), *(len(str(r.get(c, ''))) for r in filas)) for c in cols}
    print('  '.join(c.ljust(anchos[c]) for c in cols))
    for r in filas:
        print('  '.join(str(r.get(c, '')).ljust(anchos[c]) for c in cols))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--carpeta', help='carpeta de la corrida (sin esto, solo muestra el historial)')
    ap.add_argument('--historial', required=True)
    ap.add_argument('--commit', default='')
    ap.add_argument('--mapa', default='')
    ap.add_argument('--spawn', default='')
    ap.add_argument('--nota', default='')
    ap.add_argument('--ultimas', type=int, default=10)
    args = ap.parse_args()

    historial = Path(args.historial)
    if args.carpeta:
        fila = {'fecha': datetime.now().strftime('%Y-%m-%d %H:%M'), 'commit': args.commit,
                'mapa': args.mapa, 'spawn': args.spawn, 'nota': args.nota}
        fila.update(resumir(Path(args.carpeta)))
        nuevo = not historial.exists()
        historial.parent.mkdir(parents=True, exist_ok=True)
        with historial.open('a', newline='') as f:
            escritor = csv.DictWriter(f, fieldnames=COLUMNAS, extrasaction='ignore')
            if nuevo:
                escritor.writeheader()
            escritor.writerow(fila)
        print('\nEsta corrida:')
        for clave in COLUMNAS[4:]:
            print(f'  {clave:18} {fila.get(clave, "")}')
    if historial.exists():
        print(f'\nÚltimas corridas ({historial}):')
        mostrar(historial, args.ultimas)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
