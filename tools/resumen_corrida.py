#!/usr/bin/env python3
"""Resume una corrida en Markdown para el Job Summary del CI (o la terminal).

    python3 tools/resumen_corrida.py --resultado resultado.json [--salida resumen.md]

Sin `--salida` imprime por stdout. Está pensado para `>> $GITHUB_STEP_SUMMARY`.
El gráfico (PNG) no se incrusta: GitHub borra el `src` de las imágenes `data:`;
el PNG se publica vía Pages (#14) y el publicador lo linkea en un comentario
del PR.
"""
import argparse
import json
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--resultado', required=True)
    ap.add_argument('--salida', help='archivo md (por defecto, stdout)')
    args = ap.parse_args()

    d = json.loads(Path(args.resultado).read_text())
    p = d.get('puntaje', {})
    lineas = [f"## Corrida: {d.get('equipo') or 'equipo'}", '',
               f"**{d.get('resumen', 'sin resumen')}** "
               f"(fin: `{d.get('termino_por')}`, {d.get('tiempo_total_s')} s)", '']
    lineas.append('| Víctima | Puntos | Distancia (m) | Reporte # |')
    lineas.append('| --- | ---: | ---: | ---: |')
    for r in p.get('por_victima', []):
        dist = '-' if r.get('distancia') is None else f"{r['distancia']:.2f}"
        rep = '-' if r.get('indice_reporte') is None else r['indice_reporte'] + 1
        lineas.append(f"| V{r['indice'] + 1} | {r['puntos']} | {dist} | {rep} |")
    extras = []
    if p.get('bono_rapido'):
        extras.append('+200 bono 5 min')
    if p.get('bono_regreso'):
        extras.append('+100 regreso')
    if p.get('penalizacion_colisiones'):
        extras.append(f"-{p['penalizacion_colisiones']} colisiones")
    if p.get('penalizacion_salida'):
        extras.append(f"-{p['penalizacion_salida']} salida")
    if p.get('anulada'):
        extras.append('ANULADA (vuelco)')
    if extras:
        lineas += ['', ' | '.join(extras)]
    viol = d.get('violaciones', [])
    lineas += ['', f"Violaciones: {'ninguna' if not viol else ''}"]
    lineas += [f'- {v}' for v in viol]
    lineas += ['', 'Gráfico y logs completos en los artefactos de este job.']
    texto = '\n'.join(lineas) + '\n'
    if args.salida:
        Path(args.salida).write_text(texto)
    else:
        print(texto)
    return 0


if __name__ == '__main__':
    sys.exit(main())
