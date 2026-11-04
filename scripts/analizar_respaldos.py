#!/usr/bin/env python3
"""Cuenta los respaldos de percepción de cada corrida y dice si AMCL tenía razón.

    python3 scripts/analizar_respaldos.py pruebas/2026-11-0*/

Para cada corrida de pruebas/ (equipo.log + localizacion.jsonl de scripts/probar.sh):
- los respaldos que tomó el supervisor de nodo_percepcion (supervisor_localizacion.py): cuándo
  empezaron, cuánto duraron y qué dijo el log;
- el error real de AMCL durante cada uno, contra la pose real del simulador (localizacion.jsonl).
  Si AMCL estaba bien (mediana del error <= --max-error-m y --max-error-deg), el respaldo fue
  **falso**: AMCL estaba corrigiendo bien la deriva de /odom y el supervisor no le creyó. Si no,
  fue **justificado**.

Un respaldo falso cuesta: al cambiar de fuente, percepción tira las vistas sin confirmar, y en
respaldo ubica las cajas con la última referencia + /odom (peor que AMCL si AMCL estaba bien).

Necesita el log con el tiempo de simulación (" t=123.4" al final de cada "Pose: ..."), que
nodo_percepcion escribe desde que se agregó este script. localizacion.jsonl usa el mismo reloj.

Es una herramienta de prueba, NO parte del software del equipo: usa la pose real del simulador,
que nuestro código (src/) no puede usar nunca (§7.1).
"""
import argparse
import json
import re
import statistics
import sys
from pathlib import Path

RE_POSE = re.compile(r'\[nodo_percepcion\]: Pose: (?P<fuente>[^(]+?) \((?P<motivo>.*)\) '
                     r't=(?P<t>[\d.]+)')


def leer_cambios(log):
    """[(t, fuente, motivo)] de cada cambio de fuente de la pose que logueó percepción."""
    cambios = []
    for linea in open(log, errors='replace'):
        m = RE_POSE.search(linea)
        if m:
            cambios.append((float(m['t']), m['fuente'], m['motivo']))
    return cambios


def episodios(cambios, t_fin):
    """[(inicio, fin, motivo de entrada)] de cada respaldo. fin es None si siguió hasta el final."""
    salida, inicio, motivo = [], None, None
    for t, fuente, mot in cambios:
        if fuente.startswith('respaldo'):
            if inicio is None:
                inicio, motivo = t, mot
        elif inicio is not None:
            salida.append((inicio, t, motivo))
            inicio = None
    if inicio is not None:
        salida.append((inicio, None, motivo))
    return salida


def errores_entre(localizacion, desde, hasta):
    """(error_m, |error_deg|) de AMCL contra la pose real, de las mediciones en [desde, hasta]."""
    return [(d['error_m'], abs(d['error_deg'])) for d in localizacion
            if desde <= d['t'] <= hasta]


def analizar(carpeta, max_m, max_deg):
    carpeta = Path(carpeta)
    log, loc = carpeta / 'equipo.log', carpeta / 'localizacion.jsonl'
    if not log.exists() or not loc.exists():
        return None
    localizacion = [json.loads(linea) for linea in open(loc) if linea.strip()]
    cambios = leer_cambios(log)
    if not cambios:
        return {'corrida': carpeta.name, 'sin_t': True}
    t_fin = max([d['t'] for d in localizacion] + [t for t, _, _ in cambios])
    filas = []
    for inicio, fin, motivo in episodios(cambios, t_fin):
        errores = errores_entre(localizacion, inicio, fin if fin is not None else t_fin)
        if errores:
            med_m = statistics.median(e[0] for e in errores)
            med_deg = statistics.median(e[1] for e in errores)
            veredicto = 'falso' if med_m <= max_m and med_deg <= max_deg else 'justificado'
        else:
            med_m = med_deg = None
            veredicto = 'sin datos'
        filas.append({'inicio': inicio, 'duracion': None if fin is None else fin - inicio,
                      'motivo': motivo, 'error_m': med_m, 'error_deg': med_deg,
                      'veredicto': veredicto})
    return {'corrida': carpeta.name, 'respaldos': filas, 't_fin': t_fin}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('corridas', nargs='+', help='carpetas de pruebas/')
    ap.add_argument('--max-error-m', type=float, default=0.15,
                    help='AMCL "tenía razón" si la mediana de su error real es menor (m)')
    ap.add_argument('--max-error-deg', type=float, default=5.0)
    ap.add_argument('--json', action='store_true', help='salida en JSON, una corrida por línea')
    args = ap.parse_args()

    total = {'falso': 0, 'justificado': 0, 'sin datos': 0}
    duraciones = {'falso': [], 'justificado': []}
    for carpeta in args.corridas:
        r = analizar(carpeta, args.max_error_m, args.max_error_deg)
        if r is None:
            continue
        if args.json:
            print(json.dumps(r, ensure_ascii=False))
            continue
        if r.get('sin_t'):
            print(f'{r["corrida"]}: el log no tiene " t=" en las líneas de Pose (corrida vieja)')
            continue
        filas = r['respaldos']
        cuenta = {v: sum(f['veredicto'] == v for f in filas) for v in total}
        print(f'\n{r["corrida"]}: {len(filas)} respaldos ({cuenta["falso"]} falsos, '
              f'{cuenta["justificado"]} justificados, {cuenta["sin datos"]} sin datos)')
        for f in filas:
            dur = f'{f["duracion"]:5.1f} s' if f['duracion'] is not None else 'hasta el fin'
            err = (f'AMCL real {f["error_m"]:.2f} m / {f["error_deg"]:.0f}°'
                   if f['error_m'] is not None else 'sin pose real')
            print(f'  t={f["inicio"]:6.1f}  {dur:>12}  {err:24}  {f["veredicto"]:11}  '
                  f'{f["motivo"]}')
            total[f['veredicto']] += 1
            if f['duracion'] is not None and f['veredicto'] in duraciones:
                duraciones[f['veredicto']].append(f['duracion'])
    if not args.json:
        print(f'\nTotal: {sum(total.values())} respaldos: {total["falso"]} falsos, '
              f'{total["justificado"]} justificados, {total["sin datos"]} sin datos')
        for v, ds in duraciones.items():
            if ds:
                print(f'  {v}: duración mediana {statistics.median(ds):.1f} s, '
                      f'máx {max(ds):.1f} s')
    return 0


if __name__ == '__main__':
    sys.exit(main())
