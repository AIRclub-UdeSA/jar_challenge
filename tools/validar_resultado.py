#!/usr/bin/env python3
"""Valida un `resultado.json` contra el contrato de `docs/RESULTADO_JSON.md`.

    python3 tools/validar_resultado.py resultado.json

Falla (exit 1) si falta una clave que necesita `ranking.py`, si un tipo no
coincide o si `puntaje` no trae el desglose por víctima. No necesita ROS.
"""
import json
import sys
from pathlib import Path


def es_numero(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def es_entero(x):
    return isinstance(x, int) and not isinstance(x, bool)


def validar(ruta):
    """Devuelve la lista de errores del contrato (vacía si es válido).

    Además de las claves, chequea los tipos: el resultado de un PR se publica en
    Pages, y un string donde va un número rompería el ranking o terminaría en el HTML.
    """
    errores = []
    try:
        d = json.loads(Path(ruta).read_text())
    except (json.JSONDecodeError, OSError) as e:
        return [f'resultado inválido: {e}']
    if not isinstance(d, dict):
        return ['el resultado tiene que ser un objeto']

    for clave in ('equipo', 'termino_por', 'mapa_publicado', 'reportes',
                  'violaciones', 'puntaje', 'victimas',
                  'tiempo_total_s', 'tiempo_ultimo_reporte_que_puntua_s'):
        if clave not in d:
            errores.append(f'falta la clave {clave!r}')
    for clave in ('tiempo_total_s', 'tiempo_ultimo_reporte_que_puntua_s'):
        if d.get(clave) is not None and not es_numero(d[clave]):
            errores.append(f'{clave} tiene que ser número o null')
    for clave in ('equipo', 'termino_por'):
        if d.get(clave) is not None and not isinstance(d[clave], str):
            errores.append(f'{clave} tiene que ser texto o null')
    if 'mapa_publicado' in d and not isinstance(d['mapa_publicado'], bool):
        errores.append('mapa_publicado tiene que ser true o false')
    if not isinstance(d.get('violaciones', []), list):
        errores.append('violaciones tiene que ser una lista')
    if not isinstance(d.get('puntaje'), dict):
        errores.append('puntaje tiene que ser un objeto')
    else:
        p = d['puntaje']
        for clave in ('por_victima', 'total', 'penalizacion_colisiones',
                      'penalizacion_salida', 'anulada'):
            if clave not in p:
                errores.append(f'falta puntaje.{clave}')
        for clave in ('total', 'penalizacion_colisiones', 'penalizacion_salida'):
            if clave in p and not es_entero(p[clave]):
                errores.append(f'puntaje.{clave} tiene que ser entero')
        if 'anulada' in p and not isinstance(p['anulada'], bool):
            errores.append('puntaje.anulada tiene que ser true o false')
        if not isinstance(p.get('por_victima', []), list):
            errores.append('puntaje.por_victima tiene que ser una lista')
        else:
            for r in p.get('por_victima', []):
                if (not isinstance(r, dict) or not es_entero(r.get('indice'))
                        or not es_entero(r.get('puntos'))
                        or not (r.get('distancia') is None or es_numero(r['distancia']))
                        or not (r.get('indice_reporte') is None or es_entero(r['indice_reporte']))):
                    errores.append(f'reporte por víctima mal formado: {r!r}')
                    break
        reportes = d.get('reportes', [])
        if not isinstance(reportes, list):
            errores.append('reportes tiene que ser una lista')
            reportes = []
        for r in reportes:
            if not isinstance(r, dict):
                errores.append(f'reporte mal formado: {r!r}')
                break
            for k in ('x', 'y'):
                v = r.get(k)
                if v is not None and not es_numero(v):
                    errores.append(f'reporte con {k} no numérico ni null: {v!r}')
                    break
    return errores


def main():
    if len(sys.argv) != 2:
        print('uso: validar_resultado.py resultado.json')
        return 2
    errores = validar(sys.argv[1])
    for e in errores:
        print(f'::error::resultado: {e}')
    print('resultado: OK' if not errores else f'resultado: {len(errores)} error(es)')
    return 1 if errores else 0


if __name__ == '__main__':
    sys.exit(main())
