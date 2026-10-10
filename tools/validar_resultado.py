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


def validar(ruta):
    """Devuelve la lista de errores del contrato (vacía si es válido)."""
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
    if not isinstance(d.get('puntaje'), dict):
        errores.append('puntaje tiene que ser un objeto')
    else:
        p = d['puntaje']
        for clave in ('por_victima', 'total', 'penalizacion_colisiones',
                      'penalizacion_salida', 'anulada'):
            if clave not in p:
                errores.append(f'falta puntaje.{clave}')
        if isinstance(p.get('por_victima'), list):
            for r in p['por_victima']:
                if not isinstance(r, dict) or 'indice' not in r or 'puntos' not in r:
                    errores.append(f'reporte por víctima mal formado: {r!r}')
                    break
        for r in d.get('reportes', []):
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
