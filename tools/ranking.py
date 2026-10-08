#!/usr/bin/env python3
"""Tabla de posiciones a partir de los resultado.json de las corridas (reglamento §4.1, §8.2).

    python3 tools/ranking.py <carpeta_de_corridas> [--criterio mejor|suma]
                             [--desempate total|ultimo_reporte] [--finalistas 4] [--json]

Lee <carpeta>/*/resultado.json (los que deja tools/juez.py). Agrupa por equipo, arma el puntaje
según el criterio y ordena. Desempate, en el orden de §8.2:
  1. menor tiempo total de la corrida, desde la publicación de /map hasta el fin. Con
     --desempate ultimo_reporte se usa el tiempo hasta el último reporte que puntúa;
  2. menos penalizaciones por colisión;
  3. en la clasificación (`--criterio mejor`), el puntaje de la otra corrida del equipo;
  4. si siguen igual, queda marcado como empate: el reglamento no dice cómo seguir.

Para el ranking cuenta la mejor corrida de cada equipo (§4.1, `--criterio mejor`, el valor por defecto)
y el desempate es el tiempo total de la corrida (§8.2). `--criterio suma` y
`--desempate ultimo_reporte` quedan por si alguna vez se quiere comparar.
Una corrida anulada (vuelco) cuenta como usada, con 0 puntos.
"""
import argparse
import json
import sys
from pathlib import Path

INF = float('inf')


def cargar(carpeta):
    """Lista de resultados (dicts) de <carpeta>/*/resultado.json, con 'equipo' completado."""
    corridas = []
    for f in sorted(Path(carpeta).glob('*/resultado.json')):
        d = json.loads(f.read_text())
        if not d.get('equipo'):
            # Carpetas <fecha>_<hora>_<equipo>: el equipo es lo que sigue a las dos primeras partes.
            partes = f.parent.name.split('_', 2)
            d['equipo'] = partes[2] if len(partes) == 3 else f.parent.name
        d['_archivo'] = str(f)
        corridas.append(d)
    return corridas


def _tiempo(d, desempate):
    clave = 'tiempo_total_s' if desempate == 'total' else 'tiempo_ultimo_reporte_que_puntua_s'
    v = d.get(clave)
    return INF if v is None else float(v)


def _penal(d):
    # §8.2 paso 3: solo las penalizaciones por colisión, no la de salida del laberinto.
    return d['puntaje']['penalizacion_colisiones']


def _avisos(corridas):
    av = []
    if any(c['puntaje'].get('anulada') for c in corridas):
        av.append('corrida anulada')
    if any(c.get('violaciones') for c in corridas):
        av.append('violaciones (revisar)')
    if any(not c.get('mapa_publicado', True) for c in corridas):
        av.append('sin mapa')
    return av


def armar(corridas, criterio='mejor', desempate='total'):
    """Devuelve la lista de equipos ordenada, cada uno con su puntaje y sus claves de desempate."""
    por_equipo = {}
    for c in corridas:
        por_equipo.setdefault(c['equipo'], []).append(c)
    filas = []
    for equipo, cs in por_equipo.items():
        if criterio == 'mejor':
            # La mejor corrida: más puntos; a igual puntaje, la de menos tiempo y menos penalizaciones.
            mejor = min(cs, key=lambda c: (-c['puntaje']['total'], _tiempo(c, desempate), _penal(c)))
            puntos, tiempo, penal = mejor['puntaje']['total'], _tiempo(mejor, desempate), _penal(mejor)
            # §8.2 paso 4: si siguen empatados, cuenta el puntaje de la otra corrida (más es mejor).
            resto = list(cs)
            resto.remove(mejor)
            otras = tuple(sorted((c['puntaje']['total'] for c in resto), reverse=True))
        else:
            puntos = sum(c['puntaje']['total'] for c in cs)
            tiempo = sum(_tiempo(c, desempate) for c in cs)
            penal = sum(_penal(c) for c in cs)
            otras = ()
        filas.append({'equipo': equipo, 'puntos': puntos, 'tiempo_s': tiempo, 'penalizaciones': penal,
                      'otras_corridas': otras, 'corridas': len(cs), 'avisos': _avisos(cs)})

    def clave(f):
        return (-f['puntos'], f['tiempo_s'], f['penalizaciones'], tuple(-p for p in f['otras_corridas']))

    filas.sort(key=lambda f: (clave(f), f['equipo']))
    # Posiciones: los que empatan en todas las claves comparten posición y quedan marcados.
    pos = 0
    for i, f in enumerate(filas):
        if i > 0 and clave(f) == clave(filas[i - 1]):
            f['posicion'] = filas[i - 1]['posicion']
            f['empate'] = filas[i - 1]['empate'] = True
        else:
            pos = i + 1
            f['posicion'], f['empate'] = pos, False
    return filas


def finalistas(filas, k=4):
    """Los k primeros, y un aviso si el empate cruza el corte y hace falta desempatar."""
    elegidos = filas[:k]
    aviso = None
    if len(filas) > k and filas[k - 1]['posicion'] == filas[k]['posicion']:
        empatados = [f['equipo'] for f in filas if f['posicion'] == filas[k - 1]['posicion']]
        aviso = ('empate por el último lugar de la final entre ' + ', '.join(empatados)
                 + ': el reglamento no dice cómo desempatar (§8.2)')
    return elegidos, aviso


def texto(filas, k):
    def t(f):
        return '-' if f['tiempo_s'] == INF else f"{f['tiempo_s']:.1f}"
    lineas = [f"{'Pos':>3}  {'Equipo':<24} {'Puntos':>6} {'Tiempo(s)':>9} {'Penal.':>6} {'Corr.':>5}  Avisos"]
    for f in filas:
        pos = f"{f['posicion']}{'*' if f['empate'] else ''}"
        lineas.append(f"{pos:>3}  {f['equipo']:<24} {f['puntos']:>6} {t(f):>9} {f['penalizaciones']:>6} "
                      f"{f['corridas']:>5}  {', '.join(f['avisos'])}")
    elegidos, aviso = finalistas(filas, k)
    lineas.append('')
    lineas.append(f'Pasan a la final: ' + ', '.join(f['equipo'] for f in elegidos))
    if any(f['empate'] for f in filas):
        lineas.append('(* empatados en puntos, tiempo, colisiones y otra corrida)')
    if aviso:
        lineas.append('ATENCIÓN: ' + aviso)
    return '\n'.join(lineas)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('carpeta')
    ap.add_argument('--criterio', choices=['mejor', 'suma'], default='mejor')
    ap.add_argument('--desempate', choices=['total', 'ultimo_reporte'], default='total')
    ap.add_argument('--finalistas', type=int, default=4)
    ap.add_argument('--json', action='store_true')
    args = ap.parse_args()
    corridas = cargar(args.carpeta)
    if not corridas:
        print(f'no encontré resultado.json en {args.carpeta}/*/', file=sys.stderr)
        return 1
    filas = armar(corridas, args.criterio, args.desempate)
    if args.json:
        print(json.dumps(filas, indent=2, default=lambda o: None))
    else:
        print(texto(filas, args.finalistas))
    return 0


if __name__ == '__main__':
    sys.exit(main())
