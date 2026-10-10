#!/usr/bin/env python3
"""Genera el sitio del leaderboard a partir de artefactos descargados.

    python3 tools/publicar_ranking.py --artefactos <dir> --sitio <dir>
        --run-id <id> --pr <n> --repo ORG/REPO --base-url https://... [--comentario m.md]

Busca `**/resultado.json` en los artefactos, valida cada uno contra el
contrato, ignora los inválidos (o sin mapa) con un aviso, y los agrega a
`data/historial.jsonl` del sitio. Después regenera `index.html` (tabla con
`ranking.py:armar()`) y una página por corrida, y escribe el comentario para
el PR con puntaje e imagen. No necesita ROS. Los nombres de equipo se escapan:
vienen de forks y van a HTML.
"""
import argparse
import html
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ranking import armar, finalistas  # noqa: E402
from validar_resultado import validar  # noqa: E402


def cargar_historial(sitio):
    ruta = sitio / 'data' / 'historial.jsonl'
    corridas = []
    if ruta.exists():
        for linea in ruta.read_text(encoding='utf-8').splitlines():
            try:
                corridas.append(json.loads(linea))
            except json.JSONDecodeError:
                continue
    return corridas


def pagina_corrida(c):
    e = html.escape
    equipo = e(c.get('equipo') or 'equipo')
    p = c.get('puntaje', {})
    filas = []
    for r in p.get('por_victima', []):
        dist = '-' if r.get('distancia') is None else f"{r['distancia']:.2f}"
        rep = '-' if r.get('indice_reporte') is None else r['indice_reporte'] + 1
        filas.append(f'<tr><td>V{r["indice"] + 1}</td><td>{r["puntos"]}</td>'
                     f'<td>{dist}</td><td>{rep}</td></tr>')
    img = (f'<p><img src="{c["grafico"]}" alt="gráfico" style="max-width:100%"></p>'
           if c.get('grafico') else '')
    return f"""<!doctype html><html lang="es"><head><meta charset="utf-8">
<title>{equipo} · {e(c.get('mapa', ''))} · corrida {c.get('run_id', '')}</title></head>
<body><p><a href="../index.html">← tabla</a></p>
<h1>{equipo} · {e(c.get('mapa', ''))}</h1>
<p><b>{e(c.get('resumen', ''))}</b> (fin: <code>{e(str(c.get('termino_por')))}</code>,
{c.get('tiempo_total_s')} s · PR #{c.get('pr', '?')} · run {c.get('run_id', '')})</p>
{img}
<table border="1"><tr><th>Víctima</th><th>Puntos</th><th>Distancia (m)</th><th>Reporte #</th></tr>
{''.join(filas)}</table>
<p><a href="{c.get('artefacto', '#')}">artefactos del run</a></p>
</body></html>
"""


def pagina_indice(filas, corridas_por_clave):
    e = html.escape

    def tiempo(f):
        return '-' if f['tiempo_s'] == float('inf') else f"{f['tiempo_s']:.1f}"

    trs = []
    for f in filas:
        trs.append(f"<tr><td>{f['posicion']}{'*' if f['empate'] else ''}</td>"
                   f'<td>{e(f["equipo"])}</td><td>{f["puntos"]}</td>'
                   f'<td>{tiempo(f)}</td>'
                   f'<td>{f["penalizaciones"]}</td><td>{f["corridas"]}</td></tr>')
    elegidos, aviso = finalistas(filas, 3)
    detalle = ''.join(
        f'<li><a href="corridas/{k}.html">{e(c.get("equipo") or "?")} · '
        f'{e(c.get("mapa", ""))} · {c.get("puntaje", {}).get("total", "?")} pts</a></li>'
        for k, c in sorted(corridas_por_clave.items()))
    return f"""<!doctype html><html lang="es"><head><meta charset="utf-8">
<title>Leaderboard · Challenge JAR 2026 (práctica)</title></head>
<body><h1>Tabla de práctica</h1>
<table border="1"><tr><th>Pos</th><th>Equipo</th><th>Puntos</th><th>Tiempo(s)</th>
<th>Penal.</th><th>Corr.</th></tr>{''.join(trs)}</table>
<p>Pasan a la final: {e(', '.join(f['equipo'] for f in elegidos))}</p>
{f'<p>ATENCIÓN: {e(aviso)}</p>' if aviso else ''}
<h2>Corridas</h2><ul>{detalle}</ul>
</body></html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--artefactos', required=True)
    ap.add_argument('--sitio', required=True)
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--pr', required=True)
    ap.add_argument('--repo', required=True, help='ORG/REPO para links a artefactos')
    ap.add_argument('--base-url', required=True, help='URL pública del sitio, sin / final')
    ap.add_argument('--comentario', required=True, help='md del comentario para el PR')
    args = ap.parse_args()

    sitio = Path(args.sitio)
    (sitio / 'data').mkdir(parents=True, exist_ok=True)
    (sitio / 'corridas').mkdir(exist_ok=True)
    (sitio / 'runs').mkdir(exist_ok=True)
    corridas = cargar_historial(sitio)
    vistos = {(c.get('run_id'), c.get('mapa')) for c in corridas}
    nuevas = []
    for res in sorted(Path(args.artefactos).rglob('resultado.json')):
        errores = validar(res)
        if errores:
            print(f"aviso: se ignora {res}: {errores[0]}")
            continue
        d = json.loads(res.read_text(encoding='utf-8'))
        if not d.get('mapa_publicado', True):
            print(f'aviso: se ignora {res}: sin mapa')
            continue
        mapa = res.parent.name
        if (args.run_id, mapa) in vistos:
            continue  # re-publicación idempotente
        vistos.add((args.run_id, mapa))
        clave = f"{args.run_id}-{mapa}"
        destino = sitio / 'runs' / clave
        destino.mkdir(exist_ok=True)
        shutil.copy(res, destino / 'resultado.json')
        if (res.parent / 'grafico.png').exists():
            shutil.copy(res.parent / 'grafico.png', destino / 'grafico.png')
            d['grafico'] = f"{args.base_url}/runs/{clave}/grafico.png"
        d.update({'run_id': args.run_id, 'pr': int(args.pr), 'mapa': mapa,
                  'artefacto': f'https://github.com/{args.repo}/actions/runs/{args.run_id}',
                  'clave': clave})
        corridas.append(d)
        nuevas.append(d)
    (sitio / 'data' / 'historial.jsonl').write_text(
        '\n'.join(json.dumps(c) for c in corridas) + '\n', encoding='utf-8')
    por_clave = {c['clave']: c for c in corridas}
    for clave, c in por_clave.items():
        (sitio / 'corridas' / f'{clave}.html').write_text(pagina_corrida(c), encoding='utf-8')
    filas = armar(corridas)
    (sitio / 'index.html').write_text(pagina_indice(filas, por_clave), encoding='utf-8')

    lineas = [f'## Leaderboard actualizado (PR #{args.pr})', '']
    if nuevas:
        for c in nuevas:
            lineas.append(f"**{c.get('resumen', '')}** · `{c['mapa']}` · "
                          f"[corrida]({args.base_url}/corridas/{c['clave']}.html) · "
                          f"[artefactos]({c['artefacto']})")
            if c.get('grafico'):
                lineas.append(f'![gráfico]({c["grafico"]})')
            lineas.append('')
    else:
        lineas.append('Sin corridas válidas nuevas en este run.')
    Path(args.comentario).write_text('\n'.join(lineas), encoding='utf-8')
    print(f'sitio: {len(corridas)} corridas, {len(nuevas)} nuevas')
    return 0


if __name__ == '__main__':
    sys.exit(main())
