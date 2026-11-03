#!/usr/bin/env python3
"""Genera el sitio del leaderboard de práctica y el comentario para el PR.

    python3 tools/publicar_ranking.py --artefactos <dir> --sitio <dir> --comentario c.md
        --run-id <id> --pr <n> --sha <sha> --repo ORG/REPO --base-url https://...
        --equipo-id <login del fork> --equipo-yaml equipo.yaml --verde si|no --toca-infra si|no

Lo corre `publicar-ranking.yml` en contexto main. Todo lo que llega de un PR
(artefactos y `equipo.yaml`) es dato no confiable:
- cada `resultado.json` se valida contra el contrato, con tipos;
- el mapa es el nombre de la carpeta del artefacto y tiene que ser `[a-z0-9_]+`;
- el gráfico se copia solo si es un PNG de menos de 1 MB;
- el equipo es el dueño del fork, no el campo `equipo` del resultado; el nombre que
  se muestra sale de `equipo.yaml` y se escapa;
- si el PR toca `tools/` o `.github/`, los artefactos no los generó el CI de la
  organización: no se publica nada y el comentario lo explica.

Todas las corridas quedan en `data/historial.jsonl` con su página. A la tabla (una
por mapa, mejor corrida de cada equipo) entran solo las de CI en verde. No necesita ROS.
"""
import argparse
import html
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ranking import _penal, _tiempo, armar  # noqa: E402
from validar_resultado import validar  # noqa: E402

MARCA = '<!-- jar-ranking -->'
MAPA_VALIDO = re.compile(r'[a-z0-9_]{1,64}')
PNG = b'\x89PNG\r\n\x1a\n'
PNG_MAX = 1_000_000
e = html.escape


def patron(regex, que):
    def tipo(valor):
        if not re.fullmatch(regex, valor):
            raise argparse.ArgumentTypeError(f'{que} inválido: {valor!r}')
        return valor
    return tipo


def nombre_de_equipo(ruta):
    """`nombre_equipo` de equipo.yaml, o None si no está o quedó sin completar."""
    try:
        texto = Path(ruta).read_text(encoding='utf-8')
    except (OSError, UnicodeDecodeError):
        return None
    m = re.search(r'^nombre_equipo:\s*(.*?)\s*(?:#.*)?$', texto, re.MULTILINE)
    nombre = m.group(1).strip('\'"').strip() if m else ''
    if not nombre or 'COMPLETAR' in nombre:
        return None
    return nombre[:40]


def es_png(ruta):
    try:
        return ruta.stat().st_size <= PNG_MAX and ruta.read_bytes()[:8] == PNG
    except OSError:
        return False


def resumen(c):
    """Resumen armado con los números ya validados (nunca con texto del artefacto)."""
    p = c['puntaje']
    victimas = ' '.join(f"V{r['indice'] + 1}:{r['puntos']}" for r in p['por_victima'])
    return f"total {p['total']} | víctimas [{victimas}]"


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


def entrada(d, args, mapa, nombre, grafico):
    """Lo que se guarda en el historial: solo lo que usan la tabla y las páginas."""
    p = d['puntaje']
    return {
        'clave': f'{args.run_id}-{mapa}', 'run_id': args.run_id, 'pr': int(args.pr),
        'sha': args.sha, 'mapa': mapa, 'equipo': args.equipo_id, 'nombre_equipo': nombre,
        'verde': args.verde == 'si', 'grafico': grafico,
        'termino_por': d.get('termino_por'), 'mapa_publicado': d['mapa_publicado'],
        'violaciones': d.get('violaciones', []),
        'tiempo_total_s': d['tiempo_total_s'],
        'tiempo_ultimo_reporte_que_puntua_s': d['tiempo_ultimo_reporte_que_puntua_s'],
        'puntaje': {k: p[k] for k in ('por_victima', 'total', 'penalizacion_colisiones',
                                      'penalizacion_salida', 'anulada')},
    }


def segundos(v):
    return '-' if v is None or v == float('inf') else f'{v:.1f}'


def estilo():
    return ('<style>body{font-family:sans-serif;max-width:60rem;margin:2rem auto;padding:0 1rem}'
            'table{border-collapse:collapse}td,th{border:1px solid #999;padding:.3rem .6rem}'
            'img{max-width:100%}</style>')


def pagina_corrida(c, repo):
    filas = []
    for r in c['puntaje']['por_victima']:
        dist = '-' if r.get('distancia') is None else f"{r['distancia']:.2f}"
        rep = '-' if r.get('indice_reporte') is None else r['indice_reporte'] + 1
        filas.append(f"<tr><td>V{r['indice'] + 1}</td><td>{r['puntos']}</td>"
                     f'<td>{dist}</td><td>{rep}</td></tr>')
    img = f'<p><img src="../runs/{c["clave"]}/grafico.png" alt="gráfico"></p>' if c['grafico'] else ''
    estado = 'CI en verde' if c['verde'] else 'CI en rojo: no entra a la tabla'
    titulo = f"{e(c['nombre_equipo'])} · {c['mapa']}"
    return f"""<!doctype html><html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{titulo}</title>{estilo()}</head>
<body><p><a href="../index.html">← tablas</a></p>
<h1>{titulo}</h1>
<p><b>{e(resumen(c))}</b> · fin: <code>{e(str(c['termino_por']))}</code> ·
{segundos(c['tiempo_total_s'])} s · {estado}</p>
{img}
<table><tr><th>Víctima</th><th>Puntos</th><th>Distancia (m)</th><th>Reporte #</th></tr>
{''.join(filas)}</table>
<p><a href="https://github.com/{repo}/pull/{c['pr']}">PR #{c['pr']}</a> ·
<a href="https://github.com/{repo}/actions/runs/{c['run_id']}">corrida del CI y artefactos</a>
(los artefactos duran 14 días) · commit <code>{c['sha'][:7]}</code></p>
</body></html>
"""


def mejor_corrida(cs):
    # El mismo criterio que ranking.armar(): más puntos, menos tiempo, menos penalizaciones.
    return min(cs, key=lambda c: (-c['puntaje']['total'], _tiempo(c, 'total'), _penal(c)))


def pagina_indice(corridas):
    nombres = {c['equipo']: c['nombre_equipo'] for c in corridas}  # el último gana
    secciones = []
    for mapa in sorted({c['mapa'] for c in corridas}):
        verdes = [c for c in corridas if c['mapa'] == mapa and c['verde']]
        if not verdes:
            continue
        trs = []
        for f in armar(verdes):
            mejor = mejor_corrida([c for c in verdes if c['equipo'] == f['equipo']])
            trs.append(f"<tr><td>{f['posicion']}{'*' if f['empate'] else ''}</td>"
                       f"<td>{e(nombres[f['equipo']])}</td><td>{f['puntos']}</td>"
                       f"<td>{segundos(f['tiempo_s'])}</td><td>{f['penalizaciones']}</td>"
                       f"<td>{f['corridas']}</td>"
                       f"<td><a href=\"corridas/{mejor['clave']}.html\">ver</a></td></tr>")
        secciones.append(
            f'<h2><code>{mapa}</code></h2>\n<table><tr><th>Pos</th><th>Equipo</th><th>Puntos</th>'
            f'<th>Tiempo (s)</th><th>Penal.</th><th>Corr.</th><th>Detalle</th></tr>'
            f"{''.join(trs)}</table>")
    lista = ''.join(
        f"<li><a href=\"corridas/{c['clave']}.html\">{e(c['nombre_equipo'])} · {c['mapa']} · "
        f"{c['puntaje']['total']} pts</a>{'' if c['verde'] else ' (CI en rojo)'}</li>"
        for c in reversed(corridas))
    tablas = '\n'.join(secciones) or '<p>Todavía no hay corridas en verde.</p>'
    return f"""<!doctype html><html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tabla de práctica · Challenge JAR 2026</title>{estilo()}</head>
<body><h1>Tabla de práctica</h1>
<p>Corridas del CI en los mundos de práctica: una tabla por mapa, con la mejor corrida en
verde de cada equipo (desempate por tiempo total, §8.2). Es orientativa: la tabla oficial
del evento se publica aparte. * = empate.</p>
{tablas}
<h2>Todas las corridas</h2><ul>{lista}</ul>
</body></html>
"""


def texto_comentario(args, estas, avisos):
    lineas = [MARCA, f'### Corrida de práctica · commit `{args.sha[:7]}`', '']
    for c in estas:
        lineas.append(f"**`{c['mapa']}`** · {resumen(c)} · "
                      f"[detalle]({args.base_url}/corridas/{c['clave']}.html)")
        if c['grafico']:
            lineas += ['', f"![{c['mapa']}]({args.base_url}/runs/{c['clave']}/grafico.png)"]
        lineas.append('')
    lineas += [f'- {a}' for a in avisos]
    if estas:
        if args.verde == 'si':
            lineas.append(f'Entra a la [tabla de práctica]({args.base_url}/): cuenta la mejor '
                          'corrida en verde de cada equipo, por mapa.')
        else:
            lineas.append(f'El CI no está en verde: la corrida no entra a la [tabla de práctica]'
                          f'({args.base_url}/) hasta que lo esté.')
        lineas.append('El gráfico puede tardar un minuto en aparecer.')
    elif not avisos:
        lineas.append('La corrida no dejó resultados del simulador.')
    lineas += ['', f'[Corrida del CI y artefactos](https://github.com/{args.repo}/actions/runs/'
                   f'{args.run_id})']
    return '\n'.join(lineas) + '\n'


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--artefactos', required=True)
    ap.add_argument('--sitio', required=True)
    ap.add_argument('--comentario', required=True, help='md del comentario para el PR')
    ap.add_argument('--run-id', required=True, type=patron(r'[0-9]{1,20}', 'run id'))
    ap.add_argument('--pr', required=True, type=patron(r'[0-9]{1,9}', 'número de PR'))
    ap.add_argument('--sha', required=True, type=patron(r'[0-9a-f]{40}', 'sha'))
    ap.add_argument('--repo', required=True, type=patron(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', 'repo'))
    ap.add_argument('--base-url', required=True, help='URL pública del sitio, sin / final')
    ap.add_argument('--equipo-id', required=True,
                    type=patron(r'[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})', 'usuario de GitHub'))
    ap.add_argument('--equipo-yaml', required=True, help='equipo.yaml del commit del PR')
    ap.add_argument('--verde', required=True, choices=['si', 'no'])
    ap.add_argument('--toca-infra', required=True, choices=['si', 'no'])
    args = ap.parse_args()

    if args.toca_infra == 'si':
        Path(args.comentario).write_text(
            f'{MARCA}\n### Corrida de práctica · commit `{args.sha[:7]}`\n\n'
            'Este PR modifica `tools/` o `.github/`, así que estos resultados no los generó el CI '
            'de la organización: no se publican en la tabla de práctica. La validación oficial '
            'vuelve a correr los controles con la copia de la organización.\n', encoding='utf-8')
        print('el PR toca tools/ o .github/: no se publica')
        return 0

    sitio = Path(args.sitio)
    for sub in ('data', 'corridas', 'runs'):
        (sitio / sub).mkdir(parents=True, exist_ok=True)
    (sitio / '.nojekyll').touch()
    corridas = cargar_historial(sitio)
    por_clave = {c['clave']: c for c in corridas}
    nombre = nombre_de_equipo(args.equipo_yaml) or args.equipo_id
    estas, avisos, nuevas = [], [], 0
    for res in sorted(Path(args.artefactos).rglob('resultado.json')):
        mapa = res.parent.name
        if not MAPA_VALIDO.fullmatch(mapa):
            print(f'aviso: se ignora {res}: nombre de mapa inválido')
            avisos.append('Se ignoró un resultado con un nombre de mapa inválido.')
            continue
        errores = validar(res)
        if errores:
            print(f'aviso: se ignora {res}: {errores[0]}')
            avisos.append(f'`{mapa}`: el resultado no cumple el contrato y no se publica '
                          '(el detalle está en el log del publicador).')
            continue
        d = json.loads(res.read_text(encoding='utf-8'))
        if not d['mapa_publicado']:
            avisos.append(f'`{mapa}`: el juez no llegó a publicar el mapa; no se publica.')
            continue
        clave = f'{args.run_id}-{mapa}'
        if clave in por_clave:  # re-publicación: no se duplica, pero sí se comenta
            estas.append(por_clave[clave])
            continue
        grafico = res.parent / 'grafico.png'
        destino = sitio / 'runs' / clave
        destino.mkdir(exist_ok=True)
        hay_grafico = es_png(grafico)
        if hay_grafico:
            shutil.copy(grafico, destino / 'grafico.png')
        elif grafico.exists():
            print(f'aviso: se ignora {grafico}: no es un PNG de menos de 1 MB')
        c = entrada(d, args, mapa, nombre, hay_grafico)
        (destino / 'resultado.json').write_text(json.dumps(c), encoding='utf-8')
        corridas.append(c)
        por_clave[clave] = c
        estas.append(c)
        nuevas += 1

    (sitio / 'data' / 'historial.jsonl').write_text(
        ''.join(json.dumps(c) + '\n' for c in corridas), encoding='utf-8')
    for c in corridas:
        (sitio / 'corridas' / f"{c['clave']}.html").write_text(
            pagina_corrida(c, args.repo), encoding='utf-8')
    (sitio / 'index.html').write_text(pagina_indice(corridas), encoding='utf-8')
    Path(args.comentario).write_text(texto_comentario(args, estas, avisos), encoding='utf-8')
    print(f'sitio: {len(corridas)} corridas, {nuevas} nuevas')
    return 0


if __name__ == '__main__':
    sys.exit(main())
