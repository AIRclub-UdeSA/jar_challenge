"""Publicador del leaderboard con artefactos sintéticos (sin ROS ni red)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from publicar_ranking import main as publicar  # noqa: E402


def resultado(equipo, total, tiempo=100.0, mapa_ok=True):
    return {
        'equipo': equipo, 'termino_por': 'done', 'mapa_publicado': mapa_ok,
        'reportes': [], 'reportes_mal_formados': [], 'violaciones': [],
        'resumen': f'total {total}', 'victimas': [], 'eventos': [],
        'tiempo_total_s': tiempo, 'tiempo_ultimo_reporte_que_puntua_s': 10.0,
        'puntaje': {'por_victima': [], 'reportes_contados': 0, 'reportes_ignorados': 0,
                    'puntos_victimas': total, 'bono_rapido': False, 'bono_regreso': False,
                    'penalizacion_colisiones': 0, 'penalizacion_salida': 0,
                    'total': total, 'anulada': False,
                    'tiempo_ultimo_reporte_que_puntua': 10.0},
    }


def artefacto(base, run, mapa, contenido, png=True):
    d = base / f'sim-{mapa}-{run}' / mapa
    d.mkdir(parents=True)
    (d / 'resultado.json').write_text(json.dumps(contenido))
    if png:
        (d / 'grafico.png').write_bytes(b'\x89PNG fake')
    return d


def correr(args):
    sys.argv = ['publicar_ranking.py'] + args
    assert publicar() == 0


def test_publica_validos_ignora_resto_y_escapa(tmp_path):
    art = tmp_path / 'art'
    artefacto(art, 'r1', 'maze_1_6x5', resultado('<b>osos</b>', 400))
    artefacto(art, 'r1', 'maze_3_6x6', {'roto': True})                      # inválido
    artefacto(art, 'r1', 'laberinto', resultado('x', 0, mapa_ok=False))     # sin mapa
    sitio = tmp_path / 'sitio'
    correr(['--artefactos', str(art), '--sitio', str(sitio), '--run-id', 'r1',
            '--pr', '7', '--repo', 'O/R', '--base-url', 'https://p',
            '--comentario', str(tmp_path / 'c.md')])
    hist = (sitio / 'data' / 'historial.jsonl').read_text().splitlines()
    assert len(hist) == 1
    index = (sitio / 'index.html').read_text()
    assert '<b>osos</b>' not in index and '&lt;b&gt;osos&lt;/b&gt;' in index
    assert (sitio / 'corridas' / 'r1-maze_1_6x5.html').exists()
    assert (sitio / 'runs' / 'r1-maze_1_6x5' / 'grafico.png').exists()
    com = (tmp_path / 'c.md').read_text()
    assert 'https://p/runs/r1-maze_1_6x5/grafico.png' in com


def test_idempotente_y_ordena(tmp_path):
    art1 = tmp_path / 'art1'
    artefacto(art1, 'r1', 'm', resultado('B', 400, tiempo=150))
    sitio = tmp_path / 'sitio'
    base = ['--artefactos', str(art1), '--sitio', str(sitio), '--run-id', 'r1',
            '--pr', '7', '--repo', 'O/R', '--base-url', 'https://p',
            '--comentario', str(tmp_path / 'c.md')]
    correr(base)
    correr(base)  # re-publicar no duplica
    assert len((sitio / 'data' / 'historial.jsonl').read_text().splitlines()) == 1
    art2 = tmp_path / 'art2'  # cada run trae solo sus artefactos (como download-artifact)
    artefacto(art2, 'r2', 'm', resultado('A', 400, tiempo=100))
    correr(['--artefactos', str(art2), '--sitio', str(sitio), '--run-id', 'r2',
            '--pr', '8', '--repo', 'O/R', '--base-url', 'https://p',
            '--comentario', str(tmp_path / 'c.md')])
    hist = (sitio / 'data' / 'historial.jsonl').read_text().splitlines()
    assert len(hist) == 2
    index = (sitio / 'index.html').read_text()
    assert index.index('>A</td>') < index.index('>B</td>')  # menos tiempo primero
