"""Run shared-map geometry checks through colcon."""
import os
from pathlib import Path
import subprocess
import sys


def test_map_coordinate_suite():
    root = Path(__file__).resolve().parents[3]
    suites = [root / 'pruebas' / name for name in (
        'test_mapa_coordenadas.py', 'test_mapa_alcanzable.py', 'test_rayos_mapa.py')]
    assert all(suite.is_file() for suite in suites), 'Run map tests from the full source checkout'
    environment = dict(os.environ)
    environment['PYTHONPATH'] = (
        str(root / 'src' / 'equipo_jar') + os.pathsep
        + environment.get('PYTHONPATH', ''))
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', *map(str, suites), '-q', '--tb=short'],
        cwd=root, env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
