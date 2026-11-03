"""Run terminal-stop and action-acceptance regressions through colcon."""
import os
from pathlib import Path
import subprocess
import sys


def test_terminal_stop_suite():
    root = Path(__file__).resolve().parents[3]
    suites = [root / 'pruebas' / name for name in (
        'test_parada_mision.py', 'test_parada_navegacion.py',
        'test_cancelar_inspeccion_reportada.py')]
    environment = dict(os.environ)
    environment['PYTHONPATH'] = str(root / 'src/equipo_jar') + os.pathsep + environment.get('PYTHONPATH', '')
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', '-p', 'no:cacheprovider', *map(str, suites), '-q', '--tb=short'],
        cwd=root, env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
