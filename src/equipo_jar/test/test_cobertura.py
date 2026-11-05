"""Run coverage checks from colcon while generated geometry stays outside src."""
import os
from pathlib import Path
import subprocess
import sys


def test_coverage_suite():
    root = Path(__file__).resolve().parents[3]
    files = [root / 'pruebas' / name for name in (
        'test_cobertura_plan.py', 'test_cobertura_node.py', 'test_metricas_cobertura.py',
        'test_orden_cobertura.py', 'test_espacio_observacion.py',
        'test_calculo_geometrico.py', 'test_ejecucion_cobertura.py',
        'test_repaso_cobertura.py')]
    assert all(path.is_file() for path in files), 'Run coverage tests from the full source checkout'
    environment = dict(os.environ)
    environment['PYTHONPATH'] = str(root / 'src/equipo_jar') + os.pathsep + environment.get('PYTHONPATH', '')
    result = subprocess.run([sys.executable, '-m', 'pytest', *map(str, files), '-q', '--tb=short'],
                            cwd=root, env=environment, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
