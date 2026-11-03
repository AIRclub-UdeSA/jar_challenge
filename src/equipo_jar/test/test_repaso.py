import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize('name', ['test_presupuesto_mision.py', 'test_historial_vistas.py',
                                  'test_historial_mision.py', 'test_rutas_presupuesto.py',
                                  'test_cancelar_inspeccion_reportada.py', 'test_mision_repaso.py',
                                  'test_repaso_integracion.py', 'test_diagnostico_repaso.py', 'test_refresco_repaso.py'])
def test_extra_search_foundation(name):
    root = Path(__file__).resolve().parents[3]
    path = root / 'pruebas' / name
    assert path.is_file(), 'Run extra search tests from the full source checkout'
    environment = dict(os.environ)
    environment['PYTHONPATH'] = str(root / 'src/equipo_jar') + os.pathsep + environment.get('PYTHONPATH', '')
    result = subprocess.run([sys.executable, '-m', 'pytest', '-p', 'no:cacheprovider',
                             str(path), '-q', '--tb=short'], cwd=root, env=environment,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
