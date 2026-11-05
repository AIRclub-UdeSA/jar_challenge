"""Run log: where it writes, what it keeps, and that it never breaks the run."""
from pathlib import Path

import pytest
import rclpy
from rcl_interfaces.msg import Log
from rclpy.parameter import Parameter

from equipo_jar.registro import Registro, carpeta_de_logs


def test_folder_is_env_var_or_workspace_root(tmp_path):
    assert carpeta_de_logs({'JAR_LOG_DIR': str(tmp_path)}) == tmp_path
    ws = tmp_path / 'ws'
    for prefix in (ws / 'install' / 'equipo_jar', ws / 'install'):
        prefix.mkdir(parents=True, exist_ok=True)
        assert carpeta_de_logs({}, prefix) == ws.resolve() / 'logs_jar'
    assert carpeta_de_logs({}, tmp_path / 'elsewhere') is None


@pytest.fixture
def node(tmp_path, monkeypatch):
    monkeypatch.setenv('JAR_LOG_DIR', str(tmp_path / 'logs_jar'))
    rclpy.init()
    created = []

    def make():
        created.append(Registro())
        return created[-1]
    try:
        yield make, tmp_path / 'logs_jar' / 'log.txt'
    finally:
        for item in created:
            item.destroy_node()
        rclpy.shutdown()


def log(name, text, level=20):
    msg = Log()
    msg.name, msg.msg, msg.level = name, text, level
    return msg


def test_writes_logs_and_counts_repeated_lines(node):
    make, path = node
    reg = make()
    for _ in range(50):
        reg.al_log(log('controller_server', 'Control loop missed its desired rate'))
    reg.al_log(log('mision', 'Reporte 1'))
    reg.al_log(log('amcl', 'debug noise', level=10))
    reg.vaciar()
    text = path.read_text()
    assert 'run start' in text and 'params/navegacion.yaml' in text
    assert text.count('Control loop missed') == 1
    assert '[mision] Reporte 1' in text
    assert 'debug noise' not in text


def test_size_cap_stops_writing(node):
    make, path = node
    reg = make()
    reg.set_parameters([Parameter('max_bytes', value=reg.escritos + 2000)])
    for i in range(200):
        reg.al_log(log('mision', f'line {i}'))
    assert reg.archivo is None
    assert path.stat().st_size <= reg.p['max_bytes'] + 100
    assert path.read_text().endswith('log full: stopped writing =====\n')


def test_unwritable_folder_does_not_break_the_node(node, tmp_path, monkeypatch):
    make, _ = node
    blocker = tmp_path / 'file'
    blocker.write_text('not a folder')
    monkeypatch.setenv('JAR_LOG_DIR', str(blocker / 'logs_jar'))
    reg = make()
    assert reg.archivo is None
    reg.al_log(log('mision', 'still fine'))
    reg.vaciar()
