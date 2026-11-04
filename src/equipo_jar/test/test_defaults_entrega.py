"""El launch sin argumentos extra arranca la misma misión que el arnés."""
from pathlib import Path
import importlib.util
import yaml
from launch.actions import DeclareLaunchArgument


def test_defaults_de_entrega_y_perfil_real(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('entrega_launch',root/'launch/competencia.launch.py')
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'get_package_share_directory', lambda name: str(root))
    args = {a.name: ''.join(v.text for v in a.default_value)
            for a in module.generate_launch_description().entities if isinstance(a,DeclareLaunchArgument)}
    assert args['modo_navegacion'] == 'mision'
    assert args['asumir_base_cuadrada'] == 'true'
    assert args['perfil_navegacion'] == 'real'
    assert args['use_sim_time'] == 'false'
    params = yaml.safe_load((root/'params/navegacion.yaml').read_text())
    assert params['nodo_equipo']['ros__parameters']['modo_navegacion'] == 'mision'
