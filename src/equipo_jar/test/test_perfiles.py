"""What competencia.launch.py hands the nodes for each profile. A threshold set in params.yaml
that the profile overwrites never reaches the robot: reporte_p_min was lowered there and the
runs kept reporting at the profile's 0.95."""
import importlib.util
import os

import pytest
import yaml

pytest.importorskip('launch_ros')
HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)


def launched(perfil):
    spec = importlib.util.spec_from_file_location(
        'competencia', os.path.join(PKG, 'launch', 'competencia.launch.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    with open(m.con_perfil(os.path.join(PKG, 'params', 'params.yaml'), perfil)) as f:
        return yaml.safe_load(f)


@pytest.mark.parametrize('perfil', ['rapido', 'normal', 'cuidadoso'])
def test_each_profile_sets_both_thresholds(perfil):
    q = yaml.safe_load(open(os.path.join(PKG, 'params', 'perfiles.yaml')))[perfil]['ros__parameters']
    p = launched(perfil)
    mission = p['mission']['ros__parameters']['reporte_p_min']
    model = p['world_model']['ros__parameters']['reporte_p_min']
    assert mission == q['reporte_p_min'] and model == q['modelo_p_min']
    assert mission <= model  # the world model's confirmed is never looser than a report


@pytest.mark.parametrize('perfil', ['rapido', 'normal', 'cuidadoso'])
def test_the_mission_gets_the_home_of_competencia(perfil):
    p = launched(perfil)
    casa = p['competencia']['ros__parameters']['casa']
    assert casa in ('origen', 'inicio')
    assert p['mission']['ros__parameters']['casa'] == casa
