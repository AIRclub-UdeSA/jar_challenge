"""The robot clamps each command axis: 0.5 m/s in x and y, 1.0 rad/s in yaw (organizers, email
2). A command past a clamp is cut on that axis alone, so the robot drives another way than
the one planned, and the stop and turn predictions assume rates it never reaches. Nothing we
launch may ask for more, in any profile."""
import importlib.util
import math
import os

import numpy as np
import pytest
import yaml

from equipo_jar.motion import DEFAULTS, Tracker

pytest.importorskip('launch_ros')
HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
VX, VY, WZ = 0.5, 0.5, 1.0
PERFILES = ['rapido', 'normal', 'cuidadoso']


def launched(perfil):
    spec = importlib.util.spec_from_file_location(
        'competencia', os.path.join(PKG, 'launch', 'competencia.launch.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    with open(m.con_perfil(os.path.join(PKG, 'params', 'params.yaml'), perfil)) as f:
        return yaml.safe_load(f)


def paths():
    line = [(0.1 * k, 0.0, 0.0) for k in range(31)]
    strafe = [(0.1 * k, 0.0, math.pi / 2) for k in range(31)]
    diagonal = [(0.1 * k, 0.1 * k, 0.0) for k in range(21)]
    uturn = [(1.0 + 0.6 * math.cos(a), 0.6 * math.sin(a), a + math.pi / 2)
             for a in np.linspace(-math.pi / 2, math.pi / 2, 30)]
    return {'line': line, 'strafe': strafe, 'diagonal': diagonal, 'uturn': uturn,
            'turn_in_place': [(0.0, 0.0, math.pi)]}


def drive(path, params, dt=0.05, steps=600):
    """Commands of a tracker on an ideal base that follows them; the largest of each axis."""
    t = Tracker(path, **params)
    x, y, th = 0.0, 0.0, -math.pi / 2
    worst = np.zeros(3)
    for _ in range(steps):
        cmd, state = t.step((x, y, th), np.zeros((0, 2)), dt)
        worst = np.maximum(worst, np.abs(cmd))
        if state == 'llego':
            break
        c, s = math.cos(th), math.sin(th)
        x += (c * cmd[0] - s * cmd[1]) * dt
        y += (s * cmd[0] + c * cmd[1]) * dt
        th += cmd[2] * dt
    return worst


@pytest.mark.parametrize('perfil', PERFILES)
def test_motion_never_commands_past_the_robot_clamps(perfil):
    m = launched(perfil)['motion']['ros__parameters']
    tp = {k: m[k] for k in DEFAULTS if k in m}
    slow = dict(tp, v_max=min(tp['v_max'], m['v_lento']), w_max=min(tp['w_max'], m['w_lento']))
    for name, path in paths().items():
        for params in (tp, slow):
            vx, vy, w = drive(path, params)
            assert vx <= VX + 1e-9 and vy <= VY + 1e-9, (perfil, name, vx, vy)
            assert w <= WZ + 1e-9, (perfil, name, w)
