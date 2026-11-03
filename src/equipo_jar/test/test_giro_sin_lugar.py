"""Sin lugar para girar en el lugar, el explorador no hace el giro previo: arranca Nav2."""
import math
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from equipo_jar.exploracion import ELIGIENDO, NAVEGANDO, Exploracion


def nodo(holgura):
    grilla = SimpleNamespace(a_celda=lambda x, y: (0, 0), dentro=lambda f, c: True,
                             transitable=np.ones((1, 1), dtype=bool))
    return SimpleNamespace(pose_robot=lambda: (0., 0., 0.), grilla=grilla, cfg={'giro_previo_deg': 135.0},
        ruta_a_destino=lambda *a: (None, 0.), get_logger=lambda: Mock(),
        direccion_de_ruta=lambda *a: math.pi, spin_cli=Mock(server_is_ready=lambda: True),
        holgura_giro=lambda: holgura, girar=Mock(), enviar_goal=Mock(return_value=True),
        estado=ELIGIENDO, lista_negra=[])


def test_pegado_a_una_caja_no_gira_y_manda_el_goal():
    n = nodo(-0.02)
    assert Exploracion.encarar_y_enviar(n, 1.0, 0.0, 0.0, 5.0, 1.0, 'vista', NAVEGANDO)
    n.girar.assert_not_called()
    n.enviar_goal.assert_called_once()


def test_con_lugar_sigue_girando_primero():
    n = nodo(0.20)
    assert Exploracion.encarar_y_enviar(n, 1.0, 0.0, 0.0, 5.0, 1.0, 'vista', NAVEGANDO)
    n.girar.assert_called_once()
    n.enviar_goal.assert_not_called()
