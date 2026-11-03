"""Cambio de zona sin frenar y velocidad máxima salvo yendo hacia una pared."""
import math
import time
from unittest.mock import Mock
from builtin_interfaces.msg import Time

from equipo_jar.exploracion import Exploracion


class Handle:
    def __init__(self, status=2):     # 2 = EXECUTING
        self.status = status
        self.cancelada = False

    def cancel_goal_async(self):
        self.cancelada = True

    def get_result_async(self):
        f = Mock()
        f.add_done_callback = lambda cb: None
        return f


def nodo():
    n = Exploracion.__new__(Exploracion)
    n.cfg = {'cambio_sin_frenar': True, 'porcentaje_loc_mala': 100.0, 'error_loc_malo': 0.08,
             'distancia_lenta': 0.25, 'distancia_muy_lenta': 0.12,
             'porcentaje_lenta': 70.0, 'porcentaje_muy_lenta': 50.0}
    n.goal_id = 7
    n.goal_handle = Handle()
    n.solicitud_goal = None
    n.cancelacion_en_curso = None
    n.goal_heredado = None
    n.goal_actual = dict(x=1.0, y=0.0, yaw=0.0, pasos=[(1.0, 0.0, 0.0)], etiqueta='portal sector 3')
    n.get_logger = lambda: Mock()
    return n


def test_al_salir_de_zona_no_cancela_el_goal():
    n = nodo()
    h = n.goal_handle
    assert n.soltar_sin_frenar()
    assert not h.cancelada                     # Nav2 sigue moviendo el robot
    assert n.goal_handle is None and n.goal_id == 8   # su resultado ya no cuenta
    assert n.heredado_vivo()


def test_goal_navigate_to_pose_si_se_cancela():
    n = nodo()
    n.goal_actual['pasos'] = None              # otro servidor de acciones: no hay preempción
    assert not n.soltar_sin_frenar()


def test_cancelar_actual_frena_tambien_el_heredado():
    n = nodo()
    h = n.goal_handle
    n.soltar_sin_frenar()
    n.cancelar_actual()
    assert h.cancelada and n.goal_heredado is None


def test_heredado_terminado_no_cuenta():
    n = nodo()
    n.soltar_sin_frenar()
    n.goal_heredado['handle'].status = 4       # SUCCEEDED
    assert not n.heredado_vivo()


def test_velocidad_maxima_con_localizacion_regular_y_frena_hacia_pared():
    n = nodo()
    publicado = []
    n.pub_vel = Mock(publish=lambda m: publicado.append(m.speed_limit))
    n.ultimo_limite = None
    n.get_clock = lambda: Mock(now=lambda: Mock(to_msg=lambda: Time()))
    n.error_loc = 0.09                         # "empeorando", frecuente en los logs
    n.libre_hacia_donde_va = lambda: None
    n.publicar_limite_velocidad()
    assert publicado[-1] == 100.0
    n.libre_hacia_donde_va = lambda: 0.10      # yendo hacia una pared, muy cerca
    n.publicar_limite_velocidad()
    assert publicado[-1] == 50.0
    n.libre_hacia_donde_va = lambda: 1.0       # pared al costado / lejos
    n.publicar_limite_velocidad()
    assert publicado[-1] == 100.0
