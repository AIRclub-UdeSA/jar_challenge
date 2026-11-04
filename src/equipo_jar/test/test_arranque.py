from equipo_jar.nodo_arranque import transicion_para
from lifecycle_msgs.msg import State, Transition


def test_sin_configurar_se_configura():
    assert transicion_para(State.PRIMARY_STATE_UNCONFIGURED) == Transition.TRANSITION_CONFIGURE


def test_inactivo_se_activa():
    assert transicion_para(State.PRIMARY_STATE_INACTIVE) == Transition.TRANSITION_ACTIVATE


def test_en_transicion_solo_se_espera():
    assert transicion_para(State.TRANSITION_STATE_CONFIGURING) is None
    assert transicion_para(State.TRANSITION_STATE_ACTIVATING) is None


def test_activo_no_necesita_nada():
    assert transicion_para(State.PRIMARY_STATE_ACTIVE) is None
