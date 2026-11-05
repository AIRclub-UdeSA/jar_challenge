"""comun.declarar / parametros_en_vivo: a YAML 3 for a 3.0 default must not kill the node."""
import pytest

rclpy = pytest.importorskip('rclpy')
from rclpy.parameter import Parameter  # noqa: E402

from equipo_jar.comun import convertir_como, declarar, parametros_en_vivo  # noqa: E402


@pytest.fixture
def make_node():
    rclpy.init()
    nodes = []

    def make(overrides):
        node = rclpy.create_node(
            f'prueba_parametros_{len(nodes)}',
            parameter_overrides=[Parameter(n, value=v) for n, v in overrides.items()])
        nodes.append(node)
        return node

    yield make
    for node in nodes:
        node.destroy_node()
    rclpy.shutdown()


def test_an_int_for_a_float_default_is_read_as_float(make_node):
    node = make_node({'rango': 3, 'cuantos': 12.0})
    p = parametros_en_vivo(node, {'rango': 2.5, 'cuantos': 5, 'nombre': 'a', 'activo': True})
    assert p['rango'] == 3.0 and isinstance(p['rango'], float)
    assert p['cuantos'] == 12 and isinstance(p['cuantos'], int)
    assert p['nombre'] == 'a' and p['activo'] is True


def test_a_value_that_does_not_fit_keeps_the_default(make_node):
    node = make_node({'rango': 'lejos', 'cuantos': 2.5})
    p = parametros_en_vivo(node, {'rango': 2.5, 'cuantos': 5})
    assert p == {'rango': 2.5, 'cuantos': 5}


def test_live_changes_are_converted_or_refused(make_node):
    node = make_node({})
    p = parametros_en_vivo(node, {'rango': 2.5, 'cuantos': 5})
    assert node.set_parameters([Parameter('rango', value=4)])[0].successful
    assert p['rango'] == 4.0 and isinstance(p['rango'], float)
    assert not node.set_parameters([Parameter('rango', value='x')])[0].successful
    assert not node.set_parameters([Parameter('cuantos', value=2.5)])[0].successful
    assert p == {'rango': 4.0, 'cuantos': 5}


def test_declarar_alone(make_node):
    node = make_node({'plazo': 5})
    assert declarar(node, 'plazo', 2.0) == 5.0


def test_convertir_como():
    assert convertir_como(3, 1.0) == 3.0
    assert convertir_como(4.0, 1) == 4
    assert convertir_como(4.5, 1) == 4.5      # left as is: the caller's check refuses it
    assert convertir_como(True, 1.0) is True
    assert convertir_como('x', 'y') == 'x'
