import numpy as np
from equipo_jar.coordinador_mision import CoordinadorMision
from test_mision import nueva


def mission():
    _,nav=nueva();nav.largada=(.7,.7,0.)
    m=CoordinadorMision(nav,0.);m.ultima_pose=(.7,.7,0.);m.cantidad=3
    return m,nav


def test_historial_agotado_con_tiempo_reabre_busqueda_sin_borrar_obstaculos():
    m,nav=mission();original=nav.grilla.ocupada.copy()
    m.cobertura.observada[:]=True;m.descartadas.add((1,1,1));m.retrocesos_usados.add((1,1,1))
    m.retroceder(150.,'Sin avance')
    assert m.estado=='BUSCANDO_VISTA' and not nav.solicitud_regreso
    assert m.reinicios_busqueda==1 and not m.descartadas and not m.retrocesos_usados
    assert not m.cobertura.observada.any()
    np.testing.assert_array_equal(original,nav.grilla.ocupada)


def test_reserva_de_regreso_sigue_obligando_a_terminar():
    m,nav=mission();m.retroceder(590.,'Sin avance')
    assert m.estado=='REGRESANDO' and nav.solicitud_regreso
