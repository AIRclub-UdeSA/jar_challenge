"""Un objeto (o un eco en la punta de una pared) nunca deja sectores sin camino."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from equipo_jar.exploracion import Exploracion, Grilla


def mapa_pasillo():
    """Pasillo de 0.60 m de ancho (y entre 0 y 0.6), 3 m de largo, con paredes."""
    res, ancho, alto = 0.05, 60, 16
    datos = np.full((alto, ancho), 100, dtype=np.int8)
    datos[2:14, 1:59] = 0
    msg = SimpleNamespace(info=SimpleNamespace(resolution=res, width=ancho, height=alto,
                                               origin=SimpleNamespace(position=SimpleNamespace(x=0.0, y=-0.1))),
                          header=SimpleNamespace(frame_id='map'), data=datos.ravel().tolist())
    return msg


def nodo():
    n = Exploracion.__new__(Exploracion)
    n.cfg = {'lado_bloque_objeto_m': 0.30, 'celdas_max_cortadas_por_bloque': 20,
             'distancia_pared_bloque_m': 0.20}
    n.grilla = Grilla(mapa_pasillo(), 0.13, 0.17)
    n.bloques_objetos = []
    n.sectorizador = None
    n.tf_buffer = None
    n.pose_robot = lambda: (0.3, 0.2, 0.0)
    n.publicar_mapa_objetos = lambda: None
    n.get_logger = lambda: Mock()
    n.dist_pared = Exploracion.campo_distancia_paredes(n.grilla)
    return n


def test_objeto_que_cerraria_el_pasillo_no_se_bloquea():
    n = nodo()
    n.registrar_objeto_local((1.5, 0.3))           # en medio del pasillo de 60 cm: lo cierra
    assert n.bloques_objetos == []
    assert not n.grilla.bloques_conocidos.any()
    f, c = n.grilla.a_celda(2.7, 0.2)
    assert n.alcanzable_desde_robot(n.grilla.transitable)[f, c]


def test_objeto_en_zona_amplia_si_se_bloquea():
    n = nodo()
    n.grilla.bloques_conocidos[:] = False
    n.registrar_objeto_local((2.85, 0.2))          # al fondo del pasillo: no deja nada aislado
    assert n.bloques_objetos == [(2.85, 0.2)]


def test_eco_pegado_a_pared_no_se_bloquea_aunque_tenga_color():
    n = nodo()
    n.objetos_fuera_mapa = Mock()
    n.registrar_objeto_local = Mock()
    n.recordar_inspeccion(dict(punto=(1.0, 0.45), puntos=[], id_visual=None), False)  # 5 cm de la pared
    n.registrar_objeto_local.assert_not_called()
    n.recordar_inspeccion(dict(punto=(1.0, 0.45), puntos=[], id_visual=None,
                               color_confirmado=True), True)                         # azul confirmado
    n.registrar_objeto_local.assert_not_called()
