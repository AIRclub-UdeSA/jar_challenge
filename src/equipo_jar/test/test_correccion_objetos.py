"""Los objetos del mapa local se corrigen con lecturas nuevas del LiDAR."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from equipo_jar.exploracion import Exploracion, Grilla


def nodo():
    libre = np.ones((60, 60), dtype=bool)
    libre[[0, -1], :] = False
    libre[:, [0, -1]] = False
    msg = SimpleNamespace(info=SimpleNamespace(resolution=0.05, width=60, height=60,
                                               origin=SimpleNamespace(position=SimpleNamespace(x=0.0, y=0.0))),
                          header=SimpleNamespace(frame_id='map'), data=np.where(libre, 0, 100).ravel().tolist())
    n = Exploracion.__new__(Exploracion)
    n.cfg = {'lado_bloque_objeto_m': 0.26, 'celdas_max_cortadas_por_bloque': 20}
    n.grilla = Grilla(msg, 0.13, 0.17)
    n.bloques_objetos, n.lecturas_bloques, n.sectorizador = [], {}, None
    n.publicar_mapa_objetos = Mock()
    n.get_logger = lambda: Mock()
    n.pose_robot = lambda: (0.3, 0.3, 0.0)
    n.dist_pared = Exploracion.campo_distancia_paredes(n.grilla)
    return n


def test_objeto_mal_ubicado_se_corrige_con_lecturas_nuevas():
    n = nodo()
    g = n.grilla
    n.registrar_objeto_local((1.40, 1.50))                    # primera estimación, 10 cm corrida
    real = (1.50, 1.52)
    rng = np.random.default_rng(0)
    posiciones = [n.bloques_objetos[0]]
    for _ in range(60):
        lectura = (real[0] + rng.normal(0, 0.01), real[1] + rng.normal(0, 0.01))
        assert n.corregir_bloque_con_lectura(lectura, (0.6, 1.5))
        if n.bloques_objetos[0] != posiciones[-1]:
            posiciones.append(n.bloques_objetos[0])
    # nunca salta más de paso_max_correccion_m (0.05) de una vez
    assert all(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5 <= 0.05 + 1e-9
               for a, b in zip(posiciones, posiciones[1:]))
    x, y = n.bloques_objetos[0]
    assert abs(x - real[0]) < 0.02 and abs(y - real[1]) < 0.02
    f, c = g.a_celda(*real)
    assert g.bloques_conocidos[f, c]
    f, c = g.a_celda(1.26, 1.50)                               # el lado viejo quedó libre
    assert not g.bloques_conocidos[f, c]
    n.publicar_mapa_objetos.assert_called()


def test_lecturas_lejanas_o_de_otro_objeto_no_corrigen():
    n = nodo()
    n.registrar_objeto_local((1.40, 1.50))
    assert not n.corregir_bloque_con_lectura((2.30, 1.50), (0.6, 1.5))   # otro objeto
    for _ in range(10):
        n.corregir_bloque_con_lectura((1.50, 1.50), (-1.0, 1.5))         # a 2.5 m del robot
    assert n.bloques_objetos[0] == (1.40, 1.50)


def test_deriva_acumulada_tiene_tope():
    """Trabado contra la caja, lecturas sesgadas la corrían 25 cm en pasos de 5 cm."""
    n = nodo()
    n.registrar_objeto_local((1.50, 1.50))
    for k in range(200):
        # lecturas que se van corriendo hacia el robot (sesgo creciente)
        n.corregir_bloque_con_lectura((1.50 - min(0.30, 0.002 * k), 1.50), (1.0, 1.5))
    x, y = n.bloques_objetos[0]
    assert ((x - 1.50) ** 2 + (y - 1.50) ** 2) ** 0.5 <= 0.10 + 1e-9


@pytest.mark.parametrize('estado', ['NAVEGANDO', 'VOLVIENDO', 'DESENCAJANDO', 'COMPORTAMIENTO'])
def test_objeto_no_se_mueve_bajo_una_ruta_en_ejecucion(estado):
    n = nodo()
    n.estado = estado
    n.registrar_objeto_local((1.4, 1.5))
    memoria = n.grilla.bloques_conocidos.copy()
    n.publicar_mapa_objetos.reset_mock()
    for _ in range(80):
        assert n.corregir_bloque_con_lectura((1.5, 1.52), (0.6, 1.5))
    assert n.bloques_objetos == [(1.4, 1.5)]
    np.testing.assert_array_equal(n.grilla.bloques_conocidos, memoria)
    n.publicar_mapa_objetos.assert_not_called()
    n.estado = 'ELIGIENDO'
    for _ in range(80):
        n.corregir_bloque_con_lectura((1.5, 1.52), (0.6, 1.5))
    assert n.bloques_objetos[0] == pytest.approx((1.5, 1.52), abs=0.02)
