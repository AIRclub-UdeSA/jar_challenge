"""Pruebas del planificador sin ejecutar ROS ni mover el robot."""

import numpy as np

from jar_exploracion.planificador import a_estrella


def test_camino_y_extremos():
    """Conecta los extremos por celdas libres."""
    libres = np.ones((5, 5), dtype=bool)
    libres[2, :4] = False
    camino = a_estrella(libres, (0, 0), (4, 0))
    assert camino[0] == (0, 0) and camino[-1] == (4, 0)
    assert all(libres[celda] for celda in camino)
    for a, b in zip(camino, camino[1:]):
        assert max(abs(a[0] - b[0]), abs(a[1] - b[1])) == 1


def test_no_atraviesa_esquinas():
    """Rechaza una diagonal bloqueada por sus celdas laterales."""
    assert a_estrella(np.eye(2, dtype=bool), (0, 0), (1, 1)) is None


def test_extremos_invalidos():
    """Rechaza puntos fuera del mapa y puntos ocupados."""
    libres = np.ones((3, 3), dtype=bool)
    libres[1, 1] = False
    assert a_estrella(libres, (-1, 0), (2, 2)) is None
    assert a_estrella(libres, (0, 0), (1, 1)) is None
    assert a_estrella(libres, (0, 0), (0, 0)) == [(0, 0)]


def test_evitar_costos_altos():
    """Prefiere un desvío libre frente a una zona de costo alto."""
    libres = np.ones((3, 5), dtype=bool)
    costos = np.zeros((3, 5))
    costos[1, 1:4] = 100
    camino = a_estrella(libres, (1, 0), (1, 4), costos)
    assert all(costos[celda] == 0 for celda in camino)


def test_segmento_no_corta_esquinas_ni_sale_del_mapa():
    """Rechaza atajos inseguros incluso con extremos transitables."""
    from jar_exploracion.planificador import segmento_transitable

    libres = np.array([[True, False], [True, True]])
    assert not segmento_transitable(libres, (0, 0), (1, 1))
    assert segmento_transitable(libres, (0, 0), (1, 0))
    assert not segmento_transitable(libres, (-1, 0), (1, 0))
    assert not segmento_transitable(libres, None, (1, 0))


def test_busqueda_por_etapas_equivale_y_cede_control():
    """En un mapa amplio devuelve el mismo camino sin monopolizar una llamada."""
    from jar_exploracion.planificador import a_estrella_por_etapas

    libres = np.ones((50, 50), dtype=bool)
    libres[25, :45] = False
    busqueda = a_estrella_por_etapas(libres, (0, 0), (49, 0), lote=5)
    etapas = 0
    while True:
        try:
            assert next(busqueda) is None
            etapas += 1
        except StopIteration as fin:
            camino = fin.value
            break
    assert etapas > 1
    assert camino == a_estrella(libres, (0, 0), (49, 0))


def test_planificador_compilado_equivale_a_estrella():
    """Conserva optimalidad, costos de destino y prohibición de cortar esquinas."""
    from jar_exploracion.planificador import PlanificadorCompilado, segmento_transitable

    rng = np.random.default_rng(42)
    for _ in range(20):
        libres = rng.random((12, 15)) > 0.2
        libres[0, 0] = libres[-1, -1] = True
        costos = rng.random(libres.shape) * 10
        esperado = a_estrella(libres, (0, 0), (11, 14), costos)
        obtenido = PlanificadorCompilado(libres, costos).buscar((0, 0), (11, 14))
        assert (obtenido is None) == (esperado is None)
        if obtenido is None:
            continue
        assert all(segmento_transitable(libres, a, b) for a, b in zip(obtenido, obtenido[1:]))

        def costo(camino):
            return sum(np.hypot(a[0] - b[0], a[1] - b[1]) * (1 + costos[b])
                       for a, b in zip(camino, camino[1:]))

        assert abs(costo(esperado) - costo(obtenido)) < 1e-8
