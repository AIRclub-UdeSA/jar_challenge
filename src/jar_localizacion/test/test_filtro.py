"""Pruebas del filtro de partículas sin ejecutar ROS."""

import math

import numpy as np

from jar_localizacion.filtro import (calidad_scan, campo_de_probabilidad, componer_poses,
                                     dispersar, dispersion_xy, estimar_pose, invertir_pose,
                                     mover_particulas, pesar_particulas, remuestrear,
                                     sembrar_particulas, usar_filtro)

SIN_RUIDO = [0.0] * 6


def test_componer_con_inversa_es_identidad():
    """Una pose compuesta con su inversa vuelve al origen."""
    pose = (1.0, -2.0, 0.7)
    x, y, theta = componer_poses(pose, invertir_pose(pose))
    assert abs(x) < 1e-9 and abs(y) < 1e-9 and abs(theta) < 1e-9


def test_campo_maximo_en_obstaculos():
    """La probabilidad vale 100 en la pared y decae al alejarse."""
    grid = np.zeros((5, 5))
    grid[2, 2] = 100
    campo = campo_de_probabilidad(grid, 0.1, 0.2)
    assert campo[2, 2] == 100.0
    assert campo[2, 3] < 100.0 and campo[2, 4] < campo[2, 3]


def test_campo_sin_obstaculos():
    """Un mapa libre no tiene dónde terminar rayos."""
    assert not campo_de_probabilidad(np.zeros((3, 3)), 0.1, 0.2).any()


def test_mover_en_el_frame_de_cada_particula():
    """Un avance en odom se aplica según el theta de cada partícula."""
    particulas = np.array([[0.0, 0.0, 0.0, 0.5], [1.0, 1.0, math.pi / 2, 0.5]])
    # El robot miraba hacia +y en odom y avanzó 1 m.
    nuevas = mover_particulas(particulas, 0.0, 1.0, 0.0, math.pi / 2, SIN_RUIDO)
    assert np.allclose(nuevas[0, :3], [1.0, 0.0, 0.0])
    assert np.allclose(nuevas[1, :3], [1.0, 2.0, math.pi / 2])
    assert np.allclose(nuevas[:, 3], particulas[:, 3])


def test_mover_con_escalas():
    """Las escalas corrigen avance y giro antes de aplicar el ruido."""
    particulas = np.array([[0.0, 0.0, 0.0, 1.0]])
    nuevas = mover_particulas(particulas, 1.0, 0.0, 0.2, 0.0, SIN_RUIDO,
                              escala_avance=0.5, escala_giro=2.0)
    assert np.allclose(nuevas[0, :3], [0.5, 0.0, 0.4])


def test_pesar_prefiere_la_pose_que_calza():
    """La partícula que ve la pared donde está gana casi todo el peso."""
    grid = np.zeros((20, 20))
    grid[:, 15] = 100
    campo = campo_de_probabilidad(grid, 0.1, 0.05)
    puntos = np.array([[1.0, -0.2], [1.0, 0.0], [1.0, 0.2]])
    particulas = np.array([[0.55, 1.0, 0.0, 0.5], [0.15, 1.0, 0.0, 0.5]])
    pesos = pesar_particulas(particulas, puntos, campo, 0.1, 0.0, 0.0)
    assert math.isclose(pesos.sum(), 1.0)
    assert pesos[0] > 0.99


def test_remuestrear_copia_las_pesadas():
    """Sin peso, una partícula desaparece; los pesos vuelven a ser 1/N."""
    particulas = np.array([[0.0, 0, 0, 0.5], [9.0, 0, 0, 0.5]])
    nuevas = remuestrear(particulas, np.array([1.0, 0.0]))
    assert np.all(nuevas[:, 0] == 0.0)
    assert np.allclose(nuevas[:, 3], 0.5)


def test_estimar_pose_circular():
    """El promedio de ángulos cerca de pi no se va a cero."""
    particulas = sembrar_particulas(1.0, 2.0, math.pi, 0.0, 0.0, 4)
    particulas[:2, 2] = math.pi - 0.1
    particulas[2:, 2] = -math.pi + 0.1
    x, y, theta = estimar_pose(particulas)
    assert math.isclose(x, 1.0) and math.isclose(y, 2.0)
    assert abs(abs(theta) - math.pi) < 1e-9


def test_dispersar_evita_el_colapso():
    """Tras remuestrear con un solo peso, las partículas no quedan idénticas."""
    particulas = sembrar_particulas(0.0, 0.0, math.pi, 0.3, 0.3, 200)
    pesos = np.zeros(200)
    pesos[0] = 1.0
    nuevas = dispersar(remuestrear(particulas, pesos), 0.01, 0.01)
    assert 0.005 < np.std(nuevas[:, 0]) < 0.02
    assert np.all(np.abs(nuevas[:, 2]) <= math.pi)
    assert np.allclose(dispersar(nuevas, 0.0, 0.0), nuevas)


def test_calidad_scan_es_mayor_donde_calza():
    """Desde la pose correcta el scan cae sobre la pared; corrido 30 cm, no."""
    grid = np.zeros((20, 20))
    grid[:, 15] = 100
    campo = campo_de_probabilidad(grid, 0.1, 0.05)
    puntos = np.array([[1.0, -0.2], [1.0, 0.0], [1.0, 0.2]])
    buena = calidad_scan((0.55, 1.0, 0.0), puntos, campo, 0.1, 0.0, 0.0)
    corrida = calidad_scan((0.25, 1.0, 0.0), puntos, campo, 0.1, 0.0, 0.0)
    assert buena > 90.0
    assert corrida < 10.0


def test_dispersion_xy():
    particulas = np.array([[0.0, 0.0, 0, 0.5], [0.2, 0.0, 0, 0.5]])
    assert math.isclose(dispersion_xy(particulas), 0.1)


def test_usar_filtro():
    """Se usa el filtro salvo que esté abierto o calce claramente peor que la odometría."""
    assert usar_filtro(80.0, 82.0, 0.05, 0.15, 5.0)
    assert not usar_filtro(60.0, 82.0, 0.05, 0.15, 5.0)
    assert not usar_filtro(90.0, 50.0, 0.30, 0.15, 5.0)
