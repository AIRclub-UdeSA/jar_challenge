"""Seguidor holonómico: llega, respeta límites, corrige deriva y mantiene rumbo."""
import math

import pytest

from equipo_jar.navegacion.planificador import Ruta
from equipo_jar.navegacion.seguimiento import ParametrosSeguimiento, Seguidor
from equipo_jar.navegacion.tipos import Pose2D

P = ParametrosSeguimiento(velocidad_max=0.30, aceleracion_max=0.4, distancia_mira_m=0.20,
                          tolerancia_meta_m=0.03, ganancia_yaw=1.5,
                          velocidad_angular_max=0.5, espacio_lento_m=0.25,
                          espacio_rapido_m=0.45, velocidad_cerca_pared=0.10, dt_max_s=0.2)
DT = 0.1


def ruta(*puntos):
    return Ruta(list(puntos), 0.0, sum(math.dist(a, b) for a, b in zip(puntos, puntos[1:])))


def simular(seguidor, pose, pasos=600, espacio=1.0, deriva=(0.0, 0.0), dt=DT):
    """Cinemática ideal: aplica el comando en el marco del robot durante dt."""
    x, y, yaw = pose
    historia = []
    for k in range(pasos):
        cmd = seguidor.paso(k * dt, Pose2D(x, y, yaw), espacio)
        c, s = math.cos(yaw), math.sin(yaw)
        vxm, vym = c * cmd.vx - s * cmd.vy, s * cmd.vx + c * cmd.vy
        historia.append((x, y, yaw, vxm, vym, cmd))
        if seguidor.llego:
            break
        x += (vxm + deriva[0]) * dt
        y += (vym + deriva[1]) * dt
        yaw += cmd.wz * dt
    return historia


def distancia_a_polilinea(p, puntos):
    mejor = math.inf
    for a, b in zip(puntos, puntos[1:]):
        ax, ay = b[0] - a[0], b[1] - a[1]
        t = max(0.0, min(1.0, ((p[0] - a[0]) * ax + (p[1] - a[1]) * ay) / (ax * ax + ay * ay)))
        mejor = min(mejor, math.dist(p, (a[0] + t * ax, a[1] + t * ay)))
    return mejor


def test_ruta_recta_llega_dentro_de_tolerancia():
    s = Seguidor(P)
    s.reiniciar(ruta((0.0, 0.0), (2.0, 0.0)), yaw_mantener=0.3)
    h = simular(s, (0.0, 0.0, 0.3))
    assert s.llego
    assert math.dist(h[-1][:2], (2.0, 0.0)) <= P.tolerancia_meta_m
    assert h[-1][5].es_cero()


def test_nunca_supera_velocidad_ni_aceleracion():
    s = Seguidor(P)
    s.reiniciar(ruta((0.0, 0.0), (2.0, 0.0), (2.0, 1.0)), yaw_mantener=0.0)
    h = simular(s, (0.0, 0.0, 0.0))
    rapideces = [math.hypot(v[3], v[4]) for v in h]
    assert max(rapideces) <= P.velocidad_max + 1e-9
    for a, b in zip(rapideces, rapideces[1:]):
        assert b - a <= P.aceleracion_max * DT + 1e-9


def test_ruta_en_l_se_aparta_menos_de_0_10_m():
    puntos = [(0.0, 0.0), (1.5, 0.0), (1.5, 1.5)]
    s = Seguidor(P)
    s.reiniciar(ruta(*puntos), yaw_mantener=0.0)
    h = simular(s, (0.0, 0.0, 0.0))
    assert s.llego
    assert max(distancia_a_polilinea(v[:2], puntos) for v in h) < 0.10


def test_deriva_lateral_constante_se_corrige():
    s = Seguidor(P)
    s.reiniciar(ruta((0.0, 0.0), (1.5, 0.0)), yaw_mantener=0.0)
    h = simular(s, (0.0, 0.0, 0.0), deriva=(0.0, 0.02))
    assert s.llego
    assert math.dist(h[-1][:2], (1.5, 0.0)) <= P.tolerancia_meta_m
    assert max(abs(v[1]) for v in h) < 0.05


def test_mantiene_rumbo_con_wz_acotado():
    s = Seguidor(P)
    s.reiniciar(ruta((0.0, 0.0), (2.0, 0.0)), yaw_mantener=0.0)
    h = simular(s, (0.0, 0.0, 0.25))
    assert all(abs(v[5].wz) <= P.velocidad_angular_max + 1e-12 for v in h)
    assert abs(h[-1][2]) < 0.02


def test_espacio_libre_bajo_limita_velocidad():
    s = Seguidor(P)
    s.reiniciar(ruta((0.0, 0.0), (3.0, 0.0)), yaw_mantener=0.0)
    h = simular(s, (0.0, 0.0, 0.0), pasos=40, espacio=0.25)
    assert max(math.hypot(v[3], v[4]) for v in h) <= P.velocidad_cerca_pared + 1e-9


def test_dt_largo_no_genera_salto():
    s = Seguidor(P)
    s.reiniciar(ruta((0.0, 0.0), (3.0, 0.0)), yaw_mantener=0.0)
    s.paso(0.0, Pose2D(0.0, 0.0, 0.0), 1.0)
    previa = math.hypot(s.paso(0.1, Pose2D(0.0, 0.0, 0.0), 1.0).vx, 0.0)
    salto = s.paso(0.6, Pose2D(0.0, 0.0, 0.0), 1.0)
    assert math.hypot(salto.vx, salto.vy) - previa <= P.aceleracion_max * P.dt_max_s + 1e-9


def test_parametros_invalidos_fallan():
    with pytest.raises(ValueError):
        ParametrosSeguimiento(0.3, 0.4, 0.2, 0.03, 1.5, 0.5, 0.45, 0.25, 0.1, 0.2)
    with pytest.raises(ValueError):
        ParametrosSeguimiento(0.3, 0.0, 0.2, 0.03, 1.5, 0.5, 0.25, 0.45, 0.1, 0.2)
