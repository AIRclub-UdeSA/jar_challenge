"""Concordancia tolerante a obstáculos (C): no se dispara con cajas y sigue detectando poses
equivocadas. El trazador de rayos de este test es propio: no reutiliza el código evaluado."""
import math
import random
from types import SimpleNamespace as S

import pytest

from equipo_jar.localizacion_activa import concordancia_mapa, concordancia_mapa_simple

RES = 0.05
ANCHO, ALTO = 120, 100  # laberinto sintético de 6 x 5 m, paredes de 2 celdas
POSES = [(1.0, 1.2, 0.3), (3.0, 4.0, 1.0), (5.0, 3.0, -0.5), (1.5, 4.2, 2.0)]
N_RAYOS, MUESTRA = 360, 120


def laberinto():
    ocupada = [[False] * ANCHO for _ in range(ALTO)]

    def bloque(f0, f1, c0, c1):
        for f in range(f0, f1):
            for c in range(c0, c1):
                ocupada[f][c] = True

    bloque(0, 2, 0, ANCHO)
    bloque(ALTO - 2, ALTO, 0, ANCHO)
    bloque(0, ALTO, 0, 2)
    bloque(0, ALTO, ANCHO - 2, ANCHO)
    bloque(0, 66, 40, 42)       # pared vertical en x = 2,0 m
    bloque(35, ALTO, 80, 82)    # pared vertical en x = 4,0 m
    bloque(50, 52, 0, 22)       # pared horizontal en y = 2,5 m
    bloque(30, 32, 60, 101)     # pared horizontal en y = 1,5 m
    return ocupada


def como_mapa(ocupada):
    datos = [100 if ocupada[f][c] else 0 for f in range(ALTO) for c in range(ANCHO)]
    return S(info=S(resolution=RES, width=ANCHO, height=ALTO,
                    origin=S(position=S(x=0.0, y=0.0),
                             orientation=S(w=1.0, x=0.0, y=0.0, z=0.0))),
             data=datos)


def rango(ocupada, x, y, angulo, maximo=8.0):
    """Distancia hasta la primera celda ocupada, en pasos de 5 mm."""
    c, s = math.cos(angulo), math.sin(angulo)
    d = 0.0
    while d < maximo:
        f, col = math.floor((y + d * s) / RES), math.floor((x + d * c) / RES)
        if not (0 <= f < len(ocupada) and 0 <= col < len(ocupada[0])) or ocupada[f][col]:
            return d
        d += 0.005
    return math.inf


def con_cajas(ocupada, pose):
    """Cajas de 0,3 m a 0,45 m del robot en las direcciones más despejadas."""
    mundo = [fila[:] for fila in ocupada]
    libres = sorted(((rango(ocupada, pose[0], pose[1], math.radians(g)), g)
                     for g in range(0, 360, 10)), reverse=True)
    elegidas = []
    for _, g in libres:
        if all(abs((g - e + 180) % 360 - 180) >= 60 for e in elegidas):
            elegidas.append(g)
        if len(elegidas) == 4:
            break
    for g in elegidas:
        cx = pose[0] + 0.45 * math.cos(math.radians(g))
        cy = pose[1] + 0.45 * math.sin(math.radians(g))
        for f in range(ALTO):
            for col in range(ANCHO):
                if abs((col + .5) * RES - cx) <= 0.15 and abs((f + .5) * RES - cy) <= 0.15:
                    mundo[f][col] = True
    return mundo


def escanear(mundo, pose, ruido=0.0, semilla=0):
    azar = random.Random(semilla)
    inc = 2 * math.pi / N_RAYOS
    rangos = [rango(mundo, pose[0], pose[1], pose[2] - math.pi + i * inc) for i in
              range(N_RAYOS)]
    rangos = [r + azar.gauss(0.0, ruido) if math.isfinite(r) else r for r in rangos]
    validos = [(i, r) for i, r in enumerate(rangos) if math.isfinite(r) and r > 0.05]
    return validos[::max(1, len(validos) // MUESTRA)], -math.pi, inc


def evaluar(funcion, mapa, muestra, angulo_min, inc, pose_estimada):
    return funcion(mapa, muestra, angulo_min, inc, pose_estimada, 0.10)


OCUPADA = laberinto()
MAPA = como_mapa(OCUPADA)


def test_poses_de_prueba_estan_libres():
    for x, y, _ in POSES:
        assert not OCUPADA[math.floor(y / RES)][math.floor(x / RES)]


@pytest.mark.parametrize('pose', POSES)
def test_caja_no_mapeada_no_baja_concordancia(pose):
    mundo = con_cajas(OCUPADA, pose)
    muestra, amin, inc = escanear(mundo, pose)
    limpia, _, _ = escanear(OCUPADA, pose)
    tapados = sum(1 for (i, r), (_, r0) in zip(muestra, limpia) if r < r0 - 0.05)
    assert tapados / len(muestra) >= 0.35
    nueva, contados = evaluar(concordancia_mapa, MAPA, muestra, amin, inc, pose)
    assert nueva >= 0.95 and contados >= 30
    vieja, _ = evaluar(concordancia_mapa_simple, MAPA, muestra, amin, inc, pose)
    assert vieja < 0.80  # el criterio viejo sí se disparaba


@pytest.mark.parametrize('cajas', [False, True])
@pytest.mark.parametrize('pose', POSES)
def test_pose_correcta_con_ruido_no_dispara(pose, cajas):
    mundo = con_cajas(OCUPADA, pose) if cajas else OCUPADA
    muestra, amin, inc = escanear(mundo, pose, ruido=0.01, semilla=7)
    assert evaluar(concordancia_mapa, MAPA, muestra, amin, inc, pose)[0] >= 0.95


def desplazadas(pose):
    for d in (0.15, 0.30):
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            yield (f'{d}m_{dx}{dy}', (pose[0] + d * dx, pose[1] + d * dy, pose[2]))
    for g in (-10, -5, 5, 10):
        yield (f'{g}grados', (pose[0], pose[1], pose[2] + math.radians(g)))


CASOS = [(pose, nombre, estimada) for pose in POSES for nombre, estimada in desplazadas(pose)]


@pytest.mark.parametrize('pose,nombre,estimada', CASOS)
def test_c_no_detecta_menos_que_antes(pose, nombre, estimada):
    """Sin obstáculos no mapeados, todo error que el criterio viejo detectaba se detecta."""
    muestra, amin, inc = escanear(OCUPADA, pose)
    vieja, _ = evaluar(concordancia_mapa_simple, MAPA, muestra, amin, inc, estimada)
    nueva, _ = evaluar(concordancia_mapa, MAPA, muestra, amin, inc, estimada)
    if vieja < 0.80:
        assert nueva < 0.80, f'{nombre}: vieja {vieja:.2f}, nueva {nueva:.2f}'


@pytest.mark.parametrize('pose', POSES)
def test_desplazamiento_de_30_cm_y_giro_de_10_grados_detectados(pose):
    muestra, amin, inc = escanear(OCUPADA, pose)
    for nombre, estimada in desplazadas(pose):
        if nombre.startswith('0.3') or nombre.endswith('10grados'):
            nueva = evaluar(concordancia_mapa, MAPA, muestra, amin, inc, estimada)[0]
            assert nueva < 0.80, f'{nombre}: {nueva:.2f}'


def test_con_cajas_detecta_casi_todos_los_desplazamientos():
    """Límite medido: con 40 % del scan tapado, 4 de 32 corrimientos (pose (5, 3), eje y) quedan en 0,80 justo.

    Los giros pequeños con cajas no se detectan de forma confiable: las cajas tapan
    los rayos largos, que son los que informan el ángulo (ver evidencia de la tabla).
    """
    detectados = total = 0
    for pose in POSES:
        muestra, amin, inc = escanear(con_cajas(OCUPADA, pose), pose)
        for nombre, estimada in desplazadas(pose):
            if 'grados' in nombre:
                continue
            total += 1
            detectados += evaluar(concordancia_mapa, MAPA, muestra, amin, inc,
                                  estimada)[0] < 0.80
    assert detectados / total >= 28 / 32


def test_pasillo_largo_documenta_limite():
    """En un pasillo recto, un corrimiento a lo largo del eje casi no cambia el scan.

    Límite conocido: no lo detecta ni el criterio viejo ni el nuevo. Se fija el
    comportamiento para que un cambio futuro lo haga visible.
    """
    ocupada = [[f < 2 or f >= 20 for _ in range(ANCHO)] for f in range(22)]
    mapa = S(info=S(resolution=RES, width=ANCHO, height=22,
                    origin=S(position=S(x=0.0, y=0.0),
                             orientation=S(w=1.0, x=0.0, y=0.0, z=0.0))),
             data=[100 if ocupada[f][c] else 0 for f in range(22) for c in range(ANCHO)])
    muestra = [(i, r) for i, r in
               ((i, rango(ocupada, 3.0, 0.55, -math.pi + i * 2 * math.pi / 360, 2.5))
                for i in range(360)) if math.isfinite(r)][::3]
    estimada = (3.3, 0.55, 0.0)
    nueva = concordancia_mapa(mapa, muestra, -math.pi, 2 * math.pi / 360, estimada, 0.10)[0]
    vieja = concordancia_mapa_simple(mapa, muestra, -math.pi, 2 * math.pi / 360, estimada,
                                     0.10)[0]
    assert nueva >= 0.80 and vieja >= 0.80


def test_todos_cortos_no_cuenta_rayos():
    """Rodeado de obstáculos no mapeados: no hay rayos que cuenten (falta de datos)."""
    pose = (3.0, 3.0, 0.0)
    muestra = [(i, 0.25) for i in range(0, 360, 3)]
    _, contados = concordancia_mapa(MAPA, muestra, -math.pi, 2 * math.pi / 360, pose, 0.10)
    assert contados == 0
