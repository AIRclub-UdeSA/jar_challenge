"""Decisiones del evasor, sin ROS (se testean en test/test_exploracion.py).

Convención de ángulos: relativos al frente del robot, en radianes, positivo = izquierda.
"""
import math

import numpy as np

UMBRAL_OCUPADO = 65


def angulos_relativos(n: int, angle_min: float, angle_increment: float,
                      angulo_frente: float) -> np.ndarray:
    """Ángulo de cada rayo del scan respecto del frente del robot, en (-pi, pi]."""
    angs = angle_min + angle_increment * np.arange(n) - angulo_frente
    return np.arctan2(np.sin(angs), np.cos(angs))


def distancia_en_sector(rangos: np.ndarray, rel: np.ndarray, centro: float, ancho: float,
                        range_min: float) -> float:
    """Distancia al obstáculo más cercano dentro del sector centro ± ancho/2 (inf si no hay)."""
    dif = np.arctan2(np.sin(rel - centro), np.cos(rel - centro))
    validos = (np.abs(dif) <= ancho / 2) & np.isfinite(rangos) & (rangos > range_min)
    return float(rangos[validos].min()) if validos.any() else math.inf


def espacio_en_sector(rangos: np.ndarray, rel: np.ndarray, centro: float, ancho: float,
                      range_min: float, tope: float = 3.0) -> float:
    """Espacio libre típico del sector: mediana de los rangos, con 'sin retorno' = tope."""
    dif = np.arctan2(np.sin(rel - centro), np.cos(rel - centro))
    sector = np.abs(dif) <= ancho / 2
    if not sector.any():
        return 0.0
    r = rangos[sector]
    r = np.where(np.isfinite(r) & (r > range_min), np.minimum(r, tope), tope)
    return float(np.median(r))


def puntos_xy(rangos: np.ndarray, rel: np.ndarray, range_min: float):
    """Puntos válidos del scan en el frame del robot: x adelante, y a la izquierda (m)."""
    validos = np.isfinite(rangos) & (rangos > range_min)
    return rangos[validos] * np.cos(rel[validos]), rangos[validos] * np.sin(rel[validos])


def distancia_al_frente(x: np.ndarray, y: np.ndarray, medio_ancho: float) -> float:
    """Distancia (desde el centro) al punto más cercano dentro del corredor que va a barrer
    el robot avanzando derecho: |y| < medio_ancho. A diferencia de un cono, ve igual lo que
    está en diagonal pegado a las esquinas y no se asusta con las paredes de un pasillo."""
    adelante = (x > 0) & (np.abs(y) < medio_ancho)
    return float(x[adelante].min()) if adelante.any() else math.inf


def distancias_laterales(x: np.ndarray, y: np.ndarray, desde_x: float, hasta_x: float):
    """(izquierda, derecha): distancia lateral |y| a lo más cercano a cada costado del robot,
    contando solo lo que está a su altura (desde_x < x < hasta_x), esquinas incluidas."""
    costado = (x > desde_x) & (x < hasta_x)
    izq = y[costado & (y > 0)]
    der = -y[costado & (y < 0)]
    return (float(izq.min()) if izq.size else math.inf,
            float(der.min()) if der.size else math.inf)


def lado_mas_libre(rangos: np.ndarray, rel: np.ndarray, range_min: float) -> int:
    """+1 si hay más espacio a la izquierda, -1 si a la derecha."""
    izq = espacio_en_sector(rangos, rel, math.pi / 2, math.radians(120), range_min)
    der = espacio_en_sector(rangos, rel, -math.pi / 2, math.radians(120), range_min)
    return 1 if izq >= der else -1


def velocidad_lateral(d_izq: float, d_der: float, margen: float, velocidad: float,
                      ancho_pasillo: float = 0.0, ganancia: float = 0.5) -> float:
    """Base mecanum, velocidad de costado (+ = izquierda).

    - Paredes a los dos lados a menos de `ancho_pasillo`: centrarse, proporcional a la
      diferencia. Va primero: en un paso angosto, empujar a fondo contra una pared hace que el
      robot rebote de un lado al otro (y termine rozando).
    - Una pared a menos de `margen` de un solo lado (el otro abierto): alejarse a `velocidad`.
    """
    if d_izq < ancho_pasillo and d_der < ancho_pasillo:
        return float(max(-velocidad, min(velocidad, ganancia * (d_izq - d_der))))
    if d_izq < margen <= d_der:
        return -velocidad
    if d_der < margen <= d_izq:
        return velocidad
    return 0.0


def factor_velocidad(d_frente: float, d_choque: float, d_lento: float,
                     minimo: float = 0.3) -> float:
    """1 con el frente despejado; baja linealmente hasta `minimo` al acercarse a d_choque."""
    if d_lento <= d_choque:
        return 1.0
    return float(min(1.0, max(minimo, (d_frente - d_choque) / (d_lento - d_choque))))


def rampa(valor: float, desde: float, hasta: float) -> float:
    """0 por debajo de `desde`, 1 por encima de `hasta`, lineal en el medio."""
    if hasta <= desde:
        return 1.0 if valor >= hasta else 0.0
    return float(min(1.0, max(0.0, (valor - desde) / (hasta - desde))))


def velocidad_avance(d_frente: float, d_izq: float, d_der: float, d_choque: float,
                     d_lento: float, v_normal: float, v_rapida: float,
                     lateral_desde: float, lateral_hasta: float,
                     frente_desde: float, frente_hasta: float) -> float:
    """Velocidad hacia adelante: la normal (frenando cerca de lo que hay enfrente) y, en
    zonas abiertas, más rápido. Sube de a poco: hace falta tener lejos tanto el frente como
    las dos paredes de los costados; en un pasillo o cerca de algo queda en la normal."""
    v = v_normal * factor_velocidad(d_frente, d_choque, d_lento)
    if v_rapida <= v_normal:
        return v
    abierto = min(rampa(min(d_izq, d_der), lateral_desde, lateral_hasta),
                  rampa(d_frente, frente_desde, frente_hasta))
    return v + (v_rapida - v_normal) * abierto


def elegir_direccion(libres, no_vistos, distancia_minima: float, peso_libre: float,
                     peso_no_visto: float, tope_libre: float = 2.0, angulos=None,
                     peso_seguir: float = 0.0):
    """Índice de la mejor dirección candidata, o None si ninguna está libre.

    libres: espacio libre (m) en cada dirección; no_vistos: fracción 0-1 de terreno no visto;
    angulos: cuánto hay que girar para cada una (rad). Con peso_seguir > 0, a igualdad de lo
    demás se prefiere girar poco: dar la vuelta queda como último recurso.
    """
    if angulos is None:
        angulos = [0.0] * len(libres)
    mejor, puntaje_mejor = None, -math.inf
    for i, (libre, no_visto, angulo) in enumerate(zip(libres, no_vistos, angulos)):
        if libre < distancia_minima:
            continue
        puntaje = (peso_libre * min(libre, tope_libre) / tope_libre + peso_no_visto * no_visto
                   + peso_seguir * (1.0 - abs(angulo) / math.pi))
        if puntaje > puntaje_mejor:
            mejor, puntaje_mejor = i, puntaje
    return mejor


class Cobertura:
    """Qué parte del mapa ya vio la cámara (grilla del mismo tamaño que /map)."""

    def __init__(self, grilla: np.ndarray, resolucion: float, origen_x: float, origen_y: float):
        self.ocupado = (grilla >= UMBRAL_OCUPADO) | (grilla < 0)
        self.libre = ~self.ocupado
        self.visto = np.zeros(grilla.shape, dtype=bool)
        self.res = resolucion
        self.ox, self.oy = origen_x, origen_y

    def celda(self, x: float, y: float):
        return int((y - self.oy) / self.res), int((x - self.ox) / self.res)

    def _rayo(self, x, y, angulo, alcance):
        """Celdas libres que atraviesa un rayo hasta la primera pared (o el alcance)."""
        alto, ancho = self.ocupado.shape
        c, s = math.cos(angulo), math.sin(angulo)
        for d in np.arange(0.0, alcance, self.res):
            f, k = self.celda(x + d * c, y + d * s)
            if not (0 <= f < alto and 0 <= k < ancho) or self.ocupado[f, k]:
                return
            yield f, k

    def marcar(self, x, y, yaw, fov, alcance, rayos=31):
        """Marca como visto lo que cubre la cámara desde (x, y, yaw), cortado por las paredes."""
        for a in np.linspace(yaw - fov / 2, yaw + fov / 2, rayos):
            for f, k in self._rayo(x, y, a, alcance):
                self.visto[f, k] = True

    def no_visto_hacia(self, x, y, direccion, ancho, alcance, rayos=7) -> float:
        """Fracción 0-1 de terreno no visto en el cono direccion ± ancho/2."""
        celdas = set()
        for a in np.linspace(direccion - ancho / 2, direccion + ancho / 2, rayos):
            celdas.update(self._rayo(x, y, a, alcance))
        if not celdas:
            return 0.0
        return sum(1 for f, k in celdas if not self.visto[f, k]) / len(celdas)

    def no_visto_alrededor(self, x, y, radio) -> float:
        """Fracción 0-1 de celdas libres no vistas dentro de un círculo de radio `radio`."""
        f0, k0 = self.celda(x, y)
        r = int(radio / self.res)
        alto, ancho = self.ocupado.shape
        fs, ks = slice(max(0, f0 - r), min(alto, f0 + r + 1)), slice(max(0, k0 - r), min(ancho, k0 + r + 1))
        ff, kk = np.mgrid[fs, ks]
        dentro = ((ff - f0) ** 2 + (kk - k0) ** 2 <= r * r) & self.libre[fs, ks]
        if not dentro.any():
            return 0.0
        return float((~self.visto[fs, ks][dentro]).mean())

    def porcentaje_visto(self) -> float:
        return 100.0 * self.visto[self.libre].mean() if self.libre.any() else 0.0
