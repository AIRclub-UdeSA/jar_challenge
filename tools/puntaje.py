"""Puntaje del Challenge JAR 2026 (reglamento: §7.2 reportes y §8 puntaje).

Python puro, sin ROS: le entran las víctimas, los reportes y los eventos de una corrida, y
devuelve el desglose del puntaje. El nodo del juez solo junta los eventos y llama a `puntuar`.
Así se puede probar sin simulador y el mismo cálculo sirve con las víctimas de práctica y con
las de la competencia.

Reglas que aplica:
- Se cuentan como máximo N reportes en total (N = cantidad de víctimas); el resto se ignora.
- Un reporte mal formado gasta uno de los N y no puntúa.
- Cada víctima toma el reporte más cercano a su posición real; un reporte solo puede ser de
  una víctima. Con víctimas separadas por más del doble del radio no hay ambigüedad.
- 100 puntos a 0,25 m o menos, 50 hasta 0,50 m, 0 más allá.
- Bono de +200: las N víctimas con su propio reporte a 0,50 m o menos, y todos los reportes
  recibidos antes de los 5 minutos (300 segundos) desde la publicación de /map.
- Bono de +100 por regreso: /done con el robot a 0,25 m o menos de la largada y al menos una
  víctima puntuada. En el robot físico lo confirma un juez; en el simulador se mide.
- Cada colisión resta 50 y la salida del laberinto resta 50 y termina la corrida (§8).
- Un vuelco anula la corrida (0 puntos, §11.2).
- El puntaje mínimo de una corrida es 0.
"""
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

RADIO_EXACTO = 0.25          # m: hasta acá, PUNTOS_EXACTO
RADIO_MAXIMO = 0.50          # m: hasta acá, PUNTOS_APROXIMADO; más allá no puntúa
PUNTOS_EXACTO = 100
PUNTOS_APROXIMADO = 50
BONO_RAPIDO = 200
LIMITE_BONO_S = 300.0        # 5 minutos desde /map
BONO_REGRESO = 100
TOLERANCIA_LARGADA = 0.25    # m
PENALIZACION_COLISION = 50   # §8
PENALIZACION_SALIDA = 50     # §8: además termina la corrida

_EPS = 1e-9  # los bordes (0,25 m exactos) se resuelven a favor del equipo


@dataclass(frozen=True)
class Reporte:
    t: float                 # segundos desde la publicación de /map, con el reloj del juez
    x: float
    y: float
    valido: bool = True      # False: mal formado (otro frame, valores no finitos...)


@dataclass
class Corrida:
    victimas: List[Tuple[float, float]]
    reportes: List[Reporte] = field(default_factory=list)   # en orden de llegada
    colisiones: int = 0
    salio_del_laberinto: bool = False
    hizo_done: bool = False
    anulada: bool = False    # vuelco (§11.2): la corrida queda en 0 puntos
    # Robot físico: un juez confirma a mano si el robot quedó dentro del círculo de la largada.
    # Simulador: se mide con la pose real (distancia_a_largada_en_done). Si hay veredicto, manda.
    regreso_confirmado: Optional[bool] = None
    distancia_a_largada_en_done: Optional[float] = None


@dataclass
class ResultadoVictima:
    indice: int
    indice_reporte: Optional[int]    # posición en Corrida.reportes; None si nadie la reportó
    distancia: Optional[float]
    puntos: int


@dataclass
class Resultado:
    por_victima: List[ResultadoVictima]
    reportes_contados: int
    reportes_ignorados: int
    puntos_victimas: int
    bono_rapido: bool
    bono_regreso: bool
    penalizacion_colisiones: int
    penalizacion_salida: int
    total: int
    anulada: bool = False
    # Instante (s desde /map) del último reporte que se tomó para puntuar una víctima. El desempate
    # de §8.2 usa el tiempo total; este queda para `ranking.py --desempate ultimo_reporte`.
    tiempo_ultimo_reporte_que_puntua: Optional[float] = None

    def resumen(self):
        pv = ' '.join(f'V{r.indice + 1}:{r.puntos}' for r in self.por_victima)
        partes = [f'total {self.total}' + (' (ANULADA)' if self.anulada else ''), f'víctimas [{pv}]']
        if self.bono_rapido:
            partes.append(f'+{BONO_RAPIDO} bono 5 min')
        if self.bono_regreso:
            partes.append(f'+{BONO_REGRESO} regreso')
        if self.penalizacion_colisiones:
            partes.append(f'-{self.penalizacion_colisiones} colisiones')
        if self.penalizacion_salida:
            partes.append(f'-{self.penalizacion_salida} salida')
        return ' | '.join(partes)


def puntos_por_distancia(d: float) -> int:
    if d <= RADIO_EXACTO + _EPS:
        return PUNTOS_EXACTO
    if d <= RADIO_MAXIMO + _EPS:
        return PUNTOS_APROXIMADO
    return 0


def _asignar(victimas, contados):
    """Devuelve {indice_victima: (indice_reporte_en_contados, distancia)}."""
    mejor = {}
    for i, r in enumerate(contados):
        if not r.valido:
            continue
        # La víctima más cercana a este reporte, dentro del radio máximo.
        cercana, d_min = None, None
        for v, (vx, vy) in enumerate(victimas):
            d = math.hypot(r.x - vx, r.y - vy)
            if d <= RADIO_MAXIMO + _EPS and (d_min is None or d < d_min - _EPS):
                cercana, d_min = v, d
        if cercana is None:
            continue
        # Cada víctima se queda con su reporte más cercano (a igual distancia, el primero).
        if cercana not in mejor or d_min < mejor[cercana][1] - _EPS:
            mejor[cercana] = (i, d_min)
    return mejor


def puntuar(corrida: Corrida,
            penalizacion_colision: int = PENALIZACION_COLISION,
            penalizacion_salida: int = PENALIZACION_SALIDA) -> Resultado:
    n = len(corrida.victimas)
    contados = corrida.reportes[:n]
    ignorados = max(0, len(corrida.reportes) - n)

    mejor = _asignar(corrida.victimas, contados)
    por_victima = []
    for v in range(n):
        if v in mejor:
            i, d = mejor[v]
            por_victima.append(ResultadoVictima(v, i, d, puntos_por_distancia(d)))
        else:
            por_victima.append(ResultadoVictima(v, None, None, 0))
    puntos_victimas = sum(r.puntos for r in por_victima)

    todas_puntuan = n > 0 and all(r.puntos > 0 for r in por_victima)
    bono_rapido = todas_puntuan and all(r.t < LIMITE_BONO_S for r in contados)
    if corrida.regreso_confirmado is not None:
        volvio = corrida.regreso_confirmado
    else:
        volvio = (corrida.distancia_a_largada_en_done is not None
                  and corrida.distancia_a_largada_en_done <= TOLERANCIA_LARGADA + _EPS)
    bono_regreso = corrida.hizo_done and volvio and puntos_victimas > 0

    pen_col = corrida.colisiones * penalizacion_colision
    pen_sal = penalizacion_salida if corrida.salio_del_laberinto else 0
    total = (puntos_victimas + (BONO_RAPIDO if bono_rapido else 0)
             + (BONO_REGRESO if bono_regreso else 0) - pen_col - pen_sal)
    tiempos = [contados[r.indice_reporte].t for r in por_victima
               if r.puntos > 0 and r.indice_reporte is not None]
    return Resultado(por_victima, len(contados), ignorados, puntos_victimas,
                     bono_rapido, bono_regreso, pen_col, pen_sal,
                     0 if corrida.anulada else max(0, total), corrida.anulada,
                     max(tiempos) if tiempos else None)
