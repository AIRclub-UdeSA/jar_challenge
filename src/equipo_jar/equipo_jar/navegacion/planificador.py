"""A* de 8 vecinos sobre costos inflados, con poda por línea de visión."""
from dataclasses import dataclass
import heapq
import math

import numpy as np

VECINOS = [(df, dc, math.hypot(df, dc)) for df in (-1, 0, 1) for dc in (-1, 0, 1)
           if df or dc]


@dataclass(frozen=True)
class Ruta:
    puntos: list  # [(x, y)] en el marco del mapa, de la pose de inicio a la meta exacta
    costo: float
    largo_m: float
    orientaciones: object = None  # rumbo por vértice para el respaldo SE(2)
    rumbo_final: object = None  # sólo cuando la misión solicita orientación final
    precision_vertice_m: float = .02  # contrato espacial de los giros intermedios


def _ruta_escape(grilla, costos, celda, escape_m):
    """Salida corta por celdas realmente libres; nunca atraviesa una pared."""
    if celda is None or grilla.ocupada[celda]:
        return None
    if math.isfinite(costos[celda]):
        return [celda]
    distancias={celda:0.}
    padres={}
    cola=[(0.,celda)]
    while cola:
        distancia,actual=heapq.heappop(cola)
        if distancia>distancias[actual]+1e-12:
            continue
        if math.isfinite(costos[actual]):
            ruta=[actual]
            while ruta[-1]!=celda:
                ruta.append(padres[ruta[-1]])
            return list(reversed(ruta))
        f,c=actual
        for df,dc,paso in VECINOS:
            nf,nc=f+df,c+dc
            nueva=distancia+paso*grilla.resolucion
            if (nueva>escape_m+1e-12 or not(0<=nf<grilla.alto and 0<=nc<grilla.ancho)
                    or grilla.ocupada[nf,nc]):
                continue
            if df and dc and (grilla.ocupada[f,nc] or grilla.ocupada[nf,c]):
                continue
            siguiente=(nf,nc)
            if nueva+1e-12<distancias.get(siguiente,math.inf):
                distancias[siguiente]=nueva
                padres[siguiente]=actual
                heapq.heappush(cola,(nueva,siguiente))
    return None


def _escape(grilla,costos,celda,escape_m):
    ruta=_ruta_escape(grilla,costos,celda,escape_m)
    return ruta[-1] if ruta else None


def _segmento_admisible(grilla, costos, a, b, tope):
    """Todas las celdas del segmento existen y cuestan a lo sumo tope."""
    n = max(1, math.ceil(math.dist(a, b) / (grilla.resolucion / 4)))
    t = np.linspace(0.0, 1.0, n + 1)
    u, v = grilla.a_local(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
    cols = np.floor(u / grilla.resolucion).astype(int)
    filas = np.floor(v / grilla.resolucion).astype(int)
    if (cols.min() < 0 or filas.min() < 0 or cols.max() >= grilla.ancho
            or filas.max() >= grilla.alto):
        return False
    return bool(np.all(costos[filas, cols] <= tope))


def _podar(grilla, costos, puntos, celdas):
    """Atajos comprobados con búsqueda exponencial y refinamiento acotado.

    Cada segmento aceptado se comprueba completo; una visibilidad no monótona
    puede dar una poda menos agresiva, pero nunca habilita un atajo sin validar.
    """
    valores = [float(costos[c]) for c in celdas]
    podados, i = [puntos[0]], 0
    ultimo = len(puntos) - 1
    while i < ultimo:
        j, salto = i + 1, 2
        while j < ultimo:
            candidato = min(ultimo, i + salto)
            tope = max(valores[i:candidato + 1])
            if _segmento_admisible(grilla, costos, puntos[i], puntos[candidato], tope):
                j = candidato
                salto *= 2
                continue
            bajo, alto = j, candidato
            while alto - bajo > 1:
                medio = (alto + bajo) // 2
                if _segmento_admisible(grilla, costos, puntos[i], puntos[medio],
                                      max(valores[i:medio + 1])):
                    bajo = medio
                else:
                    alto = medio
            j = bajo
            break
        podados.append(puntos[j])
        i = j
    return podados


def _dijkstra(grilla, costos, origen, destino=None, diagonales_optimistas=False):
    """A*/Dijkstra con celdas planas, cierre y descarte de entradas viejas."""
    alto, ancho = costos.shape
    total = alto * ancho
    pesos = costos.ravel().tolist()
    finitas = np.isfinite(costos).ravel().tolist()
    coste = [math.inf] * total
    largos = [math.inf] * total
    padres = [-1] * total
    cerradas = bytearray(total)
    inicio = origen[0] * ancho + origen[1]
    objetivo = None if destino is None else destino[0] * ancho + destino[1]
    if destino is None:
        heuristica = [0.] * total
    else:
        filas, cols = np.indices(costos.shape)
        df, dc = np.abs(filas - destino[0]), np.abs(cols - destino[1])
        heuristica = (np.maximum(df, dc) + (math.sqrt(2) - 1) *
                      np.minimum(df, dc)).ravel().tolist()
    coste[inicio] = largos[inicio] = 0.
    abiertos = [(heuristica[inicio], 0., inicio)]
    while abiertos:
        _, acumulado, celda = heapq.heappop(abiertos)
        if cerradas[celda] or acumulado != coste[celda]:
            continue
        cerradas[celda] = 1
        if celda == objetivo:
            break
        fila, col = divmod(celda, ancho)
        for df, dc, paso in VECINOS:
            f, c = fila + df, col + dc
            if not (0 <= f < alto and 0 <= c < ancho):
                continue
            vecino = f * ancho + c
            if not finitas[vecino] or cerradas[vecino]:
                continue
            if not diagonales_optimistas and df and dc and (not finitas[fila * ancho + c] or
                              not finitas[f * ancho + col]):
                continue
            nuevo = acumulado + paso * (1. + pesos[vecino])
            if nuevo < coste[vecino]:
                coste[vecino] = nuevo
                largos[vecino] = largos[celda] + paso
                padres[vecino] = celda
                heapq.heappush(abiertos, (nuevo + heuristica[vecino], nuevo, vecino))
    previo = {divmod(i, ancho): divmod(padre, ancho)
              for i, padre in enumerate(padres) if padre >= 0}
    return (np.asarray(coste).reshape(costos.shape),
            np.asarray(largos).reshape(costos.shape), previo)


def planificar(grilla, costos, inicio_xy, meta_xy, escape_m=0.35):
    """Ruta segura de inicio a meta, o None si no hay."""
    celda_inicio = grilla.a_celda(*inicio_xy)
    celda_meta = grilla.a_celda(*meta_xy)
    if celda_inicio is None or celda_meta is None or not math.isfinite(costos[celda_meta]):
        return None
    escape = _ruta_escape(grilla,costos,celda_inicio,escape_m)
    if escape is None:
        return None
    origen = escape[-1]
    costo, _, previo = _dijkstra(grilla, costos, origen, celda_meta)
    if not math.isfinite(costo[celda_meta]):
        return None
    celdas = [celda_meta]
    while celdas[-1] != origen:
        celdas.append(previo[celdas[-1]])
    celdas.reverse()
    centros = [grilla.a_mundo(*c) for c in celdas]
    puntos = [tuple(inicio_xy)] + centros[1:-1] + [tuple(meta_xy)]
    celdas_puntos = [celda_inicio] + celdas[1:-1] + [celda_meta]
    if origen != celda_inicio:  # el tramo de escape no se poda
        prefijo=[tuple(inicio_xy)]+[grilla.a_mundo(*c) for c in escape[1:]]
        sufijo=_podar(grilla,costos,centros[:-1]+[tuple(meta_xy)],celdas)
        puntos=(prefijo[:-1]+sufijo if len(sufijo)==1 else prefijo+sufijo[1:])
    else:
        puntos = _podar(grilla, costos, puntos, celdas_puntos)
    largo = sum(math.dist(a, b) for a, b in zip(puntos, puntos[1:]))
    return Ruta(puntos, float(costo[celda_meta]) * grilla.resolucion, largo)


def costo_desde(grilla, costos, inicio_xy, escape_m=0.35, diagonales_optimistas=False):
    """Distancias desde una salida corta de la celda inicial.

    diagonales_optimistas sólo estima distancias sobre una unión de rumbos;
    no autoriza movimiento. El planificador ejecutable conserva False.
    """
    celda=grilla.a_celda(*inicio_xy)
    escape=_ruta_escape(grilla,costos,celda,escape_m)
    if escape is None:
        vacio=np.full(costos.shape,np.inf)
        return vacio,vacio.copy()
    origen=escape[-1]
    costo,largo,_=_dijkstra(grilla,costos,origen,diagonales_optimistas=diagonales_optimistas)
    puntos=[tuple(inicio_xy)]+[grilla.a_mundo(*c) for c in escape[1:]]
    extra=sum(math.dist(a,b) for a,b in zip(puntos,puntos[1:]))
    return costo*grilla.resolucion+extra*(1+costos[origen]),largo*grilla.resolucion+extra
