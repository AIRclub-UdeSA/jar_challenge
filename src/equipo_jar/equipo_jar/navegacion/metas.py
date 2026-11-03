"""Meta de prueba elegida en tiempo de ejecución desde el mapa recibido."""
import numpy as np

from equipo_jar.navegacion.planificador import costo_desde


def elegir_meta_prueba(grilla, costos, distancia, inicio_xy, distancia_min_m, distancia_max_m,
                       espacio_min_m):
    """Celda alcanzable con más espacio libre; empate (1 cm) por mayor largo de ruta."""
    _, largo = costo_desde(grilla, costos, inicio_xy)
    candidatas = ((largo >= distancia_min_m) & (largo <= distancia_max_m)
                  & (distancia >= espacio_min_m))
    filas, cols = np.nonzero(candidatas)
    if not len(filas):
        return None
    espacio = np.round(distancia[filas, cols], 2)
    orden = np.lexsort((filas * grilla.ancho + cols, -largo[filas, cols], -espacio))
    return grilla.a_mundo(int(filas[orden[0]]), int(cols[orden[0]]))
