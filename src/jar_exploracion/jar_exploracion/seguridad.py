"""Geometría y comprobación predictiva de traslación para el X3 holonómico."""

import math

import numpy as np
from scipy import ndimage


def espacio_transitable(datos, resolucion, largo, ancho, margen, yaw=0.0):
    """
    Infla obstáculos con una huella rectangular y el área de cada píxel.

    Largo y ancho envuelven todas las colisiones respecto de base_footprint.
    Yaw se expresa respecto de los ejes del mapa. El exterior es ocupado.
    No permite cambiar la orientación durante una traslación.
    """
    if (np.ndim(datos) != 2 or not np.size(datos)
            or not all(math.isfinite(v) for v in (resolucion, largo, ancho, margen, yaw))
            or min(resolucion, largo, ancho) <= 0 or margen < 0):
        raise ValueError('Mapa y dimensiones de huella inválidos')
    c, s = math.cos(yaw), math.sin(yaw)
    # Soporte del cuadrado de ocupación sobre los ejes del robot.
    pixel = resolucion * (abs(c) + abs(s)) / 2
    hx, hy = largo / 2 + margen + pixel, ancho / 2 + margen + pixel
    radio = math.ceil(math.hypot(hx, hy) / resolucion)
    fy, fx = np.mgrid[-radio:radio + 1, -radio:radio + 1] * resolucion
    bx, by = c * fx + s * fy, -s * fx + c * fy
    estructura = (np.abs(bx) <= hx + 1e-9) & (np.abs(by) <= hy + 1e-9)
    ocupado = np.asarray(datos) != 0
    return ~ndimage.binary_dilation(ocupado, structure=estructura, border_value=1)


def puntos_scan(scan, tx=0.0, ty=0.0, yaw=0.0):
    """Transforma impactos válidos del láser al marco de la base."""
    rangos = np.asarray(scan.ranges, dtype=float)
    if (not all(math.isfinite(v) for v in (tx, ty, yaw))
            or not math.isfinite(scan.range_min) or not math.isfinite(scan.range_max)
            or not 0 <= scan.range_min < scan.range_max
            or not math.isfinite(scan.angle_min) or not math.isfinite(scan.angle_increment)):
        return None
    angulos = scan.angle_min + np.arange(len(rangos)) * scan.angle_increment + yaw
    validos = (np.isfinite(rangos) & (rangos > 0)
               & (rangos >= scan.range_min) & (rangos <= scan.range_max))
    # Un scan sin mediciones ni rayos sin retorno no demuestra espacio libre.
    if not np.any(validos) and not np.any(np.isposinf(rangos)):
        return None
    return np.column_stack((tx + rangos[validos] * np.cos(angulos[validos]),
                            ty + rangos[validos] * np.sin(angulos[validos])))


def distancia_huella(puntos, largo, ancho):
    """Distancia firmada a una caja centrada: negativa dentro del chasis."""
    q = np.abs(np.asarray(puntos)) - np.array([largo / 2, ancho / 2])
    return np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(np.max(q, axis=1), 0)


# Fracciones de la orden que se prueban, de mayor a menor: 1, 3/4, 1/2, ..., 1/32.
ESCALAS_FRENADO = (1.0, 0.75) + tuple(0.5 ** k for k in range(1, 6))


def regular_traslacion(velocidad, puntos, largo, ancho, margen, velocidad_maxima,
                       mapa_seguro, horizonte=0.8, paso=0.05, reaccion=0.15,
                       desaceleracion=0.4, velocidad_pasillo=0.10):
    """
    Escala una orden sin desviarla y valida huella, mapa y distancia de frenado.

    mapa_seguro recibe desplazamientos (dx, dy) en base_footprint y debe aceptar
    el segmento completo desde el origen. El barrido usa pasos espaciales de
    hasta 5 mm para no saltar un obstáculo entre dos muestras.
    """
    v = np.asarray(velocidad, dtype=float)
    if (v.shape != (2,) or not np.all(np.isfinite(v)) or puntos is None
            or velocidad_maxima <= 0 or desaceleracion <= 0):
        return (0.0, 0.0), 'datos_invalidos'
    puntos = np.asarray(puntos, dtype=float).reshape(-1, 2)
    if not np.all(np.isfinite(puntos)):
        return (0.0, 0.0), 'datos_invalidos'
    modulo = np.linalg.norm(v)
    if modulo < 1e-6:
        return (0.0, 0.0), 'sin_orden'
    if modulo > velocidad_maxima:
        v *= velocidad_maxima / modulo
    separacion = float(np.min(distancia_huella(puntos, largo, ancho))) if len(puntos) else math.inf
    # No generar la repulsión lateral que cancela o deforma el seguimiento.
    limite = velocidad_maxima
    if separacion < margen + 0.10:
        limite = min(limite, velocidad_pasillo)
    modulo = np.linalg.norm(v)
    if modulo > limite:
        v *= limite / modulo
    for escala in ESCALAS_FRENADO:
        candidata = v * escala
        rapidez = float(np.linalg.norm(candidata))
        duracion = max(horizonte, reaccion + rapidez / (2 * desaceleracion))
        muestras = max(2, math.ceil(duracion / paso), math.ceil(rapidez * duracion / 0.005))
        libre = True
        for t in np.linspace(0.0, duracion, muestras + 1):
            delta = candidata * t
            if not mapa_seguro(float(delta[0]), float(delta[1])):
                libre = False
                break
            # Medio paso espacial cubre el intervalo entre las muestras.
            if len(puntos) and np.any(distancia_huella(puntos - delta, largo, ancho)
                                      < margen + rapidez * duracion / muestras / 2):
                libre = False
                break
        if libre:
            return tuple(map(float, candidata)), 'avance' if escala == 1 else 'velocidad_reducida'
    return (0.0, 0.0), 'obstaculo_o_margen'


def buscar_desvio(velocidad, puntos, largo, ancho, margen, rapidez,
                  angulos=(15, 30, 45, 60, 75, 90), **kwargs):
    """
    Busca la dirección libre más parecida a `velocidad` cuando esa está bloqueada.

    Gira la orden de a `angulos` grados (primero los menores, a cada lado) y la
    valida solo contra el scan, que está en el marco del robot y no depende del
    error de localización: con la pose corrida unos centímetros, el mapa marca
    bloqueado el lado que en realidad está libre. A 90° es un paso de costado.
    Devuelve ((vx, vy), grados) o None.
    """
    v = np.asarray(velocidad, dtype=float)
    modulo = float(np.linalg.norm(v))
    if modulo < 1e-6 or puntos is None:
        return None
    v = v / modulo * min(modulo, rapidez)
    puntos = np.asarray(puntos, dtype=float).reshape(-1, 2)
    if len(puntos):
        # Si ya está más cerca que el margen, vale cualquier dirección que no lo
        # acerque más: si no, ni siquiera podría alejarse de la pared.
        actual = float(np.min(distancia_huella(puntos, largo, ancho)))
        margen = max(0.0, min(margen, actual - 0.005))
    horizonte = kwargs.get('horizonte', 0.8)
    for grados in angulos:
        libres = []
        for signo in (1, -1):
            a = math.radians(grados * signo)
            c, s = math.cos(a), math.sin(a)
            candidata = (c * v[0] - s * v[1], s * v[0] + c * v[1])
            orden, motivo = regular_traslacion(
                candidata, puntos, largo, ancho, margen, rapidez,
                lambda dx, dy: True, velocidad_pasillo=rapidez, **kwargs)
            if motivo == 'avance':
                final = puntos - np.asarray(orden) * horizonte
                holgura = (float(np.min(distancia_huella(final, largo, ancho)))
                           if len(puntos) else math.inf)
                libres.append((holgura, orden, grados * signo))
        if libres:
            # Entre los dos lados, el que más se aleja de lo que ve el scan.
            _, orden, grados_elegidos = max(libres, key=lambda opcion: opcion[0])
            return orden, grados_elegidos
    return None


def celda_libre_cercana(libre, celda, radio_celdas):
    """
    Celda libre más cercana a `celda` dentro de `radio_celdas`, o None.

    Sirve para salir cuando el robot quedó dentro del margen de una pared:
    su propia celda no es transitable aunque el chasis no toque nada.
    """
    fila, col = celda
    alto, ancho = libre.shape
    f0, f1 = max(0, fila - radio_celdas), min(alto, fila + radio_celdas + 1)
    c0, c1 = max(0, col - radio_celdas), min(ancho, col + radio_celdas + 1)
    filas, cols = np.nonzero(libre[f0:f1, c0:c1])
    if not len(filas):
        return None
    distancias = np.hypot(filas + f0 - fila, cols + c0 - col)
    i = int(np.argmin(distancias))
    if distancias[i] > radio_celdas:
        return None
    return int(filas[i] + f0), int(cols[i] + c0)


def celdas_libres_cercanas(libre, celda, radio_celdas, maximo=40):
    """Hasta `maximo` celdas libres dentro de `radio_celdas`, de la más cercana a la más lejana."""
    fila, col = celda
    alto, ancho = libre.shape
    f0, f1 = max(0, fila - radio_celdas), min(alto, fila + radio_celdas + 1)
    c0, c1 = max(0, col - radio_celdas), min(ancho, col + radio_celdas + 1)
    filas, cols = np.nonzero(libre[f0:f1, c0:c1])
    filas, cols = filas + f0, cols + c0
    distancias = np.hypot(filas - fila, cols - col)
    dentro = distancias <= radio_celdas
    orden = np.argsort(distancias[dentro], kind='stable')[:maximo]
    return [(int(f), int(c)) for f, c in zip(filas[dentro][orden], cols[dentro][orden])]


def escape_seguro(puntos, largo, ancho, desplazamiento, minimo=0.01):
    """
    Indica si mover la huella `desplazamiento` (base_footprint) no la acerca al scan.

    Acepta si después del paso la distancia mínima a los impactos queda por
    encima de `minimo` o no empeora.
    """
    puntos = np.asarray(puntos, dtype=float).reshape(-1, 2)
    if not len(puntos):
        return True
    antes = float(np.min(distancia_huella(puntos, largo, ancho)))
    despues = float(np.min(distancia_huella(puntos - np.asarray(desplazamiento), largo, ancho)))
    return despues >= minimo or despues >= antes


def direccion_de_alejamiento(puntos, largo, ancho, paso=0.03, direcciones=16):
    """
    Dirección unitaria (base_footprint) que más aleja la huella del scan.

    Prueba `direcciones` rumbos con un paso de `paso` metros y devuelve el que
    deja más holgura, solo si la mejora; None si ninguno aleja el chasis.
    """
    puntos = np.asarray(puntos, dtype=float).reshape(-1, 2)
    if not len(puntos):
        return None
    actual = float(np.min(distancia_huella(puntos, largo, ancho)))
    mejor, holgura_mejor = None, actual
    for angulo in np.linspace(0.0, 2.0 * math.pi, direcciones, endpoint=False):
        u = np.array([math.cos(angulo), math.sin(angulo)])
        holgura = float(np.min(distancia_huella(puntos - u * paso, largo, ancho)))
        if holgura > holgura_mejor + 1e-4:
            mejor, holgura_mejor = (float(u[0]), float(u[1])), holgura
    return mejor
