"""Agrupa obstáculos ausentes del mapa y evita inspeccionarlos repetidamente."""

import json
import math

import numpy as np

COLORES_OBJETOS = ('rojo', 'naranja', 'amarillo', 'verde', 'cian', 'azul',
                   'violeta', 'magenta', 'neutro', 'desconocido')


def grupos_lidar(xs, ys, validos, libre, dist_pared, res, ox, oy, margen):
    """Índices de superficies nuevas; la forma y el color no intervienen."""
    validos = validos & np.isfinite(xs) & np.isfinite(ys)
    columnas = np.floor((np.where(validos, xs, ox) - ox) / res).astype(int)
    filas = np.floor((np.where(validos, ys, oy) - oy) / res).astype(int)
    alto, ancho = libre.shape
    validos &= (filas >= 0) & (filas < alto) & (columnas >= 0) & (columnas < ancho)
    indices = np.flatnonzero(validos)
    validos[indices] &= (libre[filas[indices], columnas[indices]]
                         & (dist_pared[filas[indices], columnas[indices]] >= margen))
    indices = np.flatnonzero(validos)
    saltos = ((np.diff(indices) > 1)
              | (np.hypot(np.diff(xs[indices]), np.diff(ys[indices])) > 0.20))
    return [g for g in np.split(indices, np.flatnonzero(saltos) + 1) if len(g) >= 3]


def centro_superficie_lidar(xs, ys, sensor, profundidad, tolerancia=0.015):
    """Estima el centro desde las caras observadas, sin correrlo en la dirección del rayo."""
    puntos = np.column_stack((xs, ys))
    mediana = np.median(puntos, axis=0)
    rayo = mediana - np.asarray(sensor[:2])
    norma = float(np.linalg.norm(rayo))
    if profundidad <= 0 or norma < 1e-6 or len(puntos) < 3:
        return tuple(float(v) for v in mediana)

    def recta(pts):
        media = pts.mean(axis=0)
        _, vectores = np.linalg.eigh((pts - media).T @ (pts - media))
        normal, tangente = vectores[:, 0], vectores[:, 1]
        if np.dot(normal, media - sensor[:2]) < 0:
            normal = -normal
        residuo = (pts - media) @ normal
        largo = (pts - media) @ tangente
        cara = media + tangente * (largo.min() + largo.max()) / 2
        return normal, float(normal @ media), cara, float(np.max(np.abs(residuo))), float(np.ptp(largo))

    normal, b, cara, error, ancho = recta(puntos)
    if error <= tolerancia and ancho >= 0.04 and normal @ rayo / norma > 0.25:
        return tuple(float(v) for v in cara + profundidad * normal)

    mejor = None
    # La esquina está lejos de la cuerda entre extremos. Probar sus vecinos evita
    # ajustar dos rectas por cada rayo (costoso con objetos cercanos y muchos puntos).
    cuerda = puntos[-1] - puntos[0]
    longitud = float(np.linalg.norm(cuerda))
    if longitud > 1e-6:
        distancia_cuerda = np.abs((puntos - puntos[0]) @ np.array([-cuerda[1], cuerda[0]])) / longitud
        esquina = int(np.argmax(distancia_cuerda))
        candidatos = range(max(2, esquina - 3), min(len(puntos) - 2, esquina + 4))
    else:
        candidatos = ()
    for k in candidatos:
        n1, b1, _, e1, a1 = recta(puntos[:k + 1])
        n2, b2, _, e2, a2 = recta(puntos[k:])
        if (min(a1, a2) < 0.04 or max(e1, e2) > tolerancia
                or abs(float(n1 @ n2)) > 0.25):
            continue
        centro = np.linalg.solve(np.vstack((n1, n2)), [b1 + profundidad, b2 + profundidad])
        if np.linalg.norm(centro - mediana) > 2 * profundidad + tolerancia:
            continue
        if mejor is None or e1 + e2 < mejor[0]:
            mejor = (e1 + e2, centro)
    centro = mejor[1] if mejor is not None else mediana + profundidad * rayo / norma
    return tuple(float(v) for v in centro)


def objetos_del_scan(scan, sensor, grilla, dist_pared, margen=0.12, alcance=2.5,
                     profundidad=0.0):
    """Rachas de al menos tres rayos en piso libre lejos de paredes conocidas.

    `sensor` es la pose del LiDAR en map, al instante del barrido. No depende
    de la resolución ni de la frecuencia de publicación del costmap de Nav2.
    """
    rangos = np.asarray(scan.ranges, dtype=float)
    angulos = sensor[2] + scan.angle_min + np.arange(len(rangos)) * scan.angle_increment
    validos = (np.isfinite(rangos) & (rangos >= max(0.25, scan.range_min))
               & (rangos <= min(alcance, scan.range_max)))
    seguros = np.where(validos, rangos, 0.0)
    xs = sensor[0] + seguros * np.cos(angulos)
    ys = sensor[1] + seguros * np.sin(angulos)
    grupos = grupos_lidar(xs, ys, validos, grilla.libre, dist_pared,
                          grilla.res, grilla.ox, grilla.oy, margen)
    return [centro_superficie_lidar(xs[g], ys[g], sensor, profundidad) for g in grupos]


def leer_observaciones(datos, frame):
    """Contrato interno /deteccion/observaciones (String JSON, versión 1).

    Cada mensaje corresponde a una imagen realmente procesada, incluso si
    no contiene objetos. La posición está en map y el error angular se mide
    desde el eje óptico de la cámara. Rechaza mensajes incompletos o no finitos.
    """
    try:
        mensaje = json.loads(datos)
        if (mensaje['version'] != 1 or mensaje['frame'] != frame
                or not isinstance(mensaje['stamp_ns'], int) or mensaje['stamp_ns'] <= 0
                or not isinstance(mensaje['objetos'], list)):
            return None
        for o in mensaje['objetos']:
            if (o['color'] not in COLORES_OBJETOS or o['tipo'] not in ('objeto_nuevo', 'objeto', 'caja')
                    or not isinstance(o['id'], str) or not isinstance(o['n'], int) or o['n'] < 1
                    or any(not isinstance(o[k], bool) for k in ('completo', 'confirmado', 'reportado', 'nuevo'))
                    or not all(math.isfinite(o[k]) for k in ('x', 'y', 'distancia', 'error_angular'))
                    or o['distancia'] <= 0):
                return None
        return mensaje
    except (ValueError, TypeError, KeyError, AttributeError):
        return None


def centros_obstaculos(mascara, origen, resolucion, minimo_celdas=2):
    """Centros de componentes conexas de celdas ocupadas, en metros."""
    filas, columnas = np.nonzero(mascara)
    pendientes = set(zip(columnas.tolist(), filas.tolist()))
    centros = []
    while pendientes:
        inicio = pendientes.pop()
        grupo = [inicio]
        cola = [inicio]
        while cola:
            c, f = cola.pop()
            for dc in (-1, 0, 1):
                for df in (-1, 0, 1):
                    vecino = (c + dc, f + df)
                    if vecino in pendientes:
                        pendientes.remove(vecino)
                        cola.append(vecino)
                        grupo.append(vecino)
        if len(grupo) >= minimo_celdas:
            centros.append((origen[0] + resolucion * (sum(c for c, _ in grupo) / len(grupo) + 0.5),
                            origen[1] + resolucion * (sum(f for _, f in grupo) / len(grupo) + 0.5)))
    return centros


class ObjetosFueraMapa:
    """Fusiona lecturas y conserva las inspecciones cerradas durante toda la misión."""

    def __init__(self, confirmaciones=3, radio_asociacion=0.30, caducidad_s=2.0):
        self.confirmaciones = confirmaciones
        self.radio_asociacion = radio_asociacion
        self.caducidad_s = caducidad_s
        self.rastros = []
        self.revisados = []
        self.visuales = []
        self.ids_revisados = set()

    def actualizar_vision(self, objetos, ahora):
        self.visuales = [o for o in self.visuales if ahora - o['ultimo'] <= 4.0]
        for o in objetos:
            previo = next((v for v in self.visuales if v['id'] == o['id']), None)
            if previo is not None:
                self.visuales.remove(previo)
            if o['id'] in self.ids_revisados:
                continue
            if o['reportado']:
                punto = (o['x'], o['y'])
                self.finalizar(punto, ahora, True, id_visual=o['id'])
                continue
            if o['n'] < 2:
                continue
            punto = (o['x'], o['y'])
            self.visuales.append(dict(o, punto=punto, ultimo=ahora))

    def disponible(self, punto, ahora):
        return all(math.dist(punto, p) >= 0.45 for p in self.revisados)

    def actualizar(self, centros, ahora):
        self.rastros = [r for r in self.rastros if ahora - r['ultimo'] <= self.caducidad_s]
        usados = set()
        for punto in centros:
            opciones = [(math.dist(punto, r['punto']), i) for i, r in enumerate(self.rastros)
                        if i not in usados and math.dist(punto, r['punto']) < self.radio_asociacion]
            if opciones:
                _, i = min(opciones)
                r = self.rastros[i]
                r['punto'] = punto
                r['ultimo'] = ahora
                r['n'] += 1
                usados.add(i)
            else:
                self.rastros.append(dict(punto=punto, primero=ahora, ultimo=ahora, n=1))
                usados.add(len(self.rastros) - 1)

    def confirmados(self, ahora):
        # La cámara aporta atributos; nunca sustituye la persistencia de /scan.
        return [r['punto'] for r in self.rastros
                if (r['n'] >= self.confirmaciones and ahora - r['primero'] >= 0.4
                    and ahora - r['ultimo'] <= self.caducidad_s
                    and self.disponible(r['punto'], ahora))]

    def mas_cercano(self, punto, ahora):
        opciones = [(math.dist(punto, p), p) for p in self.confirmados(ahora)
                    if math.dist(punto, p) < self.radio_asociacion]
        return min(opciones)[1] if opciones else None

    def finalizar(self, punto, ahora, observado, aliases=(), id_visual=None):
        """Cierra el intento aunque la cámara no confirme: el LiDAR ya lo localizó."""
        for p in (punto, *aliases):
            p = tuple(p)
            if all(math.dist(p, previo) >= 0.10 for previo in self.revisados):
                self.revisados.append(p)
        if id_visual is not None:
            self.ids_revisados.add(id_visual)
