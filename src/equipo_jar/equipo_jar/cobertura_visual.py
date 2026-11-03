"""Cobertura geométrica de vistas de cámara; sin ROS ni coordenadas de mundos.

La máscara predicha sirve para elegir vistas; sólo una observación válida puede
acreditar cobertura. Esto estima visibilidad horizontal, no garantiza detección
de víctimas: altura, oclusiones 3D y calidad RGB-D requieren validación adicional.
La pose recibida debe ser la de la cámara, sincronizada con la observación.
"""
from dataclasses import dataclass
from collections import OrderedDict
import hashlib
import math

import numpy as np


# Rayos geométricos independientes de rumbo y de qué muestras ya se vieron.
# Caché acotada por proceso; clave por contenido para invalidar mapas mutados.
_VISIBILIDAD = OrderedDict()
_MAX_VISIBILIDAD = 4096


@dataclass(frozen=True)
class ParametrosCobertura:
    alcance_min_m: float = .10
    alcance_max_m: float = 2.5
    campo_horizontal_rad: float = math.radians(55)
    paso_muestreo_m: float = .20
    angulo_inspeccion_pared_rad: float = math.radians(45)

    def __post_init__(self):
        values = (self.alcance_min_m, self.alcance_max_m,
                  self.campo_horizontal_rad, self.paso_muestreo_m, self.angulo_inspeccion_pared_rad)
        if (not all(math.isfinite(v) for v in values)
                or not 0 <= self.alcance_min_m < self.alcance_max_m
                or not 0 < self.campo_horizontal_rad < 2 * math.pi
                or self.paso_muestreo_m <= 0
                or not 0 < self.angulo_inspeccion_pared_rad <= math.pi/2):
            raise ValueError('Parámetros de cobertura inválidos')


def linea_libre(grilla, origen, destino, ocupada, permitir_final_ocupado=False):
    """Recorre todas las celdas del rayo; en esquinas revisa ambos lados."""
    a, b = grilla.a_celda(*origen), grilla.a_celda(*destino)
    if (a is None or b is None or ocupada[a]
            or (ocupada[b] and not permitir_final_ocupado)):
        return False
    u, v = grilla.a_local(*origen)
    w, z = grilla.a_local(*destino)
    dx, dy = w - u, z - v
    f, c = a
    sx = 1 if dx > 0 else -1
    sy = 1 if dy > 0 else -1
    delta_x = grilla.resolucion / abs(dx) if dx else math.inf
    delta_y = grilla.resolucion / abs(dy) if dy else math.inf
    tx = (((c + (dx > 0)) * grilla.resolucion - u) / dx
          if dx else math.inf)
    ty = (((f + (dy > 0)) * grilla.resolucion - v) / dy
          if dy else math.inf)

    def bloqueada(row, col):
        return not (0 <= row < grilla.alto and 0 <= col < grilla.ancho) or ocupada[row, col]

    while (f, c) != b:
        if abs(tx - ty) < 1e-12:
            if bloqueada(f, c + sx) or bloqueada(f + sy, c):
                return False
            c, f = c + sx, f + sy
            tx, ty = tx + delta_x, ty + delta_y
        elif tx < ty:
            c, tx = c + sx, tx + delta_x
        else:
            f, ty = f + sy, ty + delta_y
        if bloqueada(f, c) and not (permitir_final_ocupado and (f, c) == b):
            return False
    return True


def lineas_libres(grilla, origen, destinos, ocupada, permitir_final):
    """Cruces exactos de bordes en lotes; sin muestrear ni saltar celdas."""
    n = len(destinos)
    a = grilla.a_celda(*origen)
    if a is None or ocupada[a] or n == 0:
        return np.zeros(n, dtype=bool)
    u, v = np.asarray(grilla.a_local(*origen))/grilla.resolucion
    theta = grilla.origen_yaw
    delta = destinos - (grilla.origen_x, grilla.origen_y)
    w = (math.cos(theta)*delta[:, 0]+math.sin(theta)*delta[:, 1])/grilla.resolucion
    z = (-math.sin(theta)*delta[:, 0]+math.cos(theta)*delta[:, 1])/grilla.resolucion
    bc, bf = np.floor(w).astype(int), np.floor(z).astype(int)
    alive = (bc >= 0) & (bc < grilla.ancho) & (bf >= 0) & (bf < grilla.alto)
    alive &= ~ocupada[np.clip(bf, 0, grilla.alto-1), np.clip(bc, 0, grilla.ancho-1)] | permitir_final
    # Cada cruce vertical/horizontal toca las celdas a ambos lados del borde.
    # En esquinas se incluyen las cuatro: la misma guardia conservadora DDA.
    for horizontal in (False, True):
        base, ends, other, other_ends = (v, z, u, w) if horizontal else (u, w, v, z)
        lo, hi = math.floor(min(base, float(ends.min()))), math.ceil(max(base, float(ends.max())))
        borders = np.arange(lo, hi+1, dtype=float)
        direction = ends-base
        t = np.full((n, len(borders)), np.inf)
        np.divide(borders[None, :]-base, direction[:, None], out=t,
                  where=direction[:, None] != 0)
        crossing = (t >= 0) & (t < 1) & ((t > 0) | (direction[:, None] < 0))
        t = np.where(crossing, t, 0.)
        coordinate = other+(other_ends-other)[:, None]*t
        corner = np.abs(coordinate-np.rint(coordinate)) < 1e-10
        side = np.floor(np.where(corner, np.rint(coordinate), coordinate)).astype(int)
        for shift in (0, -1):
            for adjacent in (0, -1):
                row = side+shift
                col = np.broadcast_to(borders.astype(int)+adjacent, row.shape)
                if horizontal:
                    row, col = col, row
                valid = crossing & ((shift == 0) | corner)
                inside = (row >= 0) & (row < grilla.alto) & (col >= 0) & (col < grilla.ancho)
                blocked = ~inside | ocupada[np.clip(row, 0, grilla.alto-1),
                                             np.clip(col, 0, grilla.ancho-1)]
                endpoint = (row == bf[:, None]) & (col == bc[:, None]) & permitir_final[:, None]
                alive &= ~np.any(valid & blocked & ~endpoint, axis=1)
    return alive


class CoberturaVisual:
    def __init__(self, grilla, parametros=None):
        self.grilla = grilla
        self.p = parametros or ParametrosCobertura()
        salto = max(1, math.ceil(self.p.paso_muestreo_m / grilla.resolucion))
        suelo = {(f, c) for f in range(0, grilla.alto, salto)
                 for c in range(0, grilla.ancho, salto) if not grilla.ocupada[f, c]}
        # Buscar objetos exige inspeccionar superficies, no sólo piso libre.
        # Guardar ambas caras de paredes delgadas dentro de un mismo bloque.
        libres = np.pad(~grilla.ocupada, 1)
        bordes = {}
        for df, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            vecinos = libres[1+df:1+df+grilla.alto, 1+dc:1+dc+grilla.ancho]
            for f, c in np.argwhere(grilla.ocupada & vecinos):
                bordes.setdefault((f//salto, c//salto, df, dc), (int(f), int(c)))
        # Cada cara se conserva por separado: mirar la pared de canto
        # no inspecciona sus huecos ni acredita su cara opuesta.
        samples = [(cell, (0., 0.), False) for cell in sorted(suelo)]
        c, s = math.cos(grilla.origen_yaw), math.sin(grilla.origen_yaw)
        for key, cell in sorted(bordes.items()):
            df, dc = key[-2:]
            samples.append((cell, (c*dc-s*df, s*dc+c*df), True))
        self.celdas = [cell for cell, _, _ in samples]
        self.normales = np.asarray([normal for _, normal, _ in samples]).reshape(-1, 2)
        self.objetivos_pared = np.asarray([pared for _, _, pared in samples], dtype=bool)
        self.puntos = np.array([grilla.a_mundo(*cell) for cell in self.celdas],
                               dtype=float).reshape(-1, 2)
        self.observada = np.zeros(len(self.celdas), dtype=bool)
        # Un retorno no explicado debe ser inspeccionado por cámara.
        self.solo_camara = np.zeros(len(self.celdas), dtype=bool)
        self.vista_camara = np.zeros(len(self.celdas), dtype=bool)
        self.ultima_estampa = None
        self.ultima_visible = None  # Sólo diagnóstico: máscara ya calculada por registrar.

    def predecir(self, pose_camara, extra_ocupadas=None, solo_pendientes=False):
        """Máscara potencial: no altera lo observado ni implica ruta segura."""
        x, y, yaw = pose_camara
        if not all(math.isfinite(v) for v in pose_camara):
            raise ValueError('Pose de cámara inválida')
        occupied = self.grilla.ocupada
        if extra_ocupadas is not None:
            extra = np.asarray(extra_ocupadas, dtype=bool)
            if extra.shape != occupied.shape:
                raise ValueError('Capa de oclusiones incompatible')
            occupied = occupied | extra
        delta = self.puntos - (x, y)
        distance = np.hypot(delta[:, 0], delta[:, 1])
        bearing = np.arctan2(delta[:, 1], delta[:, 0]) - yaw
        bearing = np.arctan2(np.sin(bearing), np.cos(bearing))
        visible = ((distance >= self.p.alcance_min_m)
                   & (distance <= self.p.alcance_max_m)
                   & (np.abs(bearing) <= self.p.campo_horizontal_rad / 2))
        incidencia = np.sum(-delta*self.normales, axis=1)
        visible &= (~self.objetivos_pared | (incidencia >=
            distance*math.cos(self.p.angulo_inspeccion_pared_rad)-1e-12))
        if solo_pendientes:
            visible &= ~self.observada
        # El mismo lugar suele evaluarse con ocho o 32 rumbos. La oclusión
        # no cambia al orientar la cámara; calcularla una vez para todos ellos.
        key = (*self._firma_visibilidad(occupied), x, y)
        cached = _VISIBILIDAD.get(key)
        if cached is None:
            cached = (np.zeros(len(self.puntos), bool), np.zeros(len(self.puntos), bool))
            _VISIBILIDAD[key] = cached
            if len(_VISIBILIDAD) > _MAX_VISIBILIDAD:
                _VISIBILIDAD.popitem(last=False)
        else:
            _VISIBILIDAD.move_to_end(key)
        known, spatial = cached
        indices = np.flatnonzero(visible & ~known)
        if len(indices):
            spatial[indices] = lineas_libres(self.grilla, (x, y), self.puntos[indices],
                                            occupied, self.objetivos_pared[indices])
            known[indices] = True
        visible &= spatial
        return visible

    def registrar(self, pose_camara, estampa_s, observacion_valida, extra_ocupadas=None):
        """No acredita una visita si faltan cámara/TF/localización válidos.

        El llamador valida frescura y sincronía; se rechazan duplicados o atrasos.
        """
        if (not observacion_valida or not math.isfinite(estampa_s)
                or (self.ultima_estampa is not None and estampa_s <= self.ultima_estampa)):
            return 0
        visible = self.predecir(pose_camara, extra_ocupadas)
        self.ultima_visible = visible
        nuevas = int(np.count_nonzero(visible & ~self.observada))
        self.observada |= visible
        self.solo_camara &= ~visible
        self.vista_camara |= visible
        self.ultima_estampa = estampa_s
        return nuevas

    def registrar_lidar(self, origen, puntos, explicados, alcance_m=5.0, radio_m=.06, margen_m=.15,
                        radio_pared_m=.15, radio_objeto_m=.30):
        """Acredita lo que un barrido LiDAR demuestra vacío; no sustituye a la cámara.

        Cada rayo con retorno finito recorrió espacio libre hasta su extremo: una caja
        (más ancha que 2*radio_m) en ese trayecto lo habría cortado antes. Se acreditan
        las muestras de piso a <= radio_m del tramo [origen, extremo - margen_m], hasta
        alcance_m. Si el extremo coincide con una pared del mapa (explicados), también la
        muestra de pared de esa cara orientada al láser a <= radio_pared_m (las muestras de
        pared están cada ~0,2 m; una caja delante habría cortado el rayo antes de la pared). Un extremo no explicado por el
        mapa (posible víctima) reabre su entorno a radio_objeto_m y lo reserva
        para cámara. Una muestra ya observada por cámara no se reabre. Las reservas
        impiden que rayos vecinos declaren vacía una región con un objeto. Sin retorno, nada.
        """
        origen = np.asarray(origen, dtype=float)
        puntos = np.asarray(puntos, dtype=float).reshape((-1, 2))
        explicados = np.asarray(explicados, dtype=bool).reshape(-1)
        if origen.shape != (2,) or not np.isfinite(origen).all() or len(puntos) != len(explicados):
            return 0
        valid = np.isfinite(puntos).all(axis=1)
        puntos, explicados = puntos[valid], explicados[valid]
        if not len(puntos):
            return 0
        delta = puntos-origen
        largo = np.hypot(delta[:, 0], delta[:, 1])
        ok = largo > 1e-6
        puntos, explicados, delta, largo = puntos[ok], explicados[ok], delta[ok], largo[ok]
        if not len(puntos):
            return 0
        objetos = puntos[~explicados & (largo <= alcance_m+radio_objeto_m)]
        cerca = np.hypot(*(self.puntos-origen).T) <= alcance_m+radio_objeto_m
        idx = np.flatnonzero(cerca & ~self.vista_camara)
        if len(objetos) and len(idx):
            distancia = np.hypot(self.puntos[idx, None, 0]-objetos[None, :, 0],
                                 self.puntos[idx, None, 1]-objetos[None, :, 1])
            tocadas = idx[(distancia <= radio_objeto_m).any(axis=1)]
            self.observada[tocadas] = False
            self.solo_camara[tocadas] = True
        direccion = delta/largo[:, None]
        libre = np.clip(np.minimum(largo-margen_m, alcance_m), 0., None)
        pend = np.flatnonzero(~self.observada & ~self.solo_camara)
        if not len(pend):
            return 0
        rel = self.puntos[pend]-origen
        cerca = np.hypot(rel[:, 0], rel[:, 1]) <= alcance_m+radio_m
        pend, rel = pend[cerca], rel[cerca]
        if not len(pend):
            return 0
        t = rel@direccion.T                                    # (muestras, rayos)
        perp = np.abs(rel[:, 0:1]*direccion[None, :, 1]-rel[:, 1:2]*direccion[None, :, 0])
        piso = ~self.objetivos_pared[pend]
        libre_piso = ((t >= 0) & (t <= libre[None, :]) & (perp <= radio_m)).any(axis=1) & piso
        acreditar = libre_piso
        en_alcance = explicados & (largo <= alcance_m)
        if en_alcance.any():
            fin = puntos[en_alcance]
            d_fin = np.hypot(self.puntos[pend][:, None, 0]-fin[None, :, 0],
                             self.puntos[pend][:, None, 1]-fin[None, :, 1])
            mira = (self.normales[pend]*(origen-self.puntos[pend])).sum(axis=1) > 0
            pared = (d_fin <= radio_pared_m).any(axis=1) & ~piso & mira
            acreditar = acreditar | pared
        nuevas = pend[acreditar]
        self.observada[nuevas] = True
        return int(len(nuevas))

    def _firma_visibilidad(self, occupied):
        signature = hashlib.blake2b(digest_size=16)
        signature.update(np.ascontiguousarray(occupied).tobytes())
        signature.update(self.puntos.tobytes())
        return (signature.digest(), occupied.shape, self.grilla.resolucion,
                self.grilla.origen_x, self.grilla.origen_y, self.grilla.origen_yaw,
                self.p.alcance_min_m, self.p.alcance_max_m)

    def cotas_ganancia(self, poses):
        """Cotas vectorizadas; rayos ya conocidos afinan la cota sin repetirlos."""
        poses = np.asarray(poses, float).reshape(-1, 3)
        result = np.zeros(len(poses), int)
        pending = ~self.observada
        points, normals = self.puntos[pending], self.normales[pending]
        walls = self.objetivos_pared[pending]
        signature = self._firma_visibilidad(self.grilla.ocupada)
        for first in range(0, len(poses), 128):
            block = poses[first:first+128]
            delta = points[None, :, :]-block[:, None, :2]
            distance = np.hypot(delta[:, :, 0], delta[:, :, 1])
            projection = (delta[:, :, 0]*np.cos(block[:, 2, None])+
                          delta[:, :, 1]*np.sin(block[:, 2, None]))
            incidence = -np.sum(delta*normals, axis=2)
            possible = ((distance >= self.p.alcance_min_m-1e-12) &
                (distance <= self.p.alcance_max_m+1e-12) &
                (projection >= distance*math.cos(self.p.campo_horizontal_rad/2)-1e-12) &
                (~walls | (incidence >= distance*math.cos(self.p.angulo_inspeccion_pared_rad)-1e-12)))
            for i,(x,y,_) in enumerate(block):
                cached = _VISIBILIDAD.get((*signature, x, y))
                if cached is not None:
                    known, free = cached
                    possible[i] &= ~known[pending] | free[pending]
            result[first:first+len(block)] = np.count_nonzero(possible, axis=1)
        return result

    def cota_ganancia(self, pose_camara):
        return int(self.cotas_ganancia([pose_camara])[0])

    def ganancia(self, pose_camara, extra_ocupadas=None):
        return int(np.count_nonzero(self.predecir(pose_camara, extra_ocupadas,
                                                       solo_pendientes=True)))
