"""Percepción RGB-D y memoria de víctimas, sin ROS ni geometría de mundos.

Una región roja es un candidato. Sólo una envolvente con extensión observable
en ambos ejes permite estimar centro; una cara aislada no revela el fondo.
"""
from dataclasses import dataclass, field
import math

import cv2
import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class ParametrosPercepcion:
    tono_rojo_max: int = 8
    saturacion_min: int = 120
    brillo_min: int = 45
    pixeles_min: int = 30
    puntos_min: int = 35
    voxel_m: float = .015
    conexion_m: float = .055
    lado_min_m: float = .09  # Prior de tamaño físico; NO mínimo de fragmento visible.
    lado_prior_oculto_m: float = .09  # Sólo con prior cuadrado explícito; configurable.
    lado_max_m: float = .65
    altura_min_m: float = .07
    altura_max_m: float = .85
    distancia_max_m: float = 3.5
    error_objetivo_m: float = .15
    error_reporte_max_m: float = .15  # El YAML de competencia habilita respaldo hasta .50 m.
    error_sensor_m: float = .025
    error_centro_m: float = .025
    asociacion_m: float = .30

    def __post_init__(self):
        for name, value in vars(self).items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError('Parámetro de percepción inválido: ' + name)
        if (self.tono_rojo_max >= 90 or self.saturacion_min > 255
                or self.brillo_min > 255 or self.lado_min_m >= self.lado_max_m
                or self.lado_prior_oculto_m > self.lado_max_m
                or self.error_reporte_max_m < self.error_objetivo_m):
            raise ValueError('Rangos de percepción inválidos')


def mascara_roja(rgb, p):
    """RGB uint8 -> máscara; incluye los dos extremos del tono circular."""
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim < 2 or rgb.shape[-1] != 3:
        raise ValueError('RGB debe tener tres canales uint8')
    shape = rgb.shape[:-1]
    hsv = cv2.cvtColor(np.ascontiguousarray(rgb.reshape(-1, 1, 3)), cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, 0].T
    return (((h <= p.tono_rojo_max) | (h >= 180-p.tono_rojo_max))
            & (s >= p.saturacion_min) & (v >= p.brillo_min)).reshape(shape)


def hay_region_roja(rgb, p):
    mask = mascara_roja(rgb, p).astype(np.uint8)
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return bool(len(stats) > 1 and np.any(stats[1:, cv2.CC_STAT_AREA] >= p.pixeles_min))


def leer_imagen(msg):
    if msg.encoding not in ('rgb8', 'bgr8'):
        raise ValueError('Imagen RGB/BGR de 8 bits requerida')
    if msg.width <= 0 or msg.height <= 0 or msg.step < msg.width*3:
        raise ValueError('Dimensiones de imagen inválidas')
    if len(msg.data) < msg.step*msg.height:
        raise ValueError('Imagen truncada')
    image = np.ndarray((msg.height, msg.width, 3), dtype=np.uint8,
                       buffer=msg.data, strides=(msg.step, 3, 1))
    return image if msg.encoding == 'rgb8' else image[:, :, ::-1]


def leer_nube(msg):
    """Respeta offsets, padding, organización, endianness y RGB FLOAT32/UINT32."""
    fields = {f.name: f for f in msg.fields}
    color = 'rgb' if 'rgb' in fields else 'rgba'
    if not all(n in fields for n in ('x', 'y', 'z', color)):
        raise ValueError('Nube sin XYZ/RGB')
    names = ['x', 'y', 'z', color]
    for n in names:
        f = fields[n]
        if f.count != 1 or f.offset < 0 or f.offset+4 > msg.point_step:
            raise ValueError('Campo de nube inválido')
        if f.datatype not in ((7, 6) if n == color else (7,)):
            raise ValueError('Tipos de nube no compatibles')
    if (msg.width <= 0 or msg.height <= 0 or msg.row_step < msg.point_step*msg.width
            or len(msg.data) < msg.row_step*msg.height):
        raise ValueError('Nube truncada o vacía')
    endian = '>' if msg.is_bigendian else '<'
    dtype = np.dtype(dict(names=names, formats=[endian+'f4']*3+[endian+'u4'],
                         offsets=[fields[n].offset for n in names], itemsize=msg.point_step))
    rows = np.ndarray((msg.height, msg.width), dtype=dtype, buffer=msg.data,
                      strides=(msg.row_step, msg.point_step)).reshape(-1)
    xyz = np.column_stack([rows[n] for n in names[:3]])
    packed = rows[color].astype(np.uint32)
    rgb = np.column_stack([(packed >> 16) & 255, (packed >> 8) & 255, packed & 255])
    return xyz, rgb.astype(np.uint8)


def transformar(xyz, translation, quaternion):
    q = np.asarray(quaternion, dtype=float)
    t = np.asarray(translation, dtype=float)
    if q.shape != (4,) or t.shape != (3,) or not np.isfinite(q).all() or not np.isfinite(t).all():
        raise ValueError('Transformación inválida')
    norm = np.linalg.norm(q)
    if abs(norm-1.) > .01:
        raise ValueError('Cuaternión no unitario')
    q = q/norm
    vector = q[:3]
    # Fórmula de rotación por cuaternión; preserva el marco real de la nube.
    uv = 2*np.cross(vector, xyz)
    return xyz + q[3]*uv + np.cross(vector, uv) + t


@dataclass(frozen=True)
class Deteccion:
    x: float
    y: float
    error_estimado_m: float
    lados_m: tuple
    puntos: int
    superficie: tuple = None


def _grupos(points, p):
    _, indices = np.unique(np.floor(points/p.voxel_m).astype(np.int64), axis=0,
                           return_index=True)
    reduced = points[np.sort(indices)]
    tree = cKDTree(reduced)
    unseen = set(range(len(reduced)))
    while unseen:
        stack = [unseen.pop()]
        component = []
        while stack:
            i = stack.pop()
            component.append(i)
            neighbors = unseen.intersection(tree.query_ball_point(reduced[i], p.conexion_m))
            unseen.difference_update(neighbors)
            stack.extend(neighbors)
        if len(component) >= p.puntos_min:
            yield reduced[component]


def _envolvente_caras(xy):
    """Ajusta bordes ortogonales, no sólo área mínima de una silueta en L.

    El rectángulo de área mínima puede empatar con otro apoyado en la diagonal
    de dos caras y desplazar su centro. Elegimos el apoyo con menor residuo de
    superficie observado; búsqueda angular uniforme, independiente del mapa.
    """
    angles = np.arange(180)*math.pi/360
    axes = np.stack((np.cos(angles), np.sin(angles)), axis=1)
    perpendicular = np.stack((-np.sin(angles), np.cos(angles)), axis=1)
    u, v = xy @ axes.T, xy @ perpendicular.T
    ulo, uhi = np.percentile(u, [.5, 99.5], axis=0)
    vlo, vhi = np.percentile(v, [.5, 99.5], axis=0)
    residual = np.minimum.reduce([abs(u-ulo), abs(u-uhi), abs(v-vlo), abs(v-vhi)])
    scores = np.percentile(residual, 80, axis=0)
    i = int(np.argmin(scores))
    center = axes[i]*(ulo[i]+uhi[i])/2 + perpendicular[i]*(vlo[i]+vhi[i])/2
    return center, (uhi[i]-ulo[i], vhi[i]-vlo[i]), float(scores[i])


def detectar_cajas(xyz, rgb, translation, quaternion, std_xy, std_yaw, p, grilla=None,
                   diagnostico=None, asumir_base_cuadrada=False, observacion_rgb=None):
    """Devuelve centros observables; no supone tamaño fijo de caja.

    El radio estimado es una política conservadora de admisión, no una garantía
    estadística calibrada. Los errores sistemáticos se miden fuera de estrategia.
    """
    if (not all(math.isfinite(v) and v >= 0 for v in (std_xy, std_yaw))
            or len(xyz) != len(rgb)):
        raise ValueError('Observación/covarianza inválida')
    valid = (np.isfinite(xyz).all(axis=1) & (np.linalg.norm(xyz, axis=1) <= p.distancia_max_m)
             & mascara_roja(rgb, p))
    points = transformar(xyz[valid], translation, quaternion)
    if diagnostico is not None:
        diagnostico.update(puntos_rojos=len(points), vistas_incompletas=[],
                           incertidumbre_alta=0, grupos_evaluados=0, rechazos=[])
    def rechazo(motivo, group, **extra):
        if diagnostico is not None and len(diagnostico['rechazos']) < 8:
            diagnostico['rechazos'].append(dict(motivo=motivo, puntos=len(group), **extra))
    detections = []
    for group in _grupos(points, p):
        if diagnostico is not None:
            diagnostico['grupos_evaluados'] += 1
        # Ningún rectángulo de lados <= máximo puede tener una proyección
        # mayor que su diagonal. Evita ajustes caros sobre paredes completas.
        if np.max(np.ptp(group[:, :2], axis=0)) > math.sqrt(2)*p.lado_max_m:
            rechazo('EXTENSION_XY', group, extension_m=np.ptp(group[:, :2], axis=0).tolist())
            continue
        height = float(np.ptp(group[:, 2]))
        if not p.altura_min_m <= height <= p.altura_max_m:
            rechazo('ALTURA', group, altura_m=height)
            continue
        (x, y), sides, residual = _envolvente_caras(group[:, :2])
        if max(sides) > p.lado_max_m:
            rechazo('LADO_MAXIMO', group, lados_m=list(sides))
            continue
        inferred=False
        if min(sides) < p.lado_min_m:
            from equipo_jar.percepcion_cara import inferir_base_cuadrada
            inference_info={}
            hypothesis=(inferir_base_cuadrada(group,translation[:2],p,observacion_rgb,inference_info)
                        if asumir_base_cuadrada else None)
            if hypothesis is not None:
                (x,y),sides,residual=hypothesis
                inferred=True
                if diagnostico is not None:
                    diagnostico.setdefault('inferencias_base_cuadrada',[]).append(
                        dict(centro=[float(x),float(y)],lado_observado_m=inference_info['ancho_visible_m'],
                             lado_estimado_m=inference_info['lado_estimado_m'],
                             usando_prior_tamano=inference_info['usando_prior_tamano'],
                             supuesto='base_cuadrada',ancho_desde_rgb=observacion_rgb is not None))
            else:
                rechazo('EXTENSION_OCULTA_O_PEQUENA', group, lados_m=list(sides))
                # Una superficie visible no es todavía el centro de una víctima.
                # Se expone como candidata a otra vista, nunca como /victimas.
                if diagnostico is not None:
                    _, axes = np.linalg.eigh(np.cov(group[:, :2].T))
                    normal = axes[:, 0]
                    surface = np.median(group[:, :2], axis=0)
                    if np.dot(normal, np.asarray(translation[:2])-surface) < 0:
                        normal = -normal
                    diagnostico['vistas_incompletas'].append(dict(
                        superficie_x=float(surface[0]), superficie_y=float(surface[1]),
                        normal_x=float(normal[0]), normal_y=float(normal[1]),
                        motivo='PROFUNDIDAD_DE_CAJA_NO_OBSERVABLE'))
                    from equipo_jar.percepcion_cara import realimentacion_cara_rgb
                    diagnostico['vistas_incompletas'][-1].update(
                        realimentacion_cara_rgb(group,observacion_rgb))
                continue

        if residual > p.error_sensor_m:
            rechazo('RESIDUO', group, residuo_m=residual, lados_m=list(sides))
            continue
        if grilla is not None:
            cell = grilla.a_celda(x, y)
            if cell is None or grilla.ocupada[cell]:
                rechazo('CENTRO_EN_MAPA_OCUPADO', group, centro=[float(x), float(y)])
                continue
        distance = math.hypot(x-translation[0], y-translation[1])
        error = 2.45*math.sqrt(std_xy**2 + (distance*std_yaw)**2
                               + p.error_sensor_m**2 + p.error_centro_m**2)
        if error <= p.error_reporte_max_m:
            surface=np.median(group[:,:2],axis=0)
            detections.append(Deteccion(x, y, error, tuple(sides), len(group),
                                       tuple(float(a) for a in surface)))
        elif diagnostico is not None:
            diagnostico['incertidumbre_alta'] += 1
            rechazo('INCERTIDUMBRE', group, error_m=error,centro=[float(x),float(y)],
                    modelo='base_cuadrada' if inferred else 'dos_caras')
    return detections


@dataclass
class Victima:
    identificador: int
    centro: np.ndarray
    primera_s: float
    ultima_s: float
    error_m: float
    observaciones: list = field(default_factory=list)
    reportada: bool = False
    superficie: np.ndarray = None
    reportada_por_respaldo: bool = False


class RegistroVictimas:
    """Asociación espacial y presupuesto único de publicaciones por corrida."""
    def __init__(self, cantidad, parametros):
        if isinstance(cantidad, bool) or not isinstance(cantidad, int) or cantidad < 0:
            raise ValueError('N inválido')
        self.cantidad, self.p = cantidad, parametros
        self.victimas = []
        self.reportes = 0
        self.ultima_estampa = -math.inf

    def observar(self, detecciones, estampa):
        if not math.isfinite(estampa) or estampa <= self.ultima_estampa:
            return []
        self.ultima_estampa = estampa
        nuevas, usadas = [], set()
        for d in detecciones:
            xy = np.array([d.x, d.y])
            if not np.isfinite(xy).all() or not 0 < d.error_estimado_m <= self.p.error_reporte_max_m:
                continue
            surface=None if d.superficie is None else np.asarray(d.superficie,float)
            if surface is not None and (surface.shape!=(2,) or not np.isfinite(surface).all()):
                surface=None
            matches = [v for v in self.victimas
                       if np.linalg.norm(xy-v.centro) <= max(self.p.asociacion_m,
                              min(d.error_estimado_m,self.p.error_objetivo_m)
                              +min(v.error_m,self.p.error_objetivo_m))
                       or (surface is not None and v.superficie is not None and
                           np.linalg.norm(surface-v.superficie)<=self.p.asociacion_m)]
            if len(matches) > 1:
                continue  # no adivinar identidad entre dos cajas compatibles
            if matches:
                v = matches[0]
                if v.identificador in usadas:
                    continue  # dos regiones no cuentan como dos vistas
            else:
                v = Victima(len(self.victimas)+1, xy.copy(), estampa, estampa,
                            d.error_estimado_m, superficie=surface)
                self.victimas.append(v)
            usadas.add(v.identificador)
            if v.reportada:
                continue
            # Guardar la mejor posición; calidad preferida publica sin espera.
            v.ultima_s = estampa
            if not v.observaciones or d.error_estimado_m < v.error_m:
                v.observaciones = [xy]
                v.centro = xy
                v.error_m = d.error_estimado_m
            if v.error_m <= self.p.error_objetivo_m and self.reportes < self.cantidad:
                # Reserva el intento antes de publicar: nunca reintentar a ciegas.
                v.reportada = True
                self.reportes += 1
                nuevas.append(v)
        return nuevas

    def pendientes(self):
        if self.reportes >= self.cantidad:
            return []
        return sorted((v for v in self.victimas if not v.reportada),key=lambda v:v.error_m)

    def reportar_pendientes(self):
        """Publicar mejores centros ya localizados, sin inventar posiciones."""
        nuevas=[]
        for v in self.pendientes():
            if self.reportes >= self.cantidad:
                break
            v.reportada=True
            v.reportada_por_respaldo=v.error_m>self.p.error_objetivo_m
            self.reportes+=1
            nuevas.append(v)
        return nuevas
