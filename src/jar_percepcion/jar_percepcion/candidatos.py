"""Candidatos a víctima (ADR-0006), sin ROS.

Cualquier objeto que el LiDAR ve y el mapa no tiene, del tamaño de una caja, es un candidato:
puede ser una víctima (caja roja) o un señuelo (cubo de otro color). Lo que hay que decidir
es cuál. Para eso cada candidato se inspecciona: cuando la cámara lo tiene de frente, a una
distancia útil y sin nada en el medio, se mira si en su dirección hay un blob rojo.

Según la distancia a la que el LiDAR ve el objeto (política de detección):

    < alcance_candidato_m                   → candidato (sin_inspeccionar)
    hasta alcance_posible_m, rojo en su lugar → candidato rojo (sin_inspeccionar, sin esperar)
    hasta alcance_posible_m, sin rojo        → posible: no se inspecciona; pasa a candidato
                                               cuando el robot lo ve de cerca
    más lejos                                → nada

De lejos el LiDAR confunde pedazos de pared con cajas (fantasmas); una pared nunca es roja,
así que el rojo de la cámara confirma un objeto lejano sin ir a mirarlo.

    posible → sin_inspeccionar → victima      (votos rojos suficientes)
                               → descartado   (varias vistas claras sin rojo)

Sin esto, la detección dependía de la suerte del recorrido: si el robot pasaba pegado o de
costado a una caja, el piso de alrededor contaba como "visto" pero la caja nunca había estado
entera y de frente en la imagen, y no se detectaba nunca.
"""
import math
from dataclasses import dataclass, field
from typing import List, NamedTuple, Optional, Tuple

import numpy as np


@dataclass
class ConfigCandidatos:
    salto_m: float = 0.10            # entre puntos seguidos del scan: corta el grupo
    puntos_min: int = 3
    largo_max_m: float = 0.45        # más largo que una caja vista en diagonal: pared, no caja
    # Puntos a esto de una pared del mapa son esa pared (el error de AMCL la corre 5-10 cm).
    distancia_pared_m: float = 0.15
    centro_pared_min_m: float = 0.10  # un centro más pegado a una pared (o afuera) no es caja
    fraccion_pared_max: float = 0.5
    profundidad_m: float = 0.125     # de la cara vista al centro de la caja
    radio_union_m: float = 0.35
    distancia_min_m: float = 0.5     # inspección: más cerca, la caja no entra en la imagen
    distancia_max_m: float = 1.6     # de lejos la posición es mala: se vota de cerca
    votos_victima: int = 2
    votos_descarte: int = 4
    # Un descartado revive con más votos rojos: con 2, el rojo de una víctima vecina mal
    # ubicado (~0.3 m) revivió un distractor, gastó un reporte y la misión volvió sin la
    # última víctima.
    votos_revivir: int = 4
    # Un voto "no es rojo" sólo vale con el candidato bien centrado y cerca: no ver un blob
    # rojo no prueba nada si la caja toca el borde de la imagen (esos blobs se descartan) o
    # queda chica de lejos. En práctica, dos cajas rojas se descartaban así.
    centrado_no_rojo: float = 0.5    # fracción del medio FOV
    distancia_min_no_rojo_m: float = 0.7
    distancia_max_no_rojo_m: float = 1.6
    # Fantasmas (maze_4, regresión 3): 6 de 8 candidatos eran pedazos de pared vistos de lejos,
    # en un solo scan, y el explorador volvía a zonas ya recorridas a inspeccionarlos.
    alcance_candidato_m: float = 2.0  # más cerca: candidato
    # Hasta acá: posible, o candidato si la cámara ve rojo en su lugar. Con 2 m nada más, una
    # víctima a 2.9 m justo adelante de la largada (maze_4) no se veía y el robot se iba.
    alcance_posible_m: float = 3.5
    radio_rojo_lejano_m: float = 0.5  # rojo de lejos: la cámara lo ubica ~0.4 m corrido
    # Bordes de oclusión: en cada extremo del grupo, el rayo vecino pega más lejos (algo
    # delante del fondo). Un pedazo de pared sigue con la pared o termina en una esquina.
    salto_oclusion_m: float = 0.15
    bordes_min_cerca: int = 0        # 0: como antes (una caja en un rincón puede no tener)
    bordes_min_lejos: int = 2        # sin rojo, de lejos sólo con los dos bordes
    vistas_min: int = 3              # scans que lo ven antes de mandarlo a inspeccionar
    # Y desde posiciones distintas: un fantasma de geometría cambia con el punto de vista.
    posiciones_min: int = 2
    separacion_posiciones_m: float = 0.3
    separacion_angulo_rad: float = 0.26   # ~15°, visto desde el candidato
    # Evidencia negativa: cerca y con el LiDAR pasando de largo por donde estaría la caja.
    radio_ausencia_m: float = 1.2
    ausencias_borrar: int = 5
    # Dos candidatos de la misma caja: cada cara vista ubica el centro distinto y, si la primera
    # vista cayó a más de radio_union_m, se creaba otro que después se acercaba (laberinto_puertas:
    # la víctima de A contada dos veces, la caja gris de D inspeccionada dos veces). Se unen a
    # menos de distancia_unir_m (dos cajas de 0.25 m que se tocan quedan a >= 0.25), salvo que
    # sus votos se contradigan (rojos en uno, "no rojo" en el otro).
    unir_candidatos: bool = False
    distancia_unir_m: float = 0.25
    # Descarte por mayoría: con un solo rojo falso (de cerca, un reflejo o el rojo de otra caja)
    # el candidato no se descartaba nunca, por más "no rojo" que sumara (laberinto_puertas: rojo=1
    # no_rojo=29). Con esto se descarta si además los "no rojo" son `proporcion_descarte` veces
    # los rojos. Apagado: sólo con rojo == 0.
    descarte_por_mayoria: bool = False
    proporcion_descarte: float = 4.0
    # Para ser víctima tiene que parecer una caja, aunque haya entrado por el rojo de lejos (que
    # se saltea las vistas): visto en vistas_min scans, desde posiciones_min posiciones y con al
    # menos bordes_min_rojo bordes de oclusión alguna vez. Con rojos suficientes y sin esto
    # queda sin inspeccionar hasta que aparezca. laberinto_puertas: la punta de una pared, con
    # rojo de una víctima asomando detrás, se reportó como víctima. Apagado: como antes.
    rojo_requiere_caja: bool = False
    bordes_min_rojo: int = 1


@dataclass
class Candidato:
    id: int
    obs: List[Tuple[float, float]] = field(default_factory=list)
    rojo: int = 0
    no_rojo: int = 0
    estado: str = 'sin_inspeccionar'
    vistas: int = 0                  # scans que lo vieron (obs se recorta, esto no)
    ausencias: int = 0               # scans seguidos que miraron su lugar y no había nada
    desde: List[Tuple[float, float]] = field(default_factory=list)   # posiciones distintas
    rojo_lejano: bool = False        # la cámara vio rojo en su lugar (de lejos)
    rojo_cortado: bool = False       # de frente, rojo cortado por el borde: hay que alejarse
    bordes: int = 0                  # el máximo de bordes de oclusión con que se lo vio

    @property
    def x(self):
        return sum(p[0] for p in self.obs) / len(self.obs)

    @property
    def y(self):
        return sum(p[1] for p in self.obs) / len(self.obs)


class Objeto(NamedTuple):
    """Un objeto del scan: centro en `map`, distancia al LiDAR y bordes de oclusión (0-2, o
    None si no se pasaron los rangos del scan)."""
    x: float
    y: float
    distancia: float
    bordes: Optional[int]


def agrupar_scan(puntos, c=ConfigCandidatos(), indices=None):
    """Grupos de puntos seguidos del scan (en el orden del barrido), cortando en los saltos y,
    si se pasan los `indices` de rayo de cada punto, donde falta un rayo en el medio.
    Devuelve listas de índices (en `puntos`)."""
    p = np.asarray(puntos, dtype=float).reshape(-1, 2)
    if len(p) == 0:
        return []
    salto = np.hypot(*np.diff(p, axis=0).T) > c.salto_m
    if indices is not None:
        salto |= np.diff(np.asarray(indices)) > 1
    cortes = np.flatnonzero(salto) + 1
    return [g for g in np.split(np.arange(len(p)), cortes) if len(g) >= c.puntos_min]


def bordes_de_oclusion(rangos, i0, i1, salto, range_min=0.0):
    """Cuántos extremos del grupo de rayos [i0, i1] tienen al vecino más lejos (o sin eco):
    algo delante del fondo."""
    r = np.asarray(rangos, dtype=float)
    n = len(r)

    def mas_lejos(k, ref):
        v = r[k % n]
        return not np.isfinite(v) or v < range_min or v > r[ref] + salto
    return int(mas_lejos(i0 - 1, i0)) + int(mas_lejos(i1 + 1, i1))


def objetos_del_scan(puntos_mapa, distancia_a_pared, origen, c=ConfigCandidatos(),
                     rangos=None, indices=None, range_min=0.0):
    """Objetos del scan que no son paredes del mapa.

    `distancia_a_pared(puntos)` devuelve, para cada punto, la distancia (m) a la pared del
    mapa más cercana. `origen` es la posición del LiDAR (para correr la cara al centro).
    Con `rangos` (el scan entero) e `indices` (el rayo de cada punto) se cuentan los bordes
    de oclusión.
    """
    p = np.asarray(puntos_mapa, dtype=float).reshape(-1, 2)
    o = np.asarray(origen, dtype=float)
    objetos = []
    for k in agrupar_scan(p, c, indices):
        g = p[k]
        if float(np.hypot(*(g.max(axis=0) - g.min(axis=0)))) > c.largo_max_m:
            continue
        if np.mean(distancia_a_pared(g) < c.distancia_pared_m) > c.fraccion_pared_max:
            continue
        media = g.mean(axis=0)
        v = media - o
        norma = float(np.hypot(*v))
        centro = media + (c.profundidad_m * v / norma if norma > 1e-6 else 0.0)
        if distancia_a_pared(centro.reshape(1, 2))[0] < c.centro_pared_min_m:
            continue                              # sobre una pared o fuera del mapa
        bordes = None
        if rangos is not None and indices is not None:
            bordes = bordes_de_oclusion(rangos, int(indices[k[0]]), int(indices[k[-1]]),
                                        c.salto_oclusion_m, range_min)
        objetos.append(Objeto(float(centro[0]), float(centro[1]),
                              float(np.hypot(*(centro - o))), bordes))
    return objetos


def _se_contradicen(a, b):
    """Uno vio rojo y el otro sólo "no rojo": pueden ser una víctima pegada a un distractor."""
    return ((a.rojo > 0 and b.no_rojo > 0 and b.rojo == 0)
            or (b.rojo > 0 and a.no_rojo > 0 and a.rojo == 0))


def es_caja(cand, c):
    """Con `rojo_requiere_caja`: ¿el LiDAR lo vio como una caja (no sólo el rojo)?"""
    return (not c.rojo_requiere_caja
            or (cand.vistas >= c.vistas_min and len(cand.desde) >= c.posiciones_min
                and cand.bordes >= c.bordes_min_rojo))


def _descartable(cand, c):
    """¿Alcanzan los "no rojo" para descartarlo?"""
    if cand.no_rojo < c.votos_descarte:
        return False
    if c.descarte_por_mayoria:
        return cand.no_rojo >= c.proporcion_descarte * cand.rojo
    return cand.rojo == 0


def _absorber(a, b, c):
    """`a` se queda con lo de `b`, y el estado sale de los votos sumados."""
    a.obs = (a.obs + b.obs)[-30:]
    a.vistas += b.vistas
    a.ausencias = min(a.ausencias, b.ausencias)
    for p in b.desde:
        if p not in a.desde:
            a.desde.append(p)
    a.rojo += b.rojo
    a.no_rojo += b.no_rojo
    a.rojo_lejano = a.rojo_lejano or b.rojo_lejano
    a.rojo_cortado = a.rojo_cortado or b.rojo_cortado
    a.bordes = max(a.bordes, b.bordes)
    estados = {a.estado, b.estado}
    if 'victima' in estados:
        a.estado = 'victima'
    elif 'descartado' in estados:          # como en inspeccionar: revive con votos_revivir
        a.estado = 'victima' if a.rojo >= c.votos_revivir and es_caja(a, c) else 'descartado'
    elif 'sin_inspeccionar' in estados:
        a.estado = ('victima' if a.rojo >= c.votos_victima and es_caja(a, c)
                    else 'descartado' if _descartable(a, c) else 'sin_inspeccionar')
    else:
        a.estado = 'posible'


class Candidatos:
    def __init__(self, config=ConfigCandidatos()):
        self.config = config
        self.lista: List[Candidato] = []
        self._proximo_id = 1

    def cercano(self, x, y):
        mejor, d_min = None, self.config.radio_union_m
        for cand in self.lista:
            d = math.hypot(cand.x - x, cand.y - y)
            if d < d_min:
                mejor, d_min = cand, d
        return mejor

    def _nuevo(self, estado):
        cand = Candidato(self._proximo_id, estado=estado)
        self._proximo_id += 1
        self.lista.append(cand)
        return cand

    def _vista(self, cand, desde):
        cand.vistas += 1
        cand.ausencias = 0
        if desde is not None and self._posicion_nueva(cand, desde):
            cand.desde.append((float(desde[0]), float(desde[1])))

    def _posicion_nueva(self, cand, desde):
        """¿`desde` es otro punto de vista que los ya guardados (lejos o en otro ángulo)?"""
        c = self.config
        for px, py in cand.desde:
            if math.hypot(desde[0] - px, desde[1] - py) < c.separacion_posiciones_m:
                return False
            a = math.atan2(desde[1] - cand.y, desde[0] - cand.x)
            b = math.atan2(py - cand.y, px - cand.x)
            if abs(math.atan2(math.sin(a - b), math.cos(a - b))) < c.separacion_angulo_rad:
                return False
        return True

    def agregar(self, x, y, desde=None):
        """Suma una observación de cerca del LiDAR; devuelve el candidato (nuevo o existente).
        Un posible visto de cerca pasa a candidato."""
        cand = self.cercano(x, y) or self._nuevo('sin_inspeccionar')
        if cand.estado == 'posible':
            cand.estado = 'sin_inspeccionar'
            cand.obs.clear()                     # las posiciones de lejos eran peores
        cand.obs.append((x, y))
        del cand.obs[:-30]
        self._vista(cand, desde)
        return cand

    def observar(self, obj, desde=None, rojos_lejanos=()):
        """Aplica la política de detección a un objeto del scan. Devuelve el candidato que lo
        tomó, o None si no cuenta."""
        c = self.config
        if obj.distancia <= c.alcance_candidato_m:
            if obj.bordes is not None and obj.bordes < c.bordes_min_cerca:
                return None
            cand = self.agregar(obj.x, obj.y, desde)
            cand.bordes = max(cand.bordes, obj.bordes or 0)
            return cand
        if obj.distancia > c.alcance_posible_m:
            return None
        rojo = any(math.hypot(rx - obj.x, ry - obj.y) < c.radio_rojo_lejano_m
                   for rx, ry in rojos_lejanos)
        if not rojo and obj.bordes is not None and obj.bordes < c.bordes_min_lejos:
            return None
        cand = self.cercano(obj.x, obj.y)
        if cand is None:
            cand = self._nuevo('posible')
        if cand.estado == 'posible' or not cand.obs:
            # De lejos la posición es peor: sólo ubica a los que todavía no se vieron de cerca.
            cand.obs.append((obj.x, obj.y))
            del cand.obs[:-30]
        self._vista(cand, desde)
        cand.bordes = max(cand.bordes, obj.bordes or 0)
        if rojo:
            self.marcar_rojo_lejano([(obj.x, obj.y)])
        return cand

    def unir(self):
        """Con `unir_candidatos`: une los candidatos que quedaron pegados (ver la config). El más
        viejo se queda con las observaciones y los votos del otro. Devuelve [(quedó, se fue)]."""
        c = self.config
        if not c.unir_candidatos:
            return []
        unidos = []
        for a in list(self.lista):
            for b in list(self.lista):
                if (a is b or a not in self.lista or b not in self.lista or b.id < a.id
                        or math.hypot(a.x - b.x, a.y - b.y) >= c.distancia_unir_m
                        or _se_contradicen(a, b)):
                    continue
                _absorber(a, b, c)
                self.lista.remove(b)
                unidos.append((a.id, b.id))
        return unidos

    def marcar_rojo_lejano(self, rojos):
        """La cámara vio rojo en estos lugares (de lejos, posición aproximada): los posibles y
        candidatos sin inspeccionar que caen ahí pasan a candidatos rojos, que se inspeccionan
        sin esperar más vistas. Devuelve los que cambiaron."""
        c = self.config
        cambiados = []
        for cand in self.lista:
            if cand.estado not in ('posible', 'sin_inspeccionar') or cand.rojo_lejano:
                continue
            if any(math.hypot(rx - cand.x, ry - cand.y) < c.radio_rojo_lejano_m
                   for rx, ry in rojos):
                cand.estado = 'sin_inspeccionar'
                cand.rojo_lejano = True
                cambiados.append(cand)
        return cambiados

    def ausente(self, cand):
        """Un scan miró el lugar del candidato y no había nada. Si pasa varias veces seguidas
        (sin una vista entre medio), se borra: era un fantasma. Devuelve True si lo borró."""
        if cand.estado not in ('sin_inspeccionar', 'posible'):
            return False
        cand.ausencias += 1
        if cand.ausencias >= self.config.ausencias_borrar:
            self.lista.remove(cand)
            return True
        return False

    def listos(self, cand):
        """¿Se lo puede inspeccionar (y votar)? Un posible no; un candidato rojo, ya; el resto,
        visto en suficientes scans desde suficientes posiciones."""
        c = self.config
        if cand.estado == 'posible':
            return False
        if cand.estado != 'sin_inspeccionar' or cand.rojo_lejano:
            return True
        posiciones_ok = not cand.desde or len(cand.desde) >= c.posiciones_min
        return cand.vistas >= c.vistas_min and posiciones_ok

    def inspeccionar(self, cand, rojo):
        """Una vista clara del candidato: rojo (True), sin rojo (False) o dudoso (None: hay
        rojo en su dirección pero no se pudo ubicar; no vota)."""
        c = self.config
        if cand.estado == 'victima' or rojo is None:
            return
        if rojo:
            cand.rojo += 1                       # también un descartado: el rojo en su lugar
        elif cand.estado == 'sin_inspeccionar':  # es la evidencia fuerte y lo revive
            cand.no_rojo += 1
        necesarios = c.votos_revivir if cand.estado == 'descartado' else c.votos_victima
        if cand.rojo >= necesarios and es_caja(cand, c):
            cand.estado = 'victima'
        elif _descartable(cand, c):
            cand.estado = 'descartado'

    def resumen(self):
        """Los candidatos para el explorador: los listos para inspeccionar (o ya clasificados)
        y los posibles (estado 'posible', no se inspeccionan). Los vistos poco, todavía no."""
        return [{'id': cand.id, 'x': round(cand.x, 3), 'y': round(cand.y, 3),
                 'estado': cand.estado, 'rojo': cand.rojo, 'no_rojo': cand.no_rojo,
                 'rojo_lejano': cand.rojo_lejano, 'rojo_cortado': cand.rojo_cortado}
                for cand in self.lista if cand.estado == 'posible' or self.listos(cand)]


def en_vista(cand_xy, laser_xyyaw, rumbo_camara, medio_fov, rango_en_rumbo, c=ConfigCandidatos()):
    """¿La cámara tiene al candidato de frente, a distancia útil y sin nada en el medio?

    `rumbo_camara`: hacia dónde mira la cámara, en el marco del LiDAR. `rango_en_rumbo(a)`:
    lo que mide el LiDAR en el rumbo `a` (marco del LiDAR). Devuelve (en_vista, rumbo).
    """
    lx, ly, lyaw = laser_xyyaw
    dx, dy = cand_xy[0] - lx, cand_xy[1] - ly
    d = math.hypot(dx, dy)
    rumbo = math.atan2(dy, dx) - lyaw
    rumbo = math.atan2(math.sin(rumbo), math.cos(rumbo))
    relativo = math.atan2(math.sin(rumbo - rumbo_camara), math.cos(rumbo - rumbo_camara))
    if not (c.distancia_min_m <= d <= c.distancia_max_m) or abs(relativo) > 0.8 * medio_fov:
        return False, rumbo
    # Sin nada en el medio: el LiDAR, en ese rumbo, llega hasta la cara de la caja.
    r = rango_en_rumbo(rumbo)
    return (r is not None and r >= d - c.profundidad_m - 0.10), rumbo


def lugar_vacio(cand_xy, laser_xyyaw, rango_en_rumbo, c=ConfigCandidatos()):
    """¿El LiDAR, cerca, pasa de largo por donde estaría la caja? (evidencia de que no hay)

    Una caja real corta el rayo hacia su centro en su cara, `profundidad_m` antes del centro;
    si el rayo llega 10 cm más allá del centro, ahí no hay nada. Si corta antes (algo en el
    medio), no prueba nada.
    """
    lx, ly, lyaw = laser_xyyaw
    dx, dy = cand_xy[0] - lx, cand_xy[1] - ly
    d = math.hypot(dx, dy)
    if d > c.radio_ausencia_m or d < 1e-6:
        return False
    rumbo = math.atan2(dy, dx) - lyaw
    r = rango_en_rumbo(math.atan2(math.sin(rumbo), math.cos(rumbo)))
    return r is not None and r > d + 0.10


def vista_concluyente(cand_xy, laser_xyyaw, rumbo_camara, medio_fov, c=ConfigCandidatos(),
                      con_distancia=True):
    """¿En esta vista, no ver rojo prueba que no es rojo? (centrado y cerca).

    Con `con_distancia=False` sólo pide centrado: vale cuando "no ver rojo" se mira en los
    píxeles crudos (blobs del borde incluidos), así una caja roja de cerca no pasa por no roja
    y la distancia deja de importar (dentro del rango de `en_vista`).
    """
    lx, ly, lyaw = laser_xyyaw
    d = math.hypot(cand_xy[0] - lx, cand_xy[1] - ly)
    rumbo = math.atan2(cand_xy[1] - ly, cand_xy[0] - lx) - lyaw
    relativo = math.atan2(math.sin(rumbo - rumbo_camara), math.cos(rumbo - rumbo_camara))
    centrado = abs(relativo) <= c.centrado_no_rojo * medio_fov
    if not con_distancia:
        return centrado
    return c.distancia_min_no_rojo_m <= d <= c.distancia_max_no_rojo_m and centrado


def confirmar_con_candidatos(objetivos, candidatos, radio=0.35, sumar_candidatos=False,
                             sumar_dudosos=False):
    """Un objetivo rojo del tracker vale como víctima (confirmado) sólo si cae sobre un
    candidato que la inspección clasificó como víctima: un objeto real, rojo y mirado de
    frente. Una detección roja suelta (por ejemplo, la misma caja ubicada 0.4 m corrida)
    gastaba un reporte y dejaba a otra víctima sin buscar.

    Con `sumar_candidatos`, un candidato víctima que ningún objetivo confirmado cubre se agrega
    como objetivo confirmado con la posición del LiDAR. En práctica (dos laberintos,
    2026-10-28) el candidato quedó a 5-6 cm de la caja, el objetivo del tracker no llegó a
    confirmarse cerca, la víctima nunca se reportó y el cupo se gastó al cierre a 0.7-0.8 m.

    Con `sumar_dudosos`, un candidato sin decidir que ningún objetivo cubre se agrega sin
    confirmar, para los cupos que sobran al cierre (un cupo vacío no vale nada): con `n` = sus
    votos rojos, o 0.5 si nunca se pudo votar (va después). Uno con votos "no rojo" y ningún
    rojo no: la cámara lo vio y no era rojo."""
    clasificados_rojos = [c for c in candidatos if c['estado'] == 'victima']
    salida = []
    for o in objetivos:
        o = dict(o)
        o['confirmado'] = bool(o['confirmado']) and any(
            math.hypot(o['x'] - c['x'], o['y'] - c['y']) < radio for c in clasificados_rojos)
        salida.append(o)
    if sumar_candidatos:
        for c in clasificados_rojos:
            if not any(o['confirmado'] and math.hypot(o['x'] - c['x'], o['y'] - c['y']) < radio
                       for o in salida):
                # id entero (lo usan los marcadores de RViz), lejos de los del tracker
                salida.append({'id': 100000 + int(c['id']), 'x': c['x'], 'y': c['y'],
                               'n': c['rojo'], 'dispersion': 0.0, 'confirmado': True})
    if sumar_dudosos:
        for c in candidatos:
            if c['estado'] != 'sin_inspeccionar' or (c['rojo'] == 0 and c['no_rojo'] > 0):
                continue
            if any(math.hypot(o['x'] - c['x'], o['y'] - c['y']) < radio for o in salida):
                continue
            salida.append({'id': 200000 + int(c['id']), 'x': c['x'], 'y': c['y'],
                           'n': c['rojo'] if c['rojo'] > 0 else 0.5, 'dispersion': 0.0,
                           'confirmado': False})
    return salida
