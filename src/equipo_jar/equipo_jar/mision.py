"""Elección de vistas M4 desde mapa/sensores. Sin ROS ni fixtures de evaluación."""
from dataclasses import dataclass
import math
import numpy as np
from equipo_jar.navegacion.planificador import costo_desde


@dataclass(frozen=True)
class ParametrosMision:
    duracion_s: float = 600.
    reserva_regreso_s: float = 45.
    velocidad_regreso_m_s: float = .10
    observacion_s: float = 3.
    plazo_meta_s: float = 90.
    separacion_vistas_m: float = .20
    separacion_minima_m: float = .10
    sondeos_se2: int = 1
    plazo_sondeo_se2_s: float = 5.
    ganancia_min: int = 2
    max_candidatas: int = 100
    max_solicitudes_vista: int = 3  # inspecciones alcanzadas por objetivo
    distancia_asociacion_m: float = .30
    # Huella supuesta de una víctima reportada en planificación (lado de caja, m).
    lado_caja_reportada_m: float = .20
    exponente_distancia: float = 1.
    # Región objetivo: zonas de piso pendiente conectadas, con al menos este número de
    # muestras; la elección de vista cuenta sólo la ganancia en la mejor región (tamaño /
    # (1 + distancia de ruta)). Un valor muy alto (p. ej. 100000) la desactiva.
    region_min_muestras: int = 6
    # Cerca de reportadas, una pista ciega no interrumpe el viaje actual.
    # Sigue disponible al seleccionar otra vista; no veta detecciones RGB-D.
    radio_exclusion_reportada_m: float = .60
    interrupciones_lidar_max: int = 1
    # Cobertura por LiDAR: alcance y semiancho del corredor acreditado por cada rayo.
    cobertura_lidar_alcance_m: float = 5.0
    # Menor que medio lado de la caja más chica de práctica (0,15 m): una caja junto al rayo lo cortaría.
    cobertura_lidar_radio_m: float = .06

    def __post_init__(self):
        if (not all(math.isfinite(v) and v > 0 for v in self.__dict__.values())
                or self.reserva_regreso_s >= self.duracion_s
                or self.separacion_minima_m > self.separacion_vistas_m
                or any(type(v) is not int for v in (self.ganancia_min,
                       self.max_candidatas, self.max_solicitudes_vista, self.sondeos_se2,
                       self.region_min_muestras,
                       self.interrupciones_lidar_max))):
            raise ValueError('Parámetros de misión inválidos')


def region_objetivo(grilla, ocupada, cobertura, ida, p, diagnostico=None):
    """Máscara de muestras de la región pendiente más conveniente, o None.

    Agrupa muestras de piso pendientes vecinas en la retícula de cobertura (8-conexas,
    sin cruzar una celda ocupada entre ambas). Puntúa cada región por tamaño /
    (1 + distancia de ruta desde el robot a su muestra alcanzable más cercana). Sólo
    entran regiones con alguna muestra alcanzable y >= region_min_muestras. Incluye las
    caras de pared pendientes a <= un paso de la región. No usa mapas de práctica.
    """
    if not all(hasattr(cobertura, n) for n in ('observada', 'objetivos_pared', 'puntos', 'p')):
        return None
    pend = np.flatnonzero(~cobertura.observada & ~cobertura.objetivos_pared)
    if len(pend) < p.region_min_muestras:
        return None
    paso = cobertura.p.paso_muestreo_m
    pts = cobertura.puntos[pend]
    clave = {(int(round(x/paso)), int(round(y/paso))): i for i, (x, y) in enumerate(pts)}
    padre = list(range(len(pend)))

    def raiz(i):
        while padre[i] != i:
            padre[i] = padre[padre[i]]; i = padre[i]
        return i
    for (kx, ky), i in clave.items():
        for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
            j = clave.get((kx+dx, ky+dy))
            if j is None:
                continue
            medio = grilla.a_celda(*((pts[i]+pts[j])/2))
            if medio is None or ocupada[medio]:
                continue
            a, b = raiz(i), raiz(j)
            if a != b:
                padre[a] = b
    grupos = {}
    for i in range(len(pend)):
        grupos.setdefault(raiz(i), []).append(i)
    mejor, puntaje = None, 0.
    for miembros in grupos.values():
        if len(miembros) < p.region_min_muestras:
            continue
        dist = []
        for i in miembros:
            cell = grilla.a_celda(*pts[i])
            if cell is not None and math.isfinite(ida[cell]):
                dist.append(float(ida[cell]))
        if not dist:
            continue
        valor = len(miembros)/(1.+min(dist))
        if valor > puntaje:
            mejor, puntaje = miembros, valor
    if mejor is None:
        return None
    mascara = np.zeros(len(cobertura.observada), dtype=bool)
    mascara[pend[mejor]] = True
    # Obligaciones de cámara (objetos no explicados por el mapa) siempre cuentan, aunque
    # su parche sea demasiado chico para ser región: enfocar no debe dejarlas sin mirar.
    if hasattr(cobertura, 'solo_camara'):
        mascara |= cobertura.solo_camara & ~cobertura.observada
    paredes = np.flatnonzero(~cobertura.observada & cobertura.objetivos_pared)
    if len(paredes):
        d = np.hypot(cobertura.puntos[paredes, None, 0]-pts[mejor][None, :, 0],
                     cobertura.puntos[paredes, None, 1]-pts[mejor][None, :, 1])
        mascara[paredes[(d <= paso*1.5).any(axis=1)]] = True
    if diagnostico is not None:
        centro = pts[mejor].mean(axis=0)
        diagnostico.update(region_muestras=len(mejor), region_puntaje=round(puntaje, 3),
                           region_centro=[round(float(centro[0]), 2), round(float(centro[1]), 2)],
                           regiones=sum(len(g) >= p.region_min_muestras for g in grupos.values()))
    return mascara


def clave_vista(vista):
    # Distinguir parallax de 5 cm; rumbos equivalentes al cruzar +/-pi.
    sectores = 128
    return (round(vista[0] / .05), round(vista[1] / .05),
            round(vista[2] * sectores / (2*math.pi)) % sectores)


def elegir_vista(grilla, costos, pose, largada, cobertura, descartadas,
                 tiempo_restante_s, p, sugeridas=(), extrinseca=(0., 0., 0.),
                 diagnostico=None, metas_permitidas=None, alternativas=None, distancias=None, limite_alternativas=None, umbral_vista=None, filtro_pose=None, rumbos=8):
    """Trabajo pesado ejecutable en proceso; None: sin vista admisible.

    Se filtran rutas por costos inflados y reserva de regreso; M2 conserva la
    guardia de huella. Ganancia es visibilidad estimada, nunca crédito de reporte.
    """
    if distancias is None:
        _, ida = costo_desde(grilla, costos, pose[:2])
        _, vuelta = costo_desde(grilla, costos, largada[:2])
    else:
        ida, vuelta = distancias
    salto = max(1, math.ceil(p.separacion_vistas_m / grilla.resolucion))
    celdas = [(f, c) for f in range(0, grilla.alto, salto)
              for c in range(0, grilla.ancho, salto)
              if math.isfinite(ida[f, c]) and math.isfinite(vuelta[f, c])]
    actual = grilla.a_celda(*pose[:2])
    if actual is not None and math.isfinite(costos[actual]):
        celdas = [actual] + [c for c in celdas if c != actual]
    propuestas = []
    feedback = {}
    for f, c in celdas:
        x, y = pose[:2] if (f, c) == actual else grilla.a_mundo(f, c)
        for k in range(rumbos):
            propuestas.append((x, y, grilla.origen_yaw+k*2*math.pi/rumbos, False, None))
    for vista in sugeridas:
        try:
            values = tuple(float(vista[n]) for n in ('x', 'y', 'yaw'))
        except (KeyError, TypeError, ValueError):
            continue
        if all(math.isfinite(v) for v in values):
            mirar = tuple(float(vista[n]) for n in ('mirar_x', 'mirar_y')) if all(n in vista for n in ('mirar_x', 'mirar_y')) else None
            propuestas.append((*values, True, mirar))
            feedback[values]=vista
    diag = dict(separacion_m=p.separacion_vistas_m, posiciones=len(celdas),
                vistas_evaluadas=0, descartadas=0, sin_ruta=0, sin_tiempo=0,
                sin_ganancia=0, admisibles=0)
    candidatas = []
    for orden, (x, y, yaw, sugerida, mirar) in enumerate(propuestas):
        diag['vistas_evaluadas'] += 1
        clave = clave_vista((x,y,yaw))
        if clave in descartadas or (*clave[:2],None) in descartadas:
            diag['descartadas'] += 1
            continue
        cell = grilla.a_celda(x, y)
        if (cell is None or not math.isfinite(costos[cell]) or
                (metas_permitidas is not None and not metas_permitidas[cell])):
            diag['sin_ruta'] += 1
            continue
        camino, regreso = float(ida[cell]), float(vuelta[cell])
        if not (math.isfinite(camino) and math.isfinite(regreso)):
            diag['sin_ruta'] += 1
            continue
        requerida = ((camino+regreso)/p.velocidad_regreso_m_s
                     +p.reserva_regreso_s+p.observacion_s+math.pi/.25)
        if requerida >= tiempo_restante_s:
            diag['sin_tiempo'] += 1
            continue
        c, s = math.cos(yaw), math.sin(yaw)
        ox, oy, giro = extrinseca
        camara = (x+c*ox-s*oy, y+s*ox+c*oy, yaw+giro)
        error_yaw = abs(math.atan2(math.sin(yaw-pose[2]), math.cos(yaw-pose[2])))
        divisor = (1+camino)**p.exponente_distancia+.30*error_yaw
        opcion = dict(x=x, y=y, yaw=yaw, camino_m=camino,
                      regreso_m=regreso, sugerida=sugerida)
        if mirar is not None:
            opcion.update(mirar_x=mirar[0], mirar_y=mirar[1])
        source=feedback.get((x,y,yaw),{})
        if source.get('pista_rgb'):
            opcion.update(pista_rgb=True,recorte_rgb=source.get('recorte_rgb',False))
        if source.get('pista_lidar'):
            opcion.update(pista_lidar=True,puntos_soporte=source.get('puntos_soporte',0),
                          mapa_ocluye=source.get('mapa_ocluye',False))
        if source.get('inspeccion_local'):
            quality=source.get('prioridad_visual',0.)
            quality=max(-1.,min(1.,quality)) if type(quality) in (int,float) and math.isfinite(quality) else 0.
            opcion.update(inspeccion_local=True,prioridad_visual=quality)
        candidatas.append(((sugerida, -orden), camara, divisor, opcion))
    if hasattr(cobertura, 'cotas_ganancia'):
        cotas = cobertura.cotas_ganancia([item[1] for item in candidatas])
    else:
        cotas = [cobertura.cota_ganancia(item[1]) if hasattr(cobertura, 'cota_ganancia')
                 else math.inf for item in candidatas]
    acotadas = []
    for (prioridad, camara, divisor, opcion), cota in zip(candidatas, cotas):
        sugerida, orden = prioridad
        if cota < p.ganancia_min and not sugerida:
            diag['sin_ganancia'] += 1
            continue
        bonus=(60 if opcion.get('recorte_rgb') else (50+10*opcion.get('prioridad_visual',0.) if opcion.get('inspeccion_local') else (20 if opcion.get('pista_lidar') else (30 if sugerida else 0))))
        superior = (cota+bonus)/divisor
        if umbral_vista is not None and (sugerida, superior) <= umbral_vista:
            continue
        acotadas.append(((sugerida, superior, orden), camara, divisor, opcion))
    candidatas = acotadas
    # Evaluación exacta por cota: nunca descartar una vista que pueda superar
    # a las mejores ya comprobadas. Reduce rayos sin cambiar el criterio.
    candidatas.sort(key=lambda item: item[0], reverse=True)
    limite = (len(candidatas) if alternativas is not None and limite_alternativas is None
              else (limite_alternativas or 1))
    mejores = []
    diag['vistas_con_rayos'] = 0
    for superior, camara, divisor, opcion in candidatas:
        if len(mejores) >= limite and superior <= mejores[-1][0]:
            break
        if filtro_pose is not None and not filtro_pose(opcion['x'], opcion['y'], opcion['yaw']):
            diag['sin_huella'] = diag.get('sin_huella', 0)+1
            continue
        ganancia = cobertura.ganancia(camara)
        diag['vistas_con_rayos'] += 1
        if ganancia < p.ganancia_min and not opcion['sugerida']:
            diag['sin_ganancia'] += 1
            continue
        diag['admisibles'] += 1
        bonus=(60 if opcion.get('recorte_rgb') else (50+10*opcion.get('prioridad_visual',0.) if opcion.get('inspeccion_local') else (20 if opcion.get('pista_lidar') else (30 if opcion['sugerida'] else 0))))
        valor = (ganancia+bonus)/divisor
        if umbral_vista is not None and (opcion['sugerida'], valor) <= umbral_vista:
            continue
        opcion.update(ganancia=ganancia, valor=valor)
        mejores.append(((opcion['sugerida'], valor, superior[2]), opcion))
        mejores.sort(key=lambda item: item[0], reverse=True)
        del mejores[limite:]
    mejor = mejores[0][1] if mejores else None
    if diagnostico is not None:
        diagnostico.update(diag)
    if alternativas is not None:
        alternativas.extend(item[1] for item in mejores)
    return mejor
