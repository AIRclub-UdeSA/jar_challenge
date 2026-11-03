"""Posiciones de observación sugeridas; no publica metas ni velocidades.

Se calculan desde superficies observadas y el mapa recibido. M2 debe validar
alcanzabilidad y obstáculos vivos; la misión decide cuándo usar una sugerencia.
"""
import math
import numpy as np
from scipy.ndimage import distance_transform_edt


def espacio_libre(grilla):
    # Fuera del mapa también es ocupado.
    return distance_transform_edt(np.pad(~grilla.ocupada, 1))[1:-1, 1:-1]*grilla.resolucion


def proponer_vistas(superficies, origen, grilla, distancias, separacion=.9, despeje=.30, angulos=None, comprobar_vision=True,
                    radios_adicionales=(), desplazamiento_min=.25, desde_observador=False, incluir_frontal=False):
    """Abanico frente a la cara, con despeje y línea de visión comprobados.

    La superficie NO es el centro de la caja. No se desplaza un reporte ni se
    supone su fondo; las posiciones sólo buscan revelar una cara adicional.
    """
    if not (math.isfinite(separacion) and separacion > 0
            and math.isfinite(despeje) and despeje > 0):
        raise ValueError('Distancias de observación inválidas')
    if angulos is None:
        angulos = (-math.pi/7, math.pi/7, -math.pi/6, math.pi/6,
                   -math.pi/4, math.pi/4, -math.pi/3, math.pi/3)
    if not all(math.isfinite(a) and 0 < abs(a) < math.pi/2 for a in angulos):
        raise ValueError('Ángulos de observación inválidos')
    radios = tuple(dict.fromkeys((separacion, separacion*1.3, *radios_adicionales)))
    if (not all(math.isfinite(r) and r > 0 for r in radios)
            or not math.isfinite(desplazamiento_min) or desplazamiento_min <= 0):
        raise ValueError('Radios de observación inválidos')
    propuestas = []
    for s in superficies:
        punto = np.array([s['superficie_x'], s['superficie_y']], float)
        normal = np.array([s['normal_x'], s['normal_y']], float)
        norm = np.linalg.norm(normal)
        if not np.isfinite(punto).all() or not np.isfinite(norm) or norm < .5:
            continue
        normal /= norm
        ejes = [normal]
        if desde_observador:
            observado = np.asarray(origen[:2], float)-punto
            distancia = np.linalg.norm(observado)
            if distancia > 1e-6:
                ejes.append(observado/distancia)
        for eje in ejes:
            for radio in radios:
                for angulo in ((0., *angulos) if incluir_frontal and eje is normal else angulos):
                    c, sn = math.cos(angulo), math.sin(angulo)
                    offset = np.array([[c, -sn], [sn, c]]) @ eje
                    goal = punto+radio*offset
                    cell = grilla.a_celda(*goal)
                    if (cell is None or distancias[cell] <= despeje+grilla.resolucion/2
                            or math.dist(goal, origen[:2]) < desplazamiento_min):
                        continue
                    # Sólo construir rayos cuando se pide comprobar visión.
                    if comprobar_vision:
                        n = max(2, math.ceil(radio/(grilla.resolucion/4)))
                        cells = [grilla.a_celda(*v) for v in np.linspace(goal, punto, n)]
                        if any(c is None or grilla.ocupada[c] for c in cells):
                            continue
                    propuestas.append(dict(x=float(goal[0]), y=float(goal[1]),
                        mirar_x=float(punto[0]), mirar_y=float(punto[1]),
                        yaw=math.atan2(punto[1]-goal[1], punto[0]-goal[0])))
    propuestas.sort(key=lambda v: math.dist((v['x'], v['y']), origen[:2]))
    return propuestas


def proponer_ajustes(superficie, pose, grilla, distancias, extrinseca=(0.,0.,0.),
                     direccion=None, despeje=.16):
    """Pequeños cambios alrededor de una observación roja real.

    El mapa puede cerrar una rendija visible en RGB: aquí sólo filtra ocupación
    del destino. La ruta y huella completas siguen siendo responsabilidad de M2.
    El rumbo apunta a la superficie medida, corrigiendo la óptica de la cámara.
    """
    point=np.array([superficie['superficie_x'],superficie['superficie_y']],float)
    normal=np.array([superficie['normal_x'],superficie['normal_y']],float)
    origin=np.asarray(pose[:2],float)
    if (not np.isfinite(point).all() or not np.isfinite(normal).all()
            or np.linalg.norm(normal)<.5 or np.linalg.norm(point-origin)<.1):return []
    normal/=np.linalg.norm(normal)
    ray=(point-origin)/np.linalg.norm(point-origin)
    axes=[np.array([-normal[1],normal[0]]),np.array([-ray[1],ray[0]]),ray]
    result=[];keys=set()
    for axis in axes:
        for distance in (.10,.20,.30):
            for sign in (-1.,1.):
                shift=axis*distance*sign;goal=origin+shift;cell=grilla.a_celda(*goal)
                if (cell is None or grilla.ocupada[cell] or
                        distancias[cell]<=despeje+grilla.resolucion/2):continue
                heading=math.atan2(*(point-goal)[::-1])-extrinseca[2]
                for _ in range(2):
                    c,s=math.cos(heading),math.sin(heading)
                    camera=goal+np.array([c*extrinseca[0]-s*extrinseca[1],s*extrinseca[0]+c*extrinseca[1]])
                    heading=math.atan2(*(point-camera)[::-1])-extrinseca[2]
                key=(round(goal[0],3),round(goal[1],3),round(heading,3))
                if key in keys:continue
                keys.add(key)
                preference=0.
                if direccion is not None:
                    direction=np.asarray(direccion,float)
                    if np.isfinite(direction).all() and np.linalg.norm(direction)>.5:
                        preference=float(np.dot(shift/distance,direction/np.linalg.norm(direction)))
                result.append(dict(x=float(goal[0]),y=float(goal[1]),yaw=float(heading),
                    mirar_x=float(point[0]),mirar_y=float(point[1]),
                    inspeccion_local=True,prioridad_visual=preference))
    return result


def mirada_desde_pose(pose, point, extrinseca):
    goal=np.asarray(pose[:2],float);point=np.asarray(point,float)
    heading=math.atan2(*(point-goal)[::-1])-extrinseca[2]
    for _ in range(2):
        c,s=math.cos(heading),math.sin(heading)
        camera=goal+np.array([c*extrinseca[0]-s*extrinseca[1],s*extrinseca[0]+c*extrinseca[1]])
        heading=math.atan2(*(point-camera)[::-1])-extrinseca[2]
    return dict(x=float(goal[0]),y=float(goal[1]),yaw=float(heading),
                mirar_x=float(point[0]),mirar_y=float(point[1]))


def proponer_inspecciones_lidar(puntos_mapa, pose, grilla, distancia_mapa,
                               tolerancia_mapa, extrinseca=(0.,0.,0.), origen_laser=None):
    """Orienta cámara a superficies no explicadas por mapa; no son víctimas.

    No exige visibilidad rasterizada: una rendija real puede ocupar menos de
    una celda. La medida LiDAR proporciona la pista; M2 valida cualquier giro.
    """
    from scipy.spatial import cKDTree
    points=np.asarray(puntos_mapa,dtype=float).reshape(-1,2)
    points=points[np.isfinite(points).all(axis=1)]
    if not len(points):return []
    delta=points-np.asarray(pose[:2]);distance=np.linalg.norm(delta,axis=1)
    keep=(distance>=.65)&(distance<=2.5)
    valid=[]
    for point in points[keep]:
        cell=grilla.a_celda(*point)
        if cell is not None and not grilla.ocupada[cell] and distancia_mapa[cell]>tolerancia_mapa:
            valid.append(point)
    if not valid:return []
    points=np.unique(np.round(np.asarray(valid),3),axis=0)
    tree=cKDTree(points);pending=set(range(len(points)));views=[]
    while pending:
        seed=pending.pop();group=[seed];stack=[seed]
        while stack:
            k=stack.pop();neighbors=pending.intersection(tree.query_ball_point(points[k],.08))
            pending.difference_update(neighbors);stack.extend(neighbors);group.extend(neighbors)
        cluster=points[group]
        if np.max(np.ptp(cluster,axis=0))>.8:continue
        point=np.median(cluster,axis=0)
        from equipo_jar.cobertura_visual import linea_libre
        map_occludes=not linea_libre(grilla,pose[:2] if origen_laser is None else origen_laser,point,grilla.ocupada,True)
        view=mirada_desde_pose(pose,point,extrinseca)
        view.update(pista_lidar=True,puntos_soporte=len(cluster),mapa_ocluye=map_occludes)
        views.append(view)
    return views
