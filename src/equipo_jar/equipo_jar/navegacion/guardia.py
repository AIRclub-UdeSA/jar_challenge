"""Guardia de barrido de huella contra el LiDAR; portada de scripts/seguridad_huella.py.

El barrido conserva la geometría del banco M1; añade frenado angular y reserva de curva.
"""
import math

import numpy as np
from scipy.spatial import ConvexHull

MAX_HEADING_ERROR_RAD = 0.10
STOPPING_RESERVE_M = 0.10
STOPPING_PREVIEW_S = 1.0


def envolvente(footprint, alpha):
    if not math.isfinite(alpha):
        raise ValueError('Dirección no finita')
    c, s = math.cos(alpha), math.sin(alpha)
    points = [(c*x+s*y, -s*x+c*y) for x, y in footprint['corners_base_m']]
    # Extremos exactos de cada esquina en todo el intervalo de rumbo.
    # Se conserva +/-0.20 rad entre instantes y el mismo margen físico.
    theta = footprint.get('heading_uncertainty_rad', 2*MAX_HEADING_ERROR_RAD)
    def extrema(a,b):
        angles=[-theta,theta]
        phase=math.atan2(b,a)
        for k in range(-3,4):
            angle=phase+k*math.pi
            if -theta <= angle <= theta:
                angles.append(angle)
        return [a*math.cos(angle)+b*math.sin(angle) for angle in angles]
    xs=[];ys=[]
    for u,v in points:
        xs.extend(extrema(u,-v))
        ys.extend(extrema(v,u))
    margin=footprint['margin_m']
    return (min(xs)-margin,max(xs)+margin,min(ys)-margin,max(ys)+margin)


def puntos_validos(points):
    return len(points) >= 30 and bool(np.isfinite(np.asarray(points,dtype=float)).all())


def poligono_barrido(footprint, alpha, distance):
    """Envolvente convexa con cota conservadora del error de discretización.

    Cada esquina recorre arcos de radio r. Entre dos ángulos la distancia
    máxima al segmento es r*(1-cos(paso/2)). Inflar el casco por esa sagita
    contiene todos los arcos, además del margen físico. Trasladar ambos
    extremos contiene toda la trayectoria recta y todos los rumbos admitidos.
    """
    theta = footprint.get('heading_uncertainty_rad', 2*MAX_HEADING_ERROR_RAD)
    steps = max(1, math.ceil(2*theta/0.04))
    # Mismos puntos y casco, calculados en bloques nativos. No se reduce
    # el número de ángulos ni la reserva que contiene los arcos.
    corners=np.asarray(footprint['corners_base_m'],dtype=float)
    angles=-theta+2*theta*np.arange(steps+1)/steps-alpha
    c,s=np.cos(angles)[:,None],np.sin(angles)[:,None]
    rotated=np.stack((c*corners[:,0]-s*corners[:,1],
                      s*corners[:,0]+c*corners[:,1]),axis=-1).reshape((-1,2))
    points=np.concatenate((rotated,rotated+np.array([distance,0.])))
    hull=points[ConvexHull(points).vertices].tolist()
    radius=max(math.hypot(x,y) for x,y in footprint['corners_base_m'])
    error = radius*(1-math.cos(theta/steps))
    return hull,error


def distancias_poligono(points, hull):
    """Distancia euclídea al casco cerrado, cero para su interior."""
    data=np.asarray(points,dtype=float).reshape((-1,2))
    if not len(data):
        return np.zeros(0,dtype=float)
    vertices=np.asarray(hull,dtype=float)
    edges=np.roll(vertices,-1,axis=0)-vertices
    delta=data[:,None,:]-vertices[None,:,:]
    inside=np.all(edges[None,:,0]*delta[:,:,1]-edges[None,:,1]*delta[:,:,0]>=-1e-12,axis=1)
    t=np.clip(np.sum(delta*edges[None,:,:],axis=2)/np.sum(edges*edges,axis=1),0,1)
    residual=delta-t[:,:,None]*edges[None,:,:]
    distances=np.sqrt(np.min(np.sum(residual*residual,axis=2),axis=1))
    return np.where(inside,0.,distances)


def comprobar_barrido(points, alpha, distance, footprint):
    """Margen físico alrededor de un casco que contiene todo el barrido."""
    if not math.isfinite(distance) or not puntos_validos(points):
        return {'clear':False, 'reason':'Datos insuficientes o inválidos'}
    xmin, xmax, ymin, ymax = envolvente(footprint, alpha)
    xmin += min(0., distance)
    xmax += max(0., distance)
    c, s = math.cos(alpha), math.sin(alpha)
    base=np.asarray(points,dtype=float)
    projected=np.column_stack((c*base[:,0]+s*base[:,1],-s*base[:,0]+c*base[:,1]))
    hull,error=poligono_barrido(footprint,alpha,distance)
    data=np.asarray(projected)
    vertices=np.asarray(hull)
    lower=np.maximum(vertices.min(axis=0)-data,np.maximum(data-vertices.max(axis=0),0.))
    lower=np.linalg.norm(lower,axis=1)
    closest=int(np.argmin(lower))
    upper=float(distancias_poligono([projected[closest]],hull)[0])
    # La distancia a la caja es una cota inferior de la distancia al casco.
    # Conservar todo posible contacto y todo posible mínimo exacto.
    relevant=lower<=max(upper,footprint['margin_m']+error)+1e-12
    distances=np.full(len(data),np.inf)
    distances[relevant]=distancias_poligono(data[relevant],hull)
    indices=np.flatnonzero(distances<=footprint['margin_m']+error+1e-12)
    blocked=np.column_stack((base[indices],projected[indices])).tolist()
    return {'clear':not blocked, 'reason':'Obstáculo en envolvente barrida' if blocked else '',
            'swept_bounds_along_cross_m':[xmin, xmax, ymin, ymax],
            'swept_convex_hull_along_cross_m':hull,
            'rotation_approximation_bound_m':error,
            'physical_margin_m':footprint['margin_m'],
            'minimum_clearance_m':float(distances.min())-footprint['margin_m']-error,
            'blocking_points_base_and_along_cross_m':blocked,
            'distance_with_stopping_reserve_m':distance}


def huella_desde_parametros(x_min, x_max, semi_ancho, margen):
    """Huella rectangular en base_footprint a partir de parámetros generales."""
    valores = (x_min, x_max, semi_ancho, margen)
    if not all(math.isfinite(v) for v in valores) or x_min >= x_max or semi_ancho <= 0 \
            or margen <= 0:
        raise ValueError('Huella inválida')
    esquinas = [(x, y) for x in (x_min, x_max) for y in (-semi_ancho, semi_ancho)]
    return {'corners_base_m': esquinas,
            'radius_m': max(math.hypot(x, y) for x, y in esquinas),
            'margin_m': margen}


def puntos_desde_scan(rangos, angulo_min, incremento, rango_min, rango_max, tf_laser_base):
    """Puntos (x, y) en base_footprint; descarta inf, nan y valores fuera de rango."""
    tx, ty, yaw = tf_laser_base
    ranges=np.asarray(rangos,dtype=float)
    indices=np.flatnonzero(np.isfinite(ranges)&(ranges>=rango_min)&(ranges<rango_max))
    angle=yaw+angulo_min+indices*incremento
    x=tx+ranges[indices]*np.cos(angle)
    y=ty+ranges[indices]*np.sin(angle)
    return list(zip(x.tolist(),y.tolist()))


def limitar_velocidad(puntos_base, vx, vy, huella, desaceleracion, latencia_s,
                      escalas=(1.0, 0.75, 0.5, 0.25, 0.1), wz=0., desaceleracion_angular=.5):
    """Mayor escala del comando cuyo barrido de frenado queda libre; si no, cero."""
    rapidez = math.hypot(vx, vy)
    if rapidez < 1e-3 and abs(wz)<1e-3:
        return 0.0, 0.0, {'bloqueado': False, 'escala': 0.0}
    if not all(math.isfinite(v) for v in (vx,vy,wz,desaceleracion,latencia_s,desaceleracion_angular)) \
            or min(desaceleracion,desaceleracion_angular)<=0 or latencia_s<0:
        return 0.,0.,{'bloqueado':True,'escala':0.,'escala_angular':0.,'motivo':'Frenado inválido'}
    # El scan no cambia entre escalas: evitar convertir la misma lista
    # completa dos veces en cada comprobación. Se conservan todos los puntos.
    puntos_base=np.asarray(puntos_base,dtype=float)
    alpha = math.atan2(vy, vx)
    for escala in escalas:
        v = escala * rapidez
        w=escala*wz
        giro=abs(w)*latencia_s+w*w/(2*desaceleracion_angular)
        distancia = v * v / (2 * desaceleracion) + v * latencia_s
        if rapidez>=1e-3:distancia+=min(STOPPING_RESERVE_M,v*STOPPING_PREVIEW_S)
        curvatura=distancia*min(1.,giro)
        ampliada=dict(huella,heading_uncertainty_rad=huella.get('heading_uncertainty_rad',2*MAX_HEADING_ERROR_RAD)+giro,
                      margin_m=huella['margin_m']+curvatura)
        guardia = comprobar_barrido(puntos_base, alpha, distancia, ampliada)
        if guardia['clear']:
            return escala * vx, escala * vy, {'bloqueado': False, 'escala': escala,
                'escala_angular':escala,'reserva_curvatura_m':curvatura,
                'holgura_minima_m':guardia.get('minimum_clearance_m')}
    return 0.0, 0.0, {'bloqueado': True, 'escala': 0.0, 'escala_angular':0.,
                       'motivo': guardia['reason'],'holgura_minima_m':guardia.get('minimum_clearance_m'),
                       'bloqueantes_base_m':guardia.get('blocking_points_base_and_along_cross_m',[])[:3]}


def recuperar_margen(puntos_base, huella, desaceleracion, latencia_s):
    """Escapar de una pequeña intrusión del buffer sin acercarse al contacto.

    No habilita ruta ni cambia el margen nominal. Exige conservar durante
    todo el barrido la separación inicial (> medio margen) y mejorarla al
    finalizar. Sólo traslación lenta, sin giros; la conducción normal sigue
    exigiendo el margen completo.
    """
    if not puntos_validos(puntos_base) or not math.isfinite(desaceleracion) or desaceleracion<=0 \
            or not math.isfinite(latencia_s) or latencia_s<0:
        return None
    points=np.asarray(puntos_base,dtype=float)
    hull,error=poligono_barrido(huella,0.,0.)
    distances=distancias_poligono(points,hull)
    before=float(distances.min())-error
    margin=huella['margin_m']
    if not margin*.5 < before < margin:
        return None
    near=points[distances<=distances.min()+.002]
    vertices=np.asarray(hull);edges=np.roll(vertices,-1,axis=0)-vertices
    delta=near[:,None,:]-vertices[None,:,:]
    t=np.clip(np.sum(delta*edges[None,:,:],axis=2)/np.sum(edges*edges,axis=1),0.,1.)
    feet=vertices[None,:,:]+t[:,:,None]*edges[None,:,:]
    residual=near[:,None,:]-feet
    idx=np.argmin(np.sum(residual*residual,axis=2),axis=1)
    normals=residual[np.arange(len(near)),idx]
    normals/=np.linalg.norm(normals,axis=1)[:,None]
    away=-normals.sum(axis=0)
    if np.linalg.norm(away)<1e-6:
        return None
    alpha=math.atan2(away[1],away[0]);best=None
    hard=dict(huella,margin_m=before-1e-6)
    for angle in (alpha,alpha-math.pi/4,alpha+math.pi/4):
        for speed in (.01,.005):
            distance=speed+speed*latencia_s+speed*speed/(2*desaceleracion)
            direction=np.array([math.cos(angle),math.sin(angle)])
            after=float(distancias_poligono(points-distance*direction,hull).min())-error
            if after<=before+min(.001,.05*margin):
                continue
            sweep=comprobar_barrido(points,angle,distance,hard)
            if not sweep['clear']:
                continue
            if best is None or after>best[2]['separacion_final_m']:
                best=(speed*direction[0],speed*direction[1],dict(
                    bloqueado=False,recuperacion_margen=True,separacion_inicial_m=before,
                    separacion_final_m=after,margen_nominal_m=margin,
                    separacion_minima_barrido_m=before+sweep['minimum_clearance_m']-1e-6))
    return best


def huella_convexa_x3(x_min,x_max,chasis_semi_ancho,rueda_x,rueda_y,rueda_radio,margen,rueda_semi_ancho_visual=.018):
    """Envolvente convexa del chasis y ruedas, con círculos circunscriptos.

    Parámetros físicos del robot; ninguna geometría del escenario. Conserva
    los extremos longitudinales del respaldo rectangular. Los 32 lados
    circunscriptos contienen los círculos completos, no una aproximación menor.
    """
    values=(x_min,x_max,chasis_semi_ancho,rueda_x,rueda_y,rueda_radio,margen,rueda_semi_ancho_visual)
    if not all(math.isfinite(v) for v in values) or x_min>=x_max or min(values[2:])<=0:
        raise ValueError('Geometría convexa inválida')
    points=[(x,y) for x in (x_min,x_max) for y in (-chasis_semi_ancho,chasis_semi_ancho)]
    radius=rueda_radio/math.cos(math.pi/32)
    for x in (-rueda_x,rueda_x):
        for y in (-rueda_y,rueda_y):
            # La proyección del cilindro visual al girar sobre el eje Y
            # también queda contenida; sus esquinas sobresalen de la esfera.
            points.extend((x+dx,y+dy) for dx in (-rueda_radio,rueda_radio)
                          for dy in (-rueda_semi_ancho_visual,rueda_semi_ancho_visual))
            points.extend((x+radius*math.cos(i*2*math.pi/32),
                           y+radius*math.sin(i*2*math.pi/32)) for i in range(32))
    points=sorted(set(points))
    def cross(a,b,c):return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    lower=[];upper=[]
    for chain,sequence in ((lower,points),(upper,reversed(points))):
        for point in sequence:
            while len(chain)>1 and cross(chain[-2],chain[-1],point)<=0:chain.pop()
            chain.append(point)
    hull=lower[:-1]+upper[:-1]
    return {'corners_base_m':hull,'radius_m':max(math.hypot(x,y) for x,y in hull),
            'margin_m':margen,'model':'convexa_chasis_ruedas'}
