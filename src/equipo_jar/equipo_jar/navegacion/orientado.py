"""Respaldo A* SE(2): traslaciones holonómicas y giros con huella completa."""
from dataclasses import replace,asdict
from functools import lru_cache
import heapq
import math
import time
import numpy as np
from scipy.signal import fftconvolve
from scipy.ndimage import distance_transform_edt, label
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from equipo_jar.navegacion.guardia import poligono_barrido,distancias_poligono,STOPPING_RESERVE_M
from equipo_jar.navegacion.planificador import Ruta,VECINOS,_ruta_escape

# A* ponderado: priorizar una ruta factible dentro del plazo de cómputo.
# Sólo cambia el orden de expansión; máscaras, barridos y costos se conservan.
# No se promete costo óptimo entre todas las rutas factibles.
SE2_HEURISTIC_WEIGHT = 1.5


def _colision_con_margen(points,hull,margin):
    """Filtro conservador; sólo omite distancias que no pueden colisionar."""
    vertices=np.asarray(hull,dtype=float)
    data=np.asarray(points,dtype=float).reshape((-1,2))
    lower=np.maximum(vertices.min(axis=0)-data,np.maximum(data-vertices.max(axis=0),0.))
    relevant=np.sum(lower*lower,axis=1)<=margin*margin+1e-12
    blocked=np.zeros(len(data),dtype=bool)
    if relevant.any():
        blocked[relevant]=distancias_poligono(data[relevant],hull)<=margin
    return blocked


@lru_cache(maxsize=512)
def _mascara_por_contenido(shape, data, kernel_shape, kernel_data):
    ocupada = np.frombuffer(data, dtype=bool).reshape(shape)
    kernel = np.frombuffer(kernel_data, dtype=bool).reshape(kernel_shape)
    rf,rc=kernel.shape[0]//2,kernel.shape[1]//2
    padded=np.pad(ocupada,((rf,rf),(rc,rc)),constant_values=True)
    counts=fftconvolve(padded.astype(float),kernel[::-1,::-1].astype(float),mode='valid')
    result = counts < .5
    result.flags.writeable = False
    return result


def _libre_por_kernel(ocupada,kernel):
    """Reutiliza máscaras exactas; cambios de obstáculos/huella invalidan caché."""
    occupied = np.ascontiguousarray(ocupada, dtype=bool)
    footprint = np.ascontiguousarray(kernel, dtype=bool)
    return _mascara_por_contenido(occupied.shape, occupied.tobytes(),
                                 footprint.shape, footprint.tobytes())


def _heuristica_grilla(libre,meta,permitir_diagonales_optimistas=False,preferencia=None):
    """Dijkstra 2D nativo con la misma regla de diagonales del planificador."""
    h,w=libre.shape;ids=np.arange(h*w,dtype=np.int32).reshape((h,w))
    rows=[];cols=[];weights=[]
    for df,dc,length in VECINOS:
        f=slice(max(0,-df),min(h,h-df));c=slice(max(0,-dc),min(w,w-dc))
        nf=slice(max(0,df),min(h,h+df));nc=slice(max(0,dc),min(w,w+dc))
        valid=libre[f,c]&libre[nf,nc]
        if df and dc and not permitir_diagonales_optimistas:
            valid &= libre[f,nc]&libre[nf,c]
        rows.append(ids[f,c][valid]);cols.append(ids[nf,nc][valid])
        weights.append(np.full(int(valid.sum()),length,dtype=float) if preferencia is None else
                       length*(preferencia[f,c][valid]+preferencia[nf,nc][valid])/2)
    graph=coo_matrix((np.concatenate(weights),(np.concatenate(rows),np.concatenate(cols))),
                     shape=(h*w,h*w)).tocsr()
    return dijkstra(graph,directed=True,indices=int(ids[meta])).reshape((h,w))


def _planificar_en_grilla(grilla,ocupada,inicio,meta,huella,plazo_s=2.0,diagnostico=None,parametros_costos=None,rumbo_meta=None,solo_conectividad=False):
    """Búsqueda acotada; desconocidos y exterior siguen ocupados.

    Cada máscara contiene huella, incertidumbre de rumbo, margen físico y
    radio de celda. Cada arista traslacional comprueba su barrido completo. Los giros comprueban también todo el ángulo intermedio.
    No usa posiciones ni nombres de mundos; recibe mapa y obstáculos LiDAR.
    """
    comienzo=time.monotonic()
    expanded=0
    preparation_s=None
    def resultado(ruta,motivo):
        if diagnostico is not None:
            diagnostico.update(metodo='SE2',motivo=motivo,tiempo_s=time.monotonic()-comienzo,resolucion_busqueda_m=grilla.resolucion,estados_expandidos=expanded,preparacion_s=preparation_s,peso_heuristica=SE2_HEURISTIC_WEIGHT)
        return ruta
    paso=2*math.pi/32;res=grilla.resolucion
    # Los ejes del mapa siguen siendo candidatos aunque el robot arranque
    # girado. Conservar además su rumbo exacto evita un salto inicial.
    headings=[grilla.origen_yaw+k*paso for k in range(32)]
    closest=min(range(len(headings)),key=lambda k:abs(math.atan2(
        math.sin(headings[k]-inicio.yaw),math.cos(headings[k]-inicio.yaw))))
    if abs(math.atan2(math.sin(headings[closest]-inicio.yaw),
                      math.cos(headings[closest]-inicio.yaw)))<1e-12:
        headings[closest]=inicio.yaw
    else:headings.append(inicio.yaw)
    if rumbo_meta is not None:
        if not math.isfinite(rumbo_meta):
            return resultado(None,'Rumbo final inválido')
        if not any(abs(math.atan2(math.sin(h-rumbo_meta), math.cos(h-rumbo_meta))) < 1e-12 for h in headings):
            headings.append(rumbo_meta)
    headings.sort(key=lambda h:(h-grilla.origen_yaw)%(2*math.pi))
    n=len(headings);start_k=headings.index(inicio.yaw)
    turns_delta=[(headings[(k+1)%n]-headings[k])%(2*math.pi) for k in range(n)]
    preferencia=np.ones(ocupada.shape,dtype=float)
    if parametros_costos is not None:
        # Misma preferencia de holgura que XY, conservada como costo FINITO
        # incluso donde sólo una orientación del cuerpo permite atravesar.
        distancia=distance_transform_edt(np.pad(~ocupada,1,constant_values=False))[1:-1,1:-1]*res
        p=parametros_costos
        duro=p.radio_circunscripto_m+p.margen_m+res/2
        preferencia+=p.peso*np.maximum(0.,1.-(distancia-duro)/p.alcance_blando_m)

    start=grilla.a_celda(inicio.x,inicio.y);goal=grilla.a_celda(*meta)
    if start is None or goal is None:return resultado(None,'Fuera del mapa')
    sx,sy=grilla.a_mundo(*start)
    shift_world=(inicio.x-sx,inicio.y-sy)
    u,v=grilla.a_local(inicio.x,inicio.y)
    shift_local=np.array([u-(start[1]+.5)*res,v-(start[0]+.5)*res])
    goal=grilla.a_celda(meta[0]-shift_world[0],meta[1]-shift_world[1])
    if goal is None:return resultado(None,'Fuera del mapa')
    radius=math.ceil((huella['radius_m']+huella['margin_m']+3*res*math.sqrt(2)+STOPPING_RESERVE_M+.03)/res)
    yy,xx=np.mgrid[-radius:radius+1,-radius:radius+1]
    offsets=np.column_stack((xx.ravel()*res,yy.ravel()*res))
    def libre(k,extra=0.,df=0,dc=0):
        yaw=headings[k]+extra
        local=yaw-grilla.origen_yaw
        c,s=math.cos(local),math.sin(local)
        points=(offsets-shift_local)@np.array([[c,-s],[s,c]])
        alpha=math.atan2(df,dc)-local if df or dc else 0.
        ca,sa=math.cos(alpha),math.sin(alpha)
        projected=points@np.array([[ca,-sa],[sa,ca]])
        fp=dict(huella,heading_uncertainty_rad=huella.get('heading_uncertainty_rad',.20)+extra)
        # Arista geométrica completa. Frenado y preview dependen de la
        # velocidad real y los comprueba la guardia, no una prolongación
        # recta fija que atraviesa la siguiente esquina del camino.
        distance=res*math.hypot(df,dc)
        hull,error=poligono_barrido(fp,alpha,distance)
        # Cada obstáculo representa toda su celda, no sólo el centro.
        kernel=_colision_con_margen(projected,hull,huella['margin_m']+error+res/math.sqrt(2)).reshape(xx.shape)
        return _libre_por_kernel(ocupada,kernel)
    masks=[libre(k) for k in range(n)]
    # Preferencia blanda de holgura de la huella en cada rumbo.
    holgura=np.stack([distance_transform_edt(m)*res for m in masks])
    peso_holgura=parametros_costos.peso if parametros_costos is not None else 0.
    preferencia_rumbo=1.+peso_holgura*np.maximum(0.,1.-holgura/.10)
    movements={}
    valid=np.stack(masks)
    escape=_ruta_escape(replace(grilla,ocupada=ocupada),np.where(valid[start_k],0.,np.inf),start,.35)
    if escape is None:return resultado(None,'Sin salida corta desde la huella inicial')
    start_search=escape[-1]
    union=np.any(valid,axis=0)
    if solo_conectividad:
        componentes,_=label(union,structure=np.ones((3,3),int))
        return componentes==componentes[start_search]
    if not union[goal]:return resultado(None,'Meta sin huella admisible')
    if rumbo_meta is not None:
        final_k = min(range(n), key=lambda k: abs(math.atan2(
            math.sin(headings[k]-rumbo_meta), math.cos(headings[k]-rumbo_meta))))
        if not valid[final_k, goal[0], goal[1]]:
            return resultado(None,'Rumbo final sin huella admisible')
    turns=[libre(k,turns_delta[k]/2) for k in range(n)]
    rot=np.stack(turns)
    heuristic=_heuristica_grilla(union,goal,permitir_diagonales_optimistas=True,preferencia=preferencia*np.min(preferencia_rumbo,axis=0))
    preparation_s=time.monotonic()-comienzo
    if not math.isfinite(heuristic[start_search]):
        return resultado(None,'Sin conectividad incluso al omitir restricciones de giro')
    width=grilla.ancho;height=grilla.alto
    def encode(k,f,c):return (k*height+f)*width+c
    def decode(i):
        a,c=divmod(i,width);k,f=divmod(a,height);return k,f,c
    first=encode(start_k,*start_search);cost={first:0.};parent={};queue=[(SE2_HEURISTIC_WEIGHT*float(heuristic[start_search]),0.,first)]
    end=None
    while queue:
        if time.monotonic()-comienzo>plazo_s:return resultado(None,'Presupuesto de búsqueda agotado')
        _,g,current=heapq.heappop(queue)
        if g!=cost[current]:continue
        expanded+=1
        k,f,c=decode(current)
        if ((f,c)==goal and (rumbo_meta is None or
                abs(math.atan2(math.sin(headings[k]-rumbo_meta),
                               math.cos(headings[k]-rumbo_meta))) < 1e-6)):
            end=current;break
        successors=[]
        for df,dc,d in VECINOS:
            nf,nc=f+df,c+dc
            if not(0<=nf<height and 0<=nc<width) or not valid[k,nf,nc]:continue
            key=(k,df,dc)
            if key not in movements:movements[key]=libre(k,df=df,dc=dc)
            if movements[key][f,c]:successors.append((k,nf,nc,d*(preferencia[f,c]*preferencia_rumbo[k,f,c]+preferencia[nf,nc]*preferencia_rumbo[k,nf,nc])/2))
        if rot[k,f,c]:successors.append(((k+1)%n,f,c,.25*turns_delta[k]/res*preferencia[f,c]))
        if rot[(k-1)%n,f,c]:successors.append(((k-1)%n,f,c,.25*turns_delta[(k-1)%n]/res*preferencia[f,c]))
        for nk,nf,nc,d in successors:
            nxt=encode(nk,nf,nc);ng=g+d
            if ng<cost.get(nxt,math.inf):
                cost[nxt]=ng;parent[nxt]=current
                heapq.heappush(queue,(ng+SE2_HEURISTIC_WEIGHT*float(heuristic[nf,nc]),ng,nxt))
    if end is None:return resultado(None,'Sin conectividad de huella y rumbo')
    states=[end]
    while states[-1]!=first:states.append(parent[states[-1]])
    states.reverse()
    decoded=[(start_k,f,c) for f,c in escape[:-1]]+[decode(i) for i in states]
    # Reducir solamente tramos colineales de igual orientación.
    keep=[decoded[0]]
    for i in range(1,len(decoded)-1):
        a,b,c=decoded[i-1:i+2]
        if a[0]==b[0]==c[0] and (b[1]-a[1],b[2]-a[2])==(c[1]-b[1],c[2]-b[2]):continue
        keep.append(b)
    if len(decoded)>1:keep.append(decoded[-1])
    points=[(grilla.a_mundo(f,c)[0]+shift_world[0],
             grilla.a_mundo(f,c)[1]+shift_world[1]) for k,f,c in keep]
    headings=[headings[k] for k,f,c in keep]
    points[0]=(inicio.x,inicio.y);points[-1]=tuple(meta)
    if len(points)<2:points.append(tuple(meta));headings.append(headings[-1])
    return resultado(Ruta(points,cost[end]*res,sum(math.dist(a,b) for a,b in zip(points,points[1:])),headings,rumbo_meta,min(.02,res/4)),'Ruta encontrada')


def planificar_orientado(grilla,ocupada,inicio,meta,huella,plazo_s=5.,diagnostico=None,parametros_costos=None,rumbo_meta=None,plazo_refinado_s=0.):
    """Refina sólo discretización; nunca elimina celdas ocupadas ni márgenes."""
    start=time.monotonic()
    if diagnostico is not None:
        diagnostico['entrada']={
            'pose_mapa':[inicio.x,inicio.y,inicio.yaw],
            'meta_mapa':list(meta),
            'rumbo_meta_rad':rumbo_meta,
            'celdas_extra':np.argwhere(ocupada & ~grilla.ocupada).tolist(),
            'huella':dict(huella),
            'resolucion_m':grilla.resolucion,
            'parametros_costos':asdict(parametros_costos) if parametros_costos is not None else None}
    # El plazo inicial se conserva. Un respaldo explícito sólo se usa si
    # se agotó de verdad un componente pequeño, nunca tras un timeout de A*.
    decision=diagnostico if diagnostico is not None else {}
    route=_planificar_en_grilla(grilla,ocupada,inicio,meta,huella,plazo_s,decision,parametros_costos,rumbo_meta)
    fallo_grueso={k:decision.get(k) for k in ('motivo','estados_expandidos','tiempo_s')}
    aislado=(decision.get('motivo') in (
        'Sin conectividad de huella y rumbo',
        'Sin salida corta desde la huella inicial',
        'Sin conectividad incluso al omitir restricciones de giro')
        and decision.get('estados_expandidos',0)<=2000)
    respaldo=(plazo_refinado_s if aislado and grilla.resolucion>=.02 else 0.)
    if route is None and (grilla.resolucion>=.04 or respaldo>0.):
        remaining=max(plazo_s-(time.monotonic()-start),respaldo)
        if remaining>.1:
            fine=replace(grilla,resolucion=grilla.resolucion/2,
                         alto=grilla.alto*2,ancho=grilla.ancho*2,
                         ocupada=np.repeat(np.repeat(ocupada,2,axis=0),2,axis=1))
            route=_planificar_en_grilla(fine,fine.ocupada,inicio,meta,huella,remaining,decision,parametros_costos,rumbo_meta)
            decision['fallo_grueso']=fallo_grueso
            decision['respaldo_refinado_s']=respaldo
    if diagnostico is not None:diagnostico['tiempo_total_s']=time.monotonic()-start
    return route


def conectividad_orientada(grilla, ocupada, inicio, huella):
    """Condición necesaria para los rumbos de la grilla; no valida una ruta.

    Unión de las mismas máscaras de cuerpo/margen/banda de M2. Omite las
    restricciones de giro y barrido entre estados: conectado no implica ruta;
    desconectado evita repetir A* hacia toda una región imposible actualmente.
    """
    return _planificar_en_grilla(grilla, ocupada, inicio, (inicio.x,inicio.y),
                                huella, solo_conectividad=True)
