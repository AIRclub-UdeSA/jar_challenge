"""Hipótesis opcional de base cuadrada: ancho observado y prior físico ante oclusión.

La forma es un prior explícito; una sola cara no demuestra la profundidad.
La rama normal de dos caras conserva su estimación sin este supuesto.
"""
import numpy as np


def inferir_base_cuadrada(points, camera_xy, p, observacion_rgb=None, diagnostico=None):
    xy=np.asarray(points)[:,:2]
    values,axes=np.linalg.eigh(np.cov(xy.T))
    if values[-1]<5*max(0.,values[0]):return None
    normal=axes[:,0];surface=np.median(xy,axis=0)
    if np.dot(normal,np.asarray(camera_xy)-surface)<0:normal=-normal
    tangent=np.array([-normal[1],normal[0]])
    lo,hi=np.percentile((xy-surface)@tangent,[.5,99.5])
    side=float(hi-lo)
    residual=float(np.percentile(np.abs((xy-surface)@normal),80))
    face_center=surface+(lo+hi)/2*tangent
    if observacion_rgb is not None:
        face=geometria_cara_rgb(points,surface,normal,observacion_rgb)
        if face is None:return None
        face_center,side=face
    if not 0<side<=p.lado_max_m or residual>p.error_sensor_m:return None
    # Visible width is censored by occlusion, not evidence of a tiny/non-victim box.
    # Use the explicit, configurable square-base size prior; do not fit the
    # hidden depth to a 1cm sliver. This is an estimate, not a measured side
    # or a guaranteed physical lower bound for arbitrary competition boxes.
    visible=side
    side=max(side,p.lado_min_m,p.lado_prior_oculto_m)
    if diagnostico is not None:
        diagnostico.update(ancho_visible_m=visible,lado_estimado_m=side,
                           usando_prior_tamano=visible<side)
    center=face_center-side/2*normal
    return center,(side,side),residual


def regiones_desde_mascara(mask,min_pixels):
    import cv2
    _,_,stats,_=cv2.connectedComponentsWithStats(np.asarray(mask,dtype=np.uint8),8)
    return [dict(left=int(x),top=int(y),width=int(w),height=int(h))
            for x,y,w,h,area in stats[1:] if area>=min_pixels]


def region_cara_rgb(points,observation):
    """Asocia una sola región RGB a puntos 3D de la misma estampa."""
    k=np.asarray(observation['k'],float).reshape(3,3)
    t=np.asarray(observation['translation'],float)
    q=np.asarray(observation['quaternion'],float);q=q/np.linalg.norm(q)
    if not np.isfinite(k).all() or k[0,0]<=0 or k[1,1]<=0 or not np.isfinite(q).all():return None
    x,y,z,w=q
    rotation=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                       [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                       [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
    local=(np.median(points,axis=0)-t)@rotation
    if local[2]<=0:return None
    u=k[0,0]*local[0]/local[2]+k[0,2]
    v=k[1,1]*local[1]/local[2]+k[1,2]
    matches=[r for r in observation['regions']
        if r['left']-3<=u<=r['left']+r['width']+3 and
           r['top']-3<=v<=r['top']+r['height']+3]
    if len(matches)!=1:return None
    r=matches[0]
    return r,k,t,rotation


def geometria_cara_rgb(points,surface,normal,observation):
    """Interseca contorno calibrado con el plano; exige región no recortada."""
    match=region_cara_rgb(points,observation)
    if match is None:return None
    r,k,t,rotation=match
    if (r['left']<=0 or r['top']<=0 or
            r['left']+r['width']>=observation['width'] or
            r['top']+r['height']>=observation['height']):return None
    mid_v=r['top']+(r['height']-1)/2
    hits=[]
    for pixel in (r['left']-.5,r['left']+r['width']-.5):
        ray=np.array([(pixel-k[0,2])/k[0,0],(mid_v-k[1,2])/k[1,1],1.])@rotation.T
        denominator=float(normal@ray[:2])
        if abs(denominator)<1e-8:return None
        distance=float(normal@(surface-t[:2]))/denominator
        if distance<=0:return None
        hits.append((t+distance*ray)[:2])
    return (hits[0]+hits[1])/2,float(np.linalg.norm(hits[1]-hits[0]))


def realimentacion_cara_rgb(points,observation):
    """Calidad aparente para inspección; nunca confianza de un reporte."""
    if observation is None:return {}
    match=region_cara_rgb(points,observation)
    if match is None:return {}
    r,k,_,_=match
    return dict(calidad_visual=float(r['width']/max(1,r['height'])),
                desvio_imagen_rad=float(np.arctan((r['left']+(r['width']-1)/2-k[0,2])/k[0,0])),
                recortada=bool(r['left']<=0 or r['left']+r['width']>=observation['width']))
