"""Envolvente conservadora de giro con dimensiones del robot, sin mapa de práctica."""
import math


def limites_barrido_giro(xmin, xmax, semi_ancho, angulo, margen):
    values=(xmin,xmax,semi_ancho,angulo,margen)
    if not all(math.isfinite(v) for v in values):
        raise ValueError('Geometría de giro no finita')
    if not xmin < 0 < xmax or semi_ancho <= 0 or margen <= 0 or not 0 <= angulo <= 2*math.pi:
        raise ValueError('Geometría de giro inválida')

    def extremos(a,b):
        candidates=[0.,angulo]
        phase=math.atan2(b,a)
        for k in range(-3,5):
            theta=phase+k*math.pi
            if 0 <= theta <= angulo:
                candidates.append(theta)
        return [a*math.cos(t)+b*math.sin(t) for t in candidates]

    xs=[];ys=[]
    for x in (xmin,xmax):
        for y in (-semi_ancho,semi_ancho):
            xs.extend(extremos(x,-y))
            ys.extend(extremos(y,x))
    return (min(xs)-margen,max(xs)+margen,min(ys)-margen,max(ys)+margen)


def casco_giro(xmin, xmax, semi_ancho, angulo):
    """Casco de esquinas rotadas con cota de sagita entre muestras."""
    steps=max(1,math.ceil(angulo/.04))
    points=[]
    for i in range(steps+1):
        a=angulo*i/steps
        c,s=math.cos(a),math.sin(a)
        points.extend((c*x-s*y,s*x+c*y)
                      for x in (xmin,xmax) for y in (-semi_ancho,semi_ancho))
    points=sorted(set(points))
    def cross(a,b,c):
        return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    lower,upper=[],[]
    for chain,ordered in ((lower,points),(upper,reversed(points))):
        for point in ordered:
            while len(chain)>1 and cross(chain[-2],chain[-1],point)<=0:
                chain.pop()
            chain.append(point)
    radius=math.hypot(max(abs(xmin),abs(xmax)),semi_ancho)
    return lower[:-1]+upper[:-1],radius*(1-math.cos(angulo/(2*steps)))


def distancia_casco(point,hull):
    inside=True
    distance_sq=math.inf
    x,y=point
    for (ax,ay),(bx,by) in zip(hull,hull[1:]+hull[:1]):
        ex,ey=bx-ax,by-ay
        dx,dy=x-ax,y-ay
        inside=inside and ex*dy-ey*dx>=-1e-12
        t=max(0.,min(1.,(dx*ex+dy*ey)/(ex*ex+ey*ey)))
        distance_sq=min(distance_sq,(dx-t*ex)**2+(dy-t*ey)**2)
    return 0. if inside else math.sqrt(distance_sq)


def comprobar_giro(puntos, xmin, xmax, semi_ancho, angulo, margen):
    bounds=limites_barrido_giro(xmin,xmax,semi_ancho,angulo,margen)
    if len(puntos)<30 or not all(math.isfinite(v) for p in puntos for v in p):
        return {'seguro':False,'motivo':'Puntos insuficientes o inválidos','limites_m':bounds}
    hull,error=casco_giro(xmin,xmax,semi_ancho,angulo)
    # El casco expandido por margen+sagita contiene el barrido continuo.
    # La caja solo acelera el descarte; no decide las colisiones de sus esquinas.
    left,right,bottom,top=bounds
    blocked=[(x,y) for x,y in puntos
             if left-error <= x <= right+error and bottom-error <= y <= top+error
             and distancia_casco((x,y),hull) <= margen+error+1e-12]
    return {'seguro':not blocked,'limites_m':bounds,'puntos_bloqueantes':len(blocked),
            'bloqueantes_m':blocked,'poligono_barrido_m':hull,'cota_discretizacion_m':error,
            'angulo_con_reserva_rad':angulo,'margen_m':margen}
