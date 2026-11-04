"""Seguidor holonómico con punto de mira y perfil de velocidad acotado."""
from dataclasses import dataclass, replace
import math

from equipo_jar.navegacion.tipos import Comando


@dataclass(frozen=True)
class ParametrosSeguimiento:
    velocidad_max: float
    aceleracion_max: float
    distancia_mira_m: float
    tolerancia_meta_m: float
    ganancia_yaw: float
    velocidad_angular_max: float
    espacio_lento_m: float
    espacio_rapido_m: float
    velocidad_cerca_pared: float
    dt_max_s: float

    def __post_init__(self):
        valores = (self.velocidad_max, self.aceleracion_max, self.distancia_mira_m,
                   self.tolerancia_meta_m, self.ganancia_yaw, self.velocidad_angular_max,
                   self.espacio_lento_m, self.espacio_rapido_m, self.velocidad_cerca_pared,
                   self.dt_max_s)
        if not all(math.isfinite(v) and v > 0 for v in valores):
            raise ValueError('Parámetros de seguimiento deben ser positivos y finitos')
        if self.espacio_lento_m >= self.espacio_rapido_m:
            raise ValueError('El espacio lento debe ser menor que el rápido')
        if self.velocidad_cerca_pared > self.velocidad_max:
            raise ValueError('La velocidad cerca de pared supera la máxima')


def _diferencia_angular(a, b):
    return math.atan2(math.sin(a - b), math.cos(a - b))


class Seguidor:
    def __init__(self, p):
        self.p = p
        self.puntos = []
        # La pérdida de localización puede detener el seguidor de vuelta
        # mientras su primer plan aún está pendiente.
        self.orientaciones=None
        self.rumbo_final=None
        self.precision_vertice_m=.02
        self.indice_orientado=0
        self.tramo_orientado=None
        self.yaw_mantener=0.
        self.segmento=0
        self.rapidez=0.
        self.velocidad_mapa=(0.,0.)
        self.t_previo=None
        self.llego = False
        self.restante_m = 0.0

    def reiniciar(self, ruta, yaw_mantener):
        self.precision_vertice_m=ruta.precision_vertice_m
        if len(ruta.puntos) < 2:
            raise ValueError('La ruta necesita al menos dos puntos')
        self.orientaciones = ruta.orientaciones
        self.rumbo_final = getattr(ruta, 'rumbo_final', None)
        self.indice_orientado = 0
        self.tramo_orientado = None
        self.puntos = list(ruta.puntos)
        self.yaw_mantener = yaw_mantener
        self.segmento = 0
        self.rapidez = 0.0
        self.velocidad_mapa = (0.0,0.0)
        self.t_previo = None
        self.llego = False
        self.restante_m = ruta.largo_m

    def detener(self):
        self.rapidez=0.
        self.velocidad_mapa=(0.,0.)
        self.t_previo=None
        if self.tramo_orientado is not None:self.tramo_orientado.detener()

    def _proyectar(self, x, y):
        """Proyección sobre la ruta sin retroceder de segmento."""
        i=self.segmento
        while i<len(self.puntos)-2 and math.dist((x,y),self.puntos[i+1])<=self.p.tolerancia_meta_m:
            i+=1
        (ax,ay),(bx,by)=self.puntos[i],self.puntos[i+1]
        dx,dy=bx-ax,by-ay
        largo2=dx*dx+dy*dy
        t=0. if largo2==0 else max(0.,min(1.,((x-ax)*dx+(y-ay)*dy)/largo2))
        return i,t

    def _avanzar(self, segmento, t, distancia):
        """Punto a `distancia` sobre la ruta desde (segmento, t) y largo restante."""
        (ax, ay), (bx, by) = self.puntos[segmento], self.puntos[segmento + 1]
        largo = math.hypot(bx - ax, by - ay)
        resto = largo * (1 - t)
        restante = resto + sum(math.dist(a, b) for a, b in
                               zip(self.puntos[segmento + 1:], self.puntos[segmento + 2:]))
        # La mira permanece en el tramo actual: cruzar una esquina antes de
        # alcanzar su vértice puede atravesar la huella inflada de una pared.
        distancia=min(distancia,resto)
        f=t if largo==0 else min(1.,t+distancia/largo)
        return (ax+f*(bx-ax),ay+f*(by-ay)),restante

    def paso(self, ahora, pose, espacio_libre_m):
        """Comando en el marco del robot; cero al llegar."""
        if len(self.puntos)<2:return Comando.cero()
        if self.orientaciones is not None:
            return self._paso_orientado(ahora,pose,espacio_libre_m)
        p = self.p
        meta = self.puntos[-1]
        dt = p.dt_max_s if self.t_previo is None else min(max(ahora - self.t_previo, 0.0),
                                                           p.dt_max_s)
        self.t_previo = ahora
        if math.dist((pose.x, pose.y), meta) <= p.tolerancia_meta_m:
            self.llego, self.rapidez, self.restante_m = True, 0.0, 0.0
            return Comando.cero()
        self.segmento, t = self._proyectar(pose.x, pose.y)
        mira, restante = self._avanzar(self.segmento, t, p.distancia_mira_m)
        self.restante_m = max(restante, math.dist((pose.x, pose.y), meta))
        f = (espacio_libre_m - p.espacio_lento_m) / (p.espacio_rapido_m - p.espacio_lento_m)
        por_espacio = p.velocidad_cerca_pared + max(0.0, min(1.0, f)) * (
            p.velocidad_max - p.velocidad_cerca_pared)
        distancia_frenado = min(self.restante_m,math.dist((pose.x,pose.y),self.puntos[self.segmento+1]))
        rapidez = min(p.velocidad_max, math.sqrt(2 * p.aceleracion_max * distancia_frenado),
                      por_espacio, self.rapidez + p.aceleracion_max * dt)
        self.rapidez = max(0.0, rapidez)
        dx, dy = mira[0] - pose.x, mira[1] - pose.y
        norma = math.hypot(dx, dy)
        if norma < 1e-9:
            return Comando.cero()
        vxm, vym = self.rapidez * dx / norma, self.rapidez * dy / norma
        anterior_x,anterior_y=self.velocidad_mapa
        cambio=math.hypot(vxm-anterior_x,vym-anterior_y)
        limite=p.aceleracion_max*dt
        if cambio>limite and cambio>0:
            vxm=anterior_x+(vxm-anterior_x)*limite/cambio
            vym=anterior_y+(vym-anterior_y)*limite/cambio
        self.velocidad_mapa=(vxm,vym)
        self.rapidez=math.hypot(vxm,vym)
        c, s = math.cos(pose.yaw), math.sin(pose.yaw)
        wz = p.ganancia_yaw * _diferencia_angular(self.yaw_mantener, pose.yaw)
        wz = max(-p.velocidad_angular_max, min(p.velocidad_angular_max, wz))
        return Comando(c * vxm + s * vym, -s * vxm + c * vym, wz)


    def _paso_orientado(self,ahora,pose,espacio):
        # La misión solicita posición, igual que el seguidor XY. No exigir
        # visitar un vértice anterior ni un rumbo final no solicitado.
        if (math.dist((pose.x,pose.y),self.puntos[-1])<=self.p.tolerancia_meta_m
                and (self.rumbo_final is None or
                     abs(_diferencia_angular(self.rumbo_final,pose.yaw))<=.06)):
            self.llego=True;self.restante_m=0.;self.detener()
            return Comando.cero()
        from equipo_jar.navegacion.planificador import Ruta
        while self.indice_orientado<len(self.puntos):
            i=self.indice_orientado
            target=self.puntos[i]
            error=_diferencia_angular(self.orientaciones[i],pose.yaw)
            distance=math.dist((pose.x,pose.y),target)
            tolerance=self.p.tolerancia_meta_m if i==len(self.puntos)-1 else min(self.precision_vertice_m,self.p.tolerancia_meta_m)
            self.restante_m=distance+sum(math.dist(a,b) for a,b in zip(self.puntos[i:],self.puntos[i+1:]))
            if distance<=tolerance and abs(error)<=.06:
                self.indice_orientado+=1;self.tramo_orientado=None
                continue
            if abs(error)>.08 or (distance<=tolerance and abs(error)>.06):
                self.tramo_orientado=None
                w=max(-self.p.velocidad_angular_max,min(self.p.velocidad_angular_max,self.p.ganancia_yaw*error))
                return Comando(0.,0.,w)
            if self.tramo_orientado is None:
                self.tramo_orientado=Seguidor(replace(self.p,tolerancia_meta_m=tolerance))
                self.tramo_orientado.reiniciar(Ruta([(pose.x,pose.y),target],distance,distance),self.orientaciones[i])
            return self.tramo_orientado.paso(ahora,pose,espacio)
        self.llego=True;self.restante_m=0.;self.rapidez=0.
        return Comando.cero()
