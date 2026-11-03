"""Habilidad «ir a una meta y volver a la largada»; decide sin conocer ROS."""
from dataclasses import dataclass, replace
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
import math
import numpy as np

from equipo_jar.navegacion.costos import CapaViva, mapa_costos
from equipo_jar.navegacion.guardia import (limitar_velocidad, recuperar_margen,
    MAX_HEADING_ERROR_RAD, poligono_barrido, distancias_poligono)
from equipo_jar.navegacion.planificador import planificar
from equipo_jar.navegacion.orientado import planificar_orientado
from equipo_jar.navegacion.planificador import Ruta
from equipo_jar.navegacion.seguimiento import Seguidor
from equipo_jar.navegacion.tipos import Comando

TERMINALES = ('EN_LARGADA', 'FALLO')
EN_MARCHA = ('YENDO', 'VOLVIENDO')


@dataclass(frozen=True)
class EntradaNavegacion:
    ahora: float
    pose: object          # Pose2D en map, o None
    lista: bool           # localización utilizable (M1)
    puntos_base: object   # [(x, y)] del LiDAR en base_footprint, o None
    puntos_mapa: object   # los mismos puntos en map, o None
    std_yaw_rad: object = None  # desviación de AMCL, sin cambiar margen físico
    origen_laser_mapa: object = None
    incremento_angular_rad: object = None
    estampa_scan: object = None


@dataclass(frozen=True)
class ParametrosIdaVuelta:
    plazo_detenido_s: float = 20.0
    espera_bloqueo_s: float = 1.0
    max_replanificaciones: int = 5
    plazo_bloqueo_total_s: float = 30.0
    pausa_en_meta_s: float = 0.5
    tolerancia_largada_m: float = 0.05
    desaceleracion: float = 0.5
    latencia_s: float = 0.15
    vida_obstaculo_s: float = 600.0
    tolerancia_mapa_m: float = 0.10
    desaceleracion_angular: float = 0.5
    regreso_automatico: bool = True
    tolerancia_yaw_meta_rad: float = 0.06


def _resolver_plan(grilla,costos,p_costos,pose,fase,extra,destino,largada,replan,huella,rumbo_meta=None):
    """Entrada inmutable y resultado serializable; ningún acceso a ROS."""
    diag={'planning_process_id':os.getpid()}
    if extra.any():
        costos,_=mapa_costos(grilla,p_costos,extra_ocupadas=extra)
    ruta=None
    if replan:
        ruta=planificar_orientado(grilla,grilla.ocupada|extra,pose,destino,huella,
                                 diagnostico=diag,parametros_costos=p_costos,rumbo_meta=rumbo_meta,plazo_refinado_s=15.)
    if ruta is None:ruta=planificar(grilla,costos,(pose.x,pose.y),destino)
    if ruta is None and not replan:
        ruta=planificar_orientado(grilla,grilla.ocupada|extra,pose,destino,huella,
                                 diagnostico=diag,parametros_costos=p_costos,rumbo_meta=rumbo_meta,plazo_refinado_s=15.)
    if ruta is None:return None,None,diag
    if fase=='VOLVIENDO':regreso=ruta.largo_m
    else:
        back=planificar(grilla,costos,(pose.x,pose.y),largada[:2])
        regreso=back.largo_m if back else None
    return ruta,regreso,diag


class IdaVuelta:
    def __init__(self, grilla, p_costos, p_seguimiento, huella, p):
        self.grilla, self.p_costos, self.huella, self.p = grilla, p_costos, huella, p
        self.huella_operativa=dict(huella)
        self.costos, self.distancia = mapa_costos(grilla, p_costos)
        self.capa = CapaViva(grilla, self.distancia, p.vida_obstaculo_s, p.tolerancia_mapa_m)
        # No exigir precisión inferior a la diagonal de una celda del mapa.
        self.seguidor_ida = Seguidor(replace(p_seguimiento,
            tolerancia_meta_m=max(p_seguimiento.tolerancia_meta_m,
                                  math.sqrt(2)*grilla.resolucion)))
        self.seguidor_vuelta = Seguidor(
            replace(p_seguimiento, tolerancia_meta_m=p.tolerancia_largada_m))
        self.estado, self.motivo = 'ESPERANDO_LISTA', ''
        self.largada = self.meta = self.ruta = None
        self.regreso_estimado_m = None
        self.replanificaciones = 0
        self.reintentos_bloqueo = 0
        self.traza_ejecutada=[]
        self.traza_integra=True
        self.diagnostico_historial={}
        self.retorno_con_historial=False
        self.previo = None
        self.t_detenido = self.t_bloqueo = self.t_inicio_bloqueo = self.t_en_meta = None
        self.t_movimiento = None
        self.ahora = 0.0
        self.ultima_pose=None
        self.ultima_guardia={}
        self.t_recuperacion_margen=None
        self.ultima_propuesta=None
        self.diagnostico_plan={}
        self.executor_plan=None
        self.plan_pendiente=None
        self.ultima_estampa_capa=None
        self.meta_yaw = None
        self.yaw_trayecto = None
        self.solicitud_regreso = False
        self.largada_fijada = False

    def invalidar_plan_por_localizacion(self):
        """Descarta también resultados pendientes calculados en la pose anterior."""
        if self.estado in TERMINALES:return
        if self.ruta is not None:self.seguidor.detener()
        if self.plan_pendiente is not None:
            self.plan_pendiente[0].cancel()
            self.plan_pendiente=None
        self.ruta=None

    def cancelar(self,motivo):
        return self._fallar(motivo)

    def fijar_meta(self, xy):
        if not all(math.isfinite(v) for v in xy[:2]):
            self._fallar('Meta con coordenadas no finitas')
            return
        self.meta = (float(xy[0]), float(xy[1]))

    def continuar_meta(self, xy, rumbo=None):
        """Contrato M4: próxima vista, conservando largada y obstáculos vivos."""
        if self.p.regreso_automatico:
            raise ValueError('Metas consecutivas requieren modo misión')
        if self.estado not in ('ESPERANDO_LISTA', 'ESPERANDO_META', 'EN_META', 'FALLO'):
            raise ValueError('La navegación aún ejecuta otra meta')
        if len(xy) != 2 or not all(math.isfinite(v) for v in xy):
            raise ValueError('Meta de misión inválida')
        if rumbo is not None and not math.isfinite(rumbo):
            raise ValueError('Rumbo de misión inválido')
        if self.plan_pendiente is not None:
            self.plan_pendiente[0].cancel()
            self.plan_pendiente = None
        self.meta = tuple(float(v) for v in xy)
        self.meta_yaw = rumbo
        self.ruta = None
        self.t_recuperacion_margen=None
        self.estado = 'ESPERANDO_META' if self.largada_fijada else 'ESPERANDO_LISTA'
        self.previo = None
        self.motivo = ''
        self.replanificaciones = 0
        self.reintentos_bloqueo=0
        self.t_bloqueo = self.t_inicio_bloqueo = self.t_movimiento = None

    def pausar_inspeccion(self):
        """Pausa M4 sin perder largada ni declarar un fallo."""
        if self.p.regreso_automatico or self._fase() == 'VOLVIENDO':
            raise ValueError('Pausa de inspección sólo durante búsqueda M4')
        if self.plan_pendiente is not None:
            self.plan_pendiente[0].cancel()
            self.plan_pendiente = None
        self.seguidor_ida.detener()
        self.ruta = None
        self.meta = self.meta_yaw = None
        self.previo = None
        self.estado = 'ESPERANDO_META' if self.largada_fijada else 'ESPERANDO_LISTA'
        self.t_movimiento = None
        self.motivo = ''

    def pedir_regreso(self):
        """Cancela la meta pendiente; el siguiente paso válido planifica regreso."""
        if self.p.regreso_automatico:
            raise ValueError('Regreso explícito requiere modo misión')
        if self.largada is None or self.estado == 'EN_LARGADA':
            return
        if self.plan_pendiente is not None:
            self.plan_pendiente[0].cancel()
            self.plan_pendiente = None
        self.ruta = None
        self.solicitud_regreso = True
        self.estado = 'ESPERANDO_REGRESO'
        self.previo = None
        self.replanificaciones = 0
        self.reintentos_bloqueo=0
        self.t_bloqueo = self.t_inicio_bloqueo = self.t_movimiento = None

    @property
    def seguidor(self):
        return self.seguidor_vuelta if self._fase() == 'VOLVIENDO' else self.seguidor_ida

    def _fase(self):
        return self.previo if self.estado in ('BLOQUEADO', 'DETENIDO') else self.estado

    def _fallar(self, motivo):
        self.estado, self.motivo = 'FALLO', motivo
        return Comando.cero()

    def habilitar_planificacion_asincrona(self):
        self.executor_plan=ProcessPoolExecutor(max_workers=1,
            mp_context=multiprocessing.get_context('spawn'))

    def cerrar(self):
        if self.executor_plan is not None:self.executor_plan.shutdown(wait=False,cancel_futures=True)

    def _calcular_plan(self,pose,fase,extra,destino,largada,replan,huella,rumbo_meta=None):
        ruta,regreso,diag=_resolver_plan(self.grilla,self.costos,self.p_costos,
                                      pose,fase,extra,destino,largada,replan,huella,rumbo_meta)
        self.diagnostico_plan=diag
        return ruta,regreso

    def _registrar_traza(self,pose):
        if self.retorno_con_historial or self.largada is None:return
        actual=(pose.x,pose.y,pose.yaw)
        if not self.traza_ejecutada:
            self.traza_integra=math.dist(actual[:2],self.largada[:2])<=.20
            self.traza_ejecutada=[tuple(self.largada)]
        previo=self.traza_ejecutada[-1]
        if math.dist(actual[:2],previo[:2])>.35:
            self.traza_integra=False
        giro=abs(math.atan2(math.sin(actual[2]-previo[2]),math.cos(actual[2]-previo[2])))
        if math.dist(actual[:2],previo[:2])>=.10 or giro>=.30:
            self.traza_ejecutada.append(actual)

    def _regreso_por_historial(self, pose=None):
        self.diagnostico_historial=dict(puntos=len(self.traza_ejecutada),integra=self.traza_integra)
        if not self.traza_integra:
            self.diagnostico_historial['motivo']='Historial sin continuidad desde largada'
            return None
        if not self.traza_ejecutada or (pose is None and self.ultima_pose is None) or self.largada is None:
            self.diagnostico_historial['motivo']='Historial o pose no disponible'
            return None
        if math.dist(self.traza_ejecutada[0][:2],self.largada[:2])>.20:
            self.diagnostico_historial['motivo']='Historial no comienza en largada'
            return None
        actual=tuple(pose) if pose is not None else (self.ultima_pose.x,self.ultima_pose.y,self.ultima_pose.yaw)
        # Las visitas cercanas pueden pertenecer a vueltas muy diferentes.
        # Preferir la más antigua sólo si el enlace completo es admisible.
        ocupada=np.pad(self.grilla.ocupada|self.capa.ocupadas(self.ahora),1,
                       constant_values=True)
        filas,columnas=np.nonzero(ocupada)
        u=(columnas-.5)*self.grilla.resolucion
        v=(filas-.5)*self.grilla.resolucion
        c,s=math.cos(self.grilla.origen_yaw),math.sin(self.grilla.origen_yaw)
        obstaculos=np.column_stack((self.grilla.origen_x+c*u-s*v,
                                   self.grilla.origen_y+s*u+c*v))
        candidatos=[]
        for i,p in enumerate(self.traza_ejecutada):
            d=math.dist(actual[:2],p[:2])
            giro=abs(math.atan2(math.sin(actual[2]-p[2]),math.cos(actual[2]-p[2])))
            if d<=.20:candidatos.append((d+.05*giro,i))
        if not candidatos:
            self.diagnostico_historial.update(motivo='Pose fuera del historial',
                distancia_min_m=min(math.dist(actual[:2],p[:2]) for p in self.traza_ejecutada))
            return None
        # Si la rasterización invalida incluso el enlace al punto más próximo,
        # conservar el respaldo anterior (recorrido físico + guardia en vivo).
        # Ningún atajo nuevo se autoriza con esa excepción.
        _,index=min(candidatos)
        for _,i in candidatos:
            if self._enlace_historial_libre(actual,self.traza_ejecutada[i],obstaculos):
                index=i
                break
        estados=[actual,self.traza_ejecutada[index]]
        k=index
        saltos=0
        while k>0:
            siguiente=k-1
            for j in range(k-1):
                if (math.dist(self.traza_ejecutada[k][:2],self.traza_ejecutada[j][:2])<=.20
                        and self._enlace_historial_libre(self.traza_ejecutada[k],
                                                        self.traza_ejecutada[j],obstaculos)):
                    siguiente=j
                    saltos+=1
                    break
            estados.append(self.traza_ejecutada[siguiente])
            k=siguiente
        self.diagnostico_historial.update(motivo='Historial disponible',indice=index,
                                          saltos_sin_bucles=saltos,puntos_retorno=len(estados))
        if estados[-1]!=tuple(self.largada):estados.append(tuple(self.largada))
        largo=sum(math.dist(a[:2],b[:2]) for a,b in zip(estados,estados[1:]))
        return Ruta([p[:2] for p in estados],largo,largo,[p[2] for p in estados])

    def _enlace_historial_libre(self,a,b,obstaculos):
        """Atajo corto: contiene traslación, giro y celdas completas ocupadas.

        El seguidor gira hacia el rumbo de destino antes de trasladarse.
        El casco conserva todos los ángulos intermedios, además de la banda
        angular operativa. La guardia LiDAR sigue comprobando cada comando.
        """
        if self.grilla.a_celda(*a[:2]) is None or self.grilla.a_celda(*b[:2]) is None:
            return False
        giro=math.atan2(math.sin(b[2]-a[2]),math.cos(b[2]-a[2]))
        yaw=a[2]+giro/2
        huella=dict(self.huella_operativa,heading_uncertainty_rad=
                    self.huella_operativa.get('heading_uncertainty_rad',2*MAX_HEADING_ERROR_RAD)+abs(giro)/2)
        dx,dy=b[0]-a[0],b[1]-a[1]
        direccion=math.atan2(dy,dx)
        hull,error=poligono_barrido(huella,direccion-yaw,math.hypot(dx,dy))
        delta=obstaculos-np.asarray(a[:2])
        c,s=math.cos(direccion),math.sin(direccion)
        projected=np.column_stack((c*delta[:,0]+s*delta[:,1],-s*delta[:,0]+c*delta[:,1]))
        margen=huella['margin_m']+error+self.grilla.resolucion/math.sqrt(2)
        vertices=np.asarray(hull)
        cerca=np.all((projected>=vertices.min(axis=0)-margen)&
                     (projected<=vertices.max(axis=0)+margen),axis=1)
        return not np.any(distancias_poligono(projected[cerca],hull)<=margen)

    def _aplicar_plan(self,result,fase):
        ruta,regreso=result[:2]
        if len(result)>2:self.diagnostico_plan=result[2]
        if ruta is None and fase=='VOLVIENDO':
            if self.retorno_con_historial and self.ruta is not None:
                # Una búsqueda fallida no borra el avance del retorno conocido.
                self.diagnostico_plan['ruta_de_historial_conservada']=True
                return True
            ruta=self._regreso_por_historial()
            if ruta is not None:
                regreso=ruta.largo_m
                self.retorno_con_historial=True
                self.diagnostico_plan=dict(metodo='TRAZA_EJECUTADA',
                    motivo='Regreso por poses realmente recorridas; guardia LiDAR activa',
                    plan_primario=self.diagnostico_plan,historial_puntos=len(ruta.puntos))
        if ruta is None:
            self.ruta=None
            return False
        self.ruta=ruta
        seguidor=self.seguidor_ida if fase=='YENDO' else self.seguidor_vuelta
        rumbo = self.largada[2] if self.p.regreso_automatico else self.yaw_trayecto
        if (not self.p.regreso_automatico and fase == 'YENDO'
                and self.meta_yaw is not None and ruta.orientaciones is None):
            cell = self.grilla.a_celda(*ruta.puntos[0])
            if cell is not None and math.isfinite(self.costos[cell]):
                # Ruta XY con círculo libre: inspeccionar durante la
                # traslación mecanum, conservando la guardia del barrido.
                rumbo = self.meta_yaw
        seguidor.reiniciar(ruta, rumbo)
        self.regreso_estimado_m=regreso
        return True

    def _planificar(self,pose,fase):
        self.yaw_trayecto = pose.yaw
        destino=self.meta if fase=='YENDO' else self.largada[:2]
        args=(pose,fase,self.capa.ocupadas(self.ahora).copy(),destino,self.largada,self.estado=='BLOQUEADO',dict(self.huella_operativa),
              self.meta_yaw if fase=='YENDO' else None)
        if self.executor_plan is None:
            return self._aplicar_plan(self._calcular_plan(*args),fase)
        future=self.executor_plan.submit(_resolver_plan,self.grilla,self.costos,self.p_costos,*args)
        self.plan_pendiente=(future,fase,self.estado=='BLOQUEADO')
        return True

    def _quieto(self, cmd):
        """Registra el último movimiento y corta si lleva demasiado tiempo quieto."""
        if not cmd.es_cero() or self.t_movimiento is None:
            self.t_movimiento = self.ahora
            return cmd
        if self.ahora - self.t_movimiento > self.p.plazo_detenido_s:
            return self._fallar(f'Detenido más de {self.p.plazo_detenido_s:.0f} s')
        return cmd

    def paso(self, e):
        self.ahora = e.ahora
        self.ultima_pose=e.pose
        if self.solicitud_regreso:
            if not (e.lista and e.pose is not None and e.puntos_base is not None):
                return Comando.cero()
            self.solicitud_regreso = False
            if not self._planificar(e.pose, 'VOLVIENDO'):
                return self._fallar('Sin ruta a la largada')
            self.estado, self.t_movimiento = 'VOLVIENDO', e.ahora
            return Comando.cero()
        if self.estado in TERMINALES:
            return Comando.cero()
        if e.std_yaw_rad is not None:
            if not math.isfinite(e.std_yaw_rad) or e.std_yaw_rad<0:
                return self._fallar('Incertidumbre angular inválida')
            # Banda 3 sigma, con piso de 0,10 rad; frenado angular se añade
            # aparte en la guardia. Sin covarianza se conserva el legado 0,20.
            self.huella_operativa=dict(self.huella,
                heading_uncertainty_rad=max(MAX_HEADING_ERROR_RAD,3*e.std_yaw_rad))
        else:self.huella_operativa=dict(self.huella)
        if (e.lista and e.pose is not None and e.puntos_mapa is not None
                and e.estampa_scan is not None and (self.ultima_estampa_capa is None
                    or e.estampa_scan>self.ultima_estampa_capa)):
            self.capa.actualizar(e.ahora,e.puntos_mapa,e.origen_laser_mapa,e.incremento_angular_rad)
            self.ultima_estampa_capa=e.estampa_scan
        if self.plan_pendiente is not None:
            future,fase,replan=self.plan_pendiente
            if not future.done():return self._quieto(Comando.cero())
            self.plan_pendiente=None
            try:
                result=future.result()
            except Exception as error:
                self.diagnostico_plan={'planner_exception_type':type(error).__name__}
                return self._fallar('Error de planificación: '+type(error).__name__)
            if not self._aplicar_plan(result,fase):
                if not replan:return self._fallar('Sin ruta con huella y rumbo')
                self.estado='BLOQUEADO';self.previo=fase
                self.motivo='Sin alternativa; esperando nueva observación'
                self.t_bloqueo=e.ahora
                if self.t_inicio_bloqueo is None:self.t_inicio_bloqueo=e.ahora
            return self._quieto(Comando.cero())
        datos = e.lista and e.pose is not None and e.puntos_base is not None
        if self.estado == 'ESPERANDO_LISTA':
            if not datos:
                return Comando.cero()
            self.largada = (e.pose.x, e.pose.y, e.pose.yaw)
            self.estado = 'ESPERANDO_META'
        if self.estado == 'ESPERANDO_META':
            if not datos:
                return Comando.cero()
            # Todavía no hubo movimiento: conservar la estimación refinada por
            # las actualizaciones quietas de AMCL, hasta aceptar la primera meta.
            if not self.largada_fijada:
                self.largada = (e.pose.x,e.pose.y,e.pose.yaw)
            if self.meta is None:
                return Comando.cero()
            self.largada_fijada = True
            if not self._planificar(e.pose, 'YENDO'):
                return self._fallar('Sin ruta a la meta')
            self.estado, self.t_movimiento = 'YENDO', e.ahora
            return Comando.cero()
        if not datos:
            if self.estado != 'DETENIDO':
                self.previo = self._fase()
                self.estado, self.t_detenido = 'DETENIDO', e.ahora
                self.motivo = 'Datos o localización no utilizables'
                if self.ruta is not None:self.seguidor.detener()
            if e.ahora - self.t_detenido > self.p.plazo_detenido_s:
                return self._fallar(f'Detenido más de {self.p.plazo_detenido_s:.0f} s sin datos')
            return self._quieto(Comando.cero())
        if self.estado == 'DETENIDO':
            self.estado, self.motivo = self.previo, ''
            if self.estado in EN_MARCHA and self.ruta is None and not self._planificar(e.pose, self.estado):
                return self._fallar('Sin ruta al reanudar')
            return Comando.cero()
        if self.estado == 'EN_META':
            if not self.p.regreso_automatico:
                return Comando.cero()
            if e.ahora - self.t_en_meta < self.p.pausa_en_meta_s:
                return Comando.cero()
            if not self._planificar(e.pose, 'VOLVIENDO'):
                return self._fallar('Sin ruta a la largada')
            self.estado, self.t_movimiento = 'VOLVIENDO', e.ahora
            return Comando.cero()
        return self._quieto(self._avanzar(e))

    def _avanzar(self, e):
        self._registrar_traza(e.pose)
        fase = self._fase()
        if self.ruta is None:
            return self._bloqueado(e,fase)
        celda = self.grilla.a_celda(e.pose.x, e.pose.y)
        espacio = float(self.distancia[celda]) if celda is not None else 0.0
        if not self.p.regreso_automatico:
            # M4 propone velocidad de ruta sin penalizar toda pared lateral.
            # La guardia de huella/frenado/curva siguiente limita el comando
            # efectivo por el espacio en la dirección real del movimiento.
            espacio = self.seguidor.p.espacio_rapido_m
        cmd = self.seguidor.paso(e.ahora, e.pose, espacio)
        self.ultima_propuesta=[cmd.vx,cmd.vy,cmd.wz]
        if self.seguidor.llego:
            error = (math.atan2(math.sin(self.meta_yaw-e.pose.yaw),
                               math.cos(self.meta_yaw-e.pose.yaw))
                     if fase == 'YENDO' and self.meta_yaw is not None else 0.)
            if abs(error) <= self.p.tolerancia_yaw_meta_rad:
                self.previo = None
                if fase == 'YENDO':
                    self.estado, self.t_en_meta = 'EN_META', e.ahora
                else:
                    self.estado = 'EN_LARGADA'
                return Comando.cero()
            p = self.seguidor.p
            cmd = type(cmd)(0., 0., max(-p.velocidad_angular_max,
                min(p.velocidad_angular_max, p.ganancia_yaw*error)))
        vx, vy, diag = limitar_velocidad(e.puntos_base, cmd.vx, cmd.vy, self.huella_operativa,
                                         self.p.desaceleracion, self.p.latencia_s,
                                         wz=cmd.wz,desaceleracion_angular=self.p.desaceleracion_angular)
        stamp=e.estampa_scan
        scan_s=(stamp[0]+stamp[1]*1e-9 if isinstance(stamp,(tuple,list))
                and len(stamp)==2 else None)
        recuperacion_vigente=(self.t_recuperacion_margen is None or
            (scan_s is not None and
             0. <= scan_s-self.estampa_inicio_recuperacion <= 5. and
             e.ahora-self.t_recuperacion_margen <= self.p.plazo_detenido_s))
        if diag['bloqueado'] and recuperacion_vigente:
            escape=recuperar_margen(e.puntos_base,self.huella_operativa,
                                   self.p.desaceleracion,self.p.latencia_s)
            if escape is not None:
                # Frenar primero: medir la espera con estampas de scan,
                # no reloj de pared (Gazebo puede correr más lento).
                stamp=e.estampa_scan
                scan_s=(stamp[0]+stamp[1]*1e-9 if isinstance(stamp,(tuple,list))
                        and len(stamp)==2 else None)
                if scan_s is not None and math.isfinite(scan_s):
                    if self.t_recuperacion_margen is None:
                        self.t_recuperacion_margen=e.ahora
                        self.estampa_inicio_recuperacion=scan_s
                    rx,ry,recovery=escape
                    self.seguidor.detener()
                    espera=max(self.seguidor.p.velocidad_max/self.p.desaceleracion,
                               self.seguidor.p.velocidad_angular_max/self.p.desaceleracion_angular)+self.p.latencia_s
                    if scan_s-self.estampa_inicio_recuperacion<espera:
                        self.ultima_guardia=dict(recovery,bloqueado=True,esperando_frenado=True)
                        return Comando.cero()
                    self.ultima_guardia=recovery
                    return Comando(float(rx),float(ry),0.)
        self.ultima_guardia=diag
        if not diag['bloqueado']:
            self.reintentos_bloqueo=0
            self.t_recuperacion_margen=None
            if self.estado == 'BLOQUEADO':
                self.estado, self.previo, self.motivo = fase, None, ''
            self.t_bloqueo = self.t_inicio_bloqueo = None
            return Comando(vx, vy, cmd.wz*diag.get('escala_angular',1.))
        # Los puntos físicos que frenan el robot no pueden descartarse sólo
        # por estar a menos de la tolerancia de una pared del mapa.
        c,s=math.cos(e.pose.yaw),math.sin(e.pose.yaw)
        blocked=[(e.pose.x+c*p[0]-s*p[1],e.pose.y+s*p[0]+c*p[1])
                 for p in diag.get('bloqueantes_base_m',[])]
        self.capa.registrar_bloqueo(e.ahora,blocked)
        self.seguidor.detener()
        return self._bloqueado(e, fase)

    def _bloqueado(self, e, fase):
        if self.estado != 'BLOQUEADO':
            self.previo, self.estado = fase, 'BLOQUEADO'
            self.motivo = 'Guardia de huella: obstáculo en el barrido'
            self.t_bloqueo = e.ahora
            if self.t_inicio_bloqueo is None:
                self.t_inicio_bloqueo = e.ahora
        if e.ahora - self.t_inicio_bloqueo > self.p.plazo_bloqueo_total_s:
            return self._fallar('Bloqueo prolongado')
        if e.ahora - self.t_bloqueo >= self.p.espera_bloqueo_s:
            if self.reintentos_bloqueo >= self.p.max_replanificaciones:
                if fase=='VOLVIENDO' and not self.retorno_con_historial:
                    # Un plan puede existir y no ser ejecutable con el scan
                    # actual. Probar una vez el recorrido realmente hecho,
                    # manteniendo la misma guardia; no repetir indefinidamente.
                    if self._aplicar_plan((None,None),'VOLVIENDO'):
                        self.reintentos_bloqueo=0
                        self.t_bloqueo=self.t_inicio_bloqueo=e.ahora
                        self.estado,self.previo,self.motivo='VOLVIENDO',None,''
                        return Comando.cero()
                return self._fallar('Máximo de replanificaciones agotado')
            self.replanificaciones += 1
            self.reintentos_bloqueo += 1
            self.t_bloqueo = e.ahora
            if e.estampa_scan is None and e.puntos_mapa:
                self.capa.actualizar(e.ahora, e.puntos_mapa)
            if self._planificar(e.pose, fase):
                self.estado, self.previo, self.motivo = fase, None, ''
        return Comando.cero()

    def resumen(self):
        return {
            'estado': 'PLANIFICANDO' if self.plan_pendiente is not None and self.estado not in TERMINALES else self.estado, 'motivo': self.motivo,
            'planificacion_pendiente':self.plan_pendiente is not None,
            'pose_control':([self.ultima_pose.x,self.ultima_pose.y,self.ultima_pose.yaw]
                if self.ultima_pose is not None else None),
            'guardia':self.ultima_guardia,
            'ultima_propuesta':self.ultima_propuesta,
            'margen_huella_m':self.huella['margin_m'],
            'radio_huella_m':self.huella['radius_m'],
            'vertices_huella_m':self.huella['corners_base_m'],
            'obstaculos_recordados':int(self.capa.ocupadas(self.ahora).sum()),
            'banda_yaw_rad':self.huella_operativa.get('heading_uncertainty_rad',2*MAX_HEADING_ERROR_RAD),
            'diagnostico_plan':self.diagnostico_plan,
            'planificador':('SE2' if self.ruta and self.ruta.orientaciones is not None else 'XY'),
            'rumbos_ruta':self.ruta.orientaciones if self.ruta else None,
            'largada': list(self.largada) if self.largada else None,
            'meta': list(self.meta) if self.meta else None,
            'restante_m': self.seguidor.restante_m if self.ruta else None,
            'replanificaciones': self.replanificaciones,
            'reintentos_bloqueo':self.reintentos_bloqueo,
            'retorno_con_historial':self.retorno_con_historial,
            'diagnostico_historial':self.diagnostico_historial,
            'historial_puntos':len(self.traza_ejecutada),
            'quieto_s': (self.ahora - self.t_movimiento) if self.t_movimiento is not None
            else 0.0,
            'regreso_estimado_m': self.regreso_estimado_m,
            'ruta': [list(p) for p in self.ruta.puntos] if self.ruta else None}
