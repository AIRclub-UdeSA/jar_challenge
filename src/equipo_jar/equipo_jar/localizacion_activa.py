"""Lógica de localización activa independiente de ROS."""
from dataclasses import dataclass
import math

import numpy as np


def diferencia_angular(a, b):
    """Devuelve la diferencia circular entre dos orientaciones."""
    return math.atan2(math.sin(a - b), math.cos(a - b))


def concordancia_mapa(mapa, rangos, angulo_min, incremento, laser_pose, tolerancia=0.10,
                      deficit_obstaculo=0.40, diagnostico=None):
    """Fracción de aciertos entre los rayos que cuentan; tolera obstáculos no mapeados.

    Con e = distancia a la primera celda ocupada (>= 65) a lo largo del rayo desde la
    pose estimada y m = rango medido:
    - si e < m - tolerancia, el rayo atraviesa una pared del mapa: cuenta en contra,
      aunque su extremo caiga dentro de una pared gruesa;
    - si no, acierta cuando su extremo cae cerca de una celda ocupada (criterio de
      concordancia_mapa_simple);
    - si no acierta y e > m + deficit_obstaculo, quedó claramente antes de la pared
      esperada: puede ser un obstáculo no mapeado y no cuenta;
    - cualquier otro rayo cuenta en contra, así que un error de pose menor que
      deficit_obstaculo no se esconde.
    Devuelve (concordancia, rayos que cuentan).
    """
    resolucion = mapa.info.resolution
    if resolucion <= 0 or mapa.info.width <= 0 or mapa.info.height <= 0:
        return 0.0, 0
    rangos = list(rangos)
    rayos = [(laser_pose[0], laser_pose[1],
              laser_pose[2] + angulo_min + indice * incremento,
              distancia + deficit_obstaculo) for indice, distancia in rangos]
    esperados = _primeras_ocupadas(mapa, rayos)
    laterales = []
    indices_laterales = []
    lateral = min(tolerancia / 2, resolucion)
    for i, ((indice, distancia), esperado) in enumerate(zip(rangos, esperados)):
        if esperado < distancia - tolerancia:
            angulo = laser_pose[2] + angulo_min + indice * incremento
            dx, dy = -math.sin(angulo)*lateral, math.cos(angulo)*lateral
            laterales.extend((laser_pose[0]+signo*dx, laser_pose[1]+signo*dy,
                              angulo, distancia-tolerancia) for signo in (-1,1))
            indices_laterales.append(i)
    cruces = _primeras_ocupadas(mapa, laterales)
    contradicciones = set()
    for j, i in enumerate(indices_laterales):
        if all(e < rangos[i][1]-tolerancia for e in cruces[2*j:2*j+2]):
            contradicciones.add(i)
    extremos = _aciertos_extremos(mapa, rangos, angulo_min, incremento, laser_pose, tolerancia)
    if diagnostico is not None:
        diagnostico['concordancia_simple'] = float(extremos.mean()) if len(extremos) else 0.
    aciertos = contados = 0
    for i, ((indice, distancia), esperado, extremo) in enumerate(zip(rangos, esperados, extremos)):
        # El cruce debe persistir en las dos paralelas dentro de la tolerancia:
        # una esquina rozada por el rayo central no invalida por sí sola la pose.
        if i in contradicciones:
            contados += 1
        elif extremo:
            aciertos += 1
            contados += 1
        elif esperado <= distancia + deficit_obstaculo:
            contados += 1
    return aciertos / contados if contados else 0.0, contados


def _primeras_ocupadas(mapa, rayos):
    """El mismo muestreo de media celda, por bloques; no cambia la geometría."""
    if not rayos:
        return []
    info = mapa.info
    q = info.origin.orientation
    giro = math.atan2(2 * (q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))
    c, s = math.cos(giro), math.sin(giro)
    paso = info.resolution / 2
    datos = np.asarray(mapa.data)
    resultado = []
    for inicio in range(0, len(rayos), 32):
        bloque = rayos[inicio:inicio+32]
        origenes, direcciones, limites = [], [], []
        for x, y, angulo, largo in bloque:
            wx, wy = x-info.origin.position.x, y-info.origin.position.y
            origenes.append(((c*wx+s*wy)/info.resolution, (-s*wx+c*wy)/info.resolution))
            direcciones.append((math.cos(angulo-giro)/2, math.sin(angulo-giro)/2))
            limites.append(int(largo/paso))
        k = np.arange(max(limites)+1, dtype=np.float64)[None, :]
        origenes, direcciones = np.asarray(origenes), np.asarray(direcciones)
        ix = np.floor(origenes[:,0,None] + direcciones[:,0,None]*k).astype(np.int64)
        iy = np.floor(origenes[:,1,None] + direcciones[:,1,None]*k).astype(np.int64)
        vigente = ((k <= np.asarray(limites)[:,None]) & (ix >= 0) & (ix < info.width)
                   & (iy >= 0) & (iy < info.height))
        indices = np.clip(iy,0,info.height-1)*info.width + np.clip(ix,0,info.width-1)
        ocupada = vigente & (datos[indices] >= 65)
        primera = np.argmax(ocupada,axis=1)
        resultado.extend(np.where(ocupada.any(axis=1), primera*paso, math.inf).tolist())
    return resultado


def _primera_ocupada(mapa, x, y, angulo, largo):
    """Distancia a la primera celda ocupada (>= 65) del rayo, o inf hasta `largo`."""
    info = mapa.info
    origen = info.origin
    q = origen.orientation
    yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y ** 2 + q.z ** 2))
    c, s = math.cos(yaw), math.sin(yaw)
    paso = info.resolution / 2
    wx, wy = x - origen.position.x, y - origen.position.y
    u, v = (c * wx + s * wy) / info.resolution, (-s * wx + c * wy) / info.resolution
    dc = math.cos(angulo - yaw) * paso / info.resolution
    df = math.sin(angulo - yaw) * paso / info.resolution
    for k in range(int(largo / paso) + 1):
        d = k * paso
        ix, iy = math.floor(u + k * dc), math.floor(v + k * df)
        if (0 <= ix < info.width and 0 <= iy < info.height
                and mapa.data[iy * info.width + ix] >= 65):
            return d
    return math.inf


def _aciertos_extremos(mapa, rangos, angulo_min, incremento, laser_pose, tolerancia):
    """Extremos próximos a centros ocupados, con la misma distancia euclídea."""
    if not rangos:
        return np.zeros(0,dtype=bool)
    info = mapa.info
    q = info.origin.orientation
    giro = math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
    c, s = math.cos(giro), math.sin(giro)
    coordenadas = []
    for indice, distancia in rangos:
        angulo = laser_pose[2]+angulo_min+indice*incremento
        wx = laser_pose[0]+distancia*math.cos(angulo)-info.origin.position.x
        wy = laser_pose[1]+distancia*math.sin(angulo)-info.origin.position.y
        coordenadas.append(((c*wx+s*wy)/info.resolution,(-s*wx+c*wy)/info.resolution))
    xy = np.asarray(coordenadas)
    celda = np.floor(xy).astype(np.int64)
    radio = math.ceil(tolerancia/info.resolution)
    offsets = np.asarray([(dx,dy) for dy in range(-radio,radio+1)
                          for dx in range(-radio,radio+1)],dtype=np.int64)
    vecinos = celda[:,None,:]+offsets[None,:,:]
    valido = ((vecinos[:,:,0]>=0)&(vecinos[:,:,0]<info.width)
              &(vecinos[:,:,1]>=0)&(vecinos[:,:,1]<info.height))
    centro_valido = ((celda[:,0]>=0)&(celda[:,0]<info.width)
                     &(celda[:,1]>=0)&(celda[:,1]<info.height))
    indices = np.clip(vecinos[:,:,1],0,info.height-1)*info.width + np.clip(vecinos[:,:,0],0,info.width-1)
    delta = vecinos+.5-xy[:,None,:]
    cerca = np.hypot(delta[:,:,0],delta[:,:,1])*info.resolution <= tolerancia
    return centro_valido & (valido & cerca & (np.asarray(mapa.data)[indices]>=65)).any(axis=1)


class EvaluadorMapa:
    """Reutiliza calidad sólo ante entradas numéricas idénticas del mismo mapa.

    La vigencia de sensores, pose y TF se comprueba fuera, antes de cada llamada.
    Cada mensaje de mapa recibido construye un evaluador nuevo.
    """
    def __init__(self,mapa):
        self.mapa=mapa
        self.clave=None
        self.resultado=None
        self.hits=self.misses=0

    def calcular(self,rangos,angulo_min,incremento,laser_pose,tolerancia,deficit):
        key=(tuple(rangos),angulo_min,incremento,tuple(laser_pose),tolerancia,deficit)
        if key!=self.clave:
            self.misses+=1
            detail={}
            result=concordancia_mapa(self.mapa,rangos,angulo_min,incremento,
                laser_pose,tolerancia,deficit,diagnostico=detail)
            self.clave=key
            self.resultado=(result,detail)
        else:self.hits+=1
        result,detail=self.resultado
        return result,dict(detail)


def concordancia_mapa_simple(mapa, rangos, angulo_min, incremento, laser_pose, tolerancia=0.10):
    """Criterio M1 original, vectorizado sin cambiar extremos ni tolerancia."""
    rangos = list(rangos)
    if mapa.info.resolution<=0 or mapa.info.width<=0 or mapa.info.height<=0:
        return 0.,0
    aciertos = _aciertos_extremos(mapa,rangos,angulo_min,incremento,laser_pose,tolerancia)
    return float(aciertos.mean()) if len(aciertos) else 0.,len(rangos)


def _concordancia_mapa_simple_escalar(mapa, rangos, angulo_min, incremento, laser_pose, tolerancia=0.10):
    """Criterio original de M1: fracción de extremos válidos cerca de celdas ocupadas.

    Se conserva para comparar y diagnosticar el criterio tolerante a obstáculos.
    """
    resolucion = mapa.info.resolution
    if resolucion <= 0 or mapa.info.width <= 0 or mapa.info.height <= 0:
        return 0.0, 0
    origen = mapa.info.origin
    q = origen.orientation
    yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y ** 2 + q.z ** 2))
    c, s = math.cos(yaw), math.sin(yaw)
    radio = math.ceil(tolerancia / resolucion)
    aciertos = validos = 0
    for indice, distancia in rangos:
        validos += 1
        angulo = laser_pose[2] + angulo_min + indice * incremento
        wx = laser_pose[0] + distancia * math.cos(angulo) - origen.position.x
        wy = laser_pose[1] + distancia * math.sin(angulo) - origen.position.y
        gx, gy = (c * wx + s * wy) / resolucion, (-s * wx + c * wy) / resolucion
        ix, iy = math.floor(gx), math.floor(gy)
        encontrado = False
        if 0 <= ix < mapa.info.width and 0 <= iy < mapa.info.height:
            for dy in range(-radio, radio + 1):
                for dx in range(-radio, radio + 1):
                    nx, ny = ix + dx, iy + dy
                    if not (0 <= nx < mapa.info.width and 0 <= ny < mapa.info.height):
                        continue
                    if mapa.data[ny * mapa.info.width + nx] < 65:
                        continue
                    if math.hypot(nx + 0.5 - gx, ny + 0.5 - gy) * resolucion <= tolerancia:
                        encontrado = True
                        break
                if encontrado:
                    break
        aciertos += int(encontrado)
    return aciertos / validos if validos else 0.0, validos


@dataclass
class Observacion:
    """Datos necesarios para decidir sin referencia real del simulador."""
    yaw_odom: float
    pose: tuple
    secuencia_pose: int
    std_xy: float
    std_yaw: float
    concordancia: float
    distancia_min: float
    giro_seguro: bool | None = None


class LocalizacionActiva:
    """Controla giros acotados y verificación de calidad después de detenerse."""

    def __init__(self, velocidad=0.15, giro=math.radians(20), max_giros=3,
                 presupuesto=35.0, plazo_giro=12.0, plazo_verificacion=5.0,
                 espacio=0.35, max_std_xy=0.15, max_std_yaw=0.15,
                 concordancia_min=0.80, estabilidad_xy=0.03, estabilidad_yaw=0.035,
                 plazo_recuperacion=15.0, max_recuperaciones=2,
                 plazo_falta_datos=2.0, plazo_falla_calidad=1.0,
                 concordancia_min_supervision=0.70, recuperacion_activa=False,
                 espera_recuperacion_activa=3.0, giro_recuperacion=math.radians(35),
                 max_intentos_activos=3):
        positivos = (velocidad, giro, presupuesto, plazo_giro, plazo_verificacion,
                     espacio, max_std_xy, max_std_yaw, estabilidad_xy, estabilidad_yaw,
                     plazo_recuperacion, plazo_falta_datos, plazo_falla_calidad)
        if not all(math.isfinite(v) and v > 0 for v in positivos):
            raise ValueError('Los límites de localización deben ser positivos y finitos')
        if not isinstance(max_giros, int) or max_giros < 1 or not 0 < concordancia_min <= 1:
            raise ValueError('Cantidad de giros o concordancia inválida')
        if not 0 < concordancia_min_supervision <= 1:
            raise ValueError('Concordancia de supervisión inválida')
        if not isinstance(max_recuperaciones, int) or max_recuperaciones < 1:
            raise ValueError('Cantidad de recuperaciones inválida')
        if (not math.isfinite(espera_recuperacion_activa) or espera_recuperacion_activa <= 0
                or not math.isfinite(giro_recuperacion) or not 0 < giro_recuperacion <= math.pi/4
                or not isinstance(max_intentos_activos, int) or max_intentos_activos < 1):
            raise ValueError('Recuperación activa inválida')
        self.recuperacion_activa = bool(recuperacion_activa)
        self.espera_recuperacion_activa = espera_recuperacion_activa
        self.giro_recuperacion = giro_recuperacion
        self.max_intentos_activos = max_intentos_activos
        self.intentos_activos = self.reinicios_locales = 0
        self.ultimo_reinicio_local = None
        self.giro_activo = False
        self.direccion_activa = 1
        self.plazo_recuperacion = plazo_recuperacion
        self.plazo_falta_datos = plazo_falta_datos
        self.plazo_falla_calidad = plazo_falla_calidad
        self.t_falta_datos = self.t_falla_calidad = None
        self.max_recuperaciones = max_recuperaciones
        self.recuperaciones = 0
        self.secuencia_lista = -1
        self.t_utilizable_desde = None
        self.ultimo_ahora = None
        self.velocidad = velocidad
        self.giro = giro
        self.max_giros = max_giros
        self.presupuesto = presupuesto
        self.plazo_giro = plazo_giro
        self.plazo_verificacion = plazo_verificacion
        self.espacio = espacio
        self.max_std_xy = max_std_xy
        self.max_std_yaw = max_std_yaw
        self.concordancia_min = concordancia_min
        self.concordancia_min_supervision = concordancia_min_supervision
        self.estabilidad_xy = estabilidad_xy
        self.estabilidad_yaw = estabilidad_yaw
        self.estado = 'ESPERANDO_MAPA'
        self.motivo = ''
        self.inicio = None
        self.giros = 0
        self.muestras = []
        self.ultima_secuencia = -1

    def fallar(self, motivo):
        """Fallo inicial terminal; tras LISTA reintenta siempre con el robot detenido."""
        if self.secuencia_lista >= 0:
            return self.recuperar(self.ultimo_ahora, motivo)
        self.estado = 'INSUFICIENTE'
        self.motivo = motivo
        return 0.0

    def recuperar(self, ahora, motivo):
        """Después de LISTA, reintenta quieto sin relajar los criterios de calidad."""
        self.giro_activo = False
        self.recuperaciones += 1
        self.t_utilizable_desde = None
        self.t_falta_datos = self.t_falla_calidad = None
        self.inicio_recuperacion = ahora
        self.estado = 'RECUPERANDO'
        self.motivo = motivo
        self.muestras = []
        self.ultima_secuencia = self.secuencia_lista
        return 0.0

    def paso(self, ahora, mapa_recibido, observacion):
        """Devuelve velocidad angular; cero antes de mapa, al parar o ante fallo."""
        self.ultimo_ahora = ahora
        if not mapa_recibido:
            return 0.0
        if self.inicio is None:
            self.inicio = ahora
            self.estado = 'ESPERANDO_DATOS'
        if self.estado == 'INSUFICIENTE':
            if self.secuencia_lista >= 0:
                return self.recuperar(ahora, 'Reintento detenido tras pérdida de calidad')
            return 0.0
        if self.estado == 'LISTA':
            if self.calidad(observacion):
                if self.t_utilizable_desde is None:
                    self.t_utilizable_desde = ahora
                if ahora-self.t_utilizable_desde >= 30.:
                    self.recuperaciones = 0
                self.t_falta_datos = self.t_falla_calidad = None
                self.secuencia_lista = observacion.secuencia_pose
                return 0.0
            self.t_utilizable_desde = None
            # Una falla breve detiene (no utilizable) sin gastar una recuperación.
            if observacion is None:
                self.t_falla_calidad = None
                if self.t_falta_datos is None:
                    self.t_falta_datos = ahora
                if ahora - self.t_falta_datos <= self.plazo_falta_datos:
                    return 0.0
            else:
                self.t_falta_datos = None
                if self.t_falla_calidad is None:
                    self.t_falla_calidad = ahora
                if ahora - self.t_falla_calidad <= self.plazo_falla_calidad:
                    return 0.0
            self.t_falta_datos = self.t_falla_calidad = None
            motivo = ('Presupuesto de recuperaciones agotado: detenido y reintentando'
                      if self.recuperaciones >= self.max_recuperaciones else
                      'Localización no utilizable: detenido para recuperar observaciones')
            return self.recuperar(ahora, motivo)
        if self.giro_activo and self.estado in ('GIRANDO', 'VERIFICANDO'):
            return self.paso_giro_recuperacion(ahora, observacion)
        if self.estado == 'RECUPERANDO':
            if ahora - self.inicio_recuperacion >= self.plazo_recuperacion:
                return self.recuperar(ahora, 'Plazo de recuperación agotado: detenido y reintentando')
            if observacion is None:
                self.muestras = []
                return 0.0
            if self.estable_y_valida(observacion):
                self.estado = 'LISTA'
                self.t_utilizable_desde = ahora
                self.motivo = ''
                self.secuencia_lista = observacion.secuencia_pose
                self.intentos_activos = 0
                return 0.0
            if (self.recuperacion_activa and self.secuencia_lista >= 0
                    and ahora-self.inicio_recuperacion >= self.espera_recuperacion_activa):
                self.activar_recuperacion(ahora, observacion)
            return 0.0
        if ahora - self.inicio > self.presupuesto:
            return self.fallar('Presupuesto de localización agotado')
        if (self.estado == 'VERIFICANDO_QUIETO'
                and ahora - self.inicio_verificacion > self.plazo_verificacion):
            return self.fallar('Sin espacio para girar y sin calidad quieta suficiente')
        if observacion is None:
            self.motivo = 'Esperando sensores, pose y TF recientes'
            return 0.0
        self.motivo = ''
        if self.estado == 'ESPERANDO_DATOS':
            if (observacion.giro_seguro is False or
                    (observacion.giro_seguro is None and observacion.distancia_min < self.espacio)):
                # Sin espacio para girar, probar observaciones quietas.
                # Mantiene los mismos criterios de calidad y el plazo acotado.
                self.estado = 'VERIFICANDO_QUIETO'
                self.inicio_verificacion = ahora
                self.muestras = []
                self.ultima_secuencia = observacion.secuencia_pose
            else:
                self.estado = 'GIRANDO'
                self.inicio_giro = ahora
                self.yaw_inicial = observacion.yaw_odom
            return 0.0
        if self.estado == 'VERIFICANDO_QUIETO':
            if self.estable_y_valida(observacion):
                self.estado = 'LISTA'
                self.t_utilizable_desde = ahora
                self.secuencia_lista = observacion.secuencia_pose
                return 0.0
            return 0.0
        if self.estado == 'GIRANDO':
            if (observacion.giro_seguro is False or
                    (observacion.giro_seguro is None and observacion.distancia_min < self.espacio)):
                return self.fallar('Espacio insuficiente para el giro')
            delta = diferencia_angular(observacion.yaw_odom, self.yaw_inicial)
            if delta >= self.giro:
                self.giros += 1
                self.estado = 'VERIFICANDO'
                self.inicio_verificacion = ahora
                self.muestras = []
                self.ultima_secuencia = observacion.secuencia_pose
                return 0.0
            if ahora - self.inicio_giro > self.plazo_giro:
                return self.fallar('No hubo progreso suficiente durante el giro')
            return self.velocidad
        if self.estado == 'VERIFICANDO':
            if self.estable_y_valida(observacion):
                self.estado = 'LISTA'
                self.t_utilizable_desde = ahora
                self.secuencia_lista = observacion.secuencia_pose
                return 0.0
            if ahora - self.inicio_verificacion > self.plazo_verificacion:
                if self.giros >= self.max_giros:
                    return self.fallar('Calidad insuficiente después de los giros permitidos')
                self.estado = 'GIRANDO'
                self.inicio_giro = ahora
                self.yaw_inicial = observacion.yaw_odom
        return 0.0


    def giro_de_guardia(self, yaw_odom):
        """Plan de barrido: magnitud y sentido; sensores/guardia siempre externos."""
        if self.giro_activo:
            delta = self.direccion_activa*diferencia_angular(yaw_odom, self.yaw_inicial)
            return max(0., self.giro_recuperacion-delta), self.direccion_activa
        return self.giro_recuperacion, (1 if self.reinicios_locales % 2 == 0 else -1)

    def activar_recuperacion(self, ahora, observacion):
        if observacion is None:
            return
        if self.intentos_activos >= self.max_intentos_activos:
            if ahora-self.ultimo_reinicio_local < self.plazo_recuperacion:
                return
            self.intentos_activos = 0
        self.direccion_activa = 1 if self.reinicios_locales % 2 == 0 else -1
        self.reinicios_locales += 1
        self.intentos_activos += 1
        self.ultimo_reinicio_local = ahora
        self.inicio_recuperacion = ahora
        self.muestras = []
        self.ultima_secuencia = observacion.secuencia_pose
        self.motivo = 'Reinicialización local AMCL; esperando reconvergencia'
        if observacion.giro_seguro is True:
            self.giro_activo = True
            self.estado = 'GIRANDO'
            self.inicio_giro = ahora
            self.yaw_inicial = observacion.yaw_odom
            self.motivo = 'Giro protegido de recuperación tras reinicialización local'

    def paso_giro_recuperacion(self, ahora, observacion):
        """Giro acotado sin traslación; sólo tres poses nuevas válidas devuelven LISTA."""
        if self.estado == 'GIRANDO':
            if observacion is None or observacion.giro_seguro is not True:
                return self.recuperar(ahora, 'Guardia detuvo el giro de recuperación')
            delta = self.direccion_activa*diferencia_angular(observacion.yaw_odom, self.yaw_inicial)
            if delta >= self.giro_recuperacion or ahora-self.inicio_giro > self.plazo_giro:
                self.estado = 'VERIFICANDO'
                self.inicio_verificacion = ahora
                self.muestras = []
                self.ultima_secuencia = observacion.secuencia_pose
                return 0.0
            return self.direccion_activa*self.velocidad
        if observacion is not None and self.estable_y_valida(observacion):
            self.estado = 'LISTA'
            self.giro_activo = False
            self.intentos_activos = 0
            self.t_utilizable_desde = ahora
            self.secuencia_lista = observacion.secuencia_pose
            self.motivo = ''
        elif ahora-self.inicio_verificacion > self.plazo_verificacion:
            return self.recuperar(ahora, 'Sin reconvergencia tras giro: reintentar con límite')
        return 0.0


    @property
    def utilizable(self):
        """LISTA sin falta de datos ni falla de calidad en curso."""
        return (self.estado == 'LISTA' and self.t_falta_datos is None
                and self.t_falla_calidad is None)

    def calidad(self, observacion):
        """Exige datos finitos, covarianza y concordancia con el mapa utilizables."""
        if observacion is None:
            return False
        valores = (*observacion.pose, observacion.yaw_odom, observacion.std_xy,
                   observacion.std_yaw, observacion.concordancia)
        # secuencia_lista se fija sólo después de completar la primera LISTA.
        supervision = self.secuencia_lista >= 0
        minimo = self.concordancia_min_supervision if supervision else self.concordancia_min
        return (all(math.isfinite(v) for v in valores)
                and 0 <= observacion.std_xy <= self.max_std_xy
                and 0 <= observacion.std_yaw <= self.max_std_yaw
                and minimo <= observacion.concordancia <= 1.0)

    def estable_y_valida(self, observacion):
        """Requiere tres poses distintas, estables y con calidad suficiente."""
        if observacion.secuencia_pose != self.ultima_secuencia:
            self.ultima_secuencia = observacion.secuencia_pose
            self.muestras.append(observacion)
            self.muestras = self.muestras[-3:]
        if len(self.muestras) != 3 or not all(self.calidad(m) for m in self.muestras):
            return False
        primera = self.muestras[0].pose
        return all(
            math.hypot(m.pose[0] - primera[0], m.pose[1] - primera[1])
            <= self.estabilidad_xy
            and abs(diferencia_angular(m.pose[2], primera[2])) <= self.estabilidad_yaw
            for m in self.muestras)
