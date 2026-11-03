"""Coordinación M4; usa las habilidades de M2 y no publica velocidades."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import copy
import math
import multiprocessing

import numpy as np

from equipo_jar.cobertura_visual import CoberturaVisual
from equipo_jar.vistas_percepcion import proponer_vistas, proponer_ajustes, proponer_inspecciones_lidar, mirada_desde_pose
from equipo_jar.mision import ParametrosMision, clave_vista, elegir_vista, region_objetivo
from equipo_jar.navegacion.costos import mapa_costos
from equipo_jar.navegacion.orientado import planificar_orientado, conectividad_orientada
from equipo_jar.navegacion.guardia import poligono_barrido, distancias_poligono
from scipy.spatial import cKDTree
from equipo_jar.navegacion.tipos import Pose2D
from equipo_jar.navegacion.planificador import costo_desde


def filtro_huella_estatica(grilla, ocupada, huella):
    """Descarta destinos que ya colisionan; nunca autoriza una ruta.

    Usa las mismas celdas completas, margen y banda angular de M2. Evalúa
    la pose continua, sin redondearla al centro de otra celda.
    """
    cells = np.argwhere(ocupada)
    points = np.asarray([grilla.a_mundo(int(f), int(c)) for f, c in cells]).reshape(-1, 2)
    tree = cKDTree(points)
    hull, error = poligono_barrido(huella, 0., 0.)
    hull = np.asarray(hull)
    margen = huella['margin_m']+error+grilla.resolucion/math.sqrt(2)
    radio = float(np.max(np.linalg.norm(hull, axis=1)))+margen
    cache = {}
    def libre(x, y, yaw):
        key = (x, y, round(yaw % (2*math.pi), 10))
        if key in cache:
            return cache[key]
        c, s = math.cos(yaw), math.sin(yaw)
        rotation = np.array([[c, -s], [s, c]])
        world = hull @ rotation.T+(x, y)
        local = np.asarray([grilla.a_local(*p) for p in world])
        inside = (np.all(local.min(axis=0) >= margen) and
                  np.all(local.max(axis=0) <= np.array([grilla.ancho, grilla.alto])*grilla.resolucion-margen))
        nearby = tree.query_ball_point((x, y), radio)
        blocked = False
        if inside and nearby:
            base = (points[nearby]-(x, y)) @ rotation
            blocked = bool(np.any(distancias_poligono(base, hull) <= margen))
        cache[key] = bool(inside and not blocked)
        return cache[key]
    return libre


def resolver_busqueda(grilla, p_costos, extra, pose, largada, cobertura,
                      descartadas, restante, parametros, sugeridas, extrinseca,
                      huella=None, sondear_se2=False, regreso_historial_m=None):
    _, distancia = mapa_costos(grilla, p_costos, extra)
    ocupada = grilla.ocupada | extra
    cobertura.grilla = replace(grilla, ocupada=ocupada)
    filtro = filtro_huella_estatica(grilla, ocupada, huella) if huella is not None else None
    if huella is None:
        costos, _ = mapa_costos(grilla, p_costos, extra)
        componente = np.isfinite(costos)
    else:
        # Condición necesaria de accesibilidad del cuerpo completo. No es
        # autorización de movimiento: M2 valida ruta, giros y frenado después.
        componente = conectividad_orientada(grilla, ocupada, Pose2D(*pose), huella)
        if componente is None:
            componente = np.zeros_like(ocupada)
        duro = p_costos.radio_inscripto_m+p_costos.margen_m
        blandos = p_costos.peso*np.maximum(0., 1.-(distancia-duro)/p_costos.alcance_blando_m)
        costos = np.where(componente & ~ocupada, blandos, np.inf)
    # Una unión de rumbos es una cota de accesibilidad. Exigir además que
    # las dos poses cardinales intermedias sean libres crea desconexiones
    # falsas; M2 comprueba después las aristas reales con su barrido.
    _, ida = costo_desde(grilla, costos, pose[:2], diagonales_optimistas=True)
    _, vuelta = costo_desde(grilla, costos, largada[:2], diagonales_optimistas=True)
    retorno_por_historial = False
    if (not np.isfinite(vuelta).any() and regreso_historial_m is not None
            and math.isfinite(regreso_historial_m) and regreso_historial_m >= 0.):
        # Estimación conservadora: volver primero a la pose actual y luego
        # por la traza realmente recorrida. No autoriza ninguna maniobra;
        # M2 conserva planificación y guardia LiDAR al ejecutar el regreso.
        vuelta = ida + regreso_historial_m
        retorno_por_historial = True
    diag_vistas = {}
    diag_region = {}
    objetivo = region_objetivo(grilla, ocupada, cobertura, ida, parametros, diag_region)
    vista = None
    if objetivo is not None:
        # Comprometerse con una región: sólo cuenta la ganancia dentro de ella. Las vistas
        # sugeridas (pistas LiDAR/RGB, inspecciones) conservan su prioridad por bonus.
        enfocada = copy.copy(cobertura)
        enfocada.observada = cobertura.observada | ~objetivo
        vista = elegir_vista(grilla, costos, pose, largada, enfocada, descartadas,
            restante, parametros, sugeridas, extrinseca, diagnostico=diag_vistas,
            distancias=(ida, vuelta), filtro_pose=filtro, rumbos=(32 if sondear_se2 else 8))
        diag_region['usada'] = vista is not None
    if vista is None:
        diag_vistas = {}
        vista = elegir_vista(grilla, costos, pose, largada, cobertura, descartadas,
            restante, parametros, sugeridas, extrinseca, diagnostico=diag_vistas,
            distancias=(ida, vuelta), filtro_pose=filtro, rumbos=(32 if sondear_se2 else 8))
    objetivo_ids = (np.flatnonzero(objetivo if objetivo is not None else
                    ~getattr(cobertura, 'observada', np.zeros(0, dtype=bool))).tolist())
    return dict(vista=vista, muestras_objetivo=objetivo_ids, diagnostico=dict(modo='componente_huella',
        xy=diag_vistas, region=diag_region, retorno_estimado_por_historial=retorno_por_historial, sondeos=[], descartes_sondeo=[], mas_sondeos=False,
        celdas_componente=int(np.count_nonzero(componente)),
        celdas_descartadas_conectividad=int(np.count_nonzero(~ocupada & ~componente)),
        autorizacion_movimiento='M2: ruta y guardia física'))


class CoordinadorMision:
    def __init__(self, nav, inicio_s, parametros=None, executor=None, registrar_diagnosticos=True):
        self.nav = nav
        self.inicio_s = inicio_s
        self.p = parametros or ParametrosMision()
        self.cobertura = CoberturaVisual(nav.grilla)
        self.descartadas = set()
        self.sondeos_pendientes_descartados = set()
        self.estado = 'ESPERANDO_CAMARA'
        self.motivo = ''
        self.reportes = 0
        self.reportadas = []
        self.vista = None
        self.t_vista = None
        self.t_en_meta = None
        self.futuro = None
        self.t_plan = None
        self.sugeridas = []
        self.uso_sugeridas = 0
        self.extrinseca = None
        self.ultima_visual = -math.inf
        self.camara_ausente_desde = None
        self.estampa_visual = None
        self.datos_visuales_en_meta = 0
        self.done = False
        self.finalizar = False
        self.executor = executor
        self.ultimo_reloj = inicio_s
        self.separacion_actual = self.p.separacion_vistas_m
        self.refinamientos = 0
        self.sondear_se2 = False
        self.diagnostico_busqueda = {}
        self.cantidad = None
        self.ultima_pose = None
        self.cajas_planificacion = []
        self.intentos_objetivos = []
        self.objetivos_priorizados = []
        self.candidatos_pausados = []
        self.t_confirmacion = None
        self.pausa_busqueda_confirmacion = None
        self.ultimo_monotono_s = None
        self.reportes_al_confirmar = 0
        self.pendientes_reportes = 0
        self.reinicios_busqueda = 0
        self.confirmacion_incierta = False
        self.historial_vistas = []
        self.retrocesos_usados = set()
        self.t_busqueda_total = None
        self.pose_progreso = None
        self.pendientes_inspeccion = []
        self.pista_rgb = None
        self.centrados_rgb = []
        self.solicitudes_centrado_rgb = []
        self.pistas_lidar = []
        self.observaciones_lidar = []
        self.intentos_lidar = []
        self.ultimo_scan_lidar = None
        self.solicitudes_lidar = []
        self.superficies_lidar_sin_rojo = []
        self.rojo_en_vista = False
        self.registrar_diagnosticos = registrar_diagnosticos
        self.diagnostico_superficies_lidar = []
        self._solo_camara_generadas = np.zeros(len(self.cobertura.puntos), dtype=bool)
        self._diag_objetivo = np.zeros(len(self.cobertura.puntos), dtype=bool)
        self._diag_vistas = np.zeros(len(self.cobertura.puntos), dtype=bool)
        self._diag_vista = None

    @staticmethod
    def _rle(mask):
        ids = np.flatnonzero(mask)
        if not len(ids):
            return []
        grupos = np.split(ids, np.flatnonzero(np.diff(ids) > 1)+1)
        return [[int(g[0]), len(g)] for g in grupos]

    def _iniciar_diagnostico_vista(self, vista, objetivo):
        if not self.registrar_diagnosticos:
            return
        self._diag_objetivo[:] = False
        self._diag_objetivo[np.asarray(objetivo, dtype=int)] = True
        self._diag_vistas[:] = False
        self._diag_vista = dict(pose=[vista[k] for k in ('x', 'y', 'yaw')], resultado='asignada')

    def _anotar_superficie(self, view, motivo):
        if not self.registrar_diagnosticos:
            return
        point = self.objetivo(view)
        ids = np.hypot(*(self.cobertura.puntos-np.asarray(point)).T) <= .30
        self.diagnostico_superficies_lidar.append(dict(
            superficie=list(point), motivo=motivo, mapa_ocluye=view.get('mapa_ocluye'),
            genero_solo_camara=bool(np.any(self._solo_camara_generadas & ids)),
            muestras_generadas=int(np.count_nonzero(self._solo_camara_generadas & ids)),
            solo_camara_pendiente=int(np.count_nonzero(self.cobertura.solo_camara & ids)),
            muestras_cercanas_rle=self._rle(ids)))

    def _resumen_diagnostico_vista(self):
        if self._diag_vista is None:
            return None
        return dict(**self._diag_vista,
            muestras_objetivo=int(np.count_nonzero(self._diag_objetivo)),
            muestras_objetivo_vistas=int(np.count_nonzero(self._diag_objetivo & self._diag_vistas)),
            objetivo_rle=self._rle(self._diag_objetivo),
            vistas_objetivo_rle=self._rle(self._diag_objetivo & self._diag_vistas),
            paso_m=self.cobertura.p.paso_muestreo_m)

    def observar(self, observacion, ahora, extrinseca, extra=None):
        stamp = float(observacion['estampa_s'])
        pose = observacion['pose_camara']
        fov = float(observacion['campo_horizontal_rad'])
        if not (math.isfinite(stamp) and len(pose) == 3
                and all(math.isfinite(v) for v in pose)
                and len(extrinseca) == 3 and all(math.isfinite(v) for v in extrinseca)
                and math.isfinite(fov) and 0 < fov < math.pi):
            return
        if self.estampa_visual is not None and stamp <= self.estampa_visual:
            return
        self.cobertura.p = replace(self.cobertura.p,
            campo_horizontal_rad=fov)
        self.cobertura.registrar(pose, stamp, True, extra)
        if self.registrar_diagnosticos and self._diag_vista is not None:
            self._diag_vistas |= self.cobertura.ultima_visible
        self.estampa_visual = stamp
        self.ultima_visual = ahora
        self.extrinseca = tuple(extrinseca)
        if self.estado == 'OBSERVANDO':
            self.datos_visuales_en_meta += 1
        regions=observacion.get('regiones_rojas',[])
        if self.estado == 'OBSERVANDO' and observacion.get('region_roja'):
            self.rojo_en_vista = True
        if (self.estado=='OBSERVANDO' and self.vista is not None and
                self.vista.get('inspeccion_local')):
            objective=self.objetivo(self.vista)
            for memory in self.pendientes_inspeccion:
                point=(memory['superficie_x'],memory['superficie_y'])
                if math.dist(point,objective)>self.p.distancia_asociacion_m:continue
                expected=math.atan2(point[1]-pose[1],point[0]-pose[0])-pose[2]
                qualities=[r['calidad_visual'] for r in regions if isinstance(r,dict)
                    and all(type(r.get(k)) in (int,float) and math.isfinite(r[k])
                            for k in ('calidad_visual','desvio_rad','semiancho_rad'))
                    and r['calidad_visual']>=0 and
                    abs(math.atan2(math.sin(expected-r['desvio_rad']),math.cos(expected-r['desvio_rad'])))
                    <=max(.08,r['semiancho_rad'])]
                self.actualizar_inspeccion(memory,max(qualities,default=0.))
        if (regions and
                self.estado not in ('REGRESANDO','FINALIZADA','FALLO_RELOJ','CONFIRMANDO')):
            def semiancho(r):
                value=r.get('semiancho_rad',0.)
                return min(value,fov/2) if (type(value) in (int,float)
                    and math.isfinite(value) and value>=0.) else 0.
            regions=[r for r in regions if isinstance(r,dict) and
                type(r.get('desvio_rad')) in (int,float) and math.isfinite(r['desvio_rad']) and
                .10<abs(r['desvio_rad'])<fov/2+.02 and
                # Comparar el intervalo visible, no sólo el centro del fragmento.
                not any(max(0.,abs(math.atan2(math.sin(math.atan2(p[1]-pose[1],p[0]-pose[0])-pose[2]-r['desvio_rad']),
                    math.cos(math.atan2(p[1]-pose[1],p[0]-pose[0])-pose[2]-r['desvio_rad'])))-semiancho(r))
                    <=math.atan2(self.p.distancia_asociacion_m/2,max(.1,math.dist(p,pose[:2])))
                    for p in self.reportadas)]
            # Una inspección pendiente no puede impedir centrar el rojo que
            # está recortado por el borde de la imagen.
            def recorte(r):
                return abs(r['desvio_rad'])+semiancho(r)>=fov/2-.02
            if self.pendientes_inspeccion:
                regions=[r for r in regions if recorte(r)]
            if regions:
                r=max(regions,key=lambda r:r.get('area_relativa',0.))
                heading=pose[2]+r['desvio_rad']-extrinseca[2]
                body_yaw=pose[2]-extrinseca[2];c,s=math.cos(body_yaw),math.sin(body_yaw)
                x=pose[0]-c*extrinseca[0]+s*extrinseca[1]
                y=pose[1]-s*extrinseca[0]-c*extrinseca[1]
                tried=any(math.dist((x,y),p[:2])<.20 and
                    abs(math.atan2(math.sin(heading-p[2]),math.cos(heading-p[2])))<.20
                    for p in self.centrados_rgb)
                if not tried:
                    # Bearing only: no assumed range or victim coordinates.
                    self.pista_rgb=dict(x=x,y=y,yaw=heading,pista_rgb=True,recorte_rgb=recorte(r))
                    interrupted=any(math.dist((x,y),p[:2])<.20 and
                        abs(math.atan2(math.sin(heading-p[2]),math.cos(heading-p[2])))<.20
                        for p in self.solicitudes_centrado_rgb)
                    if (self.estado=='YENDO_VISTA' and self.vista is not None
                            and (not self.vista.get('sugerida') or self.vista.get('pista_lidar')
                                 or recorte(r)) and not interrupted):
                        # Si esta pose no admite el giro, dejar seguir el viaje:
                        # la pista permanece pendiente, sin cancelar cada frame.
                        self.solicitudes_centrado_rgb.append((x,y,heading))
                        self.solicitudes_centrado_rgb=self.solicitudes_centrado_rgb[-64:]
                        self.pausar('BUSCANDO_VISTA','Centrar región roja antes de pedir profundidad')

    @staticmethod
    def objetivo(vista):
        return (vista.get('mirar_x', vista['x']), vista.get('mirar_y', vista['y']))

    def intentos_objetivo(self, vista):
        punto = self.objetivo(vista)
        return next((r[2] for r in self.intentos_objetivos
                     if math.dist(punto, r[:2]) <= self.p.distancia_asociacion_m), 0)

    def anotar_intento(self, vista):
        punto = self.objetivo(vista)
        for r in self.intentos_objetivos:
            if math.dist(punto, r[:2]) <= self.p.distancia_asociacion_m:
                r[2] += 1
                return
        self.intentos_objetivos.append([*punto, 1])

    def pausar(self, estado, motivo):
        if self.futuro is not None:
            self.futuro.cancel()
            self.futuro = None
        self.nav.pausar_inspeccion()
        self.vista = None
        self.estado, self.motivo = estado, motivo

    def actualizar_pistas_lidar(self, entrada):
        if (entrada.estampa_scan is None or entrada.estampa_scan==self.ultimo_scan_lidar
                or entrada.puntos_mapa is None):return
        self.ultimo_scan_lidar=entrada.estampa_scan
        self.diagnostico_superficies_lidar = []
        self.registrar_cobertura_lidar(entrada)
        views=proponer_inspecciones_lidar(entrada.puntos_mapa,self.ultima_pose,
            self.nav.grilla,self.nav.capa.distancia_mapa,self.nav.capa.tolerancia,
            self.extrinseca,entrada.origen_laser_mapa)
        candidates=[]
        for view in views:
            point=self.objetivo(view)
            if any(math.dist(point,p)<=self.p.distancia_asociacion_m for p in self.reportadas):
                self._anotar_superficie(view, 'reportada')
                continue
            memory=next((m for m in self.observaciones_lidar
                if math.dist(point,m['point'])<=self.p.distancia_asociacion_m),None)
            if memory is None:
                memory=dict(point=point,count=0,last=-math.inf)
                self.observaciones_lidar.append(memory)
                self.observaciones_lidar=self.observaciones_lidar[-64:]
            memory['count']=memory['count']+1 if entrada.ahora-memory['last']<2. else 1
            memory['last']=entrada.ahora
            memory['view']=dict(view)
        for memory in self.observaciones_lidar:
            point=memory['point']
            diagnostico_view = memory['view']
            if any(math.dist(point,p)<=self.p.distancia_asociacion_m for p in self.reportadas):
                self._anotar_superficie(diagnostico_view, 'reportada')
                continue
            if any(math.dist(point,p)<=self.p.distancia_asociacion_m
                   for p in self.superficies_lidar_sin_rojo):
                self._anotar_superficie(diagnostico_view, 'ya_vista_sin_rojo')
                continue
            if not .65<=math.dist(point,self.ultima_pose[:2])<=2.5:
                self._anotar_superficie(diagnostico_view, 'fuera_de_alcance')
                continue
            view=dict(memory['view'])
            view.update(mirada_desde_pose(self.ultima_pose,point,self.extrinseca))
            # La cobertura normal ya propone vistas de superficies abiertas.
            # Interrumpirla sólo ante evidencia de una abertura que el mapa tapa.
            if not view.get('mapa_ocluye'):
                self._anotar_superficie(view, 'mapa_no_ocluye')
                continue
            if self.intentos_objetivo(view)>=self.p.max_solicitudes_vista:
                self._anotar_superficie(view, 'limite_intentos_objetivo')
                continue
            # El presupuesto pertenece a la superficie, no a la posición del
            # robot. Cambiar de lugar no habilita inspecciones ciegas infinitas.
            if sum(math.dist(point,old['point'])<=self.p.distancia_asociacion_m
                   for old in self.intentos_lidar)>=self.p.max_solicitudes_vista:
                self._anotar_superficie(view, 'limite_intentos_lidar')
                continue
            if any(math.dist(point,old['point'])<=self.p.distancia_asociacion_m and
                   math.dist(self.ultima_pose[:2],old['anchor'])<.30 for old in self.intentos_lidar):
                self._anotar_superficie(view, 'misma_posicion')
                continue
            angle=abs(math.atan2(math.sin(view['yaw']-self.ultima_pose[2]),
                                 math.cos(view['yaw']-self.ultima_pose[2])))
            if angle<max(.1,self.cobertura.p.campo_horizontal_rad/2-.1):
                self._anotar_superficie(view, 'ya_en_campo')
                continue
            self._anotar_superficie(view, 'admitida')
            candidates.append(view)
        self.pistas_lidar=candidates
        # Una pista ciega no cancela cada tramo desde una posición distinta.
        # Sigue disponible al elegir vistas: un segundo objeto cercano a una
        # caja reportada no se elimina por una asociación espacial supuesta.
        nuevas=[v for v in candidates if not self.cerca_de_reportada(self.objetivo(v))
            and sum(math.dist(self.objetivo(v),old['point'])<=self.p.distancia_asociacion_m
                    for old in self.solicitudes_lidar)<self.p.interrupciones_lidar_max]
        if (nuevas and self.estado=='YENDO_VISTA' and self.vista is not None
                and not self.vista.get('sugerida') and not self.pendientes_inspeccion
                and self.pista_rgb is None):
            self.solicitudes_lidar.extend(dict(point=self.objetivo(v),anchor=self.ultima_pose[:2]) for v in nuevas)
            self.solicitudes_lidar=self.solicitudes_lidar[-128:]
            self.pausar('BUSCANDO_VISTA','Mirar obstáculo no mapeado detectado por LiDAR')

    def registrar_cobertura_lidar(self, entrada):
        """Lo que el LiDAR barre limpio ya no exige ir a mirarlo con la cámara."""
        self._solo_camara_generadas[:] = False
        if (not entrada.lista or entrada.origen_laser_mapa is None
                or self.p.cobertura_lidar_alcance_m <= self.p.cobertura_lidar_radio_m):
            return
        puntos=np.asarray(entrada.puntos_mapa,dtype=float).reshape((-1,2))
        g=self.nav.grilla;distancia=self.nav.capa.distancia_mapa
        explicados=np.zeros(len(puntos),dtype=bool)
        for i,(x,y) in enumerate(puntos):
            if not (math.isfinite(x) and math.isfinite(y)):continue
            cell=g.a_celda(x,y)
            explicados[i]=cell is not None and distancia[cell]<=self.nav.capa.tolerancia
        antes = self.cobertura.solo_camara.copy() if self.registrar_diagnosticos else None
        self.cobertura.registrar_lidar(entrada.origen_laser_mapa,puntos,explicados,
            alcance_m=self.p.cobertura_lidar_alcance_m,radio_m=self.p.cobertura_lidar_radio_m)
        if antes is not None:
            self._solo_camara_generadas = self.cobertura.solo_camara & ~antes

    def vistas_lidar(self):
        """Misma superficie, varias poses pequeñas; todas pasan luego por M2."""
        result=[]
        for view in self.pistas_lidar:
            result.append(view)
            point=np.asarray(self.objetivo(view));ray=np.asarray(self.ultima_pose[:2])-point
            if np.linalg.norm(ray)<.1:continue
            ray/=np.linalg.norm(ray)
            surface=dict(superficie_x=float(point[0]),superficie_y=float(point[1]),
                         normal_x=float(ray[0]),normal_y=float(ray[1]))
            for alternate in proponer_ajustes(surface,self.ultima_pose,self.nav.grilla,
                    self.nav.distancia,self.extrinseca or (0.,0.,0.),
                    despeje=self.nav.p_costos.radio_inscripto_m+self.nav.p_costos.margen_m):
                if math.dist((alternate['x'],alternate['y']),point)<.65:continue
                alternate.pop('inspeccion_local',None);alternate.pop('prioridad_visual',None)
                alternate.update(pista_lidar=True,puntos_soporte=view['puntos_soporte'],
                                 mapa_ocluye=view.get('mapa_ocluye',False))
                result.append(alternate)
        return result

    def actualizar_inspeccion(self, superficie, calidad=None):
        punto=(superficie['superficie_x'],superficie['superficie_y'])
        if any(math.dist(punto,p)<=self.p.distancia_asociacion_m for p in self.reportadas):return
        memory=next((m for m in self.pendientes_inspeccion
                     if math.dist(punto,(m['superficie_x'],m['superficie_y']))<=self.p.distancia_asociacion_m),None)
        if memory is None:
            memory=dict(superficie);self.pendientes_inspeccion.append(memory)
            self.pendientes_inspeccion=self.pendientes_inspeccion[-32:]
        else:
            memory.update({k:superficie[k] for k in ('normal_x','normal_y')})
            if 'calidad_visual' in superficie:memory['calidad_visual']=superficie['calidad_visual']
        if calidad is None:calidad=superficie.get('calidad_visual')
        if (calidad is not None and type(calidad) in (int,float)
                and math.isfinite(calidad) and calidad>=0 and self.ultima_pose is not None):
            previous=memory.get('calidad_anterior');old_pose=memory.get('pose_calidad')
            stable=self.estado=='OBSERVANDO'
            if stable and previous is not None and old_pose is not None:
                delta=np.array(self.ultima_pose[:2])-old_pose
                norm=np.linalg.norm(delta)
                if norm>=.05 and abs(calidad-previous)>=.03:
                    memory['direccion']=(delta/norm*(1 if calidad>previous else -1)).tolist()
            if previous is None or stable:
                memory['calidad_anterior']=calidad;memory['pose_calidad']=list(self.ultima_pose[:2])
            memory['calidad_mejor']=max(calidad,memory.get('calidad_mejor',0.))

    def percepcion(self, datos):
        pending=datos.get('pendientes_reportes')
        if type(pending) is int and pending>=0:self.pendientes_reportes=pending
        count = datos.get('reportes', self.reportes)
        if type(count) is int and count >= self.reportes:
            if count > self.reportes:
                self.t_busqueda_total = None  # Un reporte confirmado es progreso real, incluso quieto.
                self.sugeridas = []
                self.uso_sugeridas = 0
                self.pista_rgb = None
            self.reportes = count
        reportadas = datos.get('reportadas')
        if isinstance(reportadas, list):
            self.reportadas = [(r['x'], r['y']) for r in reportadas
                               if isinstance(r, dict) and all(
                                   type(r.get(k)) in (int, float) and math.isfinite(r[k])
                                   for k in ('x', 'y'))]
            self.registrar_cajas_reportadas()
        self.pendientes_inspeccion=[m for m in self.pendientes_inspeccion
            if not any(math.dist((m['superficie_x'],m['superficie_y']),p)
                       <=self.p.distancia_asociacion_m for p in self.reportadas)]
        for surface in datos.get('diagnostico',{}).get('vistas_incompletas',[]):
            if all(type(surface.get(k)) in (int,float) and math.isfinite(surface[k])
                   for k in ('superficie_x','superficie_y','normal_x','normal_y')):
                self.actualizar_inspeccion(surface)
        if (datos.get('motivo')=='SIN_REGION_ROJA' and self.estado=='OBSERVANDO'
                and self.vista is not None and self.vista.get('inspeccion_local')):
            objective=self.objetivo(self.vista)
            for memory in self.pendientes_inspeccion:
                if math.dist(objective,(memory['superficie_x'],memory['superficie_y']))<=self.p.distancia_asociacion_m:
                    self.actualizar_inspeccion(memory,0.)
        views = datos.get('vistas_sugeridas')
        views = list(views) if isinstance(views, list) else []
        # M3 ya propone vistas laterales con despeje y visión comprobados.
        # Los abanicos estrechos son respaldo, no deben desplazar esas vistas
        # por otras frontales que no revelan profundidad.
        if self.ultima_pose is not None and not views:
            superficies = datos.get('diagnostico', {}).get('vistas_incompletas', [])
            despeje = self.nav.p_costos.radio_inscripto_m+self.nav.p_costos.margen_m
            for superficie in superficies:
                # Conservar el rango desde el que ya se vio la superficie.
                # Acercarse siempre a 0,9 m puede exigir entrar en una ranura
                # donde el robot no cabe, aunque sí pueda mirar desde fuera.
                rango = min(3.5, max(.9, math.dist(self.ultima_pose[:2],
                    (superficie['superficie_x'], superficie['superficie_y']))))
                views.extend(proponer_vistas([superficie], self.ultima_pose,
                    self.nav.grilla, self.nav.distancia, despeje=despeje,
                    angulos=tuple(sign*math.radians(a) for a in (2.5, 5, 10, 15, 20)
                                  for sign in (-1, 1)), comprobar_vision=False,
                    radios_adicionales=(rango, min(3.5, rango*1.3)),
                    desplazamiento_min=max(.10, 2*self.nav.seguidor_ida.p.tolerancia_meta_m+.05), desde_observador=True, incluir_frontal=True))
        if self.ultima_pose is not None:
            for memory in self.pendientes_inspeccion:
                if 'calidad_visual' not in memory:continue
                views.extend(proponer_ajustes(memory,self.ultima_pose,self.nav.grilla,
                    self.nav.distancia,self.extrinseca or (0.,0.,0.),
                    memory.get('direccion'),self.nav.p_costos.radio_inscripto_m+self.nav.p_costos.margen_m))
        validas = [v for v in views if isinstance(v, dict) and all(
            type(v.get(k)) in (int, float) and math.isfinite(v[k])
            for k in ('x', 'y', 'yaw', 'mirar_x', 'mirar_y'))
            and not any(math.dist((v['mirar_x'], v['mirar_y']), r)
                        <= self.p.distancia_asociacion_m for r in self.reportadas)]
        validas = [v for v in validas if self.intentos_objetivo(v) < self.p.max_solicitudes_vista]
        if validas:
            if self.ultima_pose is not None:
                validas.sort(key=lambda v: math.dist((v['x'], v['y']), self.ultima_pose[:2]))
            unicas = {}
            for v in validas:
                key = clave_vista((v['x'], v['y'], v['yaw']))
                if key not in self.descartadas:
                    unicas.setdefault(key, v)
            self.sugeridas = list(unicas.values())[:32]

        if self.ultima_pose is None or self.estado in ('REGRESANDO', 'FINALIZADA', 'FALLO_RELOJ'):
            return
        detecciones = list(datos.get('detecciones', []))
        inciertas=[]
        for rejection in datos.get('diagnostico',{}).get('rechazos',[]):
            center=rejection.get('centro')
            if (rejection.get('motivo')=='INCERTIDUMBRE' and
                    isinstance(center,(list,tuple)) and len(center)==2 and
                    all(type(v) in (int,float) and math.isfinite(v) for v in center)):
                inciertas.append(dict(x=center[0],y=center[1]))
        detecciones.extend(inciertas)
        nuevas = [(d['x'], d['y']) for d in detecciones if isinstance(d, dict)
                  and all(type(d.get(k)) in (int, float) and math.isfinite(d[k]) for k in ('x', 'y'))
                  and not any(math.dist((d['x'], d['y']), p) <= self.p.distancia_asociacion_m
                              for p in self.reportadas+self.candidatos_pausados)]
        if nuevas:
            if self.estado != 'CONFIRMANDO':
                self.pausa_busqueda_confirmacion = (self.ultimo_monotono_s, self.ultimo_reloj)
            self.confirmacion_incierta=bool(inciertas)
            self.candidatos_pausados.extend(nuevas)
            self.reportes_al_confirmar = self.reportes
            self.t_confirmacion = self.ultimo_reloj
            self.pausar('CONFIRMANDO', 'Pausa para confirmar candidato RGB-D')
            return
        nuevos_objetivos = [self.objetivo(v) for v in validas
                           if not any(math.dist(self.objetivo(v), p) <= self.p.distancia_asociacion_m
                                      for p in self.objetivos_priorizados)]
        if (nuevos_objetivos and self.estado == 'YENDO_VISTA' and self.vista is not None
                and (not self.vista['sugerida'] or self.vista.get('pista_lidar'))):
            self.objetivos_priorizados.extend(nuevos_objetivos)
            self.pausar('BUSCANDO_VISTA', 'Prioridad a otra vista de candidato rojo')

    def _descartar(self, fallo=False):
        if self.vista is None:
            return
        x, y, yaw = (self.vista[n] for n in ('x', 'y', 'yaw'))
        if self.registrar_diagnosticos and self._diag_vista is not None:
            self._diag_vista['resultado'] = 'fallo' if fallo else 'observada'
        self.descartadas.add(clave_vista((x, y, yaw)))
        if fallo and not (self.vista.get('inspeccion_local',False) or self.vista.get('pista_lidar',False)):
            # Una misma posición fallida puede reaparecer con centímetros de
            # variación por ruido. Descartar vecindad y cualquier rumbo sin
            # consumir el presupuesto de inspecciones del objeto.
            cx,cy,_ = clave_vista((x,y,yaw))
            pasos = math.ceil(self.p.separacion_minima_m/.05)
            for dx in range(-pasos,pasos+1):
                for dy in range(-pasos,pasos+1):
                    if math.hypot(dx,dy)*.05 <= self.p.separacion_minima_m+1e-9:
                        self.descartadas.add((cx+dx,cy+dy,None))

    def retroceder(self, reloj_s, motivo):
        """Salir de una búsqueda sin avance usando una posición ya alcanzada.

        El historial sólo propone la meta: M2 vuelve a comprobar la ruta viva.
        No se marca cobertura ni se simula progreso por mover el robot.
        """
        restante = self.p.duracion_s-(reloj_s-self.inicio_s)
        for vista in reversed(self.historial_vistas):
            xy = (vista['x'], vista['y'])
            key = clave_vista((*xy, vista['yaw']))
            distancia = math.dist(xy, self.ultima_pose[:2])
            tiempo = (distancia+vista['regreso_m'])/self.p.velocidad_regreso_m_s
            if (distancia < .30 or key in self.retrocesos_usados or
                    tiempo+self.p.reserva_regreso_s >= restante):
                continue
            self.retrocesos_usados.add(key)
            self.pausar('BUSCANDO_VISTA', motivo)
            self.vista = dict(vista, sugerida=False, retroceso=True, camino_m=distancia)
            self._iniciar_diagnostico_vista(self.vista, [])
            self.nav.continuar_meta(xy, vista['yaw'])
            self.estado = 'YENDO_VISTA'
            self.t_vista = reloj_s
            return
        regreso=max(self.nav.regreso_estimado_m or 0.,self.largo_historial(self.ultima_pose) or 0.)
        if restante>regreso/self.p.velocidad_regreso_m_s+self.p.reserva_regreso_s+self.p.observacion_s:
            self.pausar('BUSCANDO_VISTA',motivo+'; reiniciar exploración, faltan víctimas')
            self.descartadas.clear()
            self.sondeos_pendientes_descartados.clear()
            self.retrocesos_usados.clear()
            self.cobertura.observada.fill(False)
            self.t_busqueda_total=None
            self.reinicios_busqueda+=1
            return
        self.regresar(motivo+'; reserva de tiempo sin retroceso admisible')

    def cerca_de_reportada(self,punto):
        radio=max(self.p.distancia_asociacion_m,self.p.radio_exclusion_reportada_m)
        return any(math.dist(punto,p)<=radio for p in self.reportadas)

    def registrar_cajas_reportadas(self):
        """Una víctima reportada es una caja física: planificar alrededor de ella
        desde ya, no sólo cuando el LiDAR la registre de cerca."""
        capa=getattr(self.nav,'capa',None)
        if capa is None:return
        res=self.nav.grilla.resolucion;radio=self.p.lado_caja_reportada_m/2
        for centro in self.reportadas:
            if any(math.dist(centro,c)<=self.p.distancia_asociacion_m for c in self.cajas_planificacion):
                continue
            pasos=np.arange(-radio,radio+1e-9,res/2)
            puntos=[(centro[0]+dx,centro[1]+dy) for dx in pasos for dy in pasos]
            if self.ultima_pose is not None:
                # Nunca marcar bajo el propio robot: la planificación partiría ocupada.
                # Posponer la caja completa: no recordar como insertada una
                # huella vacía o parcial que nunca se volvería a completar.
                if any(math.dist(p,self.ultima_pose[:2])<=.25 for p in puntos):
                    continue
            capa.registrar_fija(puntos)
            self.cajas_planificacion.append(tuple(centro))

    def regresar(self, motivo):
        if self.estado in ('REGRESANDO', 'FINALIZADA'):
            return
        if self.futuro is not None:
            self.futuro.cancel()
            self.futuro = None
        self.estado, self.motivo = 'REGRESANDO', motivo
        self.nav.pedir_regreso()

    def plazo_vista(self):
        """El watchdog mínimo no puede cancelar un viaje largo aún viable."""
        largo=self.vista.get('camino_m',0.) if self.vista is not None else 0.
        viaje=(largo/self.p.velocidad_regreso_m_s+self.p.observacion_s+math.pi/.25
               if math.isfinite(largo) and largo>=0. else 0.)
        return max(self.p.plazo_meta_s,viaje)

    def paso(self, reloj_s, monotono_s, entrada, cantidad):
        self.cantidad = cantidad
        self.ultimo_monotono_s = monotono_s
        if self.done:
            return
        if reloj_s < self.ultimo_reloj - .1:
            self.estado, self.motivo = 'FALLO_RELOJ', 'Reloj de misión retrocedió'
            self.nav.cancelar(self.motivo)
            return
        self.ultimo_reloj = reloj_s
        if self.estado == 'FALLO_RELOJ':
            return
        if not entrada.lista or entrada.pose is None or self.nav.largada is None:
            return
        self.ultima_pose = (entrada.pose.x, entrada.pose.y, entrada.pose.yaw)
        if self.pose_progreso is None:
            self.pose_progreso = self.ultima_pose[:2]
        elif math.dist(self.pose_progreso, self.ultima_pose[:2]) >= .15:
            self.pose_progreso = self.ultima_pose[:2]
            self.t_busqueda_total = None
        if not self.historial_vistas:
            x,y,yaw = self.nav.largada
            self.historial_vistas.append(dict(x=x,y=y,yaw=yaw,regreso_m=0.))
        restante = max(0., self.p.duracion_s - (reloj_s-self.inicio_s))
        regreso = self.nav.regreso_estimado_m or 0.
        if self.vista is not None:
            regreso = max(regreso, self.vista['regreso_m'] +
                          (self.nav.seguidor.restante_m if self.nav.ruta else 0.))
        if (self.reportes >= cantidad or
                restante <= regreso/self.p.velocidad_regreso_m_s+self.p.reserva_regreso_s):
            self.regresar('Presupuesto N completo' if self.reportes >= cantidad
                          else 'Reserva de tiempo de regreso')
        if self.estado == 'REGRESANDO':
            if self.nav.estado == 'EN_LARGADA':
                self.estado = 'FINALIZADA'
                self.finalizar = True
            elif self.nav.estado == 'FALLO':
                self.motivo = 'Regreso sin ruta segura: '+self.nav.motivo
            return
        if self.extrinseca is None or monotono_s-self.ultima_visual > 3.:
            # No marcar cobertura ni mandar vistas sin cámara disponible.
            if self.camara_ausente_desde is None:
                self.camara_ausente_desde = monotono_s
            if monotono_s-self.camara_ausente_desde > 15.:
                self.regresar('Cámara no disponible')
            return
        self.camara_ausente_desde = None
        if self.estado not in ('CONFIRMANDO','REGRESANDO','FINALIZADA'):
            self.actualizar_pistas_lidar(entrada)
        if (self.estado in ('BUSCANDO_VISTA', 'PLANIFICANDO') and
                self.t_busqueda_total is not None and
                (monotono_s-self.t_busqueda_total[0] >= 30. or
                 reloj_s-self.t_busqueda_total[1] >= 20.)):
            self.retroceder(reloj_s, 'Búsqueda sin avance: retroceso por historial')
            return
        if self.estado == 'CONFIRMANDO':
            plazo=max(self.p.observacion_s,15.) if self.confirmacion_incierta else self.p.observacion_s
            if self.reportes == self.reportes_al_confirmar and reloj_s-self.t_confirmacion < plazo:
                return
            # El watchdog mide búsqueda estancada, no la pausa deliberada y
            # acotada de M3. Conservar todo el tiempo de búsqueda ya consumido.
            if self.t_busqueda_total is not None and self.pausa_busqueda_confirmacion is not None:
                wall, sim = self.pausa_busqueda_confirmacion
                self.t_busqueda_total = (
                    self.t_busqueda_total[0] + (max(0., monotono_s-wall) if wall is not None else 0.),
                    self.t_busqueda_total[1] + max(0., reloj_s-sim))
            self.pausa_busqueda_confirmacion = None
            self.estado, self.motivo = 'BUSCANDO_VISTA', ''
        if self.estado == 'ESPERANDO_CAMARA':
            self.estado = 'BUSCANDO_VISTA'
        if self.estado == 'OBSERVANDO':
            if reloj_s-self.t_en_meta < self.p.observacion_s:
                return
            if self.datos_visuales_en_meta < 2:
                if reloj_s-self.t_en_meta < self.p.observacion_s+5.:
                    return
                self.regresar('Sin observaciones válidas en la meta')
                return
            if self.vista is not None and self.vista.get('pista_lidar') and not self.rojo_en_vista:
                point=self.objetivo(self.vista)
                self.superficies_lidar_sin_rojo.append(point)
                self.superficies_lidar_sin_rojo=self.superficies_lidar_sin_rojo[-64:]
                self.pistas_lidar=[v for v in self.pistas_lidar
                    if math.dist(self.objetivo(v),point)>self.p.distancia_asociacion_m]
            self._descartar()
            self.estado = 'BUSCANDO_VISTA'
        if self.estado == 'YENDO_VISTA':
            if self.nav.estado == 'EN_META':
                self.t_busqueda_total = None
                self.sondeos_pendientes_descartados.clear()
                if self.vista is not None and self.vista.get('sugerida',False):
                    self.anotar_intento(self.vista)
                if self.vista is not None and not self.vista.get('retroceso'):
                    x,y,yaw = self.ultima_pose
                    self.historial_vistas.append(dict(x=x,y=y,yaw=yaw,
                        regreso_m=self.vista['regreso_m']))
                    self.historial_vistas = self.historial_vistas[-128:]
                self.estado = 'OBSERVANDO'
                self.t_en_meta = reloj_s
                self.datos_visuales_en_meta = 0
                self.rojo_en_vista = False
                return
            if (self.nav.estado == 'FALLO' or
                    reloj_s-self.t_vista > self.plazo_vista()):
                self.nav.cancelar('Vista inalcanzable o plazo agotado')
                self._descartar(fallo=True)
                self.estado = 'BUSCANDO_VISTA'
            else:
                return
        if self.estado == 'PLANIFICANDO':
            if not self.futuro.done():
                if monotono_s-self.t_plan > 20.:
                    self.retroceder(reloj_s, 'Planificación excedió plazo: retroceso por historial')
                return
            try:
                resultado = self.futuro.result()
                vista = resultado['vista']
                self.diagnostico_busqueda = resultado['diagnostico']
                self.sondeos_pendientes_descartados.update(
                    tuple(key) for key in self.diagnostico_busqueda.get('descartes_sondeo', []))
            except Exception as error:
                self.regresar('Error de búsqueda: '+type(error).__name__)
                return
            self.futuro = None
            if vista is None:
                if self.diagnostico_busqueda.get('mas_sondeos'):
                    self.estado = 'BUSCANDO_VISTA'
                    self.motivo = 'Evaluando otras poses: sondeo acotado no agota vistas'
                    return
                if self.separacion_actual > self.p.separacion_minima_m+1e-9:
                    self.separacion_actual = self.p.separacion_minima_m
                    self.refinamientos += 1
                    self.estado = 'BUSCANDO_VISTA'
                    self.motivo = 'Refinando vistas: presupuesto N aún incompleto'
                    return
                if not self.sondear_se2:
                    self.sondear_se2 = True
                    self.estado = 'BUSCANDO_VISTA'
                    self.motivo = 'Comprobando accesos con huella orientada M2'
                    return
                self.retroceder(reloj_s, 'Sin vista local admisible; búsqueda incompleta')
                return
            self.vista = vista
            self._iniciar_diagnostico_vista(vista, resultado.get('muestras_objetivo', []))
            self.sondear_se2 = False
            self.motivo = ''
            if vista['sugerida']:
                if any(math.dist(self.objetivo(vista), p) <= self.p.distancia_asociacion_m for p in self.reportadas):
                    self.estado = 'BUSCANDO_VISTA'
                    self.vista = None
                    return
                self.uso_sugeridas += 1
                if vista.get('pista_lidar'):
                    self.intentos_lidar.append(dict(point=self.objetivo(vista),anchor=(vista['x'],vista['y'])))
                    self.intentos_lidar=self.intentos_lidar[-64:]
                    self.pistas_lidar=[]
                if vista.get('pista_rgb'):
                    self.centrados_rgb.append((vista['x'],vista['y'],vista['yaw']))
                    self.centrados_rgb=self.centrados_rgb[-64:]
                    self.pista_rgb=None
            self.nav.continuar_meta((vista['x'], vista['y']), vista['yaw'])
            self.t_vista = reloj_s
            self.estado = 'YENDO_VISTA'
            return
        if self.estado == 'BUSCANDO_VISTA':
            if self.t_busqueda_total is None:
                self.t_busqueda_total = (monotono_s, reloj_s)
            if self.executor is None:
                self.executor = ProcessPoolExecutor(max_workers=1,
                    mp_context=multiprocessing.get_context('spawn'))
            pose = (entrada.pose.x, entrada.pose.y, entrada.pose.yaw)
            sugeridas = [v for v in self.sugeridas
                         if self.intentos_objetivo(v) < self.p.max_solicitudes_vista]
            if not sugeridas and self.pista_rgb is None:
                sugeridas.extend(self.vistas_lidar())
            if self.pista_rgb is not None:
                sugeridas.insert(0,self.pista_rgb)
            self.futuro = self.executor.submit(resolver_busqueda,
                self.nav.grilla, self.nav.p_costos,
                self.nav.capa.ocupadas(entrada.ahora).copy(), pose, self.nav.largada,
                copy.deepcopy(self.cobertura),
                self.descartadas | self.sondeos_pendientes_descartados, restante,
                replace(self.p, separacion_vistas_m=self.separacion_actual),
                list(sugeridas), self.extrinseca,
                dict(self.nav.huella_operativa), self.sondear_se2,
                self.largo_historial(pose))
            self.estado = 'PLANIFICANDO'
            self.t_plan = monotono_s

    def largo_historial(self, pose):
        # Presupuestar el mismo retorno que M2 ejecutaría, sin contar de
        # nuevo vueltas eliminadas ni cambiar la pose de control del robot.
        ruta = self.nav._regreso_por_historial(pose)
        return ruta.largo_m if ruta is not None else None

    def resumen(self):
        return dict(estado=self.estado, motivo=self.motivo, reportes=self.reportes,
                    diagnostico_superficies_lidar=self.diagnostico_superficies_lidar,
                    diagnostico_estampa_lidar=self.ultimo_scan_lidar,
                    diagnostico_vista=self._resumen_diagnostico_vista(),
                    pendientes_reportes=self.pendientes_reportes,reinicios_busqueda=self.reinicios_busqueda,
                    cobertura_solo_camara=int(np.count_nonzero(self.cobertura.solo_camara)),
                    cobertura_vista_camara=int(np.count_nonzero(self.cobertura.vista_camara)),
                    cobertura_lidar_tolerancia_pared_m=self.nav.capa.tolerancia,
                    cobertura_muestras=int(np.count_nonzero(self.cobertura.observada)),
                    muestras_totales=len(self.cobertura.observada),
                    vista=self.vista, vistas_descartadas=len(self.descartadas),
                    retrocesos=len(self.retrocesos_usados),
                    requests_otra_vista=self.uso_sugeridas, done=self.done,
                    candidatos_sin_resolver=len(self.pendientes_inspeccion),
                    centrados_rgb=len(self.centrados_rgb),pista_rgb_pendiente=self.pista_rgb is not None,
                    solicitudes_centrado_rgb=len(self.solicitudes_centrado_rgb),
                    inspecciones_lidar=len(self.intentos_lidar),pistas_lidar_pendientes=len(self.pistas_lidar),
                    solicitudes_lidar=len(self.solicitudes_lidar),
                    superficies_lidar_sin_rojo=len(self.superficies_lidar_sin_rojo),
                    objetivos_lidar_pendientes=[dict(x=v['mirar_x'],y=v['mirar_y'],
                        soporte=v['puntos_soporte'],mapa_ocluye=v.get('mapa_ocluye')) for v in self.pistas_lidar],
                    calidad_inspecciones=[dict(x=m['superficie_x'],y=m['superficie_y'],
                        mejor=m.get('calidad_mejor'),direccion=m.get('direccion'))
                        for m in self.pendientes_inspeccion],
                    sondeo_se2=self.sondear_se2, refinamientos=self.refinamientos, separacion_actual_m=self.separacion_actual,
                    diagnostico_busqueda=self.diagnostico_busqueda,
                    objetivo_cumplido=(self.cantidad is not None and self.reportes >= self.cantidad))

    def cerrar(self):
        if self.executor is not None:
            self.executor.shutdown(wait=False, cancel_futures=True)
