"""SE2: la orientación permite un corredor que el círculo rechaza."""
import math
import numpy as np
from equipo_jar.navegacion.grilla import Grilla
from equipo_jar.navegacion.guardia import huella_desde_parametros
from equipo_jar.navegacion.orientado import planificar_orientado
from equipo_jar.navegacion.tipos import Pose2D
from equipo_jar.navegacion.seguimiento import Seguidor,ParametrosSeguimiento


def corredor():
    occupied=np.zeros((60,100),dtype=bool)
    occupied[:24,30:70]=True
    occupied[37:,30:70]=True
    return Grilla(.05,0.,0.,0.,100,60,occupied)

def test_respaldo_gira_y_cruza_corredor_con_huella_completa():
    g=corredor();footprint=huella_desde_parametros(-.19,.13,.12,.1)
    # 55 cm: a 45 grados ni siquiera la proyección del cuerpo sin
    # incertidumbre más el margen cabe; alineado con el pasillo sí.
    g.ocupada[:25,30:70]=True
    g.ocupada[36:,30:70]=True
    assert (.32+.24)/math.sqrt(2)+.20>.55
    ruta=planificar_orientado(g,g.ocupada,Pose2D(.7,1.5,math.pi/4),(4.3,1.5),footprint)
    assert ruta is not None
    assert len(set(ruta.orientaciones))>1
    assert ruta.puntos[0]==(.7,1.5) and ruta.puntos[-1]==(4.3,1.5)
    p=ParametrosSeguimiento(.3,.4,.2,.05,1.5,.5,.25,.45,.1,.2)
    follower=Seguidor(p);follower.reiniciar(ruta,math.pi/4)
    x,y,yaw=.7,1.5,math.pi/4;rotated=False
    for k in range(2000):
        cmd=follower.paso(k*.1,Pose2D(x,y,yaw),.5)
        rotated |= abs(cmd.wz)>.05
        x+=(math.cos(yaw)*cmd.vx-math.sin(yaw)*cmd.vy)*.1
        y+=(math.sin(yaw)*cmd.vx+math.cos(yaw)*cmd.vy)*.1
        yaw+=cmd.wz*.1
        if follower.llego:break
    assert follower.llego
    assert rotated
    assert math.dist((x,y),(4.3,1.5))<=.05

def test_respaldo_no_cruza_muro_completo():
    g=corredor();occupied=g.ocupada.copy();occupied[:,50]=True
    assert planificar_orientado(g,occupied,Pose2D(.7,1.5,0.),(4.3,1.5),
           huella_desde_parametros(-.19,.13,.12,.1)) is None


def test_ida_vuelta_se2_con_guardia_fisica_sin_contacto():
    from test_nav_ida_vuelta import Mundo,correr,P_COSTO,P_SEG,HUELLA,P
    from equipo_jar.navegacion.ida_vuelta import IdaVuelta
    world=Mundo(corredor().ocupada)
    iv=IdaVuelta(world.grilla,P_COSTO,P_SEG,HUELLA,P)
    iv.fijar_meta((4.3,1.5))
    rows=correr(iv,world,(.7,1.5,math.pi/4),2500)
    assert iv.estado=='EN_LARGADA',iv.resumen()
    assert math.dist(rows[-1][1:3],(.7,1.5))<=P.tolerancia_largada_m


def test_reanudar_datos_no_repite_busqueda_con_ruta_vigente(monkeypatch):
    from test_nav_ida_vuelta import Mundo,P_COSTO,P_SEG,HUELLA,P
    from equipo_jar.navegacion.ida_vuelta import IdaVuelta,EntradaNavegacion
    world=Mundo(np.zeros((60,100),dtype=bool))
    iv=IdaVuelta(world.grilla,P_COSTO,P_SEG,HUELLA,P)
    pose=Pose2D(.7,1.5,0.);base,mapa=world.puntos(pose)
    iv.fijar_meta((3.,1.5));iv.paso(EntradaNavegacion(0.,pose,True,base,mapa))
    iv.paso(EntradaNavegacion(.1,pose,True,base,mapa))
    iv.paso(EntradaNavegacion(.2,None,False,None,None))
    assert iv.estado=='DETENIDO'
    def prohibido(*args):raise AssertionError('Replanificación redundante tras pausa breve')
    monkeypatch.setattr(iv,'_planificar',prohibido)
    iv.paso(EntradaNavegacion(.3,pose,True,base,mapa))
    assert iv.estado=='YENDO'
    cmd=iv.paso(EntradaNavegacion(.4,pose,True,base,mapa))
    assert 0.<math.hypot(cmd.vx,cmd.vy)<=P_SEG.aceleracion_max*P_SEG.dt_max_s


def test_plan_pendiente_mantiene_cero_y_aplica_resultado_en_control():
    from concurrent.futures import Future
    from types import SimpleNamespace
    from test_nav_ida_vuelta import Mundo,P_COSTO,P_SEG,HUELLA,P
    from equipo_jar.navegacion.ida_vuelta import IdaVuelta,EntradaNavegacion
    from equipo_jar.navegacion.planificador import Ruta
    world=Mundo(np.zeros((60,100),dtype=bool))
    iv=IdaVuelta(world.grilla,P_COSTO,P_SEG,HUELLA,P)
    future=Future()
    iv.executor_plan=SimpleNamespace(submit=lambda *args:future)
    pose=Pose2D(.7,1.5,0.);base,mapa=world.puntos(pose)
    iv.fijar_meta((2.,1.5))
    for k in range(10):
        cmd=iv.paso(EntradaNavegacion(k*.1,pose,True,base,mapa))
        assert cmd.es_cero()
        assert iv.resumen()['estado']=='PLANIFICANDO'
    future.set_result((Ruta([(.7,1.5),(2.,1.5)],1.3,1.3),1.3))
    assert iv.paso(EntradaNavegacion(1.,pose,True,base,mapa)).es_cero()
    assert not iv.paso(EntradaNavegacion(1.1,pose,True,base,mapa)).es_cero()


def test_fft_conserva_dilatacion_asimetrica_y_exterior_ocupado():
    from scipy.ndimage import binary_dilation
    from equipo_jar.navegacion.orientado import _libre_por_kernel
    rng=np.random.default_rng(4729)
    for shape in ((9,11),(31,29)):
        for _ in range(8):
            occupied=rng.random(shape)<.04
            kernel=rng.random((7,9))<.15
            expected=~binary_dilation(occupied,structure=kernel[::-1,::-1],border_value=1)
            assert np.array_equal(_libre_por_kernel(occupied,kernel),expected)

def test_fft_exterior_libre_no_se_filtra_por_bordes():
    from equipo_jar.navegacion.orientado import _libre_por_kernel
    occupied=np.zeros((12,14),dtype=bool)
    kernel=np.ones((5,7),dtype=bool)
    free=_libre_por_kernel(occupied,kernel)
    assert free[2:-2,3:-3].all()
    assert not free[:2].any() and not free[-2:].any()
    assert not free[:,:3].any() and not free[:,-3:].any()


def test_banda_de_yaw_usa_covarianza_sin_reducir_margen_fisico():
    from test_nav_ida_vuelta import Mundo,P_COSTO,P_SEG,HUELLA,P
    from equipo_jar.navegacion.ida_vuelta import IdaVuelta,EntradaNavegacion
    world=Mundo(np.zeros((60,100),dtype=bool))
    iv=IdaVuelta(world.grilla,P_COSTO,P_SEG,HUELLA,P)
    pose=Pose2D(.7,1.5,0.);base,mapa=world.puntos(pose)
    for std,expected in ((None,.20),(.01,.10),(.08,.24),(.03,.10)):
        iv.paso(EntradaNavegacion(0.,pose,True,base,mapa,std))
        assert iv.resumen()['banda_yaw_rad']==expected
        assert iv.huella_operativa['margin_m']==.10

def test_covarianza_invalida_nunca_habilita_movimiento():
    from test_nav_ida_vuelta import Mundo,P_COSTO,P_SEG,HUELLA,P
    from equipo_jar.navegacion.ida_vuelta import IdaVuelta,EntradaNavegacion
    world=Mundo(np.zeros((60,100),dtype=bool))
    iv=IdaVuelta(world.grilla,P_COSTO,P_SEG,HUELLA,P)
    cmd=iv.paso(EntradaNavegacion(0.,Pose2D(.7,1.5,0.),True,[],[],float('nan')))
    assert cmd.es_cero() and iv.estado=='FALLO'


def test_banda_minima_no_permite_invadir_margen_fisico():
    from equipo_jar.navegacion.guardia import limitar_velocidad
    footprint=huella_desde_parametros(-.19,.13,.12,.1)
    footprint['heading_uncertainty_rad']=.1
    points=[(.22,0.)]*30  # 9 cm del frente: dentro de los 10 cm exigidos.
    for vx,vy,w in ((.1,0.,0.),(.1,.05,.2),(0.,0.,.2)):
        x,y,diag=limitar_velocidad(points,vx,vy,footprint,.5,.15,wz=w)
        assert x==y==0. and diag['bloqueado']


def test_corredor_diagonal_usa_barrido_en_lugar_de_poses_laterales():
    from equipo_jar.navegacion.guardia import huella_convexa_x3
    rows,cols=np.indices((60,60))
    occupied=np.abs(rows-cols)*.05/math.sqrt(2)>.225
    grid=Grilla(.05,0.,0.,0.,60,60,occupied)
    footprint=huella_convexa_x3(-.19,.13,.07,.08,.0845,.033,.075,.018)
    footprint['heading_uncertainty_rad']=.1
    route=planificar_orientado(grid,occupied,Pose2D(.625,.625,math.pi/4),(2.375,2.375),footprint)
    assert route is not None
    assert route.puntos[0]==(.625,.625) and route.puntos[-1]==(2.375,2.375)

def test_esquina_realmente_cerrada_sigue_prohibida_por_barrido():
    footprint=huella_desde_parametros(-.005,.005,.005,.005)
    occupied=np.array([[False,True],[True,False]])
    grid=Grilla(.1,0.,0.,0.,2,2,occupied)
    assert planificar_orientado(grid,occupied,Pose2D(.05,.05,0.),(.15,.15),footprint) is None


def test_se2_termina_al_entrar_en_radio_final_sin_exigir_vertice_anterior():
    from equipo_jar.navegacion.planificador import Ruta
    p=ParametrosSeguimiento(.2,.4,.2,.07,1.5,.5,.25,.45,.08,.2)
    follower=Seguidor(p)
    route=Ruta([(0.,0.),(1.,0.),(1.,.05)],1.05,1.05,[0.,0.,0.])
    follower.reiniciar(route,0.)
    follower.paso(0.,Pose2D(0.,0.,0.),.5)
    cmd=follower.paso(.1,Pose2D(.97,.02,.2),.5)
    assert follower.llego and cmd.es_cero()


def test_rumbo_final_exacto_solicitado_se_planifica_y_se_ejecuta():
    g = corredor()
    fp = huella_desde_parametros(-.19, .13, .12, .1)
    route = planificar_orientado(g, g.ocupada, Pose2D(.7, 1.5, 0.),
                                (4.3, 1.5), fp, rumbo_meta=.33)
    assert route is not None
    assert route.rumbo_final == .33
    assert abs(math.atan2(math.sin(route.orientaciones[-1]-.33),
                          math.cos(route.orientaciones[-1]-.33))) < 1e-9
    p = ParametrosSeguimiento(.3, .4, .2, .05, 1.5, .5, .25, .45, .1, .2)
    follower = Seguidor(p)
    follower.reiniciar(route, 0.)
    x, y, yaw = .7, 1.5, 0.
    for k in range(2000):
        cmd = follower.paso(k*.1, Pose2D(x, y, yaw), .5)
        x += (math.cos(yaw)*cmd.vx-math.sin(yaw)*cmd.vy)*.1
        y += (math.sin(yaw)*cmd.vx+math.cos(yaw)*cmd.vy)*.1
        yaw += cmd.wz*.1
        if follower.llego:
            break
    assert follower.llego
    assert math.dist((x, y), (4.3, 1.5)) <= .05
    assert abs(math.atan2(math.sin(yaw-.33), math.cos(yaw-.33))) <= .06


def test_rechaza_rumbo_final_que_no_cabe_aunque_la_posicion_si():
    from equipo_jar.navegacion.guardia import huella_convexa_x3
    g = corredor()
    g.ocupada[:26, 30:70] = True
    g.ocupada[34:, 30:70] = True
    fp = huella_convexa_x3(-.19, .13, .070, .080, .0845, .033, .04, .018)
    fp['heading_uncertainty_rad'] = .1
    start = Pose2D(.7, 1.5, 0.)
    straight = planificar_orientado(g, g.ocupada, start, (2.5, 1.5), fp,
                                   rumbo_meta=0.)
    assert straight is not None
    diag = {}
    sideways = planificar_orientado(g, g.ocupada, start, (2.5, 1.5), fp,
                                   rumbo_meta=math.pi/2, diagnostico=diag)
    assert sideways is None
    assert diag['motivo'] == 'Rumbo final sin huella admisible'


def test_cache_mascaras_invalida_cambios_de_mapa_y_huella():
    from equipo_jar.navegacion.orientado import _libre_por_kernel
    occ=np.zeros((20,20),bool);occ[10,10]=True
    small=np.ones((3,3),bool)
    first=_libre_por_kernel(occ,small)
    assert first[10,13]
    occ[10,13]=True
    second=_libre_por_kernel(occ,small)
    assert not second[10,13]
    occ[10,13]=False
    large=np.ones((7,7),bool)
    assert not _libre_por_kernel(occ,large)[10,13]
    assert _libre_por_kernel(occ,small)[10,13]
