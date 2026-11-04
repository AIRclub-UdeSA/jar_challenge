"""Arbitraje M1/M2 y descarte de planes de una localización anterior."""
from concurrent.futures import Future
from types import SimpleNamespace
import numpy as np
from test_nav_nodo import nodo,preparar_lista,mapa_libre,cerrar,esperar_comando,M1Lista
from test_nav_ida_vuelta import Mundo,P_COSTO,P_SEG,HUELLA,P
from equipo_jar.navegacion.ida_vuelta import IdaVuelta,EntradaNavegacion
from equipo_jar.navegacion.planificador import Ruta
from equipo_jar.navegacion.tipos import Pose2D

class M1Girando(M1Lista):
    estado='GIRANDO'
    utilizable=False
    def paso(self,*args):return .15

def activo():
    node=nodo(modo_navegacion='meta_externa')
    node.al_recibir_mapa(mapa_libre());preparar_lista(node)
    node.ciclo_de_control();node.navegacion.fijar_meta((2.3,1.0))
    esperar_comando(node,node.pub_cmd_vel)
    assert node.navegacion.ruta is not None
    return node

def test_recuperacion_de_m1_conserva_su_giro_y_suspende_traslacion():
    node=activo()
    try:
        node.control=M1Girando();node.ciclo_de_control()
        cmd=node.pub_cmd_vel.mensajes[-1]
        assert cmd.linear.x==cmd.linear.y==0.
        assert cmd.angular.z==.15
        assert node.navegacion.estado=='DETENIDO'
        assert node.navegacion.ruta is None
    finally:cerrar(node)

def test_relocalizacion_planifica_desde_la_nueva_pose():
    node=activo()
    try:
        before=node.navegacion.ruta
        node.control=M1Girando();node.ciclo_de_control()
        preparar_lista(node,pose=(2.,1.,0.))
        esperar_comando(node,node.pub_cmd_vel)
        assert node.navegacion.ruta is not before
        assert node.navegacion.ruta.puntos[0]==(2.,1.)
    finally:cerrar(node)

def test_terminal_de_m2_sigue_quieto_aunque_m1_pida_giro():
    node=activo()
    try:
        node.control=M1Girando()
        for terminal in ('EN_LARGADA','FALLO'):
            node.navegacion.estado=terminal
            node.ciclo_de_control()
            cmd=node.pub_cmd_vel.mensajes[-1]
            assert cmd.linear.x==cmd.linear.y==cmd.angular.z==0.
    finally:cerrar(node)

def test_resultado_pendiente_de_pose_anterior_no_se_aplica():
    world=Mundo(np.zeros((60,100),dtype=bool))
    iv=IdaVuelta(world.grilla,P_COSTO,P_SEG,HUELLA,P)
    future=Future();future.set_running_or_notify_cancel()
    iv.executor_plan=SimpleNamespace(submit=lambda *args:future)
    old=Pose2D(.7,1.5,0.);base,mapa=world.puntos(old)
    iv.fijar_meta((2.,1.5))
    iv.paso(EntradaNavegacion(0.,old,True,base,mapa))
    iv.invalidar_plan_por_localizacion()
    iv.paso(EntradaNavegacion(.1,None,False,None,None))
    future.set_result((Ruta([(.7,1.5),(2.,1.5)],1.3,1.3),1.3))
    iv.executor_plan=None
    pose=Pose2D(1.,1.5,0.);base,mapa=world.puntos(pose)
    iv.paso(EntradaNavegacion(.2,pose,True,base,mapa))
    assert iv.ruta.puntos[0]==(1.,1.5)
