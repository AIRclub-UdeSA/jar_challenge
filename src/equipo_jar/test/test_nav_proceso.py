"""La búsqueda CPU no comparte el GIL del publicador de velocidad."""
import os
import numpy as np
from test_nav_ida_vuelta import Mundo,P_COSTO,P_SEG,HUELLA,P
from equipo_jar.navegacion.ida_vuelta import IdaVuelta,EntradaNavegacion
from equipo_jar.navegacion.tipos import Pose2D

def nav():
 return IdaVuelta(Mundo(np.zeros((60,100),dtype=bool)).grilla,P_COSTO,P_SEG,HUELLA,P)

def test_busqueda_usa_proceso_independiente():
 mission=nav()
 mission.habilitar_planificacion_asincrona()
 try:
  assert mission.executor_plan.submit(os.getpid).result(timeout=10)!=os.getpid()
 finally:mission.cerrar()

def test_plan_y_diagnostico_cruzan_proceso_y_se_aplican_en_padre():
 world=Mundo(np.zeros((60,100),dtype=bool))
 mission=IdaVuelta(world.grilla,P_COSTO,P_SEG,HUELLA,P)
 mission.habilitar_planificacion_asincrona()
 pose=Pose2D(.7,1.5,0.);base,points=world.puntos(pose)
 mission.fijar_meta((2.,1.5))
 try:
  assert mission.paso(EntradaNavegacion(0.,pose,True,base,points)).es_cero()
  result=mission.plan_pendiente[0].result(timeout=10)
  assert len(result)==3 and result[2]['planning_process_id']!=os.getpid()
  assert mission.paso(EntradaNavegacion(.1,pose,True,base,points)).es_cero()
  assert mission.ruta is not None
  assert not mission.paso(EntradaNavegacion(.2,pose,True,base,points)).es_cero()
 finally:mission.cerrar()

def test_fallo_del_worker_termina_en_cero_sin_caer_el_nodo():
 from concurrent.futures import Future
 from concurrent.futures.process import BrokenProcessPool
 mission=nav()
 future=Future();future.set_exception(BrokenProcessPool('fallo inyectado'))
 mission.estado='YENDO';mission.largada=(.7,1.5,0.)
 mission.plan_pendiente=(future,'YENDO',False)
 command=mission.paso(EntradaNavegacion(0.,Pose2D(.7,1.5,0.),True,[],[]))
 assert command.es_cero() and mission.estado=='FALLO'
 assert 'BrokenProcessPool' in mission.motivo
