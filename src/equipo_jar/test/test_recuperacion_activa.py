"""Post-LISTA recovery: bounded guarded rotation, unchanged quality, no translation."""
import math
import time
import pytest
import rclpy
from geometry_msgs.msg import Twist
from equipo_jar.localizacion_activa import LocalizacionActiva, Observacion
from equipo_jar.nodo_equipo import NodoEquipo

def obs(seq=11,yaw=0.,quality=.3,safe=True):
 return Observacion(yaw,(.2,-.1,.4),seq,.10,.10,quality,1.,safe)
def recovering(**kwargs):
 c=LocalizacionActiva(recuperacion_activa=True,**kwargs);c.secuencia_lista=10;c.inicio=0.;c.recuperar(0.,'Pérdida de calidad');return c

def test_waits_then_reinitializes_without_initial_motion():
 c=recovering();assert c.paso(2.9,True,obs())==0.;assert c.reinicios_locales==0
 assert c.paso(3.,True,obs())==0.;assert c.reinicios_locales==1 and c.estado=='GIRANDO'
 assert not c.utilizable;assert c.paso(3.1,True,obs())==c.velocidad

def test_no_guard_no_rotation_but_reinitializes():
 c=recovering();assert c.paso(3.,True,obs(safe=False))==0.;assert c.reinicios_locales==1
 assert c.estado=='RECUPERANDO' and not c.utilizable
 c.paso(6.,True,obs(safe=None));assert c.reinicios_locales==2 and c.estado=='RECUPERANDO'

def test_turn_stops_on_missing_or_unsafe_scan():
 for value in (None,obs(safe=False),obs(safe=None)):
  c=recovering();c.paso(3.,True,obs());assert c.paso(3.2,True,value)==0.
  assert c.estado=='RECUPERANDO' and not c.utilizable and not c.giro_activo

def test_turn_stops_at_bound_requires_new_stable_poses():
 c=recovering();c.recuperar(97.,'Nueva pérdida');c.paso(100.,True,obs())  # initial35s budget must not apply afterLISTA
 assert c.estado=='GIRANDO';assert c.paso(100.1,True,obs(12,yaw=.62))==0.
 assert c.estado=='VERIFICANDO' and not c.utilizable
 for t in (100.2,100.3):c.paso(t,True,obs(12,yaw=.62,quality=.75))
 assert not c.utilizable
 for seq in (13,14,15):c.paso(100.+seq*.1,True,obs(seq,yaw=.62,quality=.75))
 assert c.utilizable and c.estado=='LISTA' and not c.giro_activo
 assert c.concordancia_min_supervision==.70 and c.max_std_yaw==.15

def test_bad_quality_never_returns_to_navigation():
 c=recovering();c.paso(3,True,obs());c.paso(3.1,True,obs(12,yaw=.62))
 for seq in range(13,20):c.paso(3.+seq*.1,True,obs(seq,yaw=.62,quality=.65))
 assert not c.utilizable

def test_attempt_limit_has_cooldown_not_terminal_latch():
 c=recovering(max_intentos_activos=3)
 for t in (3.,6.,9.):c.paso(t,True,obs(safe=False))
 assert c.reinicios_locales==3;c.paso(12.,True,obs(safe=False));assert c.reinicios_locales==3
 c.paso(25.,True,obs(safe=False));c.paso(28.,True,obs(safe=False))
 assert c.reinicios_locales==4 and c.estado=='RECUPERANDO'

def test_opposite_turn_direction_and_guard_plan():
 c=recovering();c.paso(3,True,obs(safe=False));angle,direction=c.giro_de_guardia(0.)
 assert direction==-1 and angle==c.giro_recuperacion
 c.paso(6,True,obs());assert c.paso(6.1,True,obs())==-c.velocidad
 assert c.paso(6.2,True,obs(12,yaw=-.62))==0. and c.estado=='VERIFICANDO'

def test_initial_acquisition_still_fails_and_never_requests_reset():
 c=LocalizacionActiva(recuperacion_activa=True,presupuesto=1.)
 c.paso(0,True,None);c.paso(2,True,None)
 assert c.estado=='INSUFICIENTE' and c.reinicios_locales==0 and not c.utilizable

@pytest.mark.parametrize('parameter,value',[('espera_recuperacion_activa',0),('giro_recuperacion',math.pi/2),('max_intentos_activos',0)])
def test_recovery_parameter_validation(parameter,value):
 with pytest.raises(ValueError):LocalizacionActiva(**{parameter:value})

def test_local_reset_uses_current_estimate_covariance_and_publishes_once(monkeypatch):
 rclpy.init();n=NodoEquipo()
 try:
  messages=[];monkeypatch.setattr(n.pub_reinicio_local,'publish',messages.append);n.control.reinicios_locales=1
  n.reinicializar_recuperacion(obs());n.reinicializar_recuperacion(obs())
  assert len(messages)==1;m=messages[0]
  assert m.header.frame_id=='map';assert (m.pose.pose.position.x,m.pose.pose.position.y)==(.2,-.1)
  assert m.pose.covariance[0]==pytest.approx(.09);assert m.pose.covariance[7]==pytest.approx(.09)
  assert m.pose.covariance[35]==pytest.approx(.35**2)
  assert m.pose.pose.orientation.z==pytest.approx(math.sin(.4/2))
 finally:n.destroy_node();rclpy.shutdown()

def test_m2_zero_must_not_overwrite_guarded_recovery_but_no_translation(monkeypatch):
 rclpy.init();n=NodoEquipo()
 try:
  c=recovering();now=time.monotonic();c.inicio_recuperacion=now-3.;c.paso(now,True,obs());n.control=c;n.mapa=object()
  commands=[];monkeypatch.setattr(n.pub_cmd_vel,'publish',commands.append)
  monkeypatch.setattr(n,'observacion',lambda _:obs())
  bad_nav=Twist();bad_nav.linear.x=1.;monkeypatch.setattr(n,'paso_navegacion',lambda *_:bad_nav)
  n.ciclo_de_control();assert commands[-1].linear.x==commands[-1].linear.y==0.
  assert commands[-1].angular.z==.15
  monkeypatch.setattr(n,'observacion',lambda _:obs(safe=False));n.ciclo_de_control()
  assert commands[-1].linear.x==commands[-1].linear.y==commands[-1].angular.z==0.
 finally:n.destroy_node();rclpy.shutdown()
