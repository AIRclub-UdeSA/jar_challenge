"""Physical task distinctions: unseen through a wall and failed != observed."""
import math
import numpy as np
from equipo_jar.cobertura_visual import CoberturaVisual
from equipo_jar.mision_simple import siguiente_vista
from equipo_jar.mision import clave_vista
from equipo_jar.navegacion.grilla import Grilla


def scene():
    occupied=np.zeros((35,45),bool)
    occupied[[0,-1],:]=True;occupied[:,[0,-1]]=True
    occupied[1:24,22]=True
    g=Grilla(.1,0.,0.,0.,45,35,occupied)
    return g,np.where(occupied,np.inf,0.)


def test_wall_occlusion_remains_unobserved():
    g,costs=scene();cov=CoberturaVisual(g)
    cov.registrar((1.4,1.5,0.),1.,True)
    right=cov.puntos[:,0]>2.3
    assert not cov.observada[right].any()
    view=siguiente_vista(g,costs,(1.4,1.5,0.),cov,set(),(0.,0.,0.))
    assert view is not None
    assert not g.ocupada[g.a_celda(view['x'],view['y'])]


def test_attempted_view_is_not_credited_as_camera_observation():
    g,costs=scene();cov=CoberturaVisual(g)
    p=(1.4,1.5,0.)
    view=siguiente_vista(g,costs,p,cov,set(),(0.,0.,0.))
    attempted={clave_vista(tuple(view[k] for k in ('x','y','yaw')))}
    next_view=siguiente_vista(g,costs,p,cov,attempted,(0.,0.,0.))
    assert next_view is not None
    assert clave_vista(tuple(next_view[k] for k in ('x','y','yaw'))) not in attempted
    assert not cov.observada.any()


def test_live_costmap_prevents_goal_inside_new_obstacle():
    g,costs=scene();cov=CoberturaVisual(g)
    costs[:25,:22]=math.inf
    v=siguiente_vista(g,costs,(3.1,1.5,0.),cov,set(),(0.,0.,0.),
        sugeridas=[dict(x=1.4,y=1.5,yaw=0.)])
    assert v is not None
    assert math.isfinite(costs[g.a_celda(v['x'],v['y'])])
    assert not v['sugerida']


def test_useful_nearby_view_beats_tiny_gain_at_closest_site():
    g,costs=scene()
    class Coverage:
        def ganancia(self,pose,extra=None):
            return 40 if pose[0]>1.7 else 2
    v=siguiente_vista(g,costs,(1.4,1.5,0.),Coverage(),set(),(0.,0.,0.))
    assert v['x']>1.7


def test_prediction_uses_same_live_occlusions_as_observed_coverage():
    g,costs=scene();blocked=np.zeros_like(g.ocupada)
    class Coverage:
        def ganancia(self,pose,extra=None):
            assert extra is blocked
            return 2
    assert siguiente_vista(g,costs,(1.4,1.5,0.),Coverage(),set(),(0.,0.,0.),extra=blocked)


def test_distant_unseen_room_is_compared_with_low_gain_local_views():
    g,costs=scene()
    class Coverage:
        def ganancia(self,pose,extra=None):
            return 1000 if pose[0]>3. else 2
    v=siguiente_vista(g,costs,(1.4,1.5,0.),Coverage(),set(),(0.,0.,0.))
    assert v['x']>3.
