"""El respaldo SE2 prefiere holgura, igual que la planificación XY."""
import numpy as np
from test_nav_ida_vuelta import Mundo,P_COSTO,P_SEG,HUELLA,P
from equipo_jar.navegacion.ida_vuelta import IdaVuelta
from equipo_jar.navegacion.tipos import Pose2D

def test_respaldo_se2_se_aparta_de_la_pared_si_hay_espacio():
    world=Mundo(np.zeros((80,120),dtype=bool))
    nav=IdaVuelta(world.grilla,P_COSTO,P_SEG,HUELLA,P)
    pose=Pose2D(.7,.4,0.)
    route,_=nav._calcular_plan(pose,'YENDO',np.zeros_like(world.grilla.ocupada),
                               (3.7,.4),(.7,.4,0.),True,HUELLA)
    assert route is not None
    # El trayecto directo es físicamente posible; el espacio libre permite
    # alejarse de la pared en vez de rozar siempre la banda de detención.
    assert max(y for x,y in route.puntos)>=pose.y+.15-1e-9
    assert route.puntos[0]==(.7,.4)
    assert route.puntos[-1]==(3.7,.4)


def test_heuristica_ponderada_conserva_costos_de_las_aristas():
    from equipo_jar.navegacion.orientado import _heuristica_grilla
    libre=np.ones((1,3),dtype=bool)
    distancia=_heuristica_grilla(libre,(0,2),preferencia=np.array([[1.,3.,5.]]))
    np.testing.assert_allclose(distancia,[[6.,4.,0.]])
