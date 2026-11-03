import math
from equipo_jar.navegacion.planificador import Ruta
from equipo_jar.navegacion.seguimiento import Seguidor
from equipo_jar.navegacion.tipos import Pose2D
from test_nav_ida_vuelta import P_SEG


def test_no_gira_antes_de_ejecutar_microtraslacion_planificada():
    route=Ruta([(.7,.7),(.7125,.7),(.7125,.7),(1.2,.7)],1.,1.,
               [0.,0.,math.pi/2,math.pi/2],precision_vertice_m=.003125)
    # La precisión pertenece al contrato geométrico, no a la tolerancia final.
    follower=Seguidor(P_SEG);follower.reiniciar(route,0.)
    cmd=follower.paso(1.,Pose2D(.7,.7,0.),P_SEG.espacio_rapido_m)
    assert cmd.vx>0. and abs(cmd.wz)<1e-9
    assert follower.indice_orientado==1
