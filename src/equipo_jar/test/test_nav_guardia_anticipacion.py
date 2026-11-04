"""Low-speed maneuver must preserve a real braking path and physical margin."""
from equipo_jar.navegacion.guardia import limitar_velocidad,huella_desde_parametros

def wall(x):
    return [(x,-.4+i*.02) for i in range(41)]

def hull():
    return dict(huella_desde_parametros(-.19,.13,.12,.04),heading_uncertainty_rad=0.)

def test_micro_move_fits_before_wall_without_ten_cm_straight_run():
    vx,vy,d=limitar_velocidad(wall(.195),.02,0.,hull(),.5,.15)
    assert 0.<vx<=.02 and vy==0. and not d['bloqueado']

def test_micro_move_still_stops_when_braking_and_buffer_do_not_fit():
    vx,vy,d=limitar_velocidad(wall(.172),.02,0.,hull(),.5,.15)
    assert vx==vy==0. and d['bloqueado']

def test_physical_margin_is_never_waived_for_a_micro_move():
    vx,vy,d=limitar_velocidad(wall(.165),.002,0.,hull(),.5,.15)
    assert vx==vy==0. and d['bloqueado']


def test_velocidad_menor_permita_llegar_al_vertice_antes_de_girar():
    vx,vy,d=limitar_velocidad(wall(.195),.15,0.,hull(),.5,.15)
    assert 0.<vx<.03 and vy==0. and not d['bloqueado']
    assert d['escala'] <= .1
