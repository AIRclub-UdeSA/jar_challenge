"""La unión orientada estima accesibilidad; no autoriza cruzar esquinas."""
import math
import numpy as np
from equipo_jar.navegacion.grilla import Grilla
from equipo_jar.navegacion.planificador import costo_desde,planificar

def escenario():
    occ=np.ones((5,5),bool)
    occ[1,1]=occ[2,2]=occ[3,3]=False
    g=Grilla(.1,0.,0.,0.,5,5,occ)
    return g,np.where(occ,np.inf,0.)

def test_estimar_componente_diagonal_no_equivale_a_autorizar_ruta():
    g,c=escenario();a=g.a_mundo(1,1);z=g.a_mundo(3,3)
    _,normal=costo_desde(g,c,a)
    _,optimista=costo_desde(g,c,a,diagonales_optimistas=True)
    assert not math.isfinite(normal[3,3])
    assert math.isclose(optimista[3,3],2*math.sqrt(2)*g.resolucion)
    assert planificar(g,c,a,z) is None

def test_estimacion_no_salta_huecos_ni_celdas_ocupadas():
    g,c=escenario();c[2,2]=np.inf
    _,optimista=costo_desde(g,c,g.a_mundo(1,1),diagonales_optimistas=True)
    assert not math.isfinite(optimista[3,3])
