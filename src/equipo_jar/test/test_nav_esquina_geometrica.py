"""Planear una esquina no exige continuar recto después de su vértice."""
import numpy as np
from equipo_jar.navegacion.grilla import Grilla
from equipo_jar.navegacion.guardia import huella_desde_parametros
from equipo_jar.navegacion.orientado import planificar_orientado
from equipo_jar.navegacion.tipos import Pose2D

def test_esquina_con_huella_y_margen_no_exige_recta_extra():
    occ=np.ones((60,60),bool)
    occ[12:28,4:44]=False
    occ[12:52,28:44]=False
    g=Grilla(.025,0.,0.,0.,60,60,occ)
    fp=huella_desde_parametros(-.12,.12,.08,.04)
    fp['heading_uncertainty_rad']=.1
    diag={}
    r=planificar_orientado(g,occ,Pose2D(.35,.5,0.),(.9,1.1),fp,diagnostico=diag,rumbo_meta=0.)
    assert r is not None,diag
    assert r.puntos[-1]==(.9,1.1)
    assert fp['margin_m']==.04 and fp['heading_uncertainty_rad']==.1
