import numpy as np
from equipo_jar.navegacion.grilla import Grilla
from equipo_jar.navegacion.guardia import huella_desde_parametros
from equipo_jar.navegacion.orientado import planificar_orientado
from equipo_jar.navegacion.tipos import Pose2D


def test_refinado_conserva_paredes_y_margen_en_pasillo_pequeno():
    occ=np.ones((40,60),bool);occ[19:21,:]=False
    g=Grilla(.025,0.,0.,0.,60,40,occ)
    fp=huella_desde_parametros(-.014,.014,.014,.008)
    fp['heading_uncertainty_rad']=0.
    start=Pose2D(.2,.5,0.)
    assert planificar_orientado(g,occ,start,(.8,.5),fp) is None
    d={};route=planificar_orientado(g,occ,start,(.8,.5),fp,
        diagnostico=d,plazo_refinado_s=2.)
    assert route is not None,d
    assert d['resolucion_busqueda_m']==.0125
    assert fp['margin_m']==.008 and np.array_equal(g.ocupada,occ)
    # Verificación física independiente de centros de celda: todo el cuerpo
    # más margen cabe entre las caras reales de las paredes originales.
    for _,y in route.puntos:
        assert y-.014-.008>=.475-1e-9
        assert y+.014+.008<=.525+1e-9


def test_timeout_no_activa_presupuesto_extra_de_refinado():
    occ=np.zeros((40,60),bool);g=Grilla(.025,0.,0.,0.,60,40,occ)
    fp=huella_desde_parametros(-.014,.014,.014,.008)
    fp['heading_uncertainty_rad']=0.;d={}
    route=planificar_orientado(g,occ,Pose2D(.2,.5,0.),(1.2,.5),fp,
        plazo_s=0.,diagnostico=d,plazo_refinado_s=15.)
    assert route is None and d['motivo']=='Presupuesto de búsqueda agotado'
    assert 'respaldo_refinado_s' not in d
