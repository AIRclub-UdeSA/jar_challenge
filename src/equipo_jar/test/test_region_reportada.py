"""Reproducción geométrica genérica; sin coordenadas de mundos de competencia."""
import math
from test_inspeccion_visual import setup


def observe(half=.08,deviation=-.43):
    m,nav=setup();m.reportadas=[(1.05,.7)]
    m.estado='YENDO_VISTA';m.vista=dict(x=1.5,y=.7,yaw=0.,sugerida=False)
    obs=dict(estampa_s=1.,pose_camara=[.7,.7,0.],campo_horizontal_rad=1.4,
             region_roja=True,regiones_rojas=[dict(desvio_rad=deviation,
                 semiancho_rad=half,area_relativa=.02)])
    m.observar(obs,1.,(0.,0.,0.));return m


def test_borde_visible_de_caja_reportada_no_es_objetivo_nuevo():
    # Centro del fragmento fuera de tolerancia, pero intervalo visible solapa.
    assert .43>math.atan2(.15,.35)
    m=observe();assert m.estado=='YENDO_VISTA' and m.pista_rgb is None


def test_rojo_separado_del_intervalo_reportado_sigue_pidiendo_centrado():
    m=observe(.03,-.6);assert m.estado=='BUSCANDO_VISTA' and m.pista_rgb


def test_sin_ancho_valido_conserva_comparacion_original():
    for half in [0.,float('nan'),None,-1.]:
        m=observe(half);assert m.pista_rgb
