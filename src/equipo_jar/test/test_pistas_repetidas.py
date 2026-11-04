"""Las pistas LiDAR/RGB no repiten interrupciones ni persiguen cajas ya reportadas."""
import math
from equipo_jar.navegacion.ida_vuelta import EntradaNavegacion
from equipo_jar.navegacion.tipos import Pose2D
from test_inspeccion_lidar import setup


def test_caja_reportada_cercana_no_interrumpe_ni_oculta_posible_vecina():
    # Superficie a 0,45 m del centro reportado: antes pasaba el filtro de 0,30 m.
    m, nav, e = setup(); m.reportadas = [(.66, 2.25)]
    m.actualizar_pistas_lidar(e)
    assert m.pistas_lidar and m.estado == 'YENDO_VISTA'
    assert m.vistas_lidar() and not m.solicitudes_lidar


def test_superficie_lejana_de_reportadas_sigue_generando_pista():
    m, nav, e = setup(); m.reportadas = [(.66, 2.6)]
    m.actualizar_pistas_lidar(e)
    assert len(m.pistas_lidar) == 1 and m.estado == 'BUSCANDO_VISTA'


def test_misma_superficie_interrumpe_una_sola_vez_aunque_cambie_la_pose():
    m, nav, e = setup()
    m.actualizar_pistas_lidar(e)
    assert m.estado == 'BUSCANDO_VISTA' and len(m.solicitudes_lidar) == 1
    m.estado = 'YENDO_VISTA'; m.vista = dict(x=1.5, y=.7, yaw=0., sugerida=False)
    m.ultima_pose = (.9, .7, 0.)
    e2 = EntradaNavegacion(2., Pose2D(.9, .7, 0.), True, [], e.puntos_mapa, estampa_scan=(2, 0))
    m.actualizar_pistas_lidar(e2)
    assert m.estado == 'YENDO_VISTA' and len(m.solicitudes_lidar) == 1
    # La pista no se pierde: sigue disponible para el selector normal.
    assert m.pistas_lidar and m.vistas_lidar()


def test_centrado_rgb_lejos_de_reportadas_sigue_activo():
    m, nav, e = setup(); m.estado = 'YENDO_VISTA'; m.vista = dict(x=1.5, y=.7, yaw=0., sugerida=False)
    m.reportadas = [(2.7, .7)]
    obs = dict(estampa_s=1., pose_camara=[.7, .7, 0.], campo_horizontal_rad=1.,
               region_roja=True, regiones_rojas=[dict(desvio_rad=-.40, area_relativa=.01, semiancho_rad=.0)])
    m.observar(obs, 1., (0., 0., 0.))
    assert m.pista_rgb is not None
