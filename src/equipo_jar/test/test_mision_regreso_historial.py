from dataclasses import replace
import numpy as np
from equipo_jar.coordinador_mision import CoordinadorMision, resolver_busqueda
from equipo_jar.cobertura_visual import CoberturaVisual
from equipo_jar.mision import ParametrosMision
from test_mision import nueva


def test_componente_local_sin_largada_puede_estimar_regreso_por_traza():
    mundo, nav = nueva()
    g = nav.grilla
    extra = np.zeros_like(g.ocupada)
    extra[:, 30] = True
    pose = (2.2, .7, 0.)
    start = (.7, .7, 0.)
    def buscar(historial):
        return resolver_busqueda(g, nav.p_costos, extra, pose, start,
            CoberturaVisual(g), set(), 600., ParametrosMision(), [],
            (0., 0., 0.), nav.huella_operativa, False, historial)
    assert buscar(None)['vista'] is None
    result = buscar(3.)
    assert result['vista'] is not None
    assert result['diagnostico']['retorno_estimado_por_historial']
    assert result['vista']['regreso_m'] >= 3.
    assert result['vista']['x'] > 1.5


def test_historial_debe_unir_largada_y_pose_actual():
    _, nav = nueva()
    nav.largada = (.7, .7, 0.)
    m = CoordinadorMision(nav, 0.)
    assert m.largo_historial((1.2, 1.2, 0.)) is None
    nav.traza_ejecutada = [(.7,.7,0.), (1.2,.7,0.), (1.2,1.2,0.)]
    assert abs(m.largo_historial((1.2,1.25,0.)) - 1.05) < 1e-9
    assert m.largo_historial((2.,2.,0.)) is None
    nav.traza_ejecutada[0] = (2.,2.,0.)
    assert m.largo_historial((1.2,1.2,0.)) is None


def test_presupuesto_usa_mismo_regreso_sin_bucles_sin_cambiar_pose_control():
    _,nav=nueva()
    nav.largada=(.7,.7,0.)
    nav.traza_ejecutada=[nav.largada,(1.2,.7,0.),(1.7,.7,0.),
                         (1.7,1.4,0.),(1.2,1.4,0.),(1.2,.8,0.)]
    m=CoordinadorMision(nav,0.)
    anterior=nav.ultima_pose
    largo=m.largo_historial((1.2,.8,0.))
    assert abs(largo-.6)<1e-9
    assert nav.ultima_pose is anterior
    assert nav.ruta is None and not nav.retorno_con_historial
