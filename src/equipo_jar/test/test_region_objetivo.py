"""Tras agotar lo cercano, la búsqueda se compromete con la región pendiente más conveniente."""
from dataclasses import replace
import numpy as np
from equipo_jar.cobertura_visual import CoberturaVisual
from equipo_jar.coordinador_mision import resolver_busqueda
from equipo_jar.mision import ParametrosMision, region_objetivo
from equipo_jar.navegacion.planificador import costo_desde
from test_mision_componente import caso


def preparar(pendientes):
    args = list(caso()); g = args[0]
    cov = CoberturaVisual(g)
    cov.observada[:] = True
    for (x0, x1, y0, y1) in pendientes:
        sel = ((cov.puntos[:, 0] >= x0) & (cov.puntos[:, 0] <= x1)
               & (cov.puntos[:, 1] >= y0) & (cov.puntos[:, 1] <= y1))
        cov.observada[sel] = False
    args[5] = cov
    return args, cov


def test_region_grande_lejana_gana_a_resto_pequeno_cercano():
    # Robot en la sala izquierda (0,9; 1,5). Resto pequeño junto a él; sala derecha entera pendiente.
    args, cov = preparar([(.4, .9, 2.0, 2.4), (3.0, 4.6, .4, 2.6)])
    g = args[0]; costos = np.where(g.ocupada, np.inf, 1.)
    _, ida = costo_desde(g, costos, (.9, 1.5))
    diag = {}
    m = region_objetivo(g, g.ocupada, cov, ida, ParametrosMision(), diag)
    assert m is not None and diag['region_centro'][0] > 3.0


def test_resolver_va_hacia_la_region_objetivo_y_sin_ella_no():
    # Resto cercano suficiente para ganar sin región objetivo; sala derecha mucho mayor.
    args, cov = preparar([(.4, 1.4, 1.8, 2.6), (3.0, 4.6, .4, 2.6)])
    con = resolver_busqueda(*args)
    assert con['vista'] is not None and con['diagnostico']['region'].get('usada')
    args_sin = list(args); args_sin[8] = replace(ParametrosMision(), region_min_muestras=100000)
    sin = resolver_busqueda(*args_sin)
    assert sin['vista'] is not None and not sin['diagnostico']['region']
    # Con región objetivo la vista elegida está en la sala grande pendiente.
    assert con['vista']['x'] > 3.0 and con['diagnostico']['region']['region_centro'][0] > 3.0


def test_pared_entre_muestras_vecinas_separa_regiones():
    occ = np.zeros((40, 40), bool); occ[:, 20] = True
    from equipo_jar.navegacion.grilla import Grilla
    g = Grilla(.05, 0., 0., 0., 40, 40, occ)
    cov = CoberturaVisual(g)
    _, ida = costo_desde(g, np.where(occ, np.inf, 1.), (.3, 1.))
    diag = {}
    m = region_objetivo(g, occ, cov, ida, replace(ParametrosMision(), region_min_muestras=3), diag)
    # La mitad derecha es inalcanzable: la región elegida queda toda a la izquierda.
    assert m is not None and (cov.puntos[m & ~cov.objetivos_pared][:, 0] < 1.0).all()


def test_desactivada_o_sin_cobertura_real_devuelve_none():
    args, cov = preparar([(3.0, 4.6, .4, 2.6)])
    g = args[0]; _, ida = costo_desde(g, np.where(g.ocupada, np.inf, 1.), (.9, 1.5))
    assert region_objetivo(g, g.ocupada, cov, ida, replace(ParametrosMision(), region_min_muestras=100000)) is None
    assert region_objetivo(g, g.ocupada, object(), ida, ParametrosMision()) is None


def test_obligacion_de_camara_cuenta_aunque_no_sea_region():
    # Región grande a la derecha; una caja vista por LiDAR junto al robot reserva un parche
    # chico (< region_min_muestras) para la cámara: debe seguir contando en la región objetivo.
    args, cov = preparar([(3.0, 4.6, .4, 2.6)])
    g = args[0]
    cerca = np.flatnonzero(~cov.objetivos_pared & (np.hypot(*(cov.puntos-(1.3, 1.5)).T) <= .15))
    cov.observada[cerca] = False; cov.solo_camara[cerca] = True
    _, ida = costo_desde(g, np.where(g.ocupada, np.inf, 1.), (.9, 1.5))
    diag = {}
    m = region_objetivo(g, g.ocupada, cov, ida, ParametrosMision(), diag)
    assert m is not None and diag['region_centro'][0] > 3.0 and m[cerca].all()
