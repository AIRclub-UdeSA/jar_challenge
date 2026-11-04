"""El LiDAR acredita sólo lo que demuestra vacío; objetos no explicados siguen para la cámara."""
import math

import numpy as np

from equipo_jar.cobertura_visual import CoberturaVisual
from equipo_jar.navegacion.grilla import Grilla


def pasillo():
    # 4 m x 1 m libre entre paredes en filas 0 y 11 (resolución 0,1 m).
    occ = np.zeros((12, 60), dtype=bool); occ[0, :] = True; occ[11, :] = True; occ[:, 59] = True
    return CoberturaVisual(Grilla(.1, 0., 0., 0., 60, 12, occ))


def piso_cerca(cov, xy, r=.12):
    d = np.hypot(*(cov.puntos-np.asarray(xy)).T)
    return np.flatnonzero((d <= r) & ~cov.objetivos_pared)


def test_rayo_hasta_pared_explicada_acredita_su_trayecto_y_la_cara_de_pared():
    cov = pasillo(); origen = (.55, .45)
    n = cov.registrar_lidar(origen, [(5.89, .45)], [True], alcance_m=6.)
    assert n > 0
    assert cov.observada[piso_cerca(cov, (3.0, .45))].all()
    caras = np.flatnonzero(cov.objetivos_pared & (np.hypot(*(cov.puntos-(5.95, .45)).T) <= .12))
    assert len(caras) and cov.observada[caras].any()


def test_objeto_no_explicado_no_acredita_ni_su_lugar_ni_lo_que_tapa():
    cov = pasillo(); origen = (.55, .45)
    cov.registrar_lidar(origen, [(2.0, .45)], [False], alcance_m=6.)
    assert cov.observada[piso_cerca(cov, (1.2, .45))].all()      # antes del objeto: vacío
    assert not cov.observada[piso_cerca(cov, (2.0, .45))].any()  # el objeto: pendiente para la cámara
    assert not cov.observada[piso_cerca(cov, (3.0, .45))].any()  # detrás: nadie lo vio


def test_sin_retorno_o_datos_invalidos_no_acreditan():
    cov = pasillo()
    assert cov.registrar_lidar((.55, .45), [(float('nan'), .5), (float('inf'), 1.)], [True, True]) == 0
    assert cov.registrar_lidar((float('nan'), .5), [(3., .45)], [True]) == 0
    assert cov.registrar_lidar((.55, .45), [(3., .45)], [True, False]) == 0
    assert not cov.observada.any()


def test_alcance_limita_lo_acreditado():
    cov = pasillo()
    cov.registrar_lidar((.55, .45), [(5.89, .45)], [True], alcance_m=2.)
    assert cov.observada[piso_cerca(cov, (1.5, .45))].all()
    assert not cov.observada[piso_cerca(cov, (4.0, .45))].any()


def test_franja_lateral_fuera_del_radio_no_se_acredita():
    cov = pasillo()
    cov.registrar_lidar((.55, .25), [(5.89, .25)], [True], alcance_m=6., radio_m=.10)
    assert not cov.observada[piso_cerca(cov, (3.0, .65), r=.05)].any()


def test_cara_de_pared_opuesta_al_laser_no_se_acredita():
    occ = np.zeros((12, 60), dtype=bool); occ[:, 30] = True   # pared delgada vertical en x=3,0
    cov = CoberturaVisual(Grilla(.1, 0., 0., 0., 60, 12, occ))
    cov.registrar_lidar((.55, .45), [(2.99, .45)], [True], alcance_m=6.)
    lejos = np.flatnonzero(cov.objetivos_pared & (cov.normales[:, 0] > .5)
                           & (np.hypot(*(cov.puntos-(3.05, .45)).T) <= .2))
    assert len(lejos) and not cov.observada[lejos].any()


def test_caja_chica_de_15cm_junto_al_rayo_no_queda_acreditada():
    # Caja de 0,15 m centrada a 0,10 m del rayo: con radio 0,06 su muestra no se acredita.
    cov = pasillo()
    cov.registrar_lidar((.55, .45), [(5.89, .45)], [True], alcance_m=6.)
    centro = np.flatnonzero(~cov.objetivos_pared & (np.hypot(*(cov.puntos-(3.05, .65)).T) <= .11))
    assert len(centro) and not cov.observada[centro].any()


def test_objeto_no_explicado_reabre_y_reserva_para_la_camara():
    # Caja de 0,15 m en (1,125; 1,125) en espacio abierto (repro de revisión): rayos laterales
    # acreditarían sus 4 muestras vecinas; los retornos sobre la caja las reabren y reservan.
    cov = CoberturaVisual(Grilla(.05, 0., 0., 0., 60, 60, np.zeros((60, 60), bool)))
    origen = (.3, 1.125)
    laterales = [(2.9, 1.0), (2.9, 1.25)]
    sobre_caja = [(1.05, 1.10), (1.05, 1.125), (1.05, 1.15)]
    cov.registrar_lidar(origen, laterales+sobre_caja, [True, True, False, False, False], alcance_m=5.)
    vecinas = np.flatnonzero(~cov.objetivos_pared & (np.hypot(*(cov.puntos-(1.125, 1.125)).T) <= .15))
    assert len(vecinas) and not cov.observada[vecinas].any() and cov.solo_camara[vecinas].all()
    # Otro barrido limpio no las acredita: sólo la cámara puede.
    cov.registrar_lidar(origen, laterales, [True, True], alcance_m=5.)
    assert not cov.observada[vecinas].any()
    cov.registrar((.6, 1.125, 0.), 1., True)
    assert cov.observada[vecinas].all() and not cov.solo_camara[vecinas].any()


def test_caja_pegada_a_pared_reabre_la_cara_que_tapa():
    # Repro de revisión: pared x=3, caja de 0,15 m apoyada delante centrada en y=0,425.
    occ = np.zeros((12, 60), dtype=bool); occ[:, 30] = True
    cov = CoberturaVisual(Grilla(.1, 0., 0., 0., 60, 12, occ))
    origen = (.55, .425)
    rayos = [(2.99, .51), (2.99, .56), (2.83, .40), (2.83, .45)]   # dos a la pared, dos a la caja
    cov.registrar_lidar(origen, rayos, [True, True, False, False], alcance_m=6.)
    detras = np.flatnonzero(cov.objetivos_pared & (np.hypot(*(cov.puntos-(3.05, .45)).T) <= .2))
    assert len(detras) and not cov.observada[detras].any()


def test_objeto_ya_visto_por_camara_no_se_reabre_con_el_mismo_barrido():
    # Revisión 14:50: LiDAR -> cámara -> mismo LiDAR no debe reinstalar la obligación.
    cov = CoberturaVisual(Grilla(.05, 0., 0., 0., 60, 60, np.zeros((60, 60), bool)))
    origen = (.3, 1.125)
    barrido = [(2.9, 1.0), (2.9, 1.25), (1.05, 1.10), (1.05, 1.125), (1.05, 1.15)]
    expl = [True, True, False, False, False]
    cov.registrar_lidar(origen, barrido, expl, alcance_m=5.)
    vecinas = np.flatnonzero(~cov.objetivos_pared & (np.hypot(*(cov.puntos-(1.125, 1.125)).T) <= .15))
    cov.registrar((.6, 1.125, 0.), 1., True)
    assert cov.observada[vecinas].all()
    cov.registrar_lidar(origen, barrido, expl, alcance_m=5.)
    assert cov.observada[vecinas].all() and not cov.solo_camara[vecinas].any()


def test_sectores_ausentes_no_pierden_rayos_validos_frente_a_fuerza_bruta():
    # Revisión 14:50: 100 retornos consecutivos de un scan de 1080; la ventana debe ser angular.
    cov_v = CoberturaVisual(Grilla(.05, 0., 0., 0., 120, 120, np.zeros((120, 120), bool)))
    origen = np.array([3., 3.]); ang = np.linspace(-np.pi, np.pi, 1080, endpoint=False)[400:500]
    puntos = origen+2.8*np.column_stack((np.cos(ang), np.sin(ang)))
    cov_v.registrar_lidar(origen, puntos, [True]*len(puntos), alcance_m=5.)
    rel = cov_v.puntos-origen; rho = np.hypot(*rel.T)
    d = rel/np.maximum(rho, 1e-9)[:, None]; dirs = (puntos-origen)/2.8
    t = rel@dirs.T; perp = np.abs(rel[:, 0:1]*dirs[None, :, 1]-rel[:, 1:2]*dirs[None, :, 0])
    esperado = ((t >= 0) & (t <= 2.8-.15) & (perp <= .06)).any(axis=1) & ~cov_v.objetivos_pared
    assert (cov_v.observada == esperado).all()


def test_reserva_reabre_muestras_acreditadas_por_lidar():
    cov = CoberturaVisual(Grilla(.05, 0., 0., 0., 60, 60, np.zeros((60, 60), bool)))
    vecinas = piso_cerca(cov, (1.125, 1.125), .15)
    cov.observada[vecinas] = True
    cov.registrar_lidar((.3, 1.125), [(1.05, 1.125)], [False])
    assert not cov.observada[vecinas].any()
    assert cov.solo_camara[vecinas].all()
