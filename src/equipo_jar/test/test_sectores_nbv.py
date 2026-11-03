"""Regresiones del orden BCD y de la cobertura de cámara por sector."""

import numpy as np

from equipo_jar.sectores_nbv import SectoresNBV


def test_no_avanza_hasta_cubrir_sector_y_aprovecha_lo_visto_en_transito():
    libre = np.ones((20, 30), dtype=bool)
    libre[[0, -1], :] = False
    libre[:, [0, -1]] = False
    libre[8:12, 10:20] = False
    sectores = SectoresNBV(libre, libre, 0.1, (5, 5), 0.85)
    assert len(sectores.orden) > 1
    primero, segundo = sectores.orden[:2]
    visto = np.zeros_like(libre)
    visto[sectores.regiones == segundo] = True  # observado durante el traslado
    assert sectores.avanzar_cubiertos(visto) == primero
    assert sectores.cobertura(visto) == 0
    filas, columnas = np.where(sectores.regiones == primero)
    limite = int(np.ceil(0.85 * len(filas)))
    visto[filas[:limite - 1], columnas[:limite - 1]] = True
    assert sectores.avanzar_cubiertos(visto) == primero
    visto[filas[limite - 1], columnas[limite - 1]] = True
    assert sectores.avanzar_cubiertos(visto) == sectores.orden[2]
    assert sectores.resultados[primero] == 'cubierto'
    assert sectores.resultados[segundo] == 'cubierto'


def test_sector_sin_vistas_se_registra_y_no_cuenta_como_cubierto():
    libre = np.ones((20, 30), dtype=bool)
    libre[[0, -1], :] = False
    libre[:, [0, -1]] = False
    libre[8:12, 10:20] = False
    sectores = SectoresNBV(libre, libre, 0.1, (5, 5), 0.97)
    primero = sectores.actual
    sectores.avanzar_sin_vistas()
    assert sectores.resultados[primero] == 'sin_vistas_alcanzables'
    assert sectores.actual != primero


def test_sectores_omitidos_siguen_disponibles_para_observacion_exterior():
    libre = np.ones((20, 30), dtype=bool)
    libre[8:12, 10:20] = False
    s = SectoresNBV(libre, libre, 0.1, (5, 5), 0.68)
    primero, segundo = s.orden[:2]
    s.avanzar_inaccesible()
    s.avanzar_sin_vistas()
    while s.actual is not None:
        s.avanzar_cubiertos(libre)
    bloque = np.zeros_like(libre)
    f, c = np.argwhere(s.regiones == primero)[0]
    bloque[f, c] = True
    s.actualizar_bloques(bloque)
    esperado = libre & np.isin(s.regiones, [primero, segundo]) & ~bloque
    np.testing.assert_array_equal(s.mascara_observacion(), esperado)
    assert s.resultados[primero] == 'inaccesible'
    assert s.resultados[segundo] == 'sin_vistas_alcanzables'


def test_inicio_fuera_del_margen_transitable_se_ajusta_a_celda_segura():
    libre = np.ones((10, 10), dtype=bool)
    seguro = libre.copy()
    seguro[:2, :] = False
    sectores = SectoresNBV(libre, seguro, 0.1, (0, 0), 0.97,
                           area_minima_sector_m2=0, ancho_minimo_sector_m=0)
    assert sectores.actual is not None


def test_traslado_usa_portal_y_paradas_cortas_antes_de_nbv():
    libre = np.ones((20, 30), dtype=bool)
    libre[[0, -1], :] = False
    libre[:, [0, -1]] = False
    libre[8:12, 10:20] = False
    sectores = SectoresNBV(libre, libre, 0.1, (5, 5), 0.85)
    sectores.indice = 1
    ruta = sectores.ruta_entrada(libre, (5, 5), paso_maximo_m=0.30)
    assert ruta and any(es_portal for _, es_portal in ruta)
    assert sectores.dentro_actual(ruta[-1][0])
    posiciones = [(5, 5)] + [celda for celda, _ in ruta]
    assert all(np.linalg.norm(np.subtract(b, a)) * 0.1 <= 0.30 + 1e-9
               for a, b in zip(posiciones, posiciones[1:]))


def test_regreso_desde_otro_sector_usa_portales_y_paradas_cortas():
    libre = np.ones((20, 30), dtype=bool)
    libre[[0, -1], :] = False
    libre[:, [0, -1]] = False
    libre[8:12, 10:20] = False
    sectores = SectoresNBV(libre, libre, 0.1, (5, 5), 0.85)
    sectores.indice = 1
    destino_sector = sectores.ruta_entrada(libre, (5, 5), paso_maximo_m=0.30)[-1][0]
    ruta = sectores.ruta_hacia(libre, destino_sector, (5, 5), paso_maximo_m=0.30)
    assert ruta and any(es_portal for _, es_portal in ruta)
    assert ruta[-1][0] == (5, 5)
    posiciones = [destino_sector] + [celda for celda, _ in ruta]
    assert all(np.linalg.norm(np.subtract(b, a)) * 0.1 <= 0.30 + 1e-9
               for a, b in zip(posiciones, posiciones[1:]))
