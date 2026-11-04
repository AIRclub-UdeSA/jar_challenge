import math

import numpy as np
import pytest

from jar_percepcion.candidatos import (
    Candidatos, ConfigCandidatos, Objeto, agrupar_scan, bordes_de_oclusion,
    confirmar_con_candidatos, en_vista, lugar_vacio, objetos_del_scan, vista_concluyente)

C = ConfigCandidatos()


def cara(x0, y0, x1, y1, paso=0.02):
    n = max(2, int(math.hypot(x1 - x0, y1 - y0) / paso))
    return np.column_stack((np.linspace(x0, x1, n), np.linspace(y0, y1, n)))


def pared_en_y(y_pared):
    return lambda p: np.abs(np.asarray(p)[:, 1] - y_pared)


def test_una_caja_suelta_es_candidato_y_la_pared_no():
    pared = cara(-1.0, 1.0, 3.0, 1.0)                    # pared del mapa en y = 1
    caja = cara(1.875, -0.125, 1.875, 0.125)            # cara de una caja en x = 1.875
    puntos = np.vstack([pared, caja])
    centros = objetos_del_scan(puntos, pared_en_y(1.0), origen=(0.0, 0.0))
    assert len(centros) == 1
    assert math.hypot(centros[0][0] - 2.0, centros[0][1]) < 0.03   # corrido al centro


def test_algo_largo_no_es_una_caja():
    largo = cara(1.0, -0.5, 1.0, 0.5)                   # 1 m: pared que el mapa no tiene
    assert objetos_del_scan(largo, pared_en_y(5.0), origen=(0.0, 0.0)) == []


def test_agrupar_corta_en_los_saltos():
    a, b = cara(0, 0, 0, 0.2), cara(0.5, 0, 0.5, 0.2)
    assert len(agrupar_scan(np.vstack([a, b]))) == 2


def test_rojo_en_su_direccion_lo_hace_victima_y_sin_rojo_se_descarta():
    cs = Candidatos()
    roja = cs.agregar(2.0, 0.0)
    azul = cs.agregar(0.0, 2.0)
    assert cs.agregar(2.05, 0.02) is roja               # misma caja
    for _ in range(2):
        cs.inspeccionar(roja, rojo=True)
    for _ in range(4):
        cs.inspeccionar(azul, rojo=False)
    assert roja.estado == 'victima' and azul.estado == 'descartado'


def test_una_vista_sin_rojo_no_descarta_si_ya_hubo_rojo():
    cs = Candidatos()
    cand = cs.agregar(1.0, 0.0)
    cs.inspeccionar(cand, rojo=True)
    for _ in range(5):
        cs.inspeccionar(cand, rojo=False)                # p. ej. de costado, poco rojo
    assert cand.estado == 'sin_inspeccionar'


def test_por_mayoria_un_rojo_falso_no_bloquea_el_descarte():
    cs = Candidatos(ConfigCandidatos(descarte_por_mayoria=True, proporcion_descarte=4.0))
    cand = cs.agregar(1.0, 0.0)
    cs.inspeccionar(cand, rojo=True)                     # rojo falso de cerca
    for _ in range(3):
        cs.inspeccionar(cand, rojo=False)
    assert cand.estado == 'sin_inspeccionar'             # 3 < 4 × 1
    cs.inspeccionar(cand, rojo=False)
    assert cand.estado == 'descartado'
    victima = cs.agregar(0.0, 2.0)
    cs.inspeccionar(victima, rojo=False)
    cs.inspeccionar(victima, rojo=True)
    cs.inspeccionar(victima, rojo=True)
    assert victima.estado == 'victima'                   # dos rojos siguen alcanzando


def test_en_vista_pide_de_frente_distancia_util_y_nada_en_el_medio():
    libre = lambda a: 10.0                              # noqa: E731 - el LiDAR no ve nada antes
    ok, _ = en_vista((1.2, 0.0), (0.0, 0.0, 0.0), 0.0, 0.53, libre)
    assert ok
    assert not en_vista((0.3, 0.0), (0.0, 0.0, 0.0), 0.0, 0.53, libre)[0]   # muy cerca
    assert not en_vista((0.0, 1.2), (0.0, 0.0, 0.0), 0.0, 0.53, libre)[0]   # de costado
    tapado = lambda a: 0.5                              # noqa: E731 - algo a 0.5 m
    assert not en_vista((1.2, 0.0), (0.0, 0.0, 0.0), 0.0, 0.53, tapado)[0]


def test_pared_vista_corrida_por_amcl_no_es_candidato():
    # La pared del mapa en y = 1; el LiDAR la ve 12 cm más cerca (error de AMCL): un pedazo
    # corto de esa pared no tiene que pasar por caja.
    pedazo = cara(1.0, 0.88, 1.3, 0.88)
    assert objetos_del_scan(pedazo, pared_en_y(1.0), origen=(0.0, 0.0)) == []


def test_distancia_pared_020_pared_corrida_18_cm_no_y_caja_contra_la_pared_si():
    # Al principio de la corrida AMCL anda 0.13-0.20 m corrido (laberinto_puertas: un pedazo de
    # pared junto al poste de C-D se inspeccionó como candidato).
    c = ConfigCandidatos(distancia_pared_m=0.20)
    pedazo = cara(1.0, 0.82, 1.3, 0.82)
    assert objetos_del_scan(pedazo, pared_en_y(1.0), origen=(0.0, 0.0)) != []      # con 0.15
    assert objetos_del_scan(pedazo, pared_en_y(1.0), origen=(0.0, 0.0), c=c) == []
    # Caja de 0.25 m apoyada en la pared y = 1, vista de frente: la cara queda a 0.25 m.
    caja = cara(0.875, 0.75, 1.125, 0.75)
    assert len(objetos_del_scan(caja, pared_en_y(1.0), origen=(1.0, 0.0), c=c)) == 1


def test_rojo_en_la_direccion_pero_en_otro_lugar_no_vota():
    cs = Candidatos()
    esquina = cs.agregar(1.0, 0.0)
    for _ in range(5):
        cs.inspeccionar(esquina, None)                 # rojo detrás, en la misma dirección
    assert esquina.estado == 'sin_inspeccionar' and esquina.rojo == 0


def test_rojo_confirmado_sin_candidato_victima_no_se_reporta():
    # maze_2: la misma caja ubicada 0.4 m corrida se confirmaba sola y gastaba un reporte.
    objetivos = [{'id': 1, 'x': 1.09, 'y': -1.87, 'confirmado': True},
                 {'id': 2, 'x': 0.70, 'y': -1.54, 'confirmado': True}]
    candidatos = [{'id': 5, 'x': 1.12, 'y': -1.84, 'estado': 'victima'},
                  {'id': 6, 'x': 2.0, 'y': 0.0, 'estado': 'descartado'}]
    salida = confirmar_con_candidatos(objetivos, candidatos)
    assert [o['confirmado'] for o in salida] == [True, False]


def test_candidato_victima_sin_objetivo_se_ofrece_con_su_posicion():
    # laberinto_s2: candidato víctima a 6 cm de la caja; el objetivo del tracker, corrido 0.7 m.
    candidatos = [{'id': 15, 'x': 0.92, 'y': 2.89, 'estado': 'victima', 'rojo': 3},
                  {'id': 16, 'x': 3.0, 'y': 3.0, 'estado': 'descartado', 'rojo': 0}]
    objetivos = [{'id': 1, 'x': 0.48, 'y': 3.54, 'n': 6, 'dispersion': 0.2, 'confirmado': True}]
    assert not any(o['confirmado'] for o in confirmar_con_candidatos(objetivos, candidatos))
    salida = confirmar_con_candidatos(objetivos, candidatos, sumar_candidatos=True)
    confirmados = [(o['x'], o['y']) for o in salida if o['confirmado']]
    assert confirmados == [(0.92, 2.89)]                 # el descartado no se ofrece
    assert all(isinstance(o['id'], int) for o in salida)  # Marker.id de RViz es int
    # Si un objetivo confirmado ya lo cubre, no se duplica.
    cerca = [{'id': 2, 'x': 0.95, 'y': 2.85, 'n': 6, 'dispersion': 0.02, 'confirmado': True}]
    salida = confirmar_con_candidatos(cerca, candidatos, sumar_candidatos=True)
    assert [o['id'] for o in salida if o['confirmado']] == [2]


def test_los_dudosos_se_ofrecen_sin_confirmar_para_el_cierre():
    # laberinto_puertas: la caja azul de la puerta de C quedó con rojo=1 no_rojo=1, sin decidir.
    candidatos = [{'id': 1, 'x': 0.84, 'y': 2.44, 'estado': 'sin_inspeccionar', 'rojo': 1,
                   'no_rojo': 1},
                  {'id': 2, 'x': 3.0, 'y': 1.0, 'estado': 'sin_inspeccionar', 'rojo': 0,
                   'no_rojo': 0},
                  {'id': 3, 'x': 4.0, 'y': 1.0, 'estado': 'sin_inspeccionar', 'rojo': 0,
                   'no_rojo': 2},
                  {'id': 4, 'x': 5.0, 'y': 1.0, 'estado': 'descartado', 'rojo': 0,
                   'no_rojo': 4},
                  {'id': 5, 'x': 2.0, 'y': 3.0, 'estado': 'sin_inspeccionar', 'rojo': 2,
                   'no_rojo': 0}]
    objetivos = [{'id': 9, 'x': 2.05, 'y': 3.0, 'n': 4, 'dispersion': 0.1, 'confirmado': False}]
    assert len(confirmar_con_candidatos(objetivos, candidatos)) == 1
    salida = confirmar_con_candidatos(objetivos, candidatos, sumar_dudosos=True)
    dudosos = {o['id']: o['n'] for o in salida if o['id'] >= 200000}
    # El 3 (la cámara lo vio y no era rojo), el 4 (decidido) y el 5 (ya lo cubre el 9), no.
    assert dudosos == {200001: 1, 200002: 0.5}
    assert not any(o['confirmado'] for o in salida)


def dos_candidatos(c, a_xy, b_xy):
    """Dos candidatos creados por separado (la primera vista de b, lejos de a)."""
    cands = Candidatos(c)
    a = cands.agregar(*a_xy)
    b = cands.agregar(*b_xy)
    assert a is not b
    return cands, a, b


def test_unir_candidatos_pegados_de_la_misma_caja():
    # laberinto_puertas: la víctima de A como candidatos 5 (1.54, 0.15) y 17 (1.63, 0.04).
    c = ConfigCandidatos(unir_candidatos=True)
    cands, a, b = dos_candidatos(c, (1.54, 0.15), (1.95, -0.20))
    a.estado, a.rojo = 'victima', 2
    assert cands.unir() == []                           # todavía lejos
    b.obs = [(1.63, 0.04)]                              # se fue acercando
    b.no_rojo = 0
    assert cands.unir() == [(a.id, b.id)]
    assert cands.lista == [a] and a.estado == 'victima'
    assert len(a.obs) == 2


def test_unir_suma_los_votos_y_decide():
    c = ConfigCandidatos(unir_candidatos=True)
    cands, a, b = dos_candidatos(c, (2.03, 2.41), (2.50, 2.90))
    a.no_rojo, b.no_rojo = 2, 2
    b.obs = [(2.07, 2.58)]                              # la caja gris de D, a 0.17 m
    cands.unir()
    assert cands.lista == [a] and a.no_rojo == 4 and a.estado == 'descartado'


def test_no_se_unen_si_los_votos_se_contradicen_ni_con_el_flag_apagado():
    c = ConfigCandidatos(unir_candidatos=True)
    cands, a, b = dos_candidatos(c, (1.0, 1.0), (1.6, 1.0))
    b.obs = [(1.2, 1.0)]
    a.rojo, b.no_rojo = 1, 3                            # víctima pegada a un distractor
    assert cands.unir() == [] and len(cands.lista) == 2
    apagado, x, y = dos_candidatos(ConfigCandidatos(), (1.0, 1.0), (1.6, 1.0))
    y.obs = [(1.1, 1.0)]
    assert apagado.unir() == [] and len(apagado.lista) == 2


def test_un_descartado_con_rojo_en_su_lugar_pasa_a_victima():
    # maze_2: cajas rojas descartadas por vistas sin blob (borde de la imagen, lejos).
    cs = Candidatos()
    cand = cs.agregar(2.94, -1.18)
    for _ in range(4):
        cs.inspeccionar(cand, rojo=False)
    assert cand.estado == 'descartado'
    # Dos votos rojos no alcanzan para revivirlo (laberinto_s3: el rojo de una víctima vecina,
    # mal ubicado, revivió un distractor y gastó un reporte); cuatro sí.
    for _ in range(2):
        cs.inspeccionar(cand, rojo=True)
    assert cand.estado == 'descartado'
    for _ in range(2):
        cs.inspeccionar(cand, rojo=True)
    assert cand.estado == 'victima'


def test_no_ver_rojo_solo_cuenta_centrado_y_cerca():
    assert vista_concluyente((1.2, 0.0), (0.0, 0.0, 0.0), 0.0, 0.53)
    assert not vista_concluyente((1.2, 0.5), (0.0, 0.0, 0.0), 0.0, 0.53)   # al borde
    assert not vista_concluyente((1.9, 0.0), (0.0, 0.0, 0.0), 0.0, 0.53)   # lejos
    assert not vista_concluyente((0.5, 0.0), (0.0, 0.0, 0.0), 0.0, 0.53)   # muy cerca


def test_no_ver_rojo_por_pixeles_vale_a_cualquier_distancia_si_esta_centrado():
    # laberinto_s4: el robot quedó a 0.55 m del cilindro verde (tolerancia de Nav2) y nunca
    # pudo votar. Mirando los píxeles crudos la distancia no importa; el centrado sí.
    assert vista_concluyente((0.55, 0.0), (0.0, 0.0, 0.0), 0.0, 0.53, con_distancia=False)
    assert vista_concluyente((1.9, 0.0), (0.0, 0.0, 0.0), 0.0, 0.53, con_distancia=False)
    assert not vista_concluyente((1.2, 0.5), (0.0, 0.0, 0.0), 0.0, 0.53, con_distancia=False)


def test_visto_en_un_solo_scan_no_se_manda_a_inspeccionar():
    # maze_4: pedazos de pared vistos de lejos en un scan hacían volver al robot a mirarlos.
    cs = Candidatos()
    cand = cs.agregar(1.0, 0.0)
    assert cs.resumen() == [] and not cs.listos(cand)
    for _ in range(C.vistas_min - 1):
        cs.agregar(1.02, 0.0)
    assert [c['id'] for c in cs.resumen()] == [cand.id]


def test_lugar_vacio_si_el_lidar_pasa_de_largo_y_cerca():
    def rango(r):
        return lambda a: r                               # noqa: E731 - el mismo en todo rumbo
    assert lugar_vacio((1.0, 0.0), (0.0, 0.0, 0.0), rango(3.0))
    assert not lugar_vacio((1.0, 0.0), (0.0, 0.0, 0.0), rango(0.875))   # la cara de la caja
    assert not lugar_vacio((1.0, 0.0), (0.0, 0.0, 0.0), rango(0.4))     # algo en el medio
    assert not lugar_vacio((2.0, 0.0), (0.0, 0.0, 0.0), rango(5.0))     # lejos: no prueba


def test_fantasma_sin_nada_de_cerca_se_borra_y_una_vista_reinicia_la_cuenta():
    cs = Candidatos()
    cand = cs.agregar(1.0, 0.0)
    for _ in range(C.ausencias_borrar - 1):
        assert not cs.ausente(cand)
    cs.agregar(1.0, 0.0)                                 # lo vuelve a ver: no era fantasma
    for _ in range(C.ausencias_borrar - 1):
        assert not cs.ausente(cand)
    assert cs.ausente(cand) and cs.lista == []


def test_un_candidato_ya_clasificado_no_se_borra_por_ausencia():
    cs = Candidatos()
    cand = cs.agregar(1.0, 0.0)
    for _ in range(C.votos_victima):
        cs.inspeccionar(cand, rojo=True)
    for _ in range(2 * C.ausencias_borrar):
        assert not cs.ausente(cand)
    assert cs.lista == [cand]


# Política de detección -------------------------------------------------------------------

def test_bordes_de_oclusion_caja_delante_de_la_pared_y_pedazo_de_pared():
    rangos = np.full(20, 3.0)
    rangos[8:12] = 2.0                                   # algo delante del fondo
    assert bordes_de_oclusion(rangos, 8, 11, 0.15) == 2
    rangos = np.full(20, 3.0)
    rangos[8:20] = np.linspace(2.0, 2.5, 12)             # sigue con la pared a la derecha
    assert bordes_de_oclusion(rangos, 8, 11, 0.15) == 1
    rangos[7] = math.inf                                 # sin eco: también es borde
    assert bordes_de_oclusion(rangos, 8, 11, 0.15) == 1


def test_lejos_sin_rojo_es_posible_y_no_se_inspecciona():
    cs = Candidatos()
    cand = cs.observar(Objeto(3.0, 0.0, 3.0, 2), desde=(0.0, 0.0))
    assert cand.estado == 'posible' and not cs.listos(cand)
    assert [c['estado'] for c in cs.resumen()] == ['posible']


def test_lejos_sin_bordes_de_oclusion_ni_rojo_no_cuenta():
    # Un pedazo de pared visto de lejos: sigue con la pared en un extremo.
    assert Candidatos().observar(Objeto(3.0, 0.0, 3.0, 1)) is None


def test_mas_alla_del_alcance_posible_no_cuenta():
    assert Candidatos().observar(Objeto(4.0, 0.0, 4.0, 2)) is None


def test_lejos_con_rojo_es_candidato_rojo_sin_esperar_vistas():
    # maze_4: víctima a 2.9 m justo adelante de la largada; con 2 m de alcance el robot se iba.
    cs = Candidatos()
    cand = cs.observar(Objeto(2.9, 0.0, 2.9, 1), desde=(0.0, 0.0),
                       rojos_lejanos=[(2.6, 0.2)])      # ~0.36 m corrida, sin los dos bordes
    assert cand.estado == 'sin_inspeccionar' and cand.rojo_lejano and cs.listos(cand)


def test_un_posible_pasa_a_rojo_cuando_la_camara_lo_ve():
    cs = Candidatos()
    cand = cs.observar(Objeto(3.0, 0.0, 3.0, 2))
    assert cs.marcar_rojo_lejano([(3.2, 0.3)]) == [cand]
    assert cand.estado == 'sin_inspeccionar' and cs.listos(cand)


def test_un_posible_visto_de_cerca_es_candidato_con_la_posicion_de_cerca():
    cs = Candidatos()
    cand = cs.observar(Objeto(3.0, 0.1, 3.0, 2), desde=(0.0, 0.0))
    assert cs.observar(Objeto(3.05, 0.0, 1.5, None), desde=(1.5, 0.0)) is cand
    assert cand.estado == 'sin_inspeccionar' and cand.obs == [(3.05, 0.0)]


def test_de_cerca_hace_falta_verlo_desde_dos_posiciones():
    cs = Candidatos()
    for _ in range(5):
        cand = cs.observar(Objeto(1.0, 0.0, 1.0, 2), desde=(0.0, 0.0))   # robot quieto
    assert not cs.listos(cand)
    cs.observar(Objeto(1.0, 0.0, 1.0, 2), desde=(0.0, 0.5))              # se movió
    assert cs.listos(cand)


def test_bordes_min_cerca_es_configurable():
    cs = Candidatos(ConfigCandidatos(bordes_min_cerca=1))
    assert cs.observar(Objeto(1.0, 0.0, 1.0, 0)) is None
    assert cs.observar(Objeto(1.0, 0.0, 1.0, 1)) is not None


def test_un_posible_fantasma_tambien_se_borra():
    cs = Candidatos()
    cand = cs.observar(Objeto(3.0, 0.0, 3.0, 2))
    for _ in range(C.ausencias_borrar):
        cs.ausente(cand)
    assert cs.lista == []


def test_objetos_del_scan_da_distancia_y_bordes():
    pared = cara(-1.0, 3.0, 3.0, 3.0)                    # fondo en y = 3
    caja = cara(-0.125, 1.875, 0.125, 1.875)             # cara de una caja en y = 1.875
    puntos = np.vstack([pared[:40], caja, pared[60:]])
    rangos = np.hypot(puntos[:, 0], puntos[:, 1])
    indices = np.arange(len(puntos))
    (obj,) = objetos_del_scan(puntos, pared_en_y(3.0), (0.0, 0.0), rangos=rangos,
                              indices=indices)
    assert obj.distancia == pytest.approx(2.0, abs=0.03) and obj.bordes == 2


def test_victima_sin_parecer_caja_espera_hasta_verla_como_caja():
    c = ConfigCandidatos(rojo_requiere_caja=True, vistas_min=3, posiciones_min=2)
    cs = Candidatos(c)
    cand = cs.observar(Objeto(1.0, 0.0, 1.0, 0), desde=(0.0, 0.0))
    cs.marcar_rojo_lejano([(1.0, 0.0)])                 # entra por el rojo, sin más vistas
    cs.inspeccionar(cand, rojo=True)
    cs.inspeccionar(cand, rojo=True)
    assert cand.estado == 'sin_inspeccionar'            # una vista, sin bordes: no es caja
    cs.observar(Objeto(1.0, 0.0, 1.0, 1), desde=(0.0, 0.5))
    cs.observar(Objeto(1.0, 0.0, 1.0, 2), desde=(0.0, -0.5))
    cs.inspeccionar(cand, rojo=True)
    assert cand.estado == 'victima'
    apagado = Candidatos()
    otro = apagado.observar(Objeto(1.0, 0.0, 1.0, 0), desde=(0.0, 0.0))
    apagado.inspeccionar(otro, rojo=True)
    apagado.inspeccionar(otro, rojo=True)
    assert otro.estado == 'victima'                     # sin el flag, como antes
