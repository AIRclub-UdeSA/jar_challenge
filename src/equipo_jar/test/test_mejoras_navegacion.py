"""Mejoras para pasos angostos, relocalización y confirmación de cajas."""
import math
from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from builtin_interfaces.msg import Time

from equipo_jar.deteccion_victimas import Grupo, DeteccionVictimas
from equipo_jar.exploracion import (
    ELIGIENDO, NAVEGANDO, Exploracion, bfs, punto_para_mirar,
)

RES = 0.05


def _sala(alto=60, ancho=60):
    libre = np.ones((alto, ancho), dtype=bool)
    libre[[0, -1], :] = False
    libre[:, [0, -1]] = False
    return libre


def _celda(x, y):
    return int(y / RES), int(x / RES)


def test_punto_para_mirar_queda_cerca_y_encara_la_caja():
    libre = _sala()
    dist = bfs(libre, *_celda(0.3, 0.3))
    caja = (1.5, 1.5)
    x, y, yaw, metros = punto_para_mirar(libre, libre, dist, RES, (0.0, 0.0), caja)
    assert 0.6 <= math.hypot(caja[0] - x, caja[1] - y) <= 1.2
    assert abs(math.atan2(caja[1] - y, caja[0] - x) - yaw) < 1e-9
    assert metros < 1.2          # elige el lado que da hacia el robot


def test_punto_para_mirar_no_mira_a_traves_de_una_pared():
    libre = _sala()
    libre[:, 30:32] = False                      # pared en x = 1.50-1.60
    caja = (1.70, 2.0)                           # apoyada del otro lado
    dist = bfs(libre, *_celda(0.5, 2.0))
    assert punto_para_mirar(libre, libre, dist, RES, (0.0, 0.0), caja) is None
    libre[2:8, 30:32] = True                     # puerta lejos de la caja
    dist = bfs(libre, *_celda(0.5, 2.0))
    x, _, _, _ = punto_para_mirar(libre, libre, dist, RES, (0.0, 0.0), caja)
    assert x > 1.6                               # tiene que cruzar y mirarla de su lado


def test_fallos_de_navegacion_con_laser_bien_no_frenan_a_relocalizar():
    n = SimpleNamespace(error_loc=0.01, fallos_localizacion=deque(), t_ult_reloc=-1e9,
        relocalizar=Mock(), cfg={'error_loc_malo': 0.08, 'ventana_fallos_relocalizacion_s': 90.,
                                 'fallos_para_relocalizar': 3, 'espera_entre_relocalizaciones': 20.})
    for _ in range(5):
        assert not Exploracion.registrar_fallo_localizacion(n, ELIGIENDO)
    assert not n.fallos_localizacion
    n.relocalizar.assert_not_called()
    n.error_loc = 0.05                           # el láser ya se despega del mapa
    resultados = [Exploracion.registrar_fallo_localizacion(n, ELIGIENDO) for _ in range(3)]
    assert resultados == [False, False, True]
    n.relocalizar.assert_called_once()


def test_giro_pegado_a_pared_se_separa_y_despues_gira():
    n = SimpleNamespace(spin_cli=Mock(server_is_ready=lambda: True), estado=ELIGIENDO,
        giro_pendiente=None, holgura_giro=lambda: 0.0, get_logger=lambda: Mock(),
        comportamiento=Mock(), pub_cmd=Mock(), hist=[])

    def iniciar_escape(siguiente):
        n.escape = dict(siguiente=siguiente, dist=0.3)
    n.iniciar_escape = iniciar_escape
    n.girar = lambda a, s=ELIGIENDO: Exploracion.girar(n, a, s)
    Exploracion.girar(n, 1.0, NAVEGANDO)
    n.comportamiento.assert_not_called()
    assert n.giro_pendiente == (1.0, NAVEGANDO) and n.escape['dist'] == 0.10
    Exploracion.fin_escape(n, 'listo')           # aunque siga justo, gira una sola vez
    n.comportamiento.assert_called_once()
    assert n.giro_pendiente is None
    n.holgura_giro = lambda: 0.10
    Exploracion.girar(n, 1.0)
    assert n.comportamiento.call_count == 2


def test_destino_sin_guias_se_lo_pasa_directo_a_nav2():
    grilla = SimpleNamespace(a_celda=lambda x, y: (0, 0), dentro=lambda f, c: True,
                             transitable=np.ones((1, 1), dtype=bool))
    n = SimpleNamespace(pose_robot=lambda: (0., 0., 0.), grilla=grilla,
        ruta_a_destino=lambda *a: (None, 0.), get_logger=lambda: Mock(),
        direccion_de_ruta=lambda *a: 0.0, spin_cli=Mock(server_is_ready=lambda: True),
        enviar_goal=Mock(return_value=True), estado=ELIGIENDO, lista_negra=[])
    assert Exploracion.encarar_y_enviar(n, 1.0, 0.0, 0.0, 5.0, 1.0, 'vista', NAVEGANDO)
    assert n.enviar_goal.call_args.kwargs['pasos'] is None
    assert n.estado == NAVEGANDO and not n.lista_negra


def _detector(**cfg):
    base = dict(distancia_confirmacion=1.6, detecciones_minimas=8, dispersion_maxima=0.1,
                segundos_minimos_vista=1.0, detecciones_minimas_paso=4, segundos_sin_ver_paso=2.0,
                segundos_reporte_final=1e9, separacion_minima_cajas=0.40)
    base.update(cfg)
    n = SimpleNamespace(mapa=object(), cfg=base, grupos=[], t_mapa=None, reportes=0, n_max=3,
                        publicar_debug=Mock(), pub_pendientes=Mock(),
                        get_clock=lambda: Mock(now=lambda: Mock(to_msg=lambda: Time())))

    def reportar(g, motivo):
        g.reportado = True
        n.reportes += 1
    n.reportar = Mock(side_effect=reportar)
    n.publicar_pendientes = lambda: DeteccionVictimas.publicar_pendientes(n)
    n.en_recorrido = lambda x, y: False
    return n


def _grupo(x, y, n, distancia):
    g = Grupo(x, y, 0.0, distancia)
    for k in range(1, n):
        g.agregar(x, y, 0.2 * k, distancia)
    return g


def test_caja_vista_solo_de_lejos_no_se_reporta_hasta_verla_de_cerca():
    n = _detector()
    lejos = _grupo(1.0, 1.0, 20, 2.0)
    n.grupos.append(lejos)
    DeteccionVictimas.revisar_reportes(n)
    n.reportar.assert_not_called()
    pendientes = n.pub_pendientes.publish.call_args.args[0].poses
    assert len(pendientes) == 1 and pendientes[0].position.z == 20
    lejos.agregar(1.0, 1.0, 5.0, 1.0)            # el explorador se acercó a mirarla
    DeteccionVictimas.revisar_reportes(n)
    n.reportar.assert_called_once()
    assert n.pub_pendientes.publish.call_args.args[0].poses == []


def test_pendientes_no_repite_una_caja_ya_reportada():
    n = _detector()
    hecha = _grupo(1.0, 1.0, 20, 0.8)
    hecha.reportado = True
    n.grupos += [hecha, _grupo(1.2, 1.0, 5, 2.0), _grupo(3.0, 1.0, 5, 2.0), _grupo(4.0, 1.0, 1, 2.0)]
    DeteccionVictimas.publicar_pendientes(n)
    poses = n.pub_pendientes.publish.call_args.args[0].poses
    assert [(p.position.x, p.position.y) for p in poses] == [(3.0, 1.0)]


def _grilla_pasillo():
    from nav_msgs.msg import OccupancyGrid
    from equipo_jar.exploracion import Grilla
    libre = np.zeros((30, 60), dtype=bool)
    libre[5:25, 5:55] = True                      # sala de 1.0 x 2.5 m (celdas de 5 cm)
    msg = OccupancyGrid()
    msg.info.width, msg.info.height = 60, 30
    msg.info.resolution = 0.05
    msg.data = np.where(libre, 0, 100).ravel().tolist()
    return Grilla(msg, 0.10, 0.12)


def test_guias_se_alejan_de_la_pared_sin_cambiar_de_sector():
    g = _grilla_pasillo()
    regiones = np.zeros((30, 60), dtype=np.int32)
    regiones[:, 30:] = 1
    n = SimpleNamespace(grilla=g, sectorizador=SimpleNamespace(regiones=regiones),
                        cfg={'radio_alejar_guia_m': 0.25, 'holgura_guia_m': 0.30})
    n.holgura_m = lambda: Exploracion.holgura_m(n)
    inicio = (10, 15)
    guias = [((28, 8), True), ((40, 8), False), ((50, 15), False)]   # pegadas a la pared de abajo
    salida = Exploracion.alejar_guias(n, guias, inicio, fijar_ultima=True)
    holgura = n.holgura_m()
    for (antes, _), (despues, _) in zip(guias[:2], salida[:2]):
        assert holgura[despues[1], despues[0]] >= holgura[antes[1], antes[0]] + 0.09
        assert regiones[despues[1], despues[0]] == regiones[antes[1], antes[0]]
        assert math.dist(antes, despues) * g.res <= 0.25 + 1e-9
    assert salida[0][1] is True and salida[2] == guias[2]   # conserva portal y el destino fijo


def test_posible_victima_que_de_cerca_no_esta_se_descarta():
    rot = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])   # eje óptico = x del robot
    n = _detector(descartar_no_vistas=True, imagenes_para_descartar=3, margen_fov_deg=3.0,
                  radio_agrupamiento=0.40)
    n.get_logger = lambda: Mock()
    g = _grupo(1.0, 0.0, 5, 2.0)                 # vista solo de lejos
    n.grupos.append(g)
    origen = SimpleNamespace(x=0.0, y=0.0)
    puntos = (np.array([1.1, 0.4]), np.array([0.0, 0.5]))      # láser llega a la caja
    tapada = (np.array([0.5]), np.array([0.0]))
    DeteccionVictimas.revisar_ausentes(n, [g], [(0.0, -0.1, 0.1)], rot, (0.0, 0.0), np.eye(3), origen,
                                       puntos, 270.0, 320)
    DeteccionVictimas.revisar_ausentes(n, [g], [], rot, (0.0, 0.0), np.eye(3), origen, tapada, 270.0, 320)
    assert g.ausente == 0                         # hay rojo ahí / algo la tapa: no cuenta
    for _ in range(3):
        DeteccionVictimas.revisar_ausentes(n, [g], [], rot, (0.0, 0.0), np.eye(3), origen, puntos, 270.0, 320)
    assert g.descartado
    DeteccionVictimas.revisar_reportes(n)
    n.reportar.assert_not_called()
    assert n.pub_pendientes.publish.call_args.args[0].poses == []
    n.agregar_deteccion = lambda *a: DeteccionVictimas.agregar_deteccion(n, *a)
    DeteccionVictimas.agregar_deteccion(n, 1.0, 0.0, 9.0, 2.0)  # de lejos otra vez: se ignora
    assert g.descartado and g.n == 5
    DeteccionVictimas.agregar_deteccion(n, 1.0, 0.0, 9.0, 0.9)  # de cerca sí está: vuelve
    assert not g.descartado and g.n == 6


def test_pose_creida_suma_la_odometria_desde_que_se_perdio():
    odom = {'v': (1.0, 1.0, math.pi / 2)}
    n = SimpleNamespace(cfg={'odom_frame': 'odom', 'base_frame': 'base_footprint'},
                        lookup=lambda *a: odom['v'], pose_robot=lambda: None)
    r = dict(pose_creida=(2.0, 3.0, 0.0), odom_creida=(1.0, 1.0, math.pi / 2))
    odom['v'] = (1.0, 1.5, math.pi / 2)          # avanzó 0.5 m hacia adelante (según odom)
    x, y, yaw = Exploracion.pose_creida_ahora(n, r)
    assert (round(x, 6), round(y, 6), round(yaw, 6)) == (2.5, 3.0, 0.0)


def test_reubica_alrededor_de_donde_cree_estar_y_abre_en_cada_reintento(monkeypatch):
    import equipo_jar.exploracion as ex
    monkeypatch.setattr(ex.time, 'monotonic', lambda: 100.0)
    cfg = {'factor_std_reintento': 1.6, 'std_reubicar_xy_m': 0.5, 'max_std_reubicar_xy_m': 1.2,
           'std_reubicar_yaw_rad': 0.6}
    n = SimpleNamespace(cfg=cfg, pub_cmd=Mock(), publicar_pose=Mock(), get_logger=lambda: Mock(),
                        pose_creida_ahora=lambda r: (1.0, 2.0, 0.3), relocalizar_local=lambda: False)
    n.reloc = dict(fase='frenando', t0=0.0, cancelacion=None, ultimo_aviso=0.0, local_hecho=True,
                   intentos=0)
    Exploracion.paso_relocalizacion(n)
    assert n.publicar_pose.call_args.args == (1.0, 2.0, 0.3, 0.5, 0.6)
    assert n.reloc['fase'] == 'reiniciando'
    n.reloc.update(fase='frenando', intentos=2)
    Exploracion.paso_relocalizacion(n)
    assert n.publicar_pose.call_args.args[3] == 1.2          # 0.5*1.6^2 = 1.28 -> tope 1.2


def test_sin_converger_se_traslada_un_tramo_en_vez_de_girar():
    cfg = {'odom_frame': 'odom', 'base_frame': 'base_footprint', 'movimientos_reloc': 3,
           'espera_mover_reloc_s': 2.5, 'distancia_mover_reloc_m': 0.35, 'velocidad_mover_reloc': 0.12}
    odom = {'v': (0.0, 0.0, 0.0)}
    n = SimpleNamespace(cfg=cfg, pub_cmd=Mock(), get_logger=lambda: Mock(), lookup=lambda *a: odom['v'],
                        puntos_scan=lambda: (np.array([1.0]), np.array([0.0])),
                        espacio_libre=lambda p, ang: 1.0 if abs(ang) < 0.1 else 0.1)
    r = dict(mov='esperando', t_mov=0.0, movs=0, ang_mov=None, estable_desde=None)
    assert Exploracion.mover_para_relocalizar(n, r, 1.0)        # quieto: valida AMCL
    assert not Exploracion.mover_para_relocalizar(n, r, 3.0)    # pasó la espera: arranca
    assert r['mov'] == 'moviendo' and r['ang_mov'] == 0.0 and r['dist_mov'] == 0.35
    Exploracion.mover_para_relocalizar(n, r, 3.2)
    assert n.pub_cmd.publish.call_args.args[0].linear.x == 0.12
    odom['v'] = (0.36, 0.0, 0.0)
    assert not Exploracion.mover_para_relocalizar(n, r, 6.0)
    assert r['mov'] == 'esperando' and r['movs'] == 1
    assert n.pub_cmd.publish.call_args.args[0].linear.x == 0.0


def test_alrededor_de_un_objeto_queda_visto():
    g = _grilla_pasillo()
    n = SimpleNamespace(grilla=g, visto=np.zeros((g.alto, g.ancho), dtype=bool))
    Exploracion.marcar_alrededor(n, 1.0, 0.6, 0.45)
    f, c = g.a_celda(1.0, 0.6)
    assert n.visto[f, c] and n.visto[f, c + 8] and not n.visto[f, c + 10]
    assert not (n.visto & ~g.libre).any()             # nunca marca paredes


def test_al_llegar_no_gira_por_pocos_grados():
    n = SimpleNamespace(goal_actual=None, cfg={'girar_al_llegar': True, 'tolerancia_giro_llegada_deg': 35.0},
                        pose_robot=lambda: (0.0, 0.0, 0.0), girar=Mock(), lista_negra=[],
                        n_atascos=0, fallos_localizacion=deque(), sectorizador=None, fov=math.radians(70))
    Exploracion.al_llegar(n, dict(x=0.0, y=0.0, yaw=math.radians(30), etiqueta='vista'))
    n.girar.assert_not_called()
    Exploracion.al_llegar(n, dict(x=0.0, y=0.0, yaw=math.radians(60), etiqueta='vista'))
    n.girar.assert_called_once()


def test_frena_solo_si_va_hacia_la_pared():
    from equipo_jar.exploracion import Exploracion as E
    publicados = []
    pared_al_costado = (np.linspace(-1.0, 1.0, 41), np.full(41, 0.18))      # pared a 8 cm del flanco
    n = SimpleNamespace(cfg={'distancia_lenta': 0.45, 'distancia_muy_lenta': 0.25, 'porcentaje_lenta': 70.0,
                             'porcentaje_muy_lenta': 50.0, 'error_loc_malo': 0.08},
                        error_loc=0.01, ultimo_limite=None, vel_base=(0.4, 0.0),
                        puntos_scan=lambda: pared_al_costado, ESQUINAS=E.ESQUINAS,
                        pub_vel=SimpleNamespace(publish=lambda m: publicados.append(m.speed_limit)),
                        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(to_msg=lambda: Time())),
                        get_logger=lambda: Mock())
    n.espacio_libre = lambda p, a: E.espacio_libre(n, p, a)
    n.libre_hacia_donde_va = lambda: E.libre_hacia_donde_va(n)
    E.publicar_limite_velocidad(n)
    assert publicados[-1] == 100.0                # va paralelo a la pared: máxima
    n.vel_base = (0.0, 0.3)                       # ahora va de costado, hacia esa pared
    E.publicar_limite_velocidad(n)
    assert publicados[-1] == 50.0
    n.puntos_scan = lambda: (np.array([0.45]), np.array([0.0]))   # pared 33 cm delante del frente
    n.vel_base = (0.4, 0.0)
    E.publicar_limite_velocidad(n)
    assert publicados[-1] == 70.0


def test_arbol_de_portales_se_busca_junto_al_codigo():
    import os
    from equipo_jar.exploracion import buscar_arbol_portales
    ruta = buscar_arbol_portales()
    assert ruta.endswith(os.path.join('behavior_trees', 'navegar_portales.xml')) and os.path.isfile(ruta)


def test_regreso_no_manda_done_mientras_quede_tiempo():
    n = SimpleNamespace(ruta_regreso=[], reintentos_regreso=0, regreso_directo=False,
                        cfg={'max_reintentos_regreso': 1}, registrar_fallo_localizacion=lambda s: False,
                        get_logger=lambda: Mock(), limpiar_costmaps=Mock(), n_atascos=0,
                        iniciar_escape=Mock(), terminar=Mock(), tiempo_restante=lambda: 120.0)
    for _ in range(2):
        Exploracion.reintentar_regreso(n)
    assert n.regreso_directo and n.reintentos_regreso == 0         # pasa a destino directo
    for _ in range(2):
        Exploracion.reintentar_regreso(n)
    n.terminar.assert_not_called()                                 # sigue intentando
    n.tiempo_restante = lambda: 5.0
    for _ in range(4):
        Exploracion.reintentar_regreso(n)
    n.terminar.assert_called_once()


def test_cerca_del_sector_ya_cuenta_como_entrada():
    g = _grilla_pasillo()
    regiones = np.zeros((30, 60), dtype=np.int32)
    regiones[:, 30:] = 1
    n = SimpleNamespace(grilla=g, cfg={'radio_entrada_sector_m': 0.30},
                        sectorizador=SimpleNamespace(regiones=regiones, actual=1,
                                                     dentro_actual=lambda c: regiones[c[1], c[0]] == 1))
    assert Exploracion.en_sector(n, (1.60, 0.75, 0.0))          # adentro
    assert Exploracion.en_sector(n, (1.33, 0.75, 0.0))          # a 17 cm del borde
    assert not Exploracion.en_sector(n, (1.0, 0.75, 0.0))       # a 50 cm


def test_done_a_los_599_segundos_este_donde_este():
    n = SimpleNamespace(estado=NAVEGANDO, t_mapa=1.0, cfg={'done_forzado_s': 599.0},
                        tiempo_transcurrido=lambda: 599.2, terminar=Mock())
    Exploracion.tick_rapido(n)
    n.terminar.assert_called_once()


def test_no_reporta_una_caja_donde_el_robot_ya_paso():
    n = _detector()
    n.mapa = dict(res=0.05, ox=0.0, oy=0.0, alto=40, ancho=40)
    n.recorrido = np.zeros((40, 40), dtype=bool)
    n.recorrido[20, 20] = True                       # el robot pasó por (1.02, 1.02)
    n.a_celda = lambda x, y: DeteccionVictimas.a_celda(n, x, y)
    n.en_recorrido = lambda x, y: DeteccionVictimas.en_recorrido(n, x, y)
    n.get_logger = lambda: Mock()
    n.grupos += [_grupo(1.02, 1.02, 20, 0.8), _grupo(0.50, 1.50, 20, 0.8)]
    DeteccionVictimas.revisar_reportes(n)
    assert n.grupos[0].descartado and not n.grupos[0].reportado
    assert n.grupos[1].reportado


def test_sin_localizacion_buena_no_toma_detecciones():
    import equipo_jar.deteccion_victimas as dv
    n = SimpleNamespace(cfg={'error_loc_max': 0.07}, loc=None)
    assert DeteccionVictimas.localizacion_ok(n)                  # sin explorador: se asume buena
    n.loc = (dv.time.monotonic(), 0.03, False)
    assert DeteccionVictimas.localizacion_ok(n)
    n.loc = (dv.time.monotonic(), 0.10, False)
    assert not DeteccionVictimas.localizacion_ok(n)
    n.loc = (dv.time.monotonic(), 0.02, True)                    # relocalizando
    assert not DeteccionVictimas.localizacion_ok(n)


def test_campo_de_localizacion_incluye_objetos_observados():
    g = _grilla_pasillo()
    n = SimpleNamespace(grilla=g, bloques_objetos=[(1.0, 0.75)],
                        dist_pared=Exploracion.campo_distancia_paredes(g))
    n.mascara_objetos = lambda radio=0.10: Exploracion.mascara_objetos(n, radio)
    n.campo_distancia_paredes = Exploracion.campo_distancia_paredes
    campo = Exploracion.campo_loc(n)
    f, c = g.a_celda(1.0, 0.75)
    assert n.dist_pared[f, c] > 0.3 and campo[f, c] == 0.0       # el objeto cuenta como "pared"
