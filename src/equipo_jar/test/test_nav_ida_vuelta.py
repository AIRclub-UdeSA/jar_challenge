"""Máquina de ida y vuelta con cinemática ideal y LiDAR sintético."""
import json
import math

import numpy as np

from equipo_jar.navegacion.costos import ParametrosCosto
from equipo_jar.navegacion.grilla import Grilla
from equipo_jar.navegacion.guardia import huella_desde_parametros
from equipo_jar.navegacion.ida_vuelta import EntradaNavegacion, IdaVuelta, ParametrosIdaVuelta
from equipo_jar.navegacion.seguimiento import ParametrosSeguimiento
from equipo_jar.navegacion.tipos import Pose2D

RES = 0.05
DT = 0.1
P_COSTO = ParametrosCosto(0.12, math.hypot(0.19,0.12), 0.10, 0.25, 5.0)
P_SEG = ParametrosSeguimiento(0.30, 0.4, 0.20, 0.03, 1.5, 0.5, 0.25, 0.45, 0.10, 0.2)
HUELLA = huella_desde_parametros(-0.19, 0.13, 0.12, 0.10)
P = ParametrosIdaVuelta()


def anillo():
    """3 x 2 m con una isla central: dos pasillos entre izquierda y derecha."""
    ocupada = np.zeros((40, 60), dtype=bool)
    ocupada[14:26, 15:45] = True
    return ocupada


def pasillo_en_u():
    ocupada = np.zeros((40, 60), dtype=bool)
    ocupada[18:22, 0:45] = True
    return ocupada


class Mundo:
    """Mapa conocido más cajas que el mapa no tiene; genera puntos del LiDAR."""

    def __init__(self, ocupada_mapa):
        alto, ancho = ocupada_mapa.shape
        self.grilla = Grilla(RES, 0.0, 0.0, 0.0, ancho, alto, ocupada_mapa)
        self.real = ocupada_mapa.copy()

    def agregar_caja(self, x, y, lado=0.2):
        for fila in range(self.grilla.alto):
            for col in range(self.grilla.ancho):
                cx, cy = self.grilla.a_mundo(fila, col)
                if abs(cx - x) <= lado / 2 and abs(cy - y) <= lado / 2:
                    self.real[fila, col] = True

    def puntos(self, pose, alcance=2.5):
        alto, ancho = self.real.shape
        borde = np.pad(self.real, 1, constant_values=True)
        mapa = []
        for fila in range(-1, alto + 1):
            for col in range(-1, ancho + 1):
                if not borde[fila + 1, col + 1]:
                    continue
                vecinos = borde[max(0, fila):fila + 3, max(0, col):col + 3]
                if vecinos.all():
                    continue
                x, y = (col + 0.5) * RES, (fila + 0.5) * RES
                if math.hypot(x - pose.x, y - pose.y) <= alcance:
                    mapa.append((x, y))
        c, s = math.cos(pose.yaw), math.sin(pose.yaw)
        base = [(c * (x - pose.x) + s * (y - pose.y), -s * (x - pose.x) + c * (y - pose.y))
                for x, y in mapa]
        return base, mapa


def correr(iv, mundo, pose, pasos, lista=lambda k: True, datos=lambda k: True,
           al_paso=None):
    x, y, yaw = pose
    registro = []
    for k in range(pasos):
        p = Pose2D(x, y, yaw)
        base, mapa = mundo.puntos(p)
        ok = datos(k)
        e = EntradaNavegacion(k * DT, p if ok else None, lista(k), base if ok else None,
                              mapa if ok else None)
        cmd = iv.paso(e)
        registro.append((k, x, y, yaw, cmd, iv.estado))
        if al_paso:
            al_paso(k, iv)
        if iv.estado in ('EN_LARGADA', 'FALLO'):
            break
        c, s = math.cos(yaw), math.sin(yaw)
        x += (c * cmd.vx - s * cmd.vy) * DT
        y += (s * cmd.vx + c * cmd.vy) * DT
        yaw += cmd.wz * DT
    return registro


def quieto_maximo(registro):
    mayor = actual = 0.0
    for _, _, _, _, cmd, estado in registro:
        if estado in ('ESPERANDO_LISTA', 'ESPERANDO_META', 'EN_META'):
            actual = 0.0
            continue
        actual = actual + DT if cmd.es_cero() else 0.0
        mayor = max(mayor, actual)
    return mayor


def nueva(mundo):
    return IdaVuelta(mundo.grilla, P_COSTO, P_SEG, HUELLA, P)


def test_no_se_mueve_antes_de_lista():
    mundo = Mundo(anillo())
    iv = nueva(mundo)
    iv.fijar_meta((2.65, 1.0))
    registro = correr(iv, mundo, (0.35, 1.0, 0.0), 50, lista=lambda k: False)
    assert all(r[4].es_cero() for r in registro)
    assert iv.estado == 'ESPERANDO_LISTA'


def test_ida_y_vuelta_completa_termina_a_menos_de_5_cm_de_la_largada():
    mundo = Mundo(pasillo_en_u())
    iv = nueva(mundo)
    iv.fijar_meta((0.5, 1.5))
    registro = correr(iv, mundo, (0.5, 0.45, 0.1), 3000)
    assert iv.estado == 'EN_LARGADA', iv.motivo
    assert 'EN_META' in {r[5] for r in registro}
    _, x, y, _, _, _ = registro[-1]
    assert math.hypot(x - 0.5, y - 0.45) <= P.tolerancia_largada_m
    assert registro[-1][4].es_cero()


def test_datos_ausentes_detienen_y_reanudan():
    mundo = Mundo(pasillo_en_u())
    iv = nueva(mundo)
    iv.fijar_meta((0.5, 1.5))
    registro = correr(iv, mundo, (0.5, 0.45, 0.0), 3000, datos=lambda k: not 30 <= k < 40)
    hueco = [r for r in registro if 30 <= r[0] < 40]
    assert all(r[4].es_cero() and r[5] == 'DETENIDO' for r in hueco)
    assert iv.estado == 'EN_LARGADA', iv.motivo


def test_localizacion_no_utilizable_detiene():
    mundo = Mundo(pasillo_en_u())
    iv = nueva(mundo)
    iv.fijar_meta((0.5, 1.5))
    registro = correr(iv, mundo, (0.5, 0.45, 0.0), 3000, lista=lambda k: not 30 <= k < 45)
    assert all(r[4].es_cero() for r in registro if 30 <= r[0] < 45)
    assert iv.estado == 'EN_LARGADA', iv.motivo


def test_detenido_mas_de_20_s_falla_con_motivo():
    mundo = Mundo(pasillo_en_u())
    iv = nueva(mundo)
    iv.fijar_meta((0.5, 1.5))
    registro = correr(iv, mundo, (0.5, 0.45, 0.0), 3000, datos=lambda k: k < 30)
    assert iv.estado == 'FALLO'
    assert 'detenido' in iv.motivo.lower()
    assert registro[-1][0] * DT <= 30 * DT + P.plazo_detenido_s + 2 * DT


def test_obstaculo_nuevo_replanifica_por_otro_pasillo():
    mundo = Mundo(anillo())
    iv = nueva(mundo)
    iv.fijar_meta((2.65, 1.0))
    caja = []

    def poner_caja(k, iv):
        if not caja and iv.estado == 'YENDO':
            (x0, y0), (x1, y1) = iv.ruta.puntos[1], iv.ruta.puntos[2]
            caja.append(((x0 + x1) / 2, (y0 + y1) / 2))
            mundo.agregar_caja(*caja[0])

    registro = correr(iv, mundo, (0.35, 1.0, 0.0), 4000, al_paso=poner_caja)
    assert iv.estado == 'EN_LARGADA', iv.motivo
    assert iv.resumen()['replanificaciones'] >= 1
    cx, cy = caja[0]
    # Ninguna parte del robot toca la caja: centro a más de semidiagonal + medio lado.
    assert min(math.hypot(r[1] - cx, r[2] - cy) for r in registro) > 0.225 + 0.10
    assert quieto_maximo(registro) <= P.plazo_detenido_s


def test_pasillo_unico_bloqueado_falla_acotado():
    mundo = Mundo(pasillo_en_u())
    mundo.agregar_caja(2.6, 1.0, lado=0.3)
    iv = nueva(mundo)
    iv.fijar_meta((0.5, 1.5))
    registro = correr(iv, mundo, (0.5, 0.45, 0.0), 4000)
    assert iv.estado == 'FALLO'
    assert iv.motivo
    assert quieto_maximo(registro) <= P.plazo_detenido_s


def test_resumen_es_json_serializable():
    mundo = Mundo(pasillo_en_u())
    iv = nueva(mundo)
    iv.fijar_meta((0.5, 1.5))
    correr(iv, mundo, (0.5, 0.45, 0.0), 20)
    resumen = json.loads(json.dumps(iv.resumen()))
    assert resumen['estado'] == 'YENDO'
    assert resumen['regreso_estimado_m'] is not None
    for clave in ('motivo', 'largada', 'meta', 'restante_m', 'replanificaciones',
                  'quieto_s', 'ruta'):
        assert clave in resumen


def test_largada_se_refina_sin_moverse_hasta_aceptar_meta():
    mundo=Mundo(pasillo_en_u());iv=nueva(mundo)
    for t,pose in [(0.,Pose2D(.5,.45,0.)),(.1,Pose2D(.52,.46,.01))]:
        base,mapa=mundo.puntos(pose)
        assert iv.paso(EntradaNavegacion(t,pose,True,base,mapa)).es_cero()
    assert iv.largada==(.52,.46,.01)


def test_meta_no_finita_falla_sin_intentar_planificar():
    for bad in (float('nan'),float('inf'),-float('inf')):
        mundo=Mundo(pasillo_en_u());iv=nueva(mundo)
        iv.fijar_meta((bad,1.))
        assert iv.estado=='FALLO' and 'no finitas' in iv.motivo


def test_llegada_respeta_resolucion_sin_cambiar_tolerancia_de_regreso():
    iv=nueva(Mundo(pasillo_en_u()))
    assert iv.seguidor_ida.p.tolerancia_meta_m == math.sqrt(2)*RES
    assert iv.seguidor_vuelta.p.tolerancia_meta_m == P.tolerancia_largada_m
