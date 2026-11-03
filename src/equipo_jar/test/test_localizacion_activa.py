"""Pruebas de límites y criterios de calidad sin simulador."""
import math
from types import SimpleNamespace as S

import pytest

from equipo_jar.localizacion_activa import LocalizacionActiva, Observacion, concordancia_mapa


def observacion(secuencia=1, yaw=0.0, distancia=1.0, coincidencia=0.95):
    return Observacion(yaw, (0.0, 0.0, yaw), secuencia, 0.10, 0.10,
                       coincidencia, distancia)


def verificar(control):
    control.paso(0, True, observacion())
    assert control.paso(0.1, True, observacion()) > 0
    control.paso(1, True, observacion(yaw=0.4))
    assert control.estado == 'VERIFICANDO'


def test_sin_mapa_ni_sensores_no_mueve_y_el_plazo_no_se_reinicia():
    control = LocalizacionActiva(presupuesto=5)
    assert control.paso(100, False, observacion()) == 0
    assert control.estado == 'ESPERANDO_MAPA'
    assert control.paso(101, True, None) == 0
    assert control.paso(107, True, observacion()) == 0
    assert control.estado == 'INSUFICIENTE'


def test_datos_faltantes_y_obstaculo_detienen_incluso_durante_giro():
    control = LocalizacionActiva()
    control.paso(0, True, observacion())
    assert control.paso(0.1, True, observacion()) == 0.15
    assert control.paso(0.2, True, None) == 0
    assert control.paso(0.3, True, observacion(distancia=0.20)) == 0
    assert control.estado == 'INSUFICIENTE'
    assert control.paso(1, True, observacion()) == 0


def test_solo_declara_lista_con_nuevas_poses_estables_y_buena_calidad():
    control = LocalizacionActiva()
    verificar(control)
    for instante in (1.1, 1.2, 1.3):
        control.paso(instante, True, observacion(yaw=0.4))
    assert control.estado == 'VERIFICANDO'  # Repetir la misma pose no aporta evidencia.
    for secuencia in (2, 3, 4):
        assert control.paso(1 + secuencia * 0.2, True,
                            observacion(secuencia=secuencia, yaw=0.4)) == 0
    assert control.estado == 'LISTA'
    assert control.paso(10, True, observacion()) == 0


def test_covarianza_baja_sin_concordancia_no_habilita_y_respeta_maximo():
    control = LocalizacionActiva(max_giros=1, plazo_verificacion=3)
    verificar(control)
    for secuencia in (2, 3, 4):
        control.paso(1 + secuencia * 0.2, True,
                     observacion(secuencia=secuencia, yaw=0.4, coincidencia=0.1))
    assert control.estado == 'VERIFICANDO'
    assert control.paso(5, True, observacion(yaw=0.4)) == 0
    assert control.estado == 'INSUFICIENTE'


def test_giro_sin_progreso_tiene_plazo():
    control = LocalizacionActiva(plazo_giro=2)
    control.paso(0, True, observacion())
    assert control.paso(3, True, observacion()) == 0
    assert control.estado == 'INSUFICIENTE'


def test_concordancia_respeta_origen_rotado_desconocidos_y_limites():
    # Centro de la celda (1, 0) en un mapa girado 90°: mundo (-0.5, 1.5).
    data = [0] * 9
    data[1] = 100
    mapa = S(info=S(
        resolution=1.0, width=3, height=3,
        origin=S(position=S(x=0, y=0),
                 orientation=S(w=math.sqrt(0.5), x=0, y=0, z=math.sqrt(0.5)))),
        data=data)
    distancia = math.hypot(-0.5, 1.5)
    angulo = math.atan2(1.5, -0.5)
    score, count = concordancia_mapa(
        mapa, [(0, distancia)], angulo, 0, (0, 0, 0), 0.1)
    assert score == 1 and count == 1
    mapa.data[1] = -1
    assert concordancia_mapa(mapa, [(0, distancia)], angulo, 0, (0, 0, 0), 0.1)[0] == 0
    assert concordancia_mapa(mapa, [(0, 20.0)], 0, 0, (0, 0, 0), 0.1)[0] == 0


@pytest.mark.parametrize('kwargs', [
    {'velocidad': -0.1}, {'max_giros': 0}, {'concordancia_min': 1.1},
    {'presupuesto': math.nan}])
def test_rechaza_limites_invalidos(kwargs):
    with pytest.raises(ValueError):
        LocalizacionActiva(**kwargs)

def test_observacion_quieta_adicional_evitala_repeticion_de_giro():
    control = LocalizacionActiva(max_giros=1)
    verificar(control)
    for secuencia, instante in enumerate((4.1, 4.4, 4.7, 5.0), start=2):
        medida = observacion(secuencia=secuencia, yaw=0.4)
        medida.std_yaw = 0.16 if secuencia == 2 else 0.14
        assert control.paso(instante, True, medida) == 0
    assert control.estado == 'LISTA'
    assert control.giros == 1


def test_sin_espacio_inicial_localiza_quieto_con_poses_nuevas():
    control = LocalizacionActiva()
    assert control.paso(0, True, observacion(distancia=0.30)) == 0
    assert control.estado == 'VERIFICANDO_QUIETO'
    for secuencia in (2, 3, 4):
        assert control.paso(secuencia * 0.2, True,
                            observacion(secuencia=secuencia, distancia=0.30)) == 0
    assert control.estado == 'LISTA'
    assert control.giros == 0


def test_quieto_sin_poses_nuevas_no_habilita_ni_gira():
    control = LocalizacionActiva(plazo_verificacion=3)
    for instante in (0, 0.5, 1, 2):
        assert control.paso(instante, True, observacion(distancia=0.30)) == 0
    assert control.estado == 'VERIFICANDO_QUIETO'
    assert control.paso(3.1, True, observacion(distancia=0.30)) == 0
    assert control.estado == 'INSUFICIENTE'
    assert control.giros == 0


def test_quieto_sin_concordancia_no_habilita_ni_intenta_giro():
    control = LocalizacionActiva(plazo_verificacion=3)
    for secuencia, instante in enumerate((0, 0.5, 1, 2, 3.1), start=1):
        assert control.paso(instante, True, observacion(
            secuencia=secuencia, distancia=0.30, coincidencia=0.1)) == 0
    assert control.estado == 'INSUFICIENTE'
    assert control.paso(4, True, observacion(distancia=1)) == 0
    assert control.giros == 0

def test_quieto_sin_sensores_respeta_plazo():
    control = LocalizacionActiva(plazo_verificacion=3)
    control.paso(0, True, observacion(distancia=0.30))
    assert control.paso(3.1, True, None) == 0
    assert control.estado == 'INSUFICIENTE'



def lista_con_calidad_inicial():
    control = LocalizacionActiva(concordancia_min_supervision=.70)
    verificar(control)
    for secuencia in (2, 3, 4):
        control.paso(1 + secuencia*.2, True, observacion(secuencia, yaw=.4))
    assert control.utilizable
    return control


def test_supervision_acepta_078_solo_despues_de_primera_lista():
    control = lista_con_calidad_inicial()
    control.paso(2., True, observacion(5, yaw=.4, coincidencia=.78))
    assert control.estado == 'LISTA' and control.utilizable


def test_supervision_rechaza_065_y_recupera_con_078():
    control = lista_con_calidad_inicial()
    control.paso(2., True, observacion(5, yaw=.4, coincidencia=.65))
    assert not control.utilizable
    control.paso(3.1, True, observacion(6, yaw=.4, coincidencia=.65))
    assert control.estado == 'RECUPERANDO' and not control.utilizable
    for secuencia, instante in [(7, 3.3), (8, 3.5), (9, 3.7)]:
        control.paso(instante, True, observacion(secuencia, yaw=.4, coincidencia=.78))
    assert control.estado == 'LISTA' and control.utilizable


def test_inicial_078_no_llega_a_lista_aunque_supervision_permita_070():
    control = LocalizacionActiva(concordancia_min_supervision=.70)
    for secuencia in (1, 2, 3, 4):
        control.paso(secuencia*.2, True, observacion(secuencia, distancia=.30, coincidencia=.78))
        assert not control.utilizable and control.estado != 'LISTA'
    assert not control.calidad(observacion(coincidencia=.78))


@pytest.mark.parametrize('minimo', [0., -1., 1.1, math.nan, math.inf])
def test_supervision_rechaza_umbral_invalido(minimo):
    with pytest.raises(ValueError):
        LocalizacionActiva(concordancia_min_supervision=minimo)
