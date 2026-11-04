"""El diagnóstico no habilita inspecciones ni cambia las decisiones existentes."""
from dataclasses import replace
import numpy as np
from test_inspeccion_lidar import setup


def test_registro_no_cambia_decisiones_ni_cobertura():
    for occluded in (False, True):
        on, _, entry = setup(occluded)
        off, _, other = setup(occluded)
        off.registrar_diagnosticos = False
        for sequence in range(3):
            on.actualizar_pistas_lidar(replace(entry, ahora=1.+sequence*.2,
                origen_laser_mapa=(.7,.7), estampa_scan=(sequence,0)))
            off.actualizar_pistas_lidar(replace(other, ahora=1.+sequence*.2,
                origen_laser_mapa=(.7,.7), estampa_scan=(sequence,0)))
        assert on.estado == off.estado
        assert on.pistas_lidar == off.pistas_lidar
        assert on.solicitudes_lidar == off.solicitudes_lidar
        assert on.vista == off.vista
        np.testing.assert_array_equal(on.cobertura.observada, off.cobertura.observada)
        np.testing.assert_array_equal(on.cobertura.solo_camara, off.cobertura.solo_camara)
        assert on.diagnostico_superficies_lidar
        assert not off.diagnostico_superficies_lidar
        assert on.diagnostico_superficies_lidar[-1]['motivo'] == ('admitida' if occluded else 'mapa_no_ocluye')


def test_superficie_registra_deuda_generada_sin_modificar_filtro():
    mission, _, entry = setup(False)
    mission.actualizar_pistas_lidar(replace(entry, origen_laser_mapa=(.7,.7)))
    record = mission.diagnostico_superficies_lidar[-1]
    assert record['motivo'] == 'mapa_no_ocluye'
    assert record['genero_solo_camara'] and record['muestras_generadas'] > 0
    assert record['solo_camara_pendiente'] > 0
    assert not mission.pistas_lidar  # Sigue excluida por la política original.


def test_vista_registra_objetivo_y_muestras_realmente_vistas_sin_rayos_extra(monkeypatch):
    mission, _, _ = setup(False)
    ids = list(range(len(mission.cobertura.puntos)))
    mission._iniciar_diagnostico_vista(mission.vista, ids)
    predict = mission.cobertura.predecir
    calls = []
    def spy(*args, **kwargs):
        calls.append(1)
        return predict(*args, **kwargs)
    monkeypatch.setattr(mission.cobertura, 'predecir', spy)
    mission.observar(dict(estampa_s=1., pose_camara=(.7,.7,0.),
        campo_horizontal_rad=1., region_roja=False), 1., (0.,0.,0.))
    diag = mission.resumen()['diagnostico_vista']
    assert len(calls) == 1
    assert diag['muestras_objetivo'] == len(ids)
    assert diag['muestras_objetivo_vistas'] == int(mission.cobertura.ultima_visible.sum())
    assert diag['objetivo_rle'] == [[0, len(ids)]]
