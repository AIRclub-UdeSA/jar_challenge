import errno
import logging
from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor

import pytest

from equipo_jar.registro_jar import (
    LIMITE_BYTES, RegistroJarOption, RegistroLimitado, directorio_registro,
)


def mensaje(texto):
    return logging.makeLogRecord({'name': 'prueba', 'msg': texto, 'levelno': logging.INFO})


@pytest.mark.parametrize('relativo', ['install/equipo_jar', 'install'])
def test_directorio_es_workspace_compilado_y_no_el_cwd(tmp_path, relativo):
    assert directorio_registro(prefijo=tmp_path / relativo, entorno={}, cwd='/') == (
        tmp_path / 'logs_jar')


def test_detecta_install_con_nombre_personalizado(tmp_path):
    instalacion = tmp_path / 'salida'
    instalacion.mkdir()
    (instalacion / '.colcon_install_layout').touch()
    assert directorio_registro(prefijo=instalacion / 'equipo_jar', entorno={}) == (
        tmp_path / 'logs_jar')


def test_respeta_jar_log_dir(tmp_path):
    elegido = tmp_path / 'elegido'
    assert directorio_registro(entorno={'JAR_LOG_DIR': str(elegido)}, prefijo='/otro/install') == elegido


def test_crea_solo_log_txt_y_reinicia_el_archivo_por_corrida(tmp_path):
    directorio = tmp_path / 'nueva' / 'logs_jar'
    for texto in ('corrida anterior', 'corrida actual'):
        registro = RegistroLimitado(directorio)
        registro.handle(mensaje(texto))
        registro.close()
    assert sorted(p.name for p in directorio.iterdir()) == ['log.txt']
    assert (directorio / 'log.txt').read_text() == 'corrida actual\n'


def test_limite_real_de_25_mb_incluye_utf8_y_no_rota(tmp_path):
    registro = RegistroLimitado(tmp_path)
    # Cada carácter ocupa cuatro bytes, no uno.
    for _ in range(28):
        registro.handle(mensaje('🤖' * 250_000))
    registro.close()
    datos = (tmp_path / 'log.txt').read_bytes()
    assert LIMITE_BYTES == 25_000_000
    assert 24_999_900 <= len(datos) <= LIMITE_BYTES
    assert 'Limite de 25 MB' in datos.decode('utf-8')
    assert sorted(p.name for p in tmp_path.iterdir()) == ['log.txt']


def test_concurrencia_respeta_el_limite_sin_archivos_extra(tmp_path):
    registro = RegistroLimitado(tmp_path, limite=4096)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda i: registro.handle(mensaje(f'{i}: ' + 'á' * 50)), range(1000)))
    registro.close()
    datos = (tmp_path / 'log.txt').read_bytes()
    assert len(datos) <= 4096
    datos.decode('utf-8')
    assert sorted(p.name for p in tmp_path.iterdir()) == ['log.txt']


def test_no_escribir_no_impide_continuar_y_no_hay_ruta_alternativa(tmp_path):
    obstaculo = tmp_path / 'no_es_directorio'
    obstaculo.write_text('archivo existente')
    registro = RegistroLimitado(obstaculo / 'logs_jar')
    registro.handle(mensaje('continúo'))
    registro.close()
    assert registro.error
    assert sorted(p.name for p in tmp_path.iterdir()) == ['no_es_directorio']


def test_disco_lleno_desactiva_solo_el_registro(tmp_path):
    registro = RegistroLimitado(tmp_path)
    registro._cerrar_stream()

    class DiscoLleno:
        def write(self, datos):
            raise OSError(errno.ENOSPC, 'disco lleno')

        def close(self):
            raise OSError(errno.EIO, 'error al cerrar')

    registro.stream = DiscoLleno()
    registro.handle(mensaje('primer error'))
    registro.handle(mensaje('la misión continúa'))
    assert registro.stream is None
    assert registro.error
    registro.close()


def test_error_de_formato_no_se_propaga(tmp_path):
    registro = RegistroLimitado(tmp_path)
    registro.setFormatter(logging.Formatter('%(campo_inexistente)s'))
    registro.handle(mensaje('continúo'))
    assert registro.error
    registro.close()


def test_extension_no_modifica_otro_launch(monkeypatch):
    import launch.logging
    anterior = launch.logging.launch_config
    extension = RegistroJarOption()
    extension.prestart(SimpleNamespace(package_name='otro', launch_file_name='competencia.launch.py'))
    assert extension.registro is None
    assert launch.logging.launch_config is anterior


@pytest.mark.parametrize('falla', [False, True])
def test_extension_redirige_todos_los_logs_incluso_si_no_se_puede_escribir(
        tmp_path, monkeypatch, falla):
    import launch.logging
    from equipo_jar import registro_jar
    directorio = tmp_path / 'logs_jar'
    if falla:
        directorio.write_text('no es una carpeta')
    monkeypatch.setattr(registro_jar, 'directorio_registro', lambda: directorio)
    configuracion = launch.logging.LaunchConfig()
    monkeypatch.setattr(launch.logging, 'launch_config', configuracion)
    monkeypatch.setenv('OVERRIDE_LAUNCH_PROCESS_OUTPUT', 'screen')
    extension = RegistroJarOption()
    args = SimpleNamespace(package_name='equipo_jar', launch_file_name='competencia.launch.py')
    extension.prestart(args)
    import os
    assert os.environ['OVERRIDE_LAUNCH_PROCESS_OUTPUT'] == 'both'
    assert configuracion.get_log_file_handler() is configuracion.get_log_file_handler('otro.log')
    logger = launch.logging.get_logger('prueba_registro_jar')
    logger.info('mensaje del launch')
    extension.postlaunch(0, args)
    assert os.environ['OVERRIDE_LAUNCH_PROCESS_OUTPUT'] == 'screen'
    if falla:
        assert directorio.read_text() == 'no es una carpeta'
    else:
        assert sorted(p.name for p in directorio.iterdir()) == ['log.txt']
        texto = (directorio / 'log.txt').read_text()
        assert 'mensaje del launch' in texto
        assert 'Fin del launch' in texto
