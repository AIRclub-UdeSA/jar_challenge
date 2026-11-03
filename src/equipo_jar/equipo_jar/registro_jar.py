"""Registro público único y acotado; se configura antes del launch.log de ROS."""
import logging
import os
from pathlib import Path
import sys

from ros2launch.option import OptionExtension


LIMITE_BYTES = 25_000_000
AVISO_LIMITE = b'\n[JAR] Limite de 25 MB alcanzado; se omiten los mensajes siguientes.\n'


def directorio_registro(*, prefijo=None, entorno=None, cwd=None):
    """JAR_LOG_DIR o logs_jar junto al install del workspace compilado."""
    entorno = os.environ if entorno is None else entorno
    if entorno.get('JAR_LOG_DIR'):
        return Path(entorno['JAR_LOG_DIR']).expanduser().absolute()
    if prefijo is None:
        from ament_index_python.packages import get_package_prefix
        prefijo = get_package_prefix('equipo_jar')
    prefijo = Path(prefijo).absolute()
    for candidato in (prefijo, *prefijo.parents):
        if candidato.name == 'install' or (candidato / '.colcon_install_layout').is_file():
            return candidato.parent / 'logs_jar'
    return Path(cwd if cwd is not None else os.getcwd()) / 'logs_jar'


class RegistroLimitado(logging.Handler):
    """Un escritor UTF-8, sin rotación ni archivos auxiliares; errores no fatales."""

    def __init__(self, directorio, *, limite=LIMITE_BYTES):
        super().__init__()
        self.limite = max(0, int(limite))
        self.escritos = 0
        self.stream = None
        self.error = None
        self.completo = False
        self._formatters = {}
        try:
            if directorio is None:
                raise OSError('No se pudo determinar el directorio de registro')
            self.directorio = Path(directorio)
            self.directorio.mkdir(parents=True, exist_ok=True)
            # Cada ejecución empieza un log nuevo. El evento guarda el anterior.
            self.stream = (self.directorio / 'log.txt').open('wb', buffering=0)
        except Exception as exc:
            self._fallo(exc)

    def _cerrar_stream(self):
        stream, self.stream = self.stream, None
        if stream is not None:
            try:
                stream.close()
            except Exception:
                pass

    def _fallo(self, exc):
        self.error = str(exc)
        self._cerrar_stream()
        try:
            print('[JAR] Registro deshabilitado; la corrida continúa: ' + str(exc),
                  file=sys.stderr)
        except Exception:
            pass

    def setFormatterFor(self, logger, formatter):
        """Interfaz de handlers utilizada por launch en ROS 2 Humble."""
        self._formatters[logger if isinstance(logger, str) else logger.name] = formatter

    def unsetFormatterFor(self, logger):
        self._formatters.pop(logger if isinstance(logger, str) else logger.name, None)

    def format(self, record):
        formatter = self._formatters.get(record.name)
        return formatter.format(record) if formatter is not None else super().format(record)

    def _escribir(self, datos):
        while datos:
            cantidad = self.stream.write(datos)
            if not cantidad:
                raise OSError('No se pudo completar la escritura del registro')
            self.escritos += cantidad
            datos = datos[cantidad:]

    def emit(self, record):
        if self.stream is None or self.completo:
            return
        try:
            datos = (self.format(record).rstrip('\n') + '\n').encode('utf-8', errors='replace')
            restante = self.limite - self.escritos
            if len(datos) <= restante:
                self._escribir(datos)
                return
            aviso = AVISO_LIMITE[:restante]
            contenido = datos[:max(0, restante - len(aviso))]
            contenido = contenido.decode('utf-8', errors='ignore').encode('utf-8')
            self._escribir(contenido + aviso)
            self.completo = True
        except Exception as exc:
            self._fallo(exc)

    def close(self):
        self.acquire()
        try:
            self._cerrar_stream()
        finally:
            self.release()
            super().close()


def _es_competencia(args):
    return (getattr(args, 'package_name', None) == 'equipo_jar'
            and getattr(args, 'launch_file_name', None) == 'competencia.launch.py')


class RegistroJarOption(OptionExtension):
    """Usar prestart de ros2launch, conservando el comando oficial del equipo."""

    def __init__(self):
        super().__init__()
        self.registro = None
        self._entorno_anterior = {}

    def prestart(self, args):
        if not _es_competencia(args):
            return
        import launch.logging

        try:
            directorio = directorio_registro()
        except Exception:
            directorio = None
        self.registro = RegistroLimitado(directorio)
        configuracion = launch.logging.launch_config
        self._factory_anterior = configuracion.log_handler_factory
        # Los nombres solicitados por launch convergen al mismo escritor.
        configuracion.log_handler_factory = lambda *a, **kw: self.registro
        configuracion.file_handlers.clear()
        configuracion.log_dir = str(
            self.registro.directorio if self.registro.stream is not None else Path.cwd())
        self._bytecode_anterior = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        for nombre, valor in {
            'OVERRIDE_LAUNCH_PROCESS_OUTPUT': 'both',
            'PYTHONDONTWRITEBYTECODE': '1',
            'RCUTILS_COLORIZED_OUTPUT': '0',
        }.items():
            self._entorno_anterior[nombre] = os.environ.get(nombre)
            os.environ[nombre] = valor
        self.registro.handle(logging.makeLogRecord({
            'name': 'JAR', 'levelno': logging.INFO, 'levelname': 'INFO',
            'msg': 'Inicio de corrida Stack&Queue. Registro público, máximo 25 MB.',
        }))

    def postlaunch(self, launch_return_code, args):
        if self.registro is None:
            return
        import launch.logging

        self.registro.handle(logging.makeLogRecord({
            'name': 'JAR', 'levelno': logging.INFO, 'levelname': 'INFO',
            'msg': f'Fin del launch. Código de salida: {launch_return_code}.',
        }))
        loggers = [logging.getLogger(), *logging.Logger.manager.loggerDict.values()]
        for logger in loggers:
            if isinstance(logger, logging.Logger):
                logger.removeHandler(self.registro)
        launch.logging.launch_config.file_handlers.clear()
        launch.logging.launch_config.log_handler_factory = self._factory_anterior
        launch.logging.launch_config.log_dir = None
        self.registro.close()
        self.registro = None
        sys.dont_write_bytecode = self._bytecode_anterior
        for nombre, valor in self._entorno_anterior.items():
            if valor is None:
                os.environ.pop(nombre, None)
            else:
                os.environ[nombre] = valor
