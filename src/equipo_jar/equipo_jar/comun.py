"""Lo que comparten los nodos del equipo (docs/team/INTERFACES.md)."""
from rcl_interfaces.msg import ParameterDescriptor, SetParametersResult
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

# /map: el QoS que usa el juez.
QOS_MAPA = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)

# Topics /jar/ que guardan un estado (plan, candidatas): un nodo que arranca
# tarde recibe el último mensaje.
QOS_ESTADO = QOS_MAPA


def _es_numerico(valor):
    return isinstance(valor, (int, float)) and not isinstance(valor, bool)


def _convertir(valor, defecto):
    """`valor` con el tipo de `defecto` (float o int), o None si no se puede.

    Un float acepta un entero; un int acepta un float sin decimales (12.0).
    """
    if not _es_numerico(valor):
        return None
    if isinstance(defecto, float):
        return float(valor)
    if isinstance(valor, float) and not valor.is_integer():
        return None
    return int(valor)


def convertir_como(valor, defecto):
    """`valor` con el tipo de `defecto` si es numérico y se puede; si no, sin cambios."""
    if _es_numerico(defecto):
        convertido = _convertir(valor, defecto)
        if convertido is not None:
            return convertido
    return valor


def declarar(nodo, nombre, defecto):
    """declare_parameter que acepta `3` donde el defecto es `3.0` (y `12.0` por `12`).

    rclpy fija el tipo con el defecto: un entero en el YAML para un parámetro
    float tira InvalidParameterTypeException y el nodo muere al arrancar.
    Los numéricos se declaran con tipo dinámico y se convierten; un valor que
    no sirve se ignora (queda el defecto) con un aviso.
    """
    if not _es_numerico(defecto):
        return nodo.declare_parameter(nombre, defecto).value
    valor = nodo.declare_parameter(
        nombre, defecto, ParameterDescriptor(dynamic_typing=True)).value
    convertido = _convertir(valor, defecto)
    if convertido is None:
        nodo.get_logger().warning(
            f'parameter {nombre}={valor!r} does not fit {type(defecto).__name__}: '
            f'using {defecto}')
        return defecto
    return convertido


def parametros_en_vivo(nodo, valores):
    """Declara los parámetros y devuelve un dict que se actualiza en vivo.

    `valores` son los nombres con su valor por defecto. Leé siempre del dict
    (por ejemplo `self.p['umbral']`), no guardes copias: así un cambio hecho
    durante la corrida se aplica sin reiniciar el nodo. Un parámetro float
    acepta también un entero y viceversa (ver `declarar`).
    """
    actuales = {n: declarar(nodo, n, v) for n, v in valores.items()}

    def al_cambiar(parametros):
        nuevos = {}
        for p in parametros:
            if p.name not in actuales:
                continue
            valor = p.value
            if _es_numerico(valores[p.name]):
                valor = _convertir(valor, valores[p.name])
                if valor is None:
                    return SetParametersResult(
                        successful=False,
                        reason=f'{p.name} must be {type(valores[p.name]).__name__}')
            nuevos[p.name] = valor
        actuales.update(nuevos)
        return SetParametersResult(successful=True)

    nodo.add_on_set_parameters_callback(al_cambiar)
    return actuales


def correr(clase_nodo, args=None):
    """main() de todos los nodos."""
    import rclpy
    from rclpy.executors import ExternalShutdownException
    rclpy.init(args=args)
    nodo = clase_nodo()
    try:
        rclpy.spin(nodo)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass    # Ctrl+C / end of the run: Humble raises the second one on SIGINT
    except Exception:
        if rclpy.ok():
            raise
        # Also the end of the run: the context was shut down mid-spin ("failed to
        # initialize wait set: the given context is not valid").
    finally:
        nodo.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
