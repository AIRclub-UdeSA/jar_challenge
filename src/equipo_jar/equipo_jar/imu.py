"""Giro del robot medido con el IMU (giróscopo), sin depender de las ruedas.

En el robot real la odometría se equivoca sobre todo al girar: las ruedas mecanum patinan y
los motores no responden todos igual. El giróscopo mide la velocidad de giro directamente.

El IMU está montado dado vuelta y girado (en el URDF, rpy = 0 π π/2): su eje z apunta hacia
abajo, así que su angular_velocity.z tiene el signo invertido respecto del giro del robot. En
vez de suponer el montaje, la velocidad angular se rota al frame del robot con la tf
base_footprint <- frame del IMU (la publica el robot en /tf_static).
"""
from rclpy.time import Time
from sensor_msgs.msg import Imu


def fila_z_de_rotacion(x: float, y: float, z: float, w: float):
    """Tercera fila de la matriz de rotación del quaternion (x, y, z, w): multiplicada por un
    vector del frame de origen, da su componente z en el frame de destino."""
    return (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y))


class IntegradorGiro:
    """Integra la velocidad de giro (rad/s, alrededor del z del robot) en el tiempo.

    `yaw` no se normaliza a (-pi, pi]: lo que importa son las diferencias entre dos instantes.
    Si entre dos muestras pasa más de `dt_max` (se cortó el IMU), ese tramo no se integra.
    """

    def __init__(self, dt_max: float = 0.5):
        self.dt_max = dt_max
        self.yaw = 0.0
        self.velocidad = 0.0
        self.t = None

    def agregar(self, t: float, velocidad: float):
        if self.t is not None:
            dt = t - self.t
            if 0.0 < dt <= self.dt_max:
                self.yaw += 0.5 * (velocidad + self.velocidad) * dt   # regla del trapecio
        self.t = t
        self.velocidad = velocidad


class GiroImu:
    """Se suscribe a /imu/data desde `nodo` y lleva el giro del robot (z de base_frame)."""

    def __init__(self, nodo, tf_buffer, base_frame: str = 'base_footprint',
                 topic: str = '/imu/data'):
        self.nodo = nodo
        self.tf_buffer = tf_buffer
        self.base_frame = base_frame
        self.fila_z = None   # se pide a tf una vez: el IMU está fijo al robot
        self.integrador = IntegradorGiro()
        nodo.create_subscription(Imu, topic, self.recibir, 10)

    @property
    def listo(self) -> bool:
        return self.integrador.t is not None

    @property
    def yaw(self) -> float:
        return self.integrador.yaw

    @property
    def velocidad(self) -> float:
        return self.integrador.velocidad

    def instante(self) -> float:
        """Instante (s) de la última muestra, o None."""
        return self.integrador.t

    def recibir(self, msg: Imu):
        if self.fila_z is None:
            try:
                tf = self.tf_buffer.lookup_transform(
                    self.base_frame, msg.header.frame_id, Time())
            except Exception:
                return
            q = tf.transform.rotation
            self.fila_z = fila_z_de_rotacion(q.x, q.y, q.z, q.w)
        w = msg.angular_velocity
        fz = self.fila_z
        velocidad = fz[0] * w.x + fz[1] * w.y + fz[2] * w.z
        stamp = Time.from_msg(msg.header.stamp)
        if stamp.nanoseconds == 0:   # hay drivers que no completan el stamp
            stamp = self.nodo.get_clock().now()
        self.integrador.agregar(stamp.nanoseconds * 1e-9, velocidad)
