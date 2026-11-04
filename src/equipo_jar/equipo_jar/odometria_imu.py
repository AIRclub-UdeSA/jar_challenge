"""Odometría del equipo: traslación de las ruedas con el rumbo de la orientación del IMU.

En el bag real de la semana 10 las ruedas sobregiran ~25 % y la orientación del IMU coincide
con el LiDAR (0,7°); el giróscopo del robot real tiene el signo invertido, por eso se usa la
orientación. Este nodo deja el /tf oficial intacto: copia /tf y /tf_static a /equipo/tf y
/equipo/tf_static, reemplazando sólo odom->base_footprint (misma estampa). AMCL y los nodos del
equipo leen ese árbol. Si el IMU falta, envejece o no concuerda con las ruedas, sigue con las ruedas.
"""
from collections import deque
import math

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import Imu
from tf2_msgs.msg import TFMessage

QOS_TF = QoSProfile(depth=100, durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST)
QOS_TF_ESTATICO = QoSProfile(depth=100, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST)


def yaw(q):
    return math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))


def desenvolver(anterior, angulo):
    return angulo if anterior is None else anterior+math.remainder(angulo-anterior, 2*math.pi)


def interpolar(serie, t, max_hueco=math.inf):
    """Valor de una serie (t, v) ordenada en t, dentro de su rango; None fuera de él
    o entre dos muestras separadas más de `max_hueco`."""
    if not serie or t < serie[0][0] or t > serie[-1][0]:
        return None
    for i in range(len(serie)-1, 0, -1):  # lo pedido suele estar al final
        (t0, v0), (t1, v1) = serie[i-1], serie[i]
        if t0 <= t:
            if t1-t0 > max_hueco and t0 < t < t1:
                return None
            return v1 if t1 == t0 else v0+(v1-v0)*(t-t0)/(t1-t0)
    return serie[0][1]


class RumboImu:
    """Pose odom->base con rumbo del IMU; empieza igual a las ruedas.

    El IMU se habilita recién cuando, tras `giro_verificacion_rad` de giro de ruedas, la
    pendiente giro_IMU/giro_ruedas cae en [pendiente_min, pendiente_max] (real: ~0,8). Un
    signo invertido o un IMU quieto mientras las ruedas giran dejan las ruedas.
    """

    def __init__(self, max_edad_imu_s=0.5, giro_verificacion_rad=1.0, pendiente_min=0.5,
                 pendiente_max=1.5, desvio_minimo=0.10):
        if not math.isfinite(desvio_minimo) or desvio_minimo < 0:
            raise ValueError("Desvío mínimo inválido")
        self.desvio_minimo = desvio_minimo
        self.max_edad_imu_s = max_edad_imu_s
        self.giro_verificacion_rad = giro_verificacion_rad
        self.pendiente_min, self.pendiente_max = pendiente_min, pendiente_max
        self.reiniciar()

    def reiniciar(self):
        self.imu = deque(maxlen=400)
        self.ruedas = deque(maxlen=400)
        self.previa = None  # (t, x, y, rumbo de ruedas desenvuelto, rumbo IMU o None)
        self.pose = None
        self.desfase = None  # rumbo propio - rumbo IMU mientras se usa el IMU
        self.verificado = None  # Plausibilidad de la pendiente; no decide por sí sola el uso.
        self.correccion_necesaria = False
        self.suma_rr = self.suma_ri = self.giro_medido = 0.
        self.usando_imu = False

    def agregar_imu(self, t, angulo):
        if self.imu and t < self.imu[-1][0]-1.:
            self.reiniciar()  # reinicio del reloj de simulación
        if self.imu and t <= self.imu[-1][0]:
            return
        self.imu.append((t, desenvolver(self.imu[-1][1] if self.imu else None, angulo)))

    def rumbo_imu(self, t):
        """Rumbo del IMU en t: interpolado, o el último más el giro de ruedas desde entonces."""
        if self.imu and t <= self.imu[-1][0]:
            return interpolar(self.imu, t, self.max_edad_imu_s)
        if not self.imu or t-self.imu[-1][0] > self.max_edad_imu_s:
            return None
        t_imu, ultimo = self.imu[-1]
        ruedas_entonces = interpolar(self.ruedas, t_imu)
        if ruedas_entonces is None:
            return None
        return ultimo+self.ruedas[-1][1]-ruedas_entonces

    def agregar_ruedas(self, t, x, y, angulo):
        """Devuelve (x, y, rumbo) propios para la pose de ruedas en t."""
        if self.previa is not None and t < self.previa[0]-1.:
            self.reiniciar()
        if self.previa is not None and t <= self.previa[0]:
            return self.pose
        rr = desenvolver(self.previa[3] if self.previa else None, angulo)
        self.ruedas.append((t, rr))
        ri = self.rumbo_imu(t)
        if self.previa is None:
            self.previa, self.pose = (t, x, y, rr, ri), (x, y, angulo)
            return self.pose
        t0, x0, y0, rr0, ri0 = self.previa
        self._verificar(rr-rr0, None if ri is None or ri0 is None else ri-ri0)
        if not self.correccion_necesaria:
            # Hasta medir o si no hay sobregiro: passthrough exacto, sin reintegrar XY.
            self.pose = (x, y, angulo)
            self.previa = (t, x, y, rr, ri)
            self.usando_imu = False
            return self.pose
        rumbo0 = self.pose[2]
        if self.verificado and ri is not None and ri0 is not None:
            if not self.usando_imu:  # anclar sin salto al volver al IMU
                self.desfase = rumbo0-ri0
            rumbo = ri+self.desfase
            self.usando_imu = True
        else:
            rumbo = rumbo0+(rr-rr0)
            self.usando_imu = False
        # Avance en el marco del robot según las ruedas, orientado con el rumbo propio.
        c, s = math.cos(rr0), math.sin(rr0)
        bx, by = c*(x-x0)+s*(y-y0), -s*(x-x0)+c*(y-y0)
        medio = rumbo0+math.remainder(rumbo-rumbo0, 2*math.pi)/2
        c, s = math.cos(medio), math.sin(medio)
        self.pose = (self.pose[0]+c*bx-s*by, self.pose[1]+s*bx+c*by, rumbo)
        self.previa = (t, x, y, rr, ri)
        return self.pose

    def _verificar(self, d_ruedas, d_imu):
        if self.verificado is not None or d_imu is None:
            return
        self.suma_rr += d_ruedas*d_ruedas
        self.suma_ri += d_ruedas*d_imu
        self.giro_medido += abs(d_ruedas)
        if self.giro_medido >= self.giro_verificacion_rad:
            pendiente = self.suma_ri/self.suma_rr if self.suma_rr > 0 else 0.
            self.pendiente = pendiente
            self.verificado = self.pendiente_min <= pendiente <= self.pendiente_max
            self.correccion_necesaria = bool(self.verificado and
                abs(pendiente-1.)+1e-12 >= self.desvio_minimo)


def estampa_s(header):
    return header.stamp.sec+header.stamp.nanosec*1e-9


class OdometriaImu(Node):
    def __init__(self, **kwargs):
        super().__init__('odometria_imu', **kwargs)
        for nombre, valor in [('marco_odom', 'odom'), ('marco_base', 'base_footprint'),
                              ('topico_imu', '/imu/data'), ('max_edad_imu_s', 0.5),
                              ('giro_verificacion_rad', 1.0), ('pendiente_min', 0.5),
                              ('pendiente_max', 1.5), ('desvio_minimo', 0.10)]:
            self.declare_parameter(nombre, valor)
        v = lambda n: self.get_parameter(n).value
        self.odom, self.base = v('marco_odom'), v('marco_base')
        self.fusion = RumboImu(v('max_edad_imu_s'), v('giro_verificacion_rad'),
                               v('pendiente_min'), v('pendiente_max'), v('desvio_minimo'))
        self.estaticas = {}
        self.informado = None
        self.pub_tf = self.create_publisher(TFMessage, '/equipo/tf', QOS_TF)
        self.pub_estatico = self.create_publisher(TFMessage, '/equipo/tf_static', QOS_TF_ESTATICO)
        self.create_subscription(Imu, v('topico_imu'), self.al_imu, qos_profile_sensor_data)
        self.create_subscription(TFMessage, '/tf', self.al_tf, QOS_TF)
        self.create_subscription(TFMessage, '/tf_static', self.al_tf_estatico, QOS_TF_ESTATICO)

    def al_imu(self, msg):
        q = msg.orientation
        if q.x == q.y == q.z == q.w == 0.:
            return  # sin orientación
        self.fusion.agregar_imu(estampa_s(msg.header), yaw(q))

    def al_tf(self, msg):
        for tr in msg.transforms:
            if tr.header.frame_id == self.odom and tr.child_frame_id == self.base:
                a = tr.transform
                x, y, rumbo = self.fusion.agregar_ruedas(
                    estampa_s(tr.header), a.translation.x, a.translation.y, yaw(a.rotation))
                if self.fusion.correccion_necesaria:
                    a.translation.x, a.translation.y = x, y
                    a.rotation.x = a.rotation.y = 0.
                    a.rotation.z, a.rotation.w = math.sin(rumbo/2), math.cos(rumbo/2)
                # Si la compuerta queda en ruedas, ni siquiera se reescribe el quaternion.
                self.informar()
        self.pub_tf.publish(msg)

    def al_tf_estatico(self, msg):
        for tr in msg.transforms:
            self.estaticas[(tr.header.frame_id, tr.child_frame_id)] = tr
        self.pub_estatico.publish(TFMessage(transforms=list(self.estaticas.values())))

    def informar(self):
        estado = self.fusion.verificado
        if estado != self.informado:
            self.informado = estado
            if estado is True and not self.fusion.correccion_necesaria:
                self.get_logger().info(
                    f'Compuerta IMU: ruedas sin modificar (pendiente IMU/ruedas {self.fusion.pendiente:.2f})')
            elif estado is True:
                self.get_logger().info(
                    f'Rumbo del IMU en uso (pendiente IMU/ruedas {self.fusion.pendiente:.2f})')
            elif estado is False:
                self.get_logger().warn(
                    f'IMU descartado: pendiente IMU/ruedas {self.fusion.pendiente:.2f}; sigo con ruedas')


def main():
    rclpy.init()
    nodo = OdometriaImu()
    try:
        rclpy.spin(nodo)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        import signal
        # El primer SIGINT ya detuvo spin; no interrumpir otra vez la limpieza.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            nodo.destroy_node()
        except (KeyboardInterrupt, ExternalShutdownException):
            pass
        try:
            rclpy.try_shutdown()
        except (KeyboardInterrupt, ExternalShutdownException):
            pass
        finally:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
