"""Activa los nodos con ciclo de vida (AMCL y Nav2), uno por uno y en orden.

Reemplaza al lifecycle_manager de Nav2. Ese gestor espera sin límite la respuesta de cada
transición, y a veces la respuesta se pierde (`failed to send response to .../change_state`): se
queda esperando para siempre, no llega a activar el resto y el robot no se mueve en toda la corrida.

Acá ninguna respuesta es imprescindible: se mira el estado del nodo y se pide la transición que
falta. Si algo se pierde, la próxima consulta lo arregla.
"""
import time

from lifecycle_msgs.msg import State, Transition
from lifecycle_msgs.srv import ChangeState, GetState
from equipo_jar.ejecucion import main_de
from rclpy.node import Node

TRANSICION_PARA_ESTADO = {
    State.PRIMARY_STATE_UNCONFIGURED: Transition.TRANSITION_CONFIGURE,
    State.PRIMARY_STATE_INACTIVE: Transition.TRANSITION_ACTIVATE,
}


def transicion_para(estado):
    """La transición que acerca al nodo a activo, o None si solo queda esperar."""
    return TRANSICION_PARA_ESTADO.get(estado)


class NodoArranque(Node):
    def __init__(self):
        super().__init__('nodo_arranque')
        self.declare_parameter('nodos', ['amcl'])
        self.declare_parameter('periodo_s', 0.5)
        self.declare_parameter('espera_respuesta_s', 2.0)
        self.declare_parameter('espera_transicion_s', 3.0)
        self.espera_respuesta = self.get_parameter('espera_respuesta_s').value
        self.espera_transicion = self.get_parameter('espera_transicion_s').value
        self.nodos = list(self.get_parameter('nodos').value)
        self.cliente_estado = None
        self.cliente_cambio = None
        self.consulta = None
        self.t_consulta = 0.0
        self.t_transicion = -self.espera_transicion
        self.preparar_nodo()
        self.timer = self.create_timer(self.get_parameter('periodo_s').value, self.ciclo)

    def ahora(self):
        # Reloj de pared: estas esperas son de la PC, no del tiempo simulado.
        return time.monotonic()

    def preparar_nodo(self):
        nombre = self.nodos[0]
        self.cliente_estado = self.create_client(GetState, f'/{nombre}/get_state')
        self.cliente_cambio = self.create_client(ChangeState, f'/{nombre}/change_state')
        self.consulta = None

    def ciclo(self):
        if self.consulta is not None:
            if self.consulta.done():
                estado = self.consulta.result().current_state.id
                self.consulta = None
                self.atender_estado(estado)
            elif self.ahora() - self.t_consulta > self.espera_respuesta:
                # El nodo está en plena transición o se perdió la respuesta: se vuelve a mirar.
                self.cliente_estado.remove_pending_request(self.consulta)
                self.consulta = None
            return
        if self.cliente_estado.service_is_ready():
            self.consulta = self.cliente_estado.call_async(GetState.Request())
            self.t_consulta = self.ahora()

    def atender_estado(self, estado):
        nombre = self.nodos[0]
        if estado == State.PRIMARY_STATE_ACTIVE:
            self.get_logger().info(f'{nombre} activo')
            self.destroy_client(self.cliente_estado)
            self.destroy_client(self.cliente_cambio)
            self.nodos.pop(0)
            self.t_transicion = -self.espera_transicion
            if self.nodos:
                self.preparar_nodo()
            else:
                self.get_logger().info('todos los nodos activos')
                self.timer.cancel()
            return
        transicion = transicion_para(estado)
        ahora = self.ahora()
        if transicion is None or ahora - self.t_transicion < self.espera_transicion:
            return
        pedido = ChangeState.Request()
        pedido.transition.id = transicion
        self.cliente_cambio.call_async(pedido)  # no se espera: manda el estado que se lea después
        self.t_transicion = ahora
        self.get_logger().info(f'{nombre}: transición {transicion}')


def main():
    main_de(NodoArranque)  # una excepción en un callback no mata el nodo (ver ejecucion.py)


if __name__ == '__main__':
    main()
