#!/usr/bin/env python3
"""Audita en ejecución qué interfaces usa el software del equipo (§7.1, §9.2).

Compara el grafo de ROS antes y después de lanzar el software del equipo:

1. `--capturar archivo`: guarda los nodos que ya existen (simulador, juez).
2. `--auditar archivo`: con el equipo corriendo, mira todos los nodos nuevos
   (los del equipo) y marca:
   - un topic que NO es de §7.1 y que además usa la plataforma o el simulador
     (algún nodo de la línea base lo publica o lo escucha): interfaz prohibida;
   - una publicación del equipo en un topic de §7.1 donde solo puede publicar
     el robot o el juez (por ejemplo /map o /scan).

Los topics internos del equipo (los que solo usan nodos del equipo) están
permitidos: Nav2, por ejemplo, crea decenas.
"""
import argparse
import sys
import time
from pathlib import Path

import rclpy
from rclpy.node import Node

sys.path.insert(0, str(Path(__file__).parent))
from interfaces import (  # noqa: E402
    PUBLICA_EL_EQUIPO, PUBLICA_EL_EQUIPO_RELIABLE, SENSORES_BEST_EFFORT, TOPICS_INFRAESTRUCTURA,
    TOPICS_PERMITIDOS)
from rclpy.qos import ReliabilityPolicy  # noqa: E402

# En estos el equipo puede publicar aunque no estén en PUBLICA_EL_EQUIPO:
# publica map -> odom (localización) y transformadas estáticas propias.
PUBLICABLES_EXTRA = {'/tf', '/tf_static'}


def nombre_completo(ns, nombre):
    return ns.rstrip('/') + '/' + nombre


def nodos_del_grafo(nodo):
    return {nombre_completo(ns, n) for n, ns in nodo.get_node_names_and_namespaces()
            if not n.startswith('_')}


def revisar_qos(topic, pubs_info, subs_info, equipo):
    """QoS de los publicadores y suscriptores del equipo (docs/QOS.md)."""
    errores = []
    if topic in PUBLICA_EL_EQUIPO_RELIABLE:
        for i in pubs_info:
            n = nombre_completo(i.node_namespace, i.node_name)
            if n in equipo and i.qos_profile.reliability != ReliabilityPolicy.RELIABLE:
                errores.append(f'{n} publica {topic} con QoS best effort: el juez o el robot no lo '
                               'escuchan. Tiene que ser reliable (la QoS por defecto de ROS 2)')
    if topic in SENSORES_BEST_EFFORT:
        for i in subs_info:
            n = nombre_completo(i.node_namespace, i.node_name)
            if n in equipo and i.qos_profile.reliability == ReliabilityPolicy.RELIABLE:
                errores.append(f'{n} se suscribe a {topic} con QoS reliable: el robot publica best '
                               'effort y no le va a llegar nada. Usá qos_profile_sensor_data')
    return errores


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--capturar', metavar='ARCHIVO')
    g.add_argument('--auditar', metavar='ARCHIVO')
    args = ap.parse_args()

    rclpy.init()
    nodo = Node('auditor_interfaces')
    fin = time.monotonic() + 4.0  # dejar que el descubrimiento se asiente
    while time.monotonic() < fin:
        rclpy.spin_once(nodo, timeout_sec=0.2)

    propios = nombre_completo(nodo.get_namespace(), nodo.get_name())
    actuales = nodos_del_grafo(nodo) - {propios}

    if args.capturar:
        Path(args.capturar).write_text('\n'.join(sorted(actuales)) + '\n')
        print(f'auditor: línea base con {len(actuales)} nodo(s)')
        nodo.destroy_node()
        rclpy.shutdown()
        return 0

    base = set(Path(args.auditar).read_text().split())
    equipo = actuales - base
    errores = []
    if not equipo:
        errores.append('no apareció ningún nodo nuevo: ¿arrancó el launch del equipo?')

    for topic, _ in nodo.get_topic_names_and_types():
        pubs_info = nodo.get_publishers_info_by_topic(topic)
        subs_info = nodo.get_subscriptions_info_by_topic(topic)
        pubs = {nombre_completo(i.node_namespace, i.node_name) for i in pubs_info}
        subs = {nombre_completo(i.node_namespace, i.node_name) for i in subs_info}
        errores.extend(revisar_qos(topic, pubs_info, subs_info, equipo))
        del_equipo = (pubs | subs) & equipo
        if not del_equipo:
            continue

        if topic in TOPICS_PERMITIDOS:
            ilegal = (pubs & equipo) if (
                topic not in PUBLICA_EL_EQUIPO and topic not in PUBLICABLES_EXTRA) else set()
            for n in sorted(ilegal):
                errores.append(f'{n} publica en {topic}, donde solo publican el robot o el juez')
        elif topic not in TOPICS_INFRAESTRUCTURA:
            de_la_plataforma = (pubs | subs) & base
            if de_la_plataforma:
                usan = ', '.join(sorted(del_equipo))
                errores.append(
                    f'{topic} no es de §7.1 y lo usa la plataforma/simulador '
                    f'({", ".join(sorted(de_la_plataforma))}); lo toca: {usan}')

    for e in errores:
        print(f'::error::auditor: {e}')
    print(f'auditor: {len(equipo)} nodo(s) del equipo, {len(errores)} problema(s)')
    nodo.destroy_node()
    rclpy.shutdown()
    return 1 if errores else 0


if __name__ == '__main__':
    sys.exit(main())
