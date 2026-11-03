"""Entrada oficial: AMCL, Nav2, exploración, detección y registro JAR.

Espera /map y acepta victimas:=N. Los parámetros se leen de los YAML instalados
y los valores variables se pasan por argumentos ROS, sin generar YAML temporales.
"""
import json
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    paquete = get_package_share_directory('equipo_jar')
    parametros = os.path.join(paquete, 'params')
    nav2 = os.path.join(parametros, 'nav2_exploracion.yaml')
    arbol = os.path.join(paquete, 'behavior_trees', 'navegar_rapido.xml')
    remaps_tf = [('/tf', 'tf'), ('/tf_static', 'tf_static')]

    def nodo(package, executable, *, name=None, archivo=None, overrides=(), remappings=()):
        argumentos = [
            '--ros-args', '--disable-external-lib-logs',
            '-p', ['use_sim_time:=', LaunchConfiguration('use_sim_time')],
        ]
        for override in overrides:
            argumentos.extend(['-p', override])
        return Node(
            package=package, executable=executable, name=name or executable,
            output='both', parameters=[archivo] if archivo else [],
            arguments=argumentos, remappings=list(remappings),
            additional_env={'PYTHONDONTWRITEBYTECODE': '1', 'RCUTILS_COLORIZED_OUTPUT': '0'},
        )

    nodos_navegacion = [
        ('nav2_controller', 'controller_server', [('cmd_vel', 'cmd_vel_nav')]),
        ('nav2_smoother', 'smoother_server', []),
        ('nav2_planner', 'planner_server', []),
        ('nav2_behaviors', 'behavior_server', []),
        ('nav2_bt_navigator', 'bt_navigator', []),
        ('nav2_waypoint_follower', 'waypoint_follower', []),
        ('nav2_velocity_smoother', 'velocity_smoother',
         [('cmd_vel', 'cmd_vel_nav'), ('cmd_vel_smoothed', 'cmd_vel')]),
    ]
    navegacion = [
        nodo(package, executable, archivo=nav2, remappings=remaps_tf + remaps,
             overrides=[f'default_nav_to_pose_bt_xml:={arbol}']
             if executable == 'bt_navigator' else [])
        for package, executable, remaps in nodos_navegacion
    ]
    navegacion.append(nodo(
        'nav2_lifecycle_manager', 'lifecycle_manager', name='lifecycle_manager_navigation',
        overrides=['autostart:=true',
                   'node_names:=' + json.dumps([n[1] for n in nodos_navegacion])],
    ))
    cantidad = [['cantidad_victimas:=', LaunchConfiguration('victimas')]]

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false',
                              description='true solo en simulación'),
        DeclareLaunchArgument('victimas', default_value='5',
                              description='Cantidad N informada por la organización'),
        DeclareLaunchArgument('rviz', default_value='false',
                              description='Abrir RViz para visualizar la prueba'),
        nodo('nav2_amcl', 'amcl', archivo=nav2),
        TimerAction(period=3.0, actions=[
            nodo('nav2_lifecycle_manager', 'lifecycle_manager',
                 name='lifecycle_manager_localization',
                 overrides=['autostart:=true', 'node_names:=[amcl]']),
        ]),
        TimerAction(period=5.0, actions=navegacion),
        TimerAction(period=8.0, actions=[
            nodo('equipo_jar', 'exploracion',
                 archivo=os.path.join(parametros, 'exploracion.yaml'), overrides=cantidad),
        ]),
        nodo('equipo_jar', 'deteccion_victimas',
             archivo=os.path.join(parametros, 'victimas.yaml'), overrides=cantidad),
        TimerAction(period=10.0, actions=[
            Node(
                package='rviz2', executable='rviz2', name='rviz2', output='both',
                condition=IfCondition(LaunchConfiguration('rviz')),
                arguments=[
                    '-d', os.path.join(parametros, 'exploracion.rviz'),
                    '--ros-args', '--disable-external-lib-logs', '--log-level', 'fatal',
                    '-p', ['use_sim_time:=', LaunchConfiguration('use_sim_time')],
                ],
                additional_env={
                    'RCUTILS_COLORIZED_OUTPUT': '0',
                    'MESA_SHADER_CACHE_DISABLE': 'true',
                },
            ),
        ]),
    ])
