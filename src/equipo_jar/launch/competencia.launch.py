"""Launch único del equipo (reglamento §9.1).

Este es el archivo que la organización lanza en la corrida:

    ros2 launch equipo_jar competencia.launch.py victimas:=N

Levanta todo el software del equipo y se queda esperando a que el juez publique
el mapa en /map. Nada mueve al robot ni reporta antes.

- `victimas`: cantidad de víctimas a buscar (N), la informa la organización.
- `use_sim_time`: false por defecto porque el robot real no publica /clock
  (§7.3). En el simulador se pasa en true.
- `nav2`: false para levantar solo los nodos propios (útil para depurar).

Nodos (ADR-0002): supervisor (único publicador de /cmd_vel), misión (único que
reporta), explorador, percepción, AMCL y Nav2 sin map_server.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

NODOS_NAV2 = ['amcl', 'controller_server', 'planner_server', 'behavior_server', 'bt_navigator']


def generate_launch_description():
    share = get_package_share_directory('equipo_jar')
    params = LaunchConfiguration('params')
    params_nav2 = LaunchConfiguration('params_nav2')
    use_sim_time = LaunchConfiguration('use_sim_time')
    victimas = LaunchConfiguration('victimas')
    tiempo = {'use_sim_time': use_sim_time}
    remapeo_nav = [('cmd_vel', '/jar/cmd_vel_nav')]

    def propio(paquete, ejecutable, extra=None):
        return Node(package=paquete, executable=ejecutable, name=ejecutable, output='screen',
                    parameters=[params, tiempo, extra or {}])

    def nav2(paquete, ejecutable, remapeos=()):
        return Node(package=paquete, executable=ejecutable, name=ejecutable, output='screen',
                    parameters=[params_nav2, tiempo], remappings=list(remapeos))

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('victimas', default_value='5',
                              description='cantidad de víctimas a buscar (N)'),
        DeclareLaunchArgument('nav2', default_value='true'),
        DeclareLaunchArgument('params',
                              default_value=os.path.join(share, 'params', 'params.yaml'),
                              description='parámetros de los nodos propios (práctica: probar variantes)'),
        DeclareLaunchArgument('params_nav2',
                              default_value=os.path.join(share, 'params', 'nav2.yaml'),
                              description='parámetros de Nav2 (práctica: probar variantes)'),

        propio('jar_supervisor', 'supervisor'),
        propio('equipo_jar', 'mision',
               {'cantidad_victimas': ParameterValue(victimas, value_type=int)}),
        propio('jar_exploracion', 'explorador',
               {'cantidad_victimas': ParameterValue(victimas, value_type=int)}),
        propio('jar_percepcion', 'percepcion'),

        GroupAction(condition=IfCondition(LaunchConfiguration('nav2')), actions=[
            nav2('nav2_amcl', 'amcl'),
            nav2('nav2_controller', 'controller_server', remapeo_nav),
            nav2('nav2_planner', 'planner_server'),
            nav2('nav2_behaviors', 'behavior_server', remapeo_nav),
            nav2('nav2_bt_navigator', 'bt_navigator'),
            Node(package='nav2_lifecycle_manager', executable='lifecycle_manager',
                 name='lifecycle_manager_navegacion', output='screen',
                 parameters=[tiempo, {'autostart': True, 'node_names': NODOS_NAV2,
                                      'bond_timeout': 10.0}]),
        ]),
    ])
