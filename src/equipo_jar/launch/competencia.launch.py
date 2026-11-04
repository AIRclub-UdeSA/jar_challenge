"""Launch único del equipo (reglamento §9.1).

Este es el archivo que la organización lanza en la corrida:

    ros2 launch equipo_jar competencia.launch.py

Tiene que levantar TODO el software del equipo y quedarse esperando a que el
juez publique el mapa en /map. No mueve al robot ni publica reportes antes.

`victimas` es la cantidad de víctimas a buscar (N). La organización la informa en el lugar, antes de
cada corrida, y se pasa así:  ros2 launch equipo_jar competencia.launch.py victimas:=N

`use_sim_time` es false por defecto porque el robot real no publica /clock
(§7.3). En el simulador se pasa en true (lo hacen tools/correr_simulador.sh y
el CI).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    params = os.path.join(
        get_package_share_directory('equipo_jar'), 'params', 'params.yaml')
    use_sim_time = LaunchConfiguration('use_sim_time')
    victimas = LaunchConfiguration('victimas')

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('victimas', default_value='5',
                              description='cantidad de víctimas a buscar (N)'),
        # Misión: elige a dónde ir, reporta las cajas y es el único que publica en /cmd_vel.
        Node(
            package='equipo_jar',
            executable='nodo_mision',
            name='nodo_mision',
            output='screen',
            parameters=[params, {
                'use_sim_time': use_sim_time,
                'cantidad_victimas': ParameterValue(victimas, value_type=int),
            }],
        ),
        # Localización: AMCL compara el lidar con /map y publica map -> odom. nodo_pose_inicial
        # le da la pose de arranque (una gaussiana en el origen) cuando llega el mapa.
        Node(
            package='nav2_amcl',
            executable='amcl',
            name='amcl',
            output='screen',
            parameters=[params, {'use_sim_time': use_sim_time}],
        ),
        Node(
            package='equipo_jar',
            executable='nodo_pose_inicial',
            name='nodo_pose_inicial',
            output='screen',
            parameters=[params, {'use_sim_time': use_sim_time}],
        ),
        # Percepción: busca las cajas rojas y publica las confirmadas en /equipo/cajas_confirmadas.
        Node(
            package='equipo_jar',
            executable='nodo_percepcion',
            name='nodo_percepcion',
            output='screen',
            parameters=[params, {'use_sim_time': use_sim_time}],
        ),
        # Navegación: Nav2 va a un punto del mapa sin chocar. Su velocidad termina en
        # /equipo/cmd_vel_nav, no en /cmd_vel: la deja pasar nodo_mision.
        Node(
            package='nav2_controller',
            executable='controller_server',
            name='controller_server',
            output='screen',
            parameters=[params, {'use_sim_time': use_sim_time}],
            remappings=[('cmd_vel', 'cmd_vel_nav')],
        ),
        Node(
            package='nav2_planner',
            executable='planner_server',
            name='planner_server',
            output='screen',
            parameters=[params, {'use_sim_time': use_sim_time}],
        ),
        Node(
            package='nav2_behaviors',
            executable='behavior_server',
            name='behavior_server',
            output='screen',
            parameters=[params, {'use_sim_time': use_sim_time}],
            remappings=[('cmd_vel', '/equipo/cmd_vel_nav')],
        ),
        Node(
            package='nav2_bt_navigator',
            executable='bt_navigator',
            name='bt_navigator',
            output='screen',
            parameters=[params, {'use_sim_time': use_sim_time}],
        ),
        Node(
            package='nav2_velocity_smoother',
            executable='velocity_smoother',
            name='velocity_smoother',
            output='screen',
            parameters=[params, {'use_sim_time': use_sim_time}],
            remappings=[('cmd_vel', 'cmd_vel_nav'),
                        ('cmd_vel_smoothed', '/equipo/cmd_vel_nav')],
        ),
        # AMCL y Nav2 son nodos con ciclo de vida: nodo_arranque los activa, en orden.
        Node(
            package='equipo_jar',
            executable='nodo_arranque',
            name='nodo_arranque',
            output='screen',
            parameters=[params, {'use_sim_time': use_sim_time}],
        ),
    ])
