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
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = get_package_share_directory('equipo_jar')
    params = os.path.join(share, 'params', 'params.yaml')
    use_sim_time = LaunchConfiguration('use_sim_time')
    victimas = LaunchConfiguration('victimas')

    def nodo(nombre, extra=None):
        return Node(
            package='equipo_jar',
            executable=nombre,
            name=nombre,
            output='screen',
            parameters=[params, {'use_sim_time': use_sim_time, **(extra or {})}],
        )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('victimas', default_value='5',
                              description='cantidad de víctimas a buscar (N)'),
        DeclareLaunchArgument('localizacion', default_value='true',
                              description='false: sin AMCL (pruebas.launch.py)'),
        nodo('cobertura'),
        nodo('deteccion'),
        nodo('fusion'),
        nodo('mision', {'cantidad_victimas': ParameterValue(victimas, value_type=int)}),
        # Localización (AMCL) y navegación (Nav2).
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(share, 'launch', 'navegacion.launch.py')),
            launch_arguments={
                'use_sim_time': use_sim_time,
                'localizacion': LaunchConfiguration('localizacion'),
            }.items(),
        ),
    ])
