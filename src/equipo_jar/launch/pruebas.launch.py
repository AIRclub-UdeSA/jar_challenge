"""Launch de pruebas: competencia.launch.py más lo que falta para probar.

Agrega una TF fija map → odom (identidad), para que todos los módulos trabajen
en el simulador antes de que exista localización. Nunca va en
competencia.launch.py (docs/team/INTERFACES.md).

    ros2 launch equipo_jar pruebas.launch.py use_sim_time:=true victimas:=3
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    competencia = os.path.join(
        get_package_share_directory('equipo_jar'), 'launch', 'competencia.launch.py')
    use_sim_time = LaunchConfiguration('use_sim_time')

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('victimas', default_value='5'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(competencia),
            launch_arguments={
                'use_sim_time': use_sim_time,
                'victimas': LaunchConfiguration('victimas'),
                # La TF fija map -> odom de abajo reemplaza a AMCL: dos publicadores se pelearían.
                'localizacion': 'false',
            }.items(),
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='prueba_map_odom',
            arguments=['--frame-id', 'map', '--child-frame-id', 'odom'],
            parameters=[{'use_sim_time': use_sim_time}],
        ),
    ])
