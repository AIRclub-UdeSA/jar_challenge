"""Probar el software del equipo contra un bag grabado en el robot real.

Herramienta de desarrollo, NO es parte de la entrega (vive fuera de src/).

    ros2 launch dev/bag.launch.py
    ros2 launch dev/bag.launch.py bag:=~/rosmaster_ws/bags/otro mapa:=~/rosmaster_ws/bags/otro.yaml
    ros2 launch dev/bag.launch.py velocidad:=0.5      # el bag a mitad de velocidad

Qué levanta, en este orden:
  1. el bag EN PAUSA, publicando /clock: así /clock existe antes que los nodos y no hay saltos
     de tiempo (si RViz ve el reloj saltar hacia atrás se reinicia y pierde el mapa);
  2. juez_mock.py, que publica el mapa en /map (reliable + transient local, como el juez) y
     anota los reportes que llegan a /victimas;
  3. competencia.launch.py con use_sim_time y sin evasor (el bag ya trae el movimiento);
  4. RViz con dev/equipo.rviz;
  5. a los `espera` segundos, reanuda el bag. Se pausa y reanuda con:
       ros2 service call /rosbag2_player/toggle_paused rosbag2_interfaces/srv/TogglePaused

Argumentos:
  bag, mapa   carpeta del bag y yaml del mapa (fuera del repo: el CI rechaza mapas adentro)
  velocidad   factor de reproducción del bag
  victimas    N que recibe el software (cuántos reportes puede gastar)
  espera      segundos antes de reanudar el bag
  rviz        true | false
  rviz_gpu    true: RViz dibuja con la placa de video (con la Intel se colgaba)

Para volver a correr el bag desde el principio, cortá todo y relanzá: /tf_static se publica
una sola vez al principio del bag.
"""
import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription,
                            TimerAction)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node

RAIZ = Path(__file__).resolve().parent.parent
BAGS = Path.home() / 'rosmaster_ws' / 'bags'


def generate_launch_description():
    bag = LaunchConfiguration('bag')
    mapa = LaunchConfiguration('mapa')
    return LaunchDescription([
        DeclareLaunchArgument('bag', default_value=str(BAGS / 'rosbag-minilaberinto')),
        DeclareLaunchArgument('mapa',
                              default_value=str(BAGS / 'mini_laberinto' / 'minimap_map.yaml')),
        DeclareLaunchArgument('velocidad', default_value='1.0'),
        DeclareLaunchArgument('victimas', default_value='5'),
        DeclareLaunchArgument('espera', default_value='10.0'),
        DeclareLaunchArgument('rviz', default_value='true'),
        # RViz dibuja con OpenGL por software. Con el driver de la placa Intel (Mesa) RViz se colgaba
        # a los pocos segundos de recibir datos; por software anda y usa ~10 % de CPU.
        # rviz_gpu:=true vuelve a usar la placa.
        DeclareLaunchArgument('rviz_gpu', default_value='false'),

        ExecuteProcess(
            name='bag',
            cmd=['ros2', 'bag', 'play', bag, '--clock', '50', '--start-paused',
                 '--rate', LaunchConfiguration('velocidad')],
            output='screen',
        ),
        ExecuteProcess(
            name='mapa',
            cmd=['python3', str(RAIZ / 'tools' / 'juez_mock.py'), '--mapa', mapa,
                 '--retardo', '3', '--duracion', '1e9'],
            output='screen',
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                get_package_share_directory('equipo_jar'), 'launch', 'competencia.launch.py')),
            launch_arguments={'use_sim_time': 'true', 'evasor': 'false',
                              'victimas': LaunchConfiguration('victimas')}.items(),
        ),
        Node(
            package='rviz2', executable='rviz2', name='rviz2', output='log',
            arguments=['-d', str(RAIZ / 'dev' / 'equipo.rviz')],
            parameters=[{'use_sim_time': True}],
            condition=IfCondition(LaunchConfiguration('rviz')),
            additional_env={'LIBGL_ALWAYS_SOFTWARE': PythonExpression(
                ["'0' if '", LaunchConfiguration('rviz_gpu'), "' == 'true' else '1'"])},
        ),
        TimerAction(
            period=LaunchConfiguration('espera'),
            actions=[ExecuteProcess(
                name='reanudar_bag',
                cmd=['ros2', 'service', 'call', '/rosbag2_player/resume',
                     'rosbag2_interfaces/srv/Resume'],
                output='log',
            )],
        ),
    ])
