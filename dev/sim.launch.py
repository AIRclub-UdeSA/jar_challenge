"""Sesión de práctica en el simulador: Gazebo + juez de práctica + RViz + software del equipo.

Herramienta de desarrollo, NO es parte de la entrega (vive fuera de src/).

    ros2 launch dev/sim.launch.py                         # maze_1_6x5, largada con /juez/iniciar
    ros2 launch dev/sim.launch.py mundo:=maze_3_6x6 spawn_x:=0.15 spawn_y:=-0.1 spawn_yaw:=0.2
    ros2 launch dev/sim.launch.py equipo:=false           # levantás competencia.launch.py aparte
    ros2 launch dev/sim.launch.py libre:=false manual:=false   # corrida con los tiempos del reglamento

Argumentos:
  mundo          mundo de práctica (worlds/<mundo>_victimas.world y maps/<mundo>.yaml)
  spawn_x/y/yaw  pose de largada del robot en el simulador (m, m, rad)
  motion_profile ideal | stress   (stress se parece más al robot real)
  motion_bias    true | false     (drift aleatorio de una base mecanum sin calibrar)
  gui            ventana de Gazebo (false por defecto: cuesta ~50 % de CPU y RViz alcanza)
  rviz           RViz con dev/equipo.rviz
  equipo         levantar también competencia.launch.py (victimas:=N sale del mundo)
  manual         true: el mapa se publica con  ros2 service call /juez/iniciar std_srvs/srv/Trigger
                 false: 5 s después del primer /scan
  libre          true: el juez no corta por tiempo ni por robot quieto (para teleoperar)
  evasor         false: no levanta el evasor, para teleoperar
  rviz_gpu       true: RViz dibuja con la placa de video (con la Intel se colgaba)
  cloud_decimation  1 de cada N píxeles en la nube de profundidad del simulador (4 por defecto).
                 El equipo no la usa, pero el simulador la arma igual y era lo más caro después de
                 Gazebo. Con 1 queda como en el robot.

El teleop va en otra terminal, porque necesita el teclado (con evasor:=false):
    ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r cmd_vel:=cmd_vel_evasor

El resultado de cada corrida queda en build/practica/<mundo>.json.
"""
import os
import sys
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription,
                            OpaqueFunction)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / 'tools'))


def es_true(context, nombre):
    return LaunchConfiguration(nombre).perform(context).lower() in ('true', '1')


def armar(context):
    from juez import resolver_mundo, victimas_de_mundo

    mundo = LaunchConfiguration('mundo').perform(context)
    pkg_sim = get_package_share_directory('yahboom_rosmaster_gazebo')
    ruta_mundo = os.path.join(pkg_sim, 'worlds', f'{mundo}_victimas.world')
    ruta_mapa = os.path.join(pkg_sim, 'maps', f'{mundo}.yaml')
    n_victimas = len(victimas_de_mundo(resolver_mundo(mundo)))

    acciones = [IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_sim, 'launch', 'rosmaster_gazebo_fortress.launch.py')),
        launch_arguments={
            'world': ruta_mundo,
            'rviz': 'false',
            'gui': LaunchConfiguration('gui'),
            'motion_profile': LaunchConfiguration('motion_profile'),
            'motion_bias': LaunchConfiguration('motion_bias'),
            'spawn_x': LaunchConfiguration('spawn_x'),
            'spawn_y': LaunchConfiguration('spawn_y'),
            'spawn_yaw': LaunchConfiguration('spawn_yaw'),
            'cloud_decimation': LaunchConfiguration('cloud_decimation'),
        }.items(),
    )]

    salida = RAIZ / 'build' / 'practica'
    salida.mkdir(parents=True, exist_ok=True)
    cmd_juez = [sys.executable, str(RAIZ / 'tools' / 'juez.py'),
                '--mundo', mundo, '--mapa', ruta_mapa, '--ground-truth',
                '--salida', str(salida / f'{mundo}.json')]
    if es_true(context, 'manual'):
        cmd_juez.append('--manual')
    else:
        cmd_juez += ['--esperar-topic', '/scan']
    if es_true(context, 'libre'):
        cmd_juez += ['--duracion', '1e9', '--inmovil', '1e9']
    acciones.append(ExecuteProcess(cmd=cmd_juez, name='juez', output='screen'))

    if es_true(context, 'rviz'):
        acciones.append(Node(
            package='rviz2', executable='rviz2', name='rviz2', output='log',
            arguments=['-d', str(RAIZ / 'dev' / 'equipo.rviz')],
            parameters=[{'use_sim_time': True}],
            # RViz dibuja con OpenGL por software. Con el driver de la placa Intel (Mesa) RViz se colgaba
            # a los pocos segundos de recibir datos; por software anda y usa ~10 % de CPU.
            # rviz_gpu:=true vuelve a usar la placa.
            additional_env={'LIBGL_ALWAYS_SOFTWARE': '0' if es_true(context, 'rviz_gpu') else '1'},
        ))

    if es_true(context, 'equipo'):
        acciones.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                get_package_share_directory('equipo_jar'), 'launch', 'competencia.launch.py')),
            launch_arguments={'use_sim_time': 'true', 'victimas': str(n_victimas),
                              'evasor': LaunchConfiguration('evasor')}.items(),
        ))
    return acciones


def generate_launch_description():
    argumentos = [
        ('mundo', 'maze_1_6x5'),
        ('spawn_x', '0.0'), ('spawn_y', '0.0'), ('spawn_yaw', '0.0'),
        ('motion_profile', 'ideal'), ('motion_bias', 'false'),
        ('gui', 'false'), ('rviz', 'true'), ('equipo', 'true'),
        ('manual', 'true'), ('libre', 'true'), ('evasor', 'true'),
        ('cloud_decimation', '4'), ('rviz_gpu', 'false'),
    ]
    return LaunchDescription(
        [DeclareLaunchArgument(nombre, default_value=valor) for nombre, valor in argumentos]
        + [OpaqueFunction(function=armar)])
