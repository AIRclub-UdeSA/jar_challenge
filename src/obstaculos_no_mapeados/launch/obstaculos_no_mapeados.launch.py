"""
Semana 07 — launch de detección de obstáculos no mapeados.

Levanta todo lo de semana 06 (simulador, map_server, lifecycle_manager,
campo_verosimilitud + localizador) más detector_obstaculos y RViz, con un
solo comando:

    ros2 launch obstaculos_no_mapeados obstaculos_no_mapeados.launch.py

Por defecto usa el mundo `laberinto_simple_victimas.world` (el laberinto con
los cubos que NO están en el mapa) contra `laberinto_simple.yaml` (el mapa
de siempre, sin cubos). Para otro par mundo+mapa:

    ros2 launch obstaculos_no_mapeados obstaculos_no_mapeados.launch.py \\
        world:=<ruta al _victimas.world> map:=<ruta al .yaml>

teleop_twist_keyboard queda afuera a propósito: necesita una terminal propia
para leer el teclado, así que se sigue corriendo a mano:

    ros2 run teleop_twist_keyboard teleop_twist_keyboard
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

    pkg_sim = get_package_share_directory('yahboom_rosmaster_bringup')
    pkg_gazebo = get_package_share_directory('yahboom_rosmaster_gazebo')
    pkg_obstaculos = get_package_share_directory('obstaculos_no_mapeados')

    launch_simulador = os.path.join(
        pkg_sim, 'launch', 'rosmaster_x3_sim.launch.py')
    # El mundo tiene víctimas que el mapa no tiene: eso es justo lo que el
    # detector tiene que encontrar. Fuera de eso, mundo y mapa tienen que
    # ser el mismo laberinto.
    mundo_por_defecto = os.path.join(
        pkg_gazebo, 'worlds', 'laberinto_simple_victimas.world')
    mapa_por_defecto = os.path.join(pkg_gazebo, 'maps', 'laberinto_simple.yaml')
    config_rviz = os.path.join(pkg_obstaculos, 'rviz', 'obstaculos_no_mapeados.rviz')

    declarar_world = DeclareLaunchArgument(
        'world', default_value=mundo_por_defecto,
        description='Ruta al .world de Gazebo que se va a cargar (con víctimas).')
    declarar_mapa = DeclareLaunchArgument(
        'map', default_value=mapa_por_defecto,
        description='Ruta al .yaml del mapa que sirve map_server (sin víctimas).')
    declarar_sigma = DeclareLaunchArgument(
        'sigma_sensor', default_value='0.2',
        description='Ancho (m) del halo de probabilidad alrededor de cada pared.')
    declarar_particulas = DeclareLaunchArgument(
        'num_particulas', default_value='300',
        description='Cantidad de partículas del filtro.')
    declarar_gamma = DeclareLaunchArgument(
        'gamma', default_value='0.0025',
        description='Varianza de ruido del modelo de movimiento (gamma1-gamma6).')
    declarar_umbral = DeclareLaunchArgument(
        'umbral_probabilidad', default_value='25.0',
        description='Probabilidad (0-100) por debajo de la cual un punto es "no mapeado".')
    declarar_racha = DeclareLaunchArgument(
        'racha_minima', default_value='3',
        description='Puntos contiguos mínimos para que una racha cuente como objeto.')
    declarar_distancia = DeclareLaunchArgument(
        'distancia_minima_valida', default_value='0.25',
        description='Rango mínimo (m) válido, para filtrar el rebote contra el propio robot.')

    simulador = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(launch_simulador),
        launch_arguments={
            'world': LaunchConfiguration('world'),
            'motion_profile': 'ideal',
            'rviz': 'false',
        }.items(),
    )

    # map_server es un nodo lifecycle: arranca sin publicar nada y espera que
    # lifecycle_manager (con autostart=True) lo pase a configure → activate.
    map_server = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'yaml_filename': LaunchConfiguration('map'),
        }],
    )

    lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_localizacion',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'autostart': True,
            'node_names': ['map_server'],
        }],
    )

    # /map y /likelihood_map son transient local: no hace falta esperar a que
    # map_server esté activo antes de arrancar los nodos que los consumen.
    campo_verosimilitud = Node(
        package='localizacion',
        executable='campo_verosimilitud',
        name='campo_verosimilitud',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'sigma_sensor': ParameterValue(
                LaunchConfiguration('sigma_sensor'), value_type=float),
        }],
    )

    gamma = ParameterValue(LaunchConfiguration('gamma'), value_type=float)
    localizador = Node(
        package='localizacion',
        executable='localizador',
        name='localizador',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'num_particulas': ParameterValue(
                LaunchConfiguration('num_particulas'), value_type=int),
            'gamma1': gamma,
            'gamma2': gamma,
            'gamma3': gamma,
            'gamma4': gamma,
            'gamma5': gamma,
            'gamma6': gamma,
        }],
    )

    # Hasta que localizador publique la primera map → odom, este nodo avisa
    # "Todavía no hay tf" y descarta los scans; después arranca solo.
    detector_obstaculos = Node(
        package='obstaculos_no_mapeados',
        executable='obstaculos_no_mapeados',
        name='detector_obstaculos',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'umbral_probabilidad': ParameterValue(
                LaunchConfiguration('umbral_probabilidad'), value_type=float),
            'racha_minima': ParameterValue(
                LaunchConfiguration('racha_minima'), value_type=int),
            'distancia_minima_valida': ParameterValue(
                LaunchConfiguration('distancia_minima_valida'), value_type=float),
        }],
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', config_rviz],
        parameters=[{'use_sim_time': True}],
    )

    return LaunchDescription([
        declarar_world,
        declarar_mapa,
        declarar_sigma,
        declarar_particulas,
        declarar_gamma,
        declarar_umbral,
        declarar_racha,
        declarar_distancia,
        simulador,
        map_server,
        lifecycle_manager,
        campo_verosimilitud,
        localizador,
        detector_obstaculos,
        rviz,
    ])
