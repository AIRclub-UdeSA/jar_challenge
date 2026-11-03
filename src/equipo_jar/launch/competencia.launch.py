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
from launch.conditions import IfCondition
from launch.substitutions import (LaunchConfiguration, PathJoinSubstitution, PythonExpression,
                                  TextSubstitution)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    params = os.path.join(
        get_package_share_directory('equipo_jar'), 'params', 'params.yaml')
    localizacion = os.path.join(
        get_package_share_directory('equipo_jar'), 'params', 'localizacion.yaml')
    localizacion_params = LaunchConfiguration('localizacion_params')
    compartido = os.path.join(get_package_share_directory('equipo_jar'), 'params')
    navegacion_params = LaunchConfiguration('navegacion_params')
    perfil = PathJoinSubstitution([compartido, [TextSubstitution(text='navegacion_'),
                                                LaunchConfiguration('perfil_navegacion'),
                                                TextSubstitution(text='.yaml')]])
    use_sim_time = LaunchConfiguration('use_sim_time')
    victimas = LaunchConfiguration('victimas')
    rumbo_imu = LaunchConfiguration('rumbo_imu')
    # Con rumbo_imu, AMCL y los nodos del equipo leen el árbol de odometria_imu; /tf queda intacto.
    arbol_tf = [(f'/{t}', PythonExpression([f"'/equipo/{t}' if '", rumbo_imu, f"' == 'true' else '/{t}'"]))
                for t in ('tf', 'tf_static')]

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('asumir_base_cuadrada',default_value='true',
                              description='Prior opcional: inferir profundidad desde ancho de cara'),
        # La compuerta sólo corrige si la pendiente válida difiere de 1 al menos 0,10.
        DeclareLaunchArgument('rumbo_imu', default_value='true',
                              description='Rumbo de la orientación del IMU sobre la traslación de ruedas'),
        DeclareLaunchArgument('percepcion', default_value='true',
                              description='Activa detección RGB-D y reportes M3'),
        DeclareLaunchArgument('localizacion_params', default_value=localizacion,
                              description='Parámetros generales de localización'),
        DeclareLaunchArgument('victimas', default_value='5',
                              description='cantidad de víctimas a buscar (N)'),
        DeclareLaunchArgument('navegacion_params',
                              default_value=os.path.join(compartido, 'navegacion.yaml'),
                              description='Parámetros generales de navegación'),
        DeclareLaunchArgument('mision_params',
                              default_value=os.path.join(compartido, 'mision.yaml'),
                              description='Política general de búsqueda y regreso M4'),
        DeclareLaunchArgument('perfil_navegacion', default_value='real',
                              description='real (por defecto, más lento) o sim'),
        DeclareLaunchArgument('modo_navegacion', default_value='mision',
                              description='inactivo, ida_vuelta_prueba, meta_externa o mision'),
        DeclareLaunchArgument('prueba_en_seco', default_value='false',
                              description='true: calcula pero publica cero en /cmd_vel'),
        Node(
            package='equipo_jar',
            executable='nodo_equipo',
            name='nodo_equipo',
            output='screen',
            parameters=[params, localizacion_params, navegacion_params, perfil,
                        LaunchConfiguration('mision_params'), {
                'use_sim_time': use_sim_time,
                'cantidad_victimas': ParameterValue(victimas, value_type=int),
                'modo_navegacion': LaunchConfiguration('modo_navegacion'),
                'prueba_en_seco': ParameterValue(LaunchConfiguration('prueba_en_seco'),
                                                 value_type=bool),
            }],
            remappings=arbol_tf,
        ),
        Node(
            package='equipo_jar', executable='percepcion', name='percepcion',
            condition=IfCondition(LaunchConfiguration('percepcion')),
            output='screen',
            parameters=[os.path.join(compartido, 'percepcion.yaml'), {
                'use_sim_time': ParameterValue(use_sim_time, value_type=bool),
                'cantidad_victimas': ParameterValue(victimas, value_type=int),
                'asumir_base_cuadrada': ParameterValue(LaunchConfiguration('asumir_base_cuadrada'),value_type=bool),
            }],
            remappings=arbol_tf,
        ),
        Node(
            package='nav2_amcl',
            executable='amcl',
            name='amcl',
            output='screen',
            parameters=[localizacion_params, {'use_sim_time': use_sim_time}],
            remappings=[('initialpose', '/equipo/initialpose'),
                        ('amcl_pose', '/equipo/amcl_pose'),
                        ('particle_cloud', '/equipo/particle_cloud'),
                        ('request_nomotion_update', '/equipo/request_nomotion_update')]+arbol_tf,
        ),
        Node(
            package='equipo_jar', executable='odometria_imu', name='odometria_imu',
            condition=IfCondition(rumbo_imu), output='screen',
            parameters=[{'use_sim_time': ParameterValue(use_sim_time, value_type=bool)}],
        ),
        Node(
            package='equipo_jar',
            executable='inicializar_localizacion',
            name='inicializar_localizacion',
            output='screen',
            parameters=[localizacion_params, {'use_sim_time': use_sim_time}],
        ),
    ])
