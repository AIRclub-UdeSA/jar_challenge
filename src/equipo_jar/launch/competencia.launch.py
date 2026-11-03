"""Entrada oficial: AMCL, Nav2, exploración y detección; espera /map."""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from nav2_common.launch import RewrittenYaml


def generate_launch_description():
    paquete = get_package_share_directory('equipo_jar')
    parametros = os.path.join(paquete, 'params')
    use_sim_time = LaunchConfiguration('use_sim_time')
    victimas = LaunchConfiguration('victimas')
    reloj = ParameterValue(use_sim_time, value_type=bool)
    cantidad = ParameterValue(victimas, value_type=int)
    remaps_tf = [('/tf', 'tf'), ('/tf_static', 'tf_static')]
    argumentos_ros = ['--disable-external-lib-logs']

    nav2_params = RewrittenYaml(
        source_file=os.path.join(parametros, 'nav2_exploracion.yaml'),
        root_key='',
        param_rewrites={
            'use_sim_time': use_sim_time,
            'default_nav_to_pose_bt_xml': os.path.join(
                paquete, 'behavior_trees', 'navegar_rapido.xml'),
        },
        convert_types=True,
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false',
                              description='true solo en simulación'),
        DeclareLaunchArgument('victimas', default_value='5',
                              description='Cantidad N informada por la organización'),
        Node(
            package='nav2_amcl', executable='amcl', name='amcl',
            output='both', ros_arguments=argumentos_ros,
            parameters=[nav2_params],
        ),
        Node(
            package='nav2_lifecycle_manager', executable='lifecycle_manager',
            name='lifecycle_manager_localization',
            output='both', ros_arguments=argumentos_ros,
            parameters=[{
                'use_sim_time': reloj,
                'autostart': True,
                'node_names': ['amcl'],
            }],
        ),
        Node(
            package='nav2_controller', executable='controller_server', name='controller_server',
            output='both', ros_arguments=argumentos_ros,
            parameters=[nav2_params],
            remappings=remaps_tf + [('cmd_vel', 'cmd_vel_nav')],
        ),
        Node(
            package='nav2_smoother', executable='smoother_server', name='smoother_server',
            output='both', ros_arguments=argumentos_ros,
            parameters=[nav2_params],
            remappings=remaps_tf,
        ),
        Node(
            package='nav2_planner', executable='planner_server', name='planner_server',
            output='both', ros_arguments=argumentos_ros,
            parameters=[nav2_params],
            remappings=remaps_tf,
        ),
        Node(
            package='nav2_behaviors', executable='behavior_server', name='behavior_server',
            output='both', ros_arguments=argumentos_ros,
            parameters=[nav2_params],
            remappings=remaps_tf,
        ),
        Node(
            package='nav2_bt_navigator', executable='bt_navigator', name='bt_navigator',
            output='both', ros_arguments=argumentos_ros,
            parameters=[nav2_params],
            remappings=remaps_tf,
        ),
        Node(
            package='nav2_waypoint_follower', executable='waypoint_follower', name='waypoint_follower',
            output='both', ros_arguments=argumentos_ros,
            parameters=[nav2_params],
            remappings=remaps_tf,
        ),
        Node(
            package='nav2_velocity_smoother', executable='velocity_smoother', name='velocity_smoother',
            output='both', ros_arguments=argumentos_ros,
            parameters=[nav2_params],
            remappings=remaps_tf + [
                ('cmd_vel', 'cmd_vel_nav'),
                ('cmd_vel_smoothed', 'cmd_vel'),
            ],
        ),
        Node(
            package='nav2_lifecycle_manager', executable='lifecycle_manager',
            name='lifecycle_manager_navigation',
            output='both', ros_arguments=argumentos_ros,
            parameters=[{
                'use_sim_time': reloj,
                'autostart': True,
                'node_names': [
                    'controller_server', 'smoother_server', 'planner_server',
                    'behavior_server', 'bt_navigator', 'waypoint_follower', 'velocity_smoother',
                ],
            }],
        ),
        Node(
            package='equipo_jar', executable='exploracion', name='exploracion',
            output='both', ros_arguments=argumentos_ros,
            parameters=[os.path.join(parametros, 'exploracion.yaml'), {
                'use_sim_time': reloj,
                'cantidad_victimas': cantidad,
            }],
        ),
        Node(
            package='equipo_jar', executable='deteccion_victimas', name='deteccion_victimas',
            output='both', ros_arguments=argumentos_ros,
            parameters=[os.path.join(parametros, 'victimas.yaml'), {
                'use_sim_time': reloj,
                'cantidad_victimas': cantidad,
            }],
        ),
    ])
