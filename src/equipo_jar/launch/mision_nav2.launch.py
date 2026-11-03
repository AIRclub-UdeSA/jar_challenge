"""Opt-in workshop comparison. No simulator, map server or second AMCL."""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share=get_package_share_directory('equipo_jar')
    params=os.path.join(share,'params')
    sim=ParameterValue(LaunchConfiguration('use_sim_time'),value_type=bool)
    count=ParameterValue(LaunchConfiguration('victimas'),value_type=int)
    common={'use_sim_time':sim}
    local=os.path.join(params,'localizacion.yaml')
    nav=os.path.join(params,'nav2_workshop.yaml')
    lattice=os.path.join(get_package_share_directory('nav2_smac_planner'),
        'sample_primitives','5cm_resolution','0.5m_turning_radius','omni','output.json')
    servers=[('nav2_controller','controller_server'),('nav2_planner','planner_server'),
             ('nav2_behaviors','behavior_server'),('nav2_bt_navigator','bt_navigator')]
    nodes=[Node(package=p,executable=n,name=n,output='screen',
                parameters=[nav,common]+([{'GridBased.lattice_filepath':lattice}]
                    if n=='planner_server' else []),
                remappings=[('/cmd_vel','/equipo/cmd_nav2')])
           for p,n in servers]
    nodes += [
        Node(package='nav2_lifecycle_manager',executable='lifecycle_manager',
             name='lifecycle_manager_navigation',output='screen',
             parameters=[common,{'autostart':True,'node_names':[n for _,n in servers]}]),
        Node(package='equipo_jar',executable='nodo_equipo',name='nodo_equipo',output='screen',
             parameters=[os.path.join(params,'params.yaml'),local,common,
                         {'modo_navegacion':'inactivo','cantidad_victimas':count}],
             remappings=[('/cmd_vel','/equipo/cmd_localizacion')]),
        Node(package='nav2_amcl',executable='amcl',name='amcl',output='screen',
             parameters=[local,common],remappings=[
                 ('initialpose','/equipo/initialpose'),('amcl_pose','/equipo/amcl_pose'),
                 ('particle_cloud','/equipo/particle_cloud'),
                 ('request_nomotion_update','/equipo/request_nomotion_update')]),
        Node(package='equipo_jar',executable='inicializar_localizacion',
             name='inicializar_localizacion',output='screen',parameters=[local,common]),
        Node(package='equipo_jar',executable='percepcion',name='percepcion',output='screen',
             parameters=[os.path.join(params,'percepcion.yaml'),common,{'cantidad_victimas':count}]),
        Node(package='equipo_jar',executable='mision_nav2',name='mision_nav2',output='screen',
             parameters=[common,{'cantidad_victimas':count,
                 'automatico':ParameterValue(LaunchConfiguration('automatico'),value_type=bool),
                 'duracion_s':ParameterValue(LaunchConfiguration('duracion_s'),value_type=float)}]),
    ]
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time',default_value='false'),
        DeclareLaunchArgument('victimas',default_value='5'),
        DeclareLaunchArgument('automatico',default_value='true'),
        DeclareLaunchArgument('duracion_s',default_value='600.0'),*nodes])
