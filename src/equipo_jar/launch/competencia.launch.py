"""Launch único del equipo (reglamento §9.1).

Este es el archivo que la organización lanza en la corrida:

    ros2 launch equipo_jar competencia.launch.py victimas:=N

Levanta todo el software y espera a que el juez publique /map: sin mapa, el
explorador no mueve el robot y no hay reportes.

- Localización: filtro de partículas propio (jar_localizacion) sobre /map.
- Percepción: rachas del scan que la cámara ve rojas -> objetos rojos en el mapa
  -> un reporte por caja en /victimas (jar_percepcion, jar_mision).
- Exploración: cobertura con A*, miradas de 360° para la cámara, regreso a la
  largada y /done (jar_exploracion).

`use_sim_time` es false por defecto porque el robot real no publica /clock
(§7.3). Con true (simulador) se carga además params/simulacion.yaml y los nodos
escuchan /clock reenviado a 100 Hz por jar_reloj.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node, SetRemap
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    carpeta = os.path.join(get_package_share_directory('equipo_jar'), 'params')
    params = os.path.join(carpeta, 'params.yaml')
    use_sim_time = LaunchConfiguration('use_sim_time')
    victimas = LaunchConfiguration('victimas')
    en_simulador = IfCondition(use_sim_time)
    # En el robot real se repite params.yaml: no cambia nada.
    ajustes = PythonExpression([
        "'", os.path.join(carpeta, 'simulacion.yaml'), "' if '", use_sim_time,
        "' == 'true' else '", params, "'"])
    reloj = {'use_sim_time': ParameterValue(use_sim_time, value_type=bool)}
    comunes = [params, ajustes, reloj]

    def nodo(paquete, ejecutable, nombre=None, **kwargs):
        return Node(package=paquete, executable=ejecutable, name=nombre or ejecutable,
                    output='screen', parameters=comunes + kwargs.pop('extra', []),
                    **kwargs)

    nodos = [
        nodo('jar_localizacion', 'campo_verosimilitud'),
        nodo('jar_localizacion', 'localizador'),
        nodo('jar_percepcion', 'detector_scan', 'detector_scan_color'),
        nodo('jar_percepcion', 'detector_obstaculos', 'detector_rojos',
             remappings=[('scan', 'scan_rojo'),
                         ('scan_no_mapeado', 'scan_victimas'),
                         ('objetos_no_mapeados', 'objetos_rojos'),
                         ('objetos_no_mapeados_markers', 'objetos_rojos_markers')]),
        nodo('jar_mision', 'juntar_victimas',
             extra=[{'victimas': ParameterValue(victimas, value_type=int)}]),
        nodo('jar_exploracion', 'grilla_cobertura'),
        nodo('jar_exploracion', 'explorador'),
    ]
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('victimas', default_value='5',
                              description='cantidad de víctimas a buscar (N)'),
        # Gazebo publica /clock a 1000 Hz; reenviado a 100 Hz, los nodos de Python
        # no gastan CPU en recibirlo. En el robot real no hay /clock.
        Node(package='jar_reloj', executable='reloj_lento', name='reloj_lento',
             output='screen', condition=en_simulador,
             parameters=[{'frecuencia_hz': 100.0}]),
        GroupAction([SetRemap(src='/clock', dst='/clock_lento', condition=en_simulador)]
                    + nodos),
    ])
