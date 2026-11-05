"""Launch único del equipo (reglamento §9.1).

Este es el archivo que la organización lanza en la corrida:

    ros2 launch equipo_jar competencia.launch.py victimas:=N

Levanta cuatro nodos y se queda esperando a que el juez publique el mapa en /map; no mueve
al robot ni reporta antes. Sin map_server: todos leen el /map del juez.
- `localizer` (equipo_jar/localizer.py, algoritmo en scan_matcher.py): ubica el scan en el
  mapa y publica map -> odom (docs/LOCALIZATION.md).
- `world_model` (equipo_jar/world_model_node.py): objetos que el mapa no tiene, cuáles son
  rojos y dónde puede quedar una víctima sin ver (docs/WORLD_MODEL.md).
- `mission` (equipo_jar/mission.py, decisiones en planner.py): planifica con el modelo del
  mundo, reporta y vuelve a casa (docs/MISSION.md).
- `motion` (equipo_jar/motion_node.py): sigue los caminos de la misión (planeados para la
  forma del chasis); es el único que publica /cmd_vel.

El perfil (`perfil:=rapido|normal|cuidadoso`, o `competencia.perfil` de params.yaml) fija
cuánto arriesga el robot: params/perfiles.yaml. El launch deriva de él los parámetros de los
nodos motion, mission y world_model que dependen unos de otros y lanza todo con ese
params.yaml combinado.

`use_sim_time` es false por defecto porque el robot real no publica /clock
(§7.3). En el simulador se pasa en true (lo hacen tools/correr_simulador.sh y
el CI).

Todo lo que imprimen los procesos del launch va también a un archivo, el log.txt que la
organización guarda después de cada corrida (equipo_jar/run_log.py, docs/LOGGING.md).
"""
import os
import sys
import tempfile

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
import yaml


RADIO_GIRO = 0.207  # m, base_link to the chassis' farthest corner (yahboom_rosmaster_description)


def con_perfil(params, perfil):
    """params.yaml with the profile's values and what follows from them, as a new file."""
    with open(params) as f:
        p = yaml.safe_load(f)
    perfil = perfil or p['competencia']['ros__parameters'].get('perfil', 'normal')
    with open(os.path.join(os.path.dirname(params), 'perfiles.yaml')) as f:
        perfiles = yaml.safe_load(f)
    if perfil not in perfiles:
        raise ValueError(f'perfil must be one of {sorted(perfiles)}, not {perfil}')
    q = perfiles[perfil]['ros__parameters']
    v = q['v_max']
    margen = q['margen'] + q['holgura_control']
    p['mission']['ros__parameters'].update(
        velocidad=0.65 * v,  # average along a path: 0.63 of vx_max measured on 72 runs
        # Paths: where the footprint grown by this fits (planner.py, over x, y and heading).
        # Goals: room to turn there, the chassis' corner farthest from base_link + margin.
        huella_margen=round(margen, 3), holgura_meta=round(RADIO_GIRO + margen, 3),
        atasco_s=q['atasco_s'], atasco_dist=q['atasco_dist'],
        reporte_p_min=q['reporte_p_min'])
    # What the world model calls confirmed (red seeds; N confirmed: every other object drops
    # to 0): stricter than what the mission reports.
    p['world_model']['ros__parameters']['reporte_p_min'] = q.get('modelo_p_min',
                                                                 q['reporte_p_min'])
    p.setdefault('motion', {'ros__parameters': {}})['ros__parameters'].update(
        v_max=v, margen_seguridad=q['margen'])
    p['competencia']['ros__parameters']['perfil'] = perfil
    casa = p['competencia']['ros__parameters'].get('casa', 'origen')
    if casa not in ('origen', 'inicio'):
        raise ValueError(f'competencia.casa must be origen or inicio, not {casa}')
    p['mission']['ros__parameters']['casa'] = casa
    out = tempfile.NamedTemporaryFile('w', prefix=f'equipo_jar_{perfil}_', suffix='.yaml',
                                      delete=False)
    yaml.safe_dump(p, out, sort_keys=False)
    out.close()
    return out.name


def abrir_log():
    """The run's log and the actions that feed it. Whatever fails here, the launch goes on."""
    try:
        from equipo_jar import run_log
        log = run_log.start()
        return log, run_log.event_handlers(log)
    except Exception as e:
        print(f'[competencia] running without a log file: {e!r}', file=sys.stderr)
        return None, []


def equipo(context, log=None):
    params = con_perfil(LaunchConfiguration('params_file').perform(context),
                        LaunchConfiguration('perfil').perform(context))
    if log is not None:
        log.header({k: LaunchConfiguration(k).perform(context)
                    for k in ('victimas', 'perfil', 'use_sim_time')}, params)
    comunes = [params, {'use_sim_time': LaunchConfiguration('use_sim_time')}]
    # victimas: the world model and the mission use N (docs/WORLD_MODEL.md §5.3).
    n = {'victimas': ParameterValue(LaunchConfiguration('victimas'), value_type=int)}
    # No log files of each node: the launch's log has their lines, and rcl's file would be
    # somewhere else on the disk.
    sin_archivos = ['--disable-external-lib-logs']
    return [
        Node(package='equipo_jar', executable='localizer', name='localizer', output='screen',
             parameters=comunes, ros_arguments=sin_archivos),
        Node(package='equipo_jar', executable='world_model', name='world_model',
             output='screen', parameters=comunes + [n], ros_arguments=sin_archivos),
        Node(package='equipo_jar', executable='motion', name='motion', output='screen',
             parameters=comunes, ros_arguments=sin_archivos),
        Node(package='equipo_jar', executable='mission', output='screen',
             parameters=comunes + [n], ros_arguments=sin_archivos),
    ]


def generate_launch_description():
    log, registro = abrir_log()
    return LaunchDescription(registro + [
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('victimas', default_value='5',
                              description='cantidad de víctimas a buscar (N)'),
        DeclareLaunchArgument('perfil', default_value='',
                              description='rapido, normal o cuidadoso (params/perfiles.yaml); '
                              'vacío: competencia.perfil de params.yaml'),
        # Para probar variantes en el simulador; en la corrida es el del paquete.
        DeclareLaunchArgument('params_file', default_value=os.path.join(
            get_package_share_directory('equipo_jar'), 'params', 'params.yaml')),
        OpaqueFunction(function=equipo, kwargs={'log': log}),
    ])
