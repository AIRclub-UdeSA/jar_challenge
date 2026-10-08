"""Interfaz oficial del reglamento §7.1, en un solo lugar.

Si cambia §7.1, se cambia acá y los chequeos siguen a la par.
"""

# Topics que el software del equipo puede usar (§7.1).
TOPICS_PERMITIDOS = {
    '/map',
    '/victimas',
    '/done',
    '/cmd_vel',
    '/scan',
    '/odom',
    '/imu/data',
    '/cam_1/color/image_raw',
    '/cam_1/color/camera_info',
    '/cam_1/depth/color/points',
    '/joint_states',
    '/tf',
    '/tf_static',
}

# Infraestructura de ROS que aparece sola, sin que el equipo la pida.
TOPICS_INFRAESTRUCTURA = {
    '/clock',  # el simulador lo publica; use_sim_time lo suscribe solo
    '/rosout',
    '/parameter_events',
}

# Sentido de cada topic oficial, para detectar equipos que publican donde no deben.
# Solo el equipo publica en estos:
PUBLICA_EL_EQUIPO = {'/victimas', '/done', '/cmd_vel'}
# Solo el juez publica en este:
PUBLICA_EL_JUEZ = {'/map'}

# Patrones (regex) de interfaces que existen en el simulador o en el robot y que
# el equipo NO puede usar. No es una lista completa: el chequeo real en
# ejecución (auditor_interfaces.py) compara contra el grafo. Esto agarra lo obvio
# leyendo el código.
PATRONES_PROHIBIDOS = [
    (r'ground_truth', 'ground truth del simulador (§7.1, §10)'),
    (r'/internal/', 'topics internos del simulador'),
    (r'cmd_vel_gz', 'canal directo al simulador, se pasa por /cmd_vel'),
    (r'''["']/(model|world|gz|ign)/''', 'topics de Gazebo'),
    (r'\b(ign|gz)\s+(topic|service|sdf|model)\b', 'acceso a Gazebo por línea de comandos'),
]

# Archivos que delatan que se metió información del mapa de competencia (§10).
EXTENSIONES_PROHIBIDAS = {'.pgm', '.world', '.sdf'}

# QoS (docs/QOS.md, reglamento §7.1). Un suscriptor reliable no se conecta con un publicador best
# effort, y un publicador best effort no es escuchado por un suscriptor reliable.
# Topics que publica el equipo: tienen que ser reliable.
PUBLICA_EL_EQUIPO_RELIABLE = {'/victimas', '/done', '/cmd_vel'}
# Topics del robot que son best effort: el equipo se suscribe con best effort.
SENSORES_BEST_EFFORT = {
    '/scan', '/cam_1/color/image_raw', '/cam_1/color/camera_info', '/cam_1/depth/color/points',
}
