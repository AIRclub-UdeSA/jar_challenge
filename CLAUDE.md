# CLAUDE.md — Equipo CHANGUITRON, Challenge JAR 2026

Contexto compartido del equipo para Claude Code. Si algo acá contradice al
[reglamento](https://airclub-udesa.github.io/jar_site/reglamento/), gana el reglamento.

## Proyecto

- Competencia de búsqueda y rescate autónomo (AIR Club UdeSA, JAR 2026, Rosario). Clasificación
  4/11, final 5/11. Corrida de 10 min: localizarse en `/map`, explorar un laberinto desconocido,
  reportar N cajas rojas en `/victimas`, opcionalmente volver a la largada y publicar `/done`.
- Robot: Yahboom ROSMASTER X3 (mecanum, LiDAR 2D 360°, cámara RGB-D, IMU).
- Stack: **ROS 2 Humble + Gazebo Fortress** (Ubuntu 22.04 o contenedor Docker). Todo tiene que
  compilar y correr en Humble: el CI usa `ros:humble-ros-base-jammy`.
- Este repo parte del template oficial `AIRclub-UdeSA/jar_challenge` (remote `upstream`).

## Estructura

- `src/equipo_jar/`: nuestro paquete (`ament_python`). Se pueden agregar más paquetes en `src/`.
  - `launch/competencia.launch.py`: **el único launch** de la corrida. Recibe `victimas:=N` y
    `use_sim_time`. Todos los nodos van acá, con `parameters=[params, {...use_sim_time...}]`.
  - `params/`: solo valores generales (umbrales, ganancias, velocidades, rangos de color).
  - `equipo_jar/reportes.py`: `armar_reporte(x, y, stamp)`; usarlo siempre para `/victimas`.
  - `equipo_jar/nodo_mision.py`: elige metas (`cobertura.py`), manda a Nav2, sale de las trabas
    (`escape.py`), reporta y vuelve. Único dueño de `/cmd_vel` y `/victimas`.
  - `equipo_jar/nodo_equipo.py`: seguidor de pared del ejemplo. **Ya no se lanza.**
  - Navegación (hecha): Nav2 con Smac y RPP, activado por `nodo_arranque.py`. Su velocidad pasa por
    `/equipo/cmd_vel_nav` y `nodo_mision` la deja llegar a `/cmd_vel`.
  - Localización (hecha): AMCL + `nodo_arranque.py` + `nodo_pose_inicial.py` (gaussiana en el
    origen por `/initialpose`). Detalle en `docs/ARQUITECTURA.md` §0.
- `tools/`, `.github/`: controles y CI de la organización. **No modificarlos.**
- `docs/GUIA_EQUIPO.md` (flujo de trabajo y entrega), `docs/ARQUITECTURA.md` (estado de cada
  módulo), `docs/QOS.md` (QoS de cada topic).

## Reglas que no se negocian

- **Interfaz fija:** solo `/map`, `/victimas`, `/done`, `/cmd_vel`, `/scan`, `/odom`, `/imu/data`,
  `/cam_1/color/image_raw`, `/cam_1/color/camera_info`, `/cam_1/depth/color/points`,
  `/joint_states`, `/tf`, `/tf_static` (lista en `tools/interfaces.py`). Topics internos entre
  nuestros nodos: libres. Nunca ground truth del simulador.
- **QoS:** `/scan` y cámaras → `qos_profile_sensor_data`. `/map` → reliable + transient local,
  depth 1. Lo que publicamos (`/victimas`, `/done`, `/cmd_vel`) → QoS por defecto (reliable, 10).
- **Nada antes de `/map`:** no mover el robot ni reportar hasta recibir el mapa.
- **Exactamente N reportes:** cada mensaje a `/victimas` gasta uno. Un solo nodo publica ahí y lleva
  la cuenta. `frame_id: map`, sin NaN/inf. Se puntúa contra el centro de la caja (0,25 m de lado).
- **Sin hardcodeo:** ni mapas, ni mundos, ni coordenadas, ni waypoints, en código ni en `params/`.
  N llega como argumento del launch, no va en `params/`.
- **`use_sim_time`** del launch en todos los nodos (el robot real no publica `/clock`).
- `/cmd_vel` tiene watchdog de 0,5 s: publicar continuamente mientras se mueve. Tras `/done`,
  frenar y no volver a mover. Robot quieto 60 s = fin de corrida.
- **Sin internet en la corrida:** dependencias de ROS en `package.xml` (`<exec_depend>`), de Python
  en `requirements.txt` con versión fija.

## Comandos

Comandos de ROS dentro del contenedor (`./container.sh enter`) o con Humble sourceado. El repo vive
en `~/rosmaster_ws/src/CHANGUITRON---JAR-2026`.

```bash
cd ~/rosmaster_ws && colcon build --symlink-install --packages-select equipo_jar && source install/setup.bash
ros2 launch equipo_jar competencia.launch.py use_sim_time:=true victimas:=N
python3 tools/juez.py --mundo maze_1_6x5 --mapa $(ros2 pkg prefix --share yahboom_rosmaster_gazebo)/maps/maze_1_6x5.yaml --ground-truth
tools/verificar.sh --template                  # controles del CI, sin simulador
tools/verificar.sh --template --sim maze_1_6x5 # además, corrida headless puntuada
```

Simulador: `ros2 launch yahboom_rosmaster_gazebo rosmaster_gazebo_fortress.launch.py world:=<...>_victimas.world`
(con `spawn_x`, `spawn_y`, `spawn_yaw` para no arrancar exacto en el origen).

## Convenciones

- Código, comentarios, nombres y commits en español, como el template.
- El callback de un sensor solo guarda el último dato; la lógica va en un timer a frecuencia fija.
- Nodos chicos con una responsabilidad cada uno; un solo dueño de `/cmd_vel` y de `/victimas`.
- Tests con pytest en `src/equipo_jar/test/` para la lógica pura (geometría, filtros, agrupado).

## Git

- **Nunca commitear ni pushear directo a `main`.** Una branch por tarea (`feature/...`, `fix/...`)
  y PR hacia `main` con revisión de otra persona y CI en verde.
- Antes de proponer un PR, correr `tools/verificar.sh --template`.
- La entrega es otro repo (fork `JoacoFernandezB/jar_challenge-changuitron`, remote `entrega`):
  solo se le pushea `main` cuando está estable. No pushear ahí sin que lo pida quien entrega.

## Cómo ayudar

- Muchos en el equipo están aprendiendo: explicar el porqué, no solo el qué.
- Para debuggear, primero diagnóstico (`ros2 topic list`, `ros2 node list`, `ros2 topic echo`,
  `ros2 topic info -v` para la QoS, logs de `colcon build`) y después cambios de código.
- En navegación y percepción, contar los trade-offs antes de escribir código.
- No inventar topics, nodos ni archivos: verificar en el repo antes de nombrarlos.
