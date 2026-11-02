# Guía del equipo CHANGUITRON

Cómo trabajamos en este repo. El [README](../README.md) es el del template oficial
(`AIRclub-UdeSA/jar_challenge`) y el [reglamento](https://airclub-udesa.github.io/jar_site/reglamento/)
es la fuente de verdad: si algo de acá los contradice, ganan ellos.

**Fechas:** clasificación el miércoles 4 de noviembre, final el jueves 5 (Rosario).
La organización toma el commit vigente del PR de entrega al cierre de entregas.

**En una línea:** el equipo trabaja en este repo (CHANGUITRON) con branches y PRs; cuando `main`
tiene algo bueno, se copia al fork de entrega; el PR del fork hacia la organización es la entrega.

---

## 1. Setup (una vez por persona)

Necesitás ROS 2 Humble + Gazebo Fortress + el simulador, ya sea en Ubuntu 22.04 nativo o con el
contenedor Docker del simulador ([guía de setup](https://airclub-udesa.github.io/jar_site/setup/simulador/)).

Este repo va **dentro de `src/` del workspace**, al lado de `yahboom_rosmaster`. Con Docker es
obligatorio: el contenedor solo ve la carpeta del workspace (`~/rosmaster_ws` →
`/home/alumno/rosmaster_ws`).

```text
rosmaster_ws/
└── src/
    ├── yahboom_rosmaster/          simulador oficial, en el tag jar2026-v1
    └── CHANGUITRON---JAR-2026/     este repo
        └── src/equipo_jar/         nuestro paquete ROS 2
```

```bash
cd ~/rosmaster_ws/src
git clone git@github.com:JoacoFernandezB/CHANGUITRON---JAR-2026.git
cd CHANGUITRON---JAR-2026
git remote add upstream https://github.com/AIRclub-UdeSA/jar_challenge.git
```

El CI valida contra el simulador `jar2026-v1`: conviene que `yahboom_rosmaster` esté en ese tag o
apenas después (`git -C ~/rosmaster_ws/src/yahboom_rosmaster describe --tags`).

## 2. Los repos y cómo se conectan

### Qué es un "remote"

Un repo de git vive **en tu compu** (copia local) y **en GitHub** (copias en internet). Un
*remote* es un nombre corto que tu copia local usa para referirse a un repo de GitHub:
`git push <remote>` sube tus commits ahí, y `git pull` / `git fetch` traen los de ahí.

| Remote | Repo en GitHub | Para qué | Quién lo tiene |
| --- | --- | --- | --- |
| `origin` | `JoacoFernandezB/CHANGUITRON---JAR-2026` | Repo de trabajo del equipo | Todos |
| `upstream` | `AIRclub-UdeSA/jar_challenge` | Template oficial de la organización | Todos (opcional) |
| `entrega` | `JoacoFernandezB/jar_challenge-changuitron` | Fork para la entrega | Solo quien entrega |

### ¿Hay conexión entre los repos?

**Entre CHANGUITRON y el fork, en GitHub, no.** Ninguno sabe del otro: si cambiás uno, el otro no
se entera. La única conexión es una copia local que tenga a los dos como remotes: trae de uno y
sube al otro.

```text
          GitHub                                   GitHub
  ┌────────────────────┐                  ┌──────────────────────────┐
  │ CHANGUITRON        │                  │ jar_challenge-changuitron│──PR──▶ AIRclub-UdeSA/
  │ (trabajo, equipo)  │                  │ (fork, entrega)          │        jar_challenge
  └─────────▲──────────┘                  └────────────▲─────────────┘        (oficial)
            │ git pull origin main                     │ git push entrega main
            │ git push origin ...                      │
            └────────────── copia local ───────────────┘
                     (de quien entrega, 3 remotes)
```

**El fork sí está conectado con el repo oficial:** GitHub sabe que es una copia del oficial, y por
eso desde ahí se puede abrir el PR de entrega. Esa es la única razón por la que existe.

Los dos repos comparten historia (ambos parten del template), así que pasar código de uno al otro
con `git push` no genera conflictos.

### Cómo se armó (2026-10-25)

1. Se trajo el template oficial a CHANGUITRON, con su historial (paquete `equipo_jar`, launch, juez
   de práctica, controles del CI).
2. Se agregaron esta guía, [la propuesta de arquitectura](ARQUITECTURA.md), el `CLAUDE.md` y el
   nombre del equipo en `equipo.yaml`.
3. Se creó el fork `jar_challenge-changuitron` y se le subió el mismo `main`.

## 3. El día a día

Todos los comandos de ROS van **dentro del contenedor** (`./container.sh enter`) o en una terminal
con Humble.

**Compilar** (desde la raíz del workspace, como en los workshops):

```bash
cd ~/rosmaster_ws
colcon build --symlink-install --packages-select equipo_jar
source install/setup.bash
```

Con `--symlink-install` los cambios a `.py`, launch y `params/` se ven sin recompilar. Hay que
volver a compilar si agregás un ejecutable nuevo en `setup.py` o un archivo nuevo en `launch/` o
`params/`.

**Probar como en la corrida** (tres terminales):

```bash
# 1. Simulador (spawn_x/y/yaw: los jueces nos ponen cerca del origen, no exacto)
ros2 launch yahboom_rosmaster_gazebo rosmaster_gazebo_fortress.launch.py \
  world:=$(ros2 pkg prefix --share yahboom_rosmaster_gazebo)/worlds/maze_1_6x5_victimas.world \
  spawn_x:=0.15 spawn_y:=-0.1 spawn_yaw:=0.2

# 2. Juez de práctica: publica /map a los 5 s, puntúa en vivo y dice cuántas víctimas hay (N)
cd ~/rosmaster_ws/src/CHANGUITRON---JAR-2026
SIM=$(ros2 pkg prefix --share yahboom_rosmaster_gazebo)
python3 tools/juez.py --mundo maze_1_6x5 --mapa $SIM/maps/maze_1_6x5.yaml --ground-truth

# 3. Nuestro software, con el N que dijo el juez
ros2 launch equipo_jar competencia.launch.py use_sim_time:=true victimas:=N
```

**Prueba rápida con un solo comando** (hace lo mismo que las tres terminales, mide y guarda):

```bash
cd ~/rosmaster_ws/src/CHANGUITRON---JAR-2026
scripts/probar.sh --nota "qué cambié"            # maze_1_6x5, 300 s, arranque corrido
scripts/probar.sh --mapa maze_3_6x6 --tiempo 600 # otro mapa, corrida completa de 10 min
scripts/probar.sh --ventana                      # con Gazebo y RViz para mirar
scripts/probar.sh --help                         # todas las opciones
python3 scripts/historial.py --historial pruebas/historial.csv   # ver corridas anteriores
```

Compila `equipo_jar`, levanta el simulador sin ventana (o con `--ventana`), el juez con puntaje y
nuestro launch, y mientras corre muestra solo los mensajes importantes (mapa, localización,
víctimas). Al final imprime el puntaje, la distancia de cada reporte y **el error de la
localización contra la pose real** (cuánto tarda AMCL en converger, error medio y máximo después),
y suma una línea a `pruebas/historial.csv` con el commit y la nota: así se ve si un cambio mejoró o
empeoró. Logs y `grafico.png` de cada corrida en `pruebas/<fecha>_<mapa>/` (ignorado por git).

La medición usa `/ground_truth/odom`, que solo existe en el simulador: por eso vive en
`scripts/` y no en `src/`. Nuestro software nunca lo usa (§7.1).

Mapas de práctica: `laberinto_simple`, `laberinto_1`, `maze_1_6x5`, `maze_2_6x5`, `maze_3_6x6`,
`maze_4_metal_6x6`. Probá en **todos**: en Rosario el laberinto es otro.

**Antes de abrir un PR**, los mismos controles que el CI:

```bash
cd ~/rosmaster_ws/src/CHANGUITRON---JAR-2026
tools/verificar.sh --template                  # rápido, sin simulador (~1 min)
tools/verificar.sh --template --sim maze_1_6x5 # además, una corrida headless puntuada
```

`--template` saltea el chequeo de `equipo.yaml`. Como ya está completo, también se puede correr sin
`--template`, que es como corre el CI del PR de entrega. `verificar.sh` compila adentro del repo
(`build/`, `install/`, `log/`, todo ignorado por git); no molesta al build del workspace.

## 4. Reglas que no se negocian

Romper cualquiera de estas cuesta puntos o la entrega entera:

- **Interfaz fija** (§7.1). Solo `/map`, `/victimas`, `/done`, `/cmd_vel`, `/scan`, `/odom`,
  `/imu/data`, `/cam_1/...`, `/joint_states`, `/tf`, `/tf_static`. Entre nuestros nodos, topics libres.
  Nada de ground truth del simulador.
- **QoS** ([docs/QOS.md](QOS.md)). `/scan` y cámaras: `qos_profile_sensor_data`. `/map`: reliable +
  transient local. Lo que publicamos: la QoS por defecto (`create_publisher(msg, topic, 10)`).
  Una QoS mal puesta no tira error: simplemente no llega nada.
- **Nada antes de `/map`.** No mover el robot ni reportar hasta recibir el mapa.
- **Exactamente N reportes** (§7.2). Cada mensaje a `/victimas` gasta uno, puntúe o no. Siempre
  con `armar_reporte` de `reportes.py` (`frame_id: map`, sin NaN). Que **un solo nodo** publique en
  `/victimas` y lleve la cuenta.
- **Nada hardcodeado** (§10): ni mapas, ni coordenadas, ni waypoints. En `params/` solo umbrales,
  ganancias, velocidades, rangos de color. N llega por `victimas:=N`, no va en `params/`.
- **`use_sim_time`** (§7.3): todos los nodos lo reciben del launch. El robot real no publica `/clock`.
- **Watchdog de `/cmd_vel`**: el robot se frena si pasan 0,5 s sin comando. Tras `/done`, frenar y
  no volver a mover.
- **Robot quieto 60 s = fin de corrida** (§11.3). Si un nodo se cae, no se puede reiniciar (§11.4).
- **Todo adentro de la imagen.** Corre sin internet: dependencias de ROS en `package.xml`
  (`<exec_depend>`), de Python en `requirements.txt` con versión fija.
- **No tocar `tools/` ni `.github/`.** La organización usa su propia copia, y un PR que los modifica
  no entra al leaderboard de práctica.

## 5. Workflow de implementación (todos)

Todo pasa en **CHANGUITRON**. El fork no se toca. **Nadie trabaja ni pushea directo en `main`**:
`main` siempre tiene que funcionar, porque es lo que se entrega.

```bash
git checkout main && git pull origin main         # 1. partir de lo último
git checkout -b feature/deteccion-rojo            # 2. una branch para tu tarea
# ... programar, compilar, probar con simulador + juez ...
tools/verificar.sh --template                     # 3. mismos controles que el CI
git add -A && git commit -m "Detector de rojo"    # 4. guardar (commits chicos, mensaje claro)
git push origin feature/deteccion-rojo            # 5. subir la branch
```

6. En GitHub, abrir un **PR de tu branch hacia `main`** de CHANGUITRON.
7. Otra persona lo revisa, el CI (`Validar entrega`) corre solo, y si está todo bien se **mergea**.
8. Después del merge: `git checkout main && git pull origin main` y una branch nueva para la
   siguiente tarea.

Nombres de branch: `feature/<tema>` para algo nuevo, `fix/<tema>` para un arreglo. No commitear
`build/`, `install/`, `log/` ni rosbags.

**Si `main` avanzó mientras trabajabas en tu branch**, traé esos cambios antes de abrir el PR:

```bash
git checkout main && git pull origin main
git checkout feature/mi-tarea && git merge main
```

**Traer cambios del template oficial** (la organización lo sigue actualizando), por PR como
cualquier otro cambio:

```bash
git fetch upstream
git checkout main && git pull origin main
git checkout -b sync/template && git merge upstream/main
git push origin sync/template                     # y PR hacia main
```

## 6. La entrega (§9.1)

La entrega es un PR desde el fork **`JoacoFernandezB/jar_challenge-changuitron`** hacia
**`AIRclub-UdeSA/jar_challenge`**. La hace una sola persona, desde una copia local con el remote
`entrega`:

```bash
git remote add entrega git@github.com:JoacoFernandezB/jar_challenge-changuitron.git   # una vez
```

Cada vez que `main` tenga una versión buena:

```bash
git checkout main && git pull origin main         # traer lo último del equipo
git push entrega main                             # copiarlo al fork
```

- **La primera vez**, con `codigo_inscripcion` cargado en `equipo.yaml`, abrir el PR desde el fork
  hacia `AIRclub-UdeSA/jar_challenge`, título `[Equipo] CHANGUITRON`, con el checklist del template
  completo. Sin el código el CI del PR da rojo, y a la tabla de práctica solo entran corridas en
  verde: no tiene sentido abrirlo antes.
- **Las siguientes veces** alcanza con los dos comandos de arriba: el PR se actualiza solo y el CI
  corre en los mundos de práctica, con puntaje en el
  [leaderboard de práctica](https://airclub-udesa.github.io/jar_challenge/).
- **El PR nunca se mergea.** Al cierre de entregas, la organización toma el último commit del fork.
- El PR es público: nada de mails ni datos personales.

`codigo_inscripcion` quedó en `100`: la organización confirmó que vale cualquier código.

**Pendiente:** confirmar con la organización la fecha de cierre de entregas.
