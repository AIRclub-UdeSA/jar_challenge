# jar_challenge

Repositorio base del **Challenge JAR 2026** (búsqueda y rescate, Jornada Argentina de Robótica, Rosario, 3 al 6 de noviembre). Lo organiza el [AIR Club UdeSA](https://github.com/AIRclub-UdeSA).

Este repositorio es el punto de partida y a la vez la vía de entrega: hacés un **fork**, desarrollás tu software ahí y abrís un **pull request**. Ese PR es tu entrega.

> El reglamento oficial es la fuente de verdad. Este README explica cómo trabajar con el repo; si algo acá contradice al reglamento, gana el reglamento. [POR CONFIRMAR: enlace al reglamento.]

## Qué hay acá

```text
jar_challenge/
├── equipo.yaml                     nombre de tu equipo y código de inscripción (completalo)
├── Dockerfile                      imagen que la organización construye y corre sin internet
├── requirements.txt                dependencias de Python que no vienen con ROS
├── src/
│   └── equipo_jar/                 tu paquete ROS 2 (podés agregar más paquetes en src/)
│       ├── launch/competencia.launch.py    EL launch: levanta todo tu software
│       ├── params/params.yaml              parámetros generales (umbrales, ganancias...)
│       └── equipo_jar/nodo_equipo.py       nodo esqueleto: espera /map y no hace más
└── tools/                          los mismos controles que corre el CI (no los modifiques)
```

## Empezar

Necesitás Ubuntu 22.04, ROS 2 Humble y el simulador. Seguí la [guía de setup](https://airclub-udesa.github.io/jar_site/setup/simulador/) (hay alternativa en Docker).

```bash
# 1. Fork en GitHub (botón "Fork"), y después:
# Ojo: usá Fork, no "Use this template" (un repo creado desde un template no puede abrir PR).
cd ~/rosmaster_ws/src
git clone https://github.com/<tu-usuario>/jar_challenge.git

# 2. Compilar
cd ~/rosmaster_ws/src/jar_challenge
source /opt/ros/humble/setup.bash
colcon build --base-paths src
source install/setup.bash

# 3. En otra terminal, el simulador
ros2 launch yahboom_rosmaster_gazebo rosmaster_gazebo_fortress.launch.py \
  world:=$(ros2 pkg prefix --share yahboom_rosmaster_gazebo)/worlds/maze_1_6x5_victimas.world

# 4. Tu software (use_sim_time:=true solo en el simulador; victimas:=N es la cantidad a buscar)
ros2 launch equipo_jar competencia.launch.py use_sim_time:=true victimas:=3
```

El simulador pone al robot justo en el origen del mapa, mirando a +x. En la competencia los jueces lo ubican a mano cerca del origen, pero no exactamente ahí (§1 del reglamento). Para practicar eso, agregale al launch del simulador `spawn_x` y `spawn_y` (en metros, en el marco del mapa) y `spawn_yaw` (en radianes), por ejemplo `spawn_x:=0.15 spawn_y:=-0.1 spawn_yaw:=0.2`. En los mundos `maze_*` la pared más cercana al origen está a unos 0,45 m, así que alcanza con moverlo hasta ±0,2 m.

El nodo esqueleto queda esperando el mapa. Para probar tu software como en la corrida, usá el **juez de práctica**. Publica el mapa, escucha tus reportes y te va diciendo cuántos puntos sumás:

```bash
SIM=$(ros2 pkg prefix --share yahboom_rosmaster_gazebo)
python3 tools/juez.py --mundo maze_1_6x5 --mapa $SIM/maps/maze_1_6x5.yaml --ground-truth
```

- **Largada.** Publica `/map` a los 5 s (`--retardo`), o cuando llamás a `ros2 service call /juez/iniciar std_srvs/srv/Trigger` si lo levantás con `--manual`.
- **Puntaje en vivo.** Cada reporte se puntúa al llegar y se publica en `/juez/puntaje`. Al terminar imprime el desglose y, con `--salida resultado.json`, lo guarda con todos los eventos.
- **Fin de corrida.** Por tu `/done`, a los 10 minutos, si el robot queda quieto 60 s o si sale del laberinto.
- **Colisiones, salida del laberinto y vuelco** no se detectan solos: los marca una persona, en el simulador y en el evento (`ros2 service call /juez/colision std_srvs/srv/Trigger`, `/juez/salida`, `/juez/vuelco`). **Regreso a la largada:** después de tu `/done`, `ros2 service call /juez/regreso std_srvs/srv/SetBool "{data: true}"` (o `false`). Con `--ground-truth` el juez mide el regreso con la pose real del simulador, para practicar.
- **Las víctimas son las del mundo de práctica** (las cajas rojas del `.world`), que tienen 2 o 3 cajas. Al arrancar, el juez te dice cuántas son: lanzá tu software con `victimas:=N` con ese N. En la competencia el mapa y las víctimas son otros, y la cantidad la informa la organización (llega a tu launch como `victimas:=N`).

Es un juez de práctica, no el oficial: en el evento las colisiones, el regreso a la largada y el registro de la corrida los hace la organización.

## Cómo es la entrega

1. **Un único launch.** `ros2 launch equipo_jar competencia.launch.py` tiene que levantar todo y quedarse esperando el mapa en `/map`. No mueve el robot ni reporta nada antes de recibirlo. **Acepta la cantidad de víctimas como argumento** (`victimas:=N`): la organización informa N en el lugar, antes de cada corrida, y tu software tiene que buscar esa cantidad. El template ya lo recibe como parámetro `cantidad_victimas`.
2. **Parámetros en `params/`.** Solo valores generales: umbrales, ganancias, límites de velocidad, rangos de color. Nada de coordenadas, rutas ni posiciones. Es la única carpeta que se puede retocar después del cierre de entregas.
3. **Todo dentro de la imagen.** Tu software corre en un contenedor sin internet. Las dependencias y los pesos de los modelos tienen que quedar en la imagen al construirla con el `Dockerfile`. Probalo con `docker build -t equipo-jar . && docker run --rm --network none equipo-jar`.
4. **Nada del mapa de competencia.** No incluyas mapas, mundos ni posiciones de víctimas, señuelos u obstáculos, ni rutas derivadas de observar la arena. El mapa llega por `/map` en la corrida.
5. **Solo la interfaz oficial.** Tu software usa los topics del reglamento (`/map`, `/victimas`, `/done`, `/cmd_vel`, `/scan`, `/odom`, `/imu/data`, `/cam_1/...`, `/joint_states`, `/tf`, `/tf_static`). Los topics internos entre tus propios nodos son libres. Nada de ground truth del simulador. **Usá la QoS de cada topic** ([docs/QOS.md](docs/QOS.md)): `/scan` y las cámaras son *best effort*, y `/victimas`, `/done` y `/cmd_vel` se publican *reliable*. Con otra, no te llega nada o no te escuchan, y no da derecho a repetir la corrida.
6. **CI en verde.** El CI corre en cada push a tu PR. Corré lo mismo en tu máquina antes:

   ```bash
   tools/verificar.sh                      # controles rápidos, sin simulador
   tools/verificar.sh --sim maze_1_6x5     # además, una corrida en el simulador
   ```

7. **`equipo.yaml` completo** y el título del PR con el formato `[Equipo] <nombre>`. El fork y el PR son públicos: no pongas ahí mails ni datos personales; van en el formulario de inscripción.

### Qué revisa el CI

| Control | Qué comprueba |
| --- | --- |
| Estructura | Launch, `params/`, `Dockerfile` y `equipo.yaml` en su lugar; sin mapas ni mundos embebidos. |
| Parámetros | `params/` sin coordenadas, poses ni rutas a mapas. |
| Revisión estática | Sin acceso a interfaces fuera del reglamento. Además marca posibles hardcodeos para que las mire el juez técnico: una marca **no es una sanción**. |
| Compilación y tests | `colcon build` y `colcon test`. |
| El launch espera el mapa | Con el juez de práctica: no mueve el robot ni reporta antes de `/map`, los reportes tienen el formato oficial, y ningún nodo publica donde no debe, y la QoS de tus publicadores y suscriptores es la correcta. |
| Imagen sin internet | La imagen se construye y el launch sigue vivo sin red. |
| Simulador | Corrida en mundos de práctica, y auditoría de las interfaces usadas. |

El detalle fino de las revisiones contra el hardcodeo no se publica. La organización vuelve a correr los controles desde una copia propia de `tools/` y `.github/`: modificarlos en tu fork no cambia el resultado.

## Reportar una víctima

Usá `armar_reporte` de [reportes.py](src/equipo_jar/equipo_jar/reportes.py): construye el mensaje con el formato oficial (`PointStamped`, `frame_id: map`, `x` e `y` en metros) y rechaza valores inválidos. Recordá que **cada mensaje en `/victimas` gasta un reporte**, puntúe o no.

## Contacto

[POR CONFIRMAR: casilla de contacto del club.]
