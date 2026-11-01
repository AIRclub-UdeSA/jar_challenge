# Challenge JAR 2026 — reglas, objetivos y estado del equipo PoliBots

Fuente de las reglas: reglamento oficial, https://airclub-udesa.github.io/jar_site/reglamento/
(si algo de acá lo contradice, gana el reglamento). Repo del equipo:
`/root/humble_ws/jar_challenge_PoliBots` (fork de `AIRclub-UdeSA/jar_challenge`, rama `experimentos`).

## 1. Fechas

| Qué | Cuándo |
| --- | --- |
| Cierre de entregas (se evalúa el commit del PR a esa hora) | **martes 3/11, 9:00** |
| Pedir dependencias nuevas (Discussions del repo) | hasta lunes 2/11, 23:59 |
| Clasificación: 2 corridas, cuenta la mejor; pasan 4 equipos | miércoles 4/11 |
| Final: 1 corrida, layout nuevo (víctimas y obstáculos movidos) | jueves 5/11, 17:00 |

## 2. Cómo es una corrida (§1, §5, §11)

- Robot: Yahboom ROSMASTER X3, **mecanum** (se puede desplazar de costado).
- Los jueces lo ponen **cerca** del origen (0, 0, 0), no exacto.
- El launch (`ros2 launch equipo_jar competencia.launch.py victimas:=N`) tiene que quedar
  esperando `/map`. No puede mover el robot ni reportar antes.
- El juez publica `/map` → arranca el reloj de **10 minutos**.
- La corrida termina con lo primero que pase:
  - el equipo publica en `/done`;
  - pasan 10 minutos;
  - el robot queda **quieto 60 s seguidos** (sin trasladarse ni girar);
  - el robot vuelca (más de 45°) o sale del laberinto.
- Si el software se cae, la corrida sigue hasta los 10 min o los 60 s quieto. No se puede reiniciar.
- Autonomía total: sin teleoperación ni intervención humana.

## 3. Puntaje (§8)

| Criterio | Puntos |
| --- | --- |
| Víctima reportada a ≤ 0.25 m del centro de la caja | **+100** |
| Víctima reportada entre 0.25 y 0.50 m | +50 |
| Reporte a más de 0.50 m | 0 (y gasta un reporte) |
| **Rescate rápido: las N víctimas puntuadas antes de los 5 minutos** | **+200** |
| Volver a la largada: `/done` con el robot a ≤ 0.25 m del inicio y al menos 1 víctima puntuada | +100 |
| Choque con pared, obstáculo o víctima (cada evento; retroceder y volver a chocar es otro) | **−50** |
| Salir del laberinto | −50 y fin de la corrida |
| Vuelco | corrida anulada con 0 puntos |

- El total no baja de 0.
- El tiempo no resta puntos. Importa por el bono de 5 minutos y como desempate.
- **Desempate**: 1) más puntos; 2) **menos tiempo** (de `/map` a `/done` o al fin); 3) menos choques; 4) en la clasificación, el puntaje de la otra corrida.
- **Máximo teórico con N víctimas: 100·N + 200 + 100.**

## 4. Víctimas, obstáculos y distractores (§6)

- **Víctimas**: cajas **rojas**, detectables con la cámara RGB frontal y el lidar. Hay N por
  layout y se informan antes de la corrida. Están separadas lo suficiente como para que ningún
  reporte pueda corresponder a dos. Su posición es el centro geométrico de la caja proyectado al
  piso, en el frame `map`. En los mundos de práctica miden 0.25 × 0.25 × 0.3 m.
- **Obstáculos no mapeados**: no están en `/map`. Hay que detectarlos y esquivarlos en tiempo
  real, y chocarlos resta puntos. En el mundo de práctica son cajas **azules** de 0.25 m, ubicadas
  en pasillos, que dejan huecos de unos 0.4 m.
- **Distractores**: objetos de otras formas y colores (azul, amarillo, verde, neutro), **nunca
  rojos**. Reportar uno gasta un reporte.

## 5. Reportes (§7.2)

- Tópico `/victimas`, tipo `geometry_msgs/PointStamped`, `frame_id: map`, Reliable.
- **Se permiten exactamente N reportes por corrida**; los que sobran se ignoran.
- Un reporte mal formado (frame incorrecto, NaN o infinito) gasta el lugar y da 0.
- Reportar dos veces la misma víctima gasta dos lugares; cuenta el mejor de los dos.

## 6. Interfaz permitida (§7) y prohibiciones (§10)

- Solo se pueden usar estos tópicos: `/map`, `/victimas`, `/done`, `/cmd_vel`, `/scan`
  (Best Effort, lidar de 360°), `/odom` (odometría de ruedas), `/imu/data`,
  `/cam_1/color/image_raw`, `/cam_1/color/camera_info`, `/cam_1/depth/color/points`
  (nube RGB-D; el reglamento avisa que tenerla activa toda la corrida puede agregar latencia),
  `/tf` y `/tf_static`. Entre los nodos propios se puede usar cualquier tópico.
- **QoS**: los sensores son Best Effort, así que hay que suscribirse con `qos_profile_sensor_data`.
- Prohibido usar ground truth del simulador, mapas o coordenadas hardcodeados, o waypoints
  sacados de observar la arena.
- `params/` solo admite valores generales (umbrales, ganancias, límites). Es la única carpeta que
  se puede modificar después del cierre de entregas.
- **Dependencias**: solo las de `tools/dependencias_permitidas.yaml` (ROS desktop, Nav2,
  slam_toolbox, robot_localization, numpy, OpenCV, scipy, scikit-learn, pandas, etc.). El código
  corre en una computadora de la organización, sin internet.
- El CI (`tools/verificar.sh`) tiene que estar en verde.

## 7. Datos del robot y del simulador

- Chasis de 0.30 × 0.20 m con ruedas (URDF), corrido 3 cm hacia atrás de `base_footprint`.
- **Topes de velocidad del robot real: 0.5 m/s en x, 0.5 m/s en y, 1 rad/s de giro.**
- `/odom` sale solo de los encoders de las ruedas. No se fusiona con el IMU ni con el lidar.
- El simulador por defecto usa `motion_profile:=stress` (patinamiento de ruedas) y
  `motion_bias:=true` (desvío aleatorio en cada lanzamiento). El watchdog frena el robot si pasan
  0.5 s sin `/cmd_vel`.
- Velocidades reales medidas en el simulador al pedir el máximo:

  | Movimiento | Velocidad real |
  | --- | --- |
  | Adelante | ~0.40 m/s |
  | De costado | ~0.35 m/s |
  | Diagonal | ~0.6 m/s |
  | Girando | rinde solo ~65% del giro pedido |

## 8. Objetivo del equipo

**Maximizar el puntaje y minimizar el tiempo**:
1. Encontrar y reportar las N víctimas con error ≤ 0.25 m, **antes de los 5 minutos**, para sumar el bono de +200.
2. **No chocar**: cada choque resta 50.
3. Volver a la largada y publicar `/done` (+100). Eso además corta el reloj para el desempate.
4. No gastar reportes en distractores ni en posiciones malas.

## 9. Estado actual del software (1/11)

Launch `competencia.launch.py`:
- **Localización**: `amcl` con `OmniMotionModel`, ajustado para confiar más en el lidar (`max_beams` 120, `sigma_hit` 0.1, `z_hit`/`z_rand` 0.85/0.05). Error medido de ~0.03 m.
- **Navegación**: Nav2 con `controller_server` (DWB holonómico, topes 0.5/0.5/1.0, footprint
  rectangular, crítico `ObstacleFootprint`), `planner_server` (NavFn), `behavior_server` y
  `bt_navigator`.
- **Percepción**:
  - `detector` y `detector_scan` (máscara HSV de rojo; los rayos del lidar que caen sobre rojo se publican en `/scan_rojo`);
  - `detector_obstaculos` (`/scan_rojo` → `/objetos_rojos`, con `umbral_probabilidad` 101, como en el workshop de la semana 10);
  - `juntar_victimas` (agrupa por mediana, corre cada detección de la cara al centro de la caja con `mitad_caja` 0.125, publica en `/victimas` como mucho N veces).

  Probado: con el robot frente a una víctima, reporta a ~6 cm y el juez da 100.
- **Exploración: NO EXISTE.** `cobertura_mapa.py` no genera objetivos y el robot se queda quieto,
  así que hoy la corrida termina a los 60 s con 0 puntos. Es la pieza que falta.
- **Problemas conocidos de navegación**:
  - DWB a veces queda con el footprint tocando la esquina de una pared y descarta todas las trayectorias ("No valid trajectories"). Se está probando castigar más la cercanía a las paredes.
  - Llegar a objetivos pegados a una pared (a ~0.17 m) puede tardar mucho.
- Hay material de los workshops (semanas 06-10) en `/root/humble_ws/rosmaster_ws/src/jar_workshops`:
  explorador con grilla de cobertura y A* (semana 08), localizador propio, `juntar_victimas`.
