# Mejoras propuestas: PoliBots, Challenge JAR 2026

Investigación del 1/11/2026. No toqué ningún archivo de `jar_challenge_PoliBots` ni de `rosmaster_ws`
y no lancé ROS. Lo que sigue sale de leer el código, el reglamento, el código fuente de Nav2 Humble
y los forks públicos, más un script offline (numpy/scipy, sin ROS) que corrí sobre los `.pgm` de
práctica para medir cuántos puntos de vista hacen falta.

## Estado (2/11/2026)

| Punto | Estado |
| --- | --- |
| P0, P1, P2, P3, P6, P7 | Hechos. CI en verde salvo `codigo_inscripcion` (lo completa el equipo). |
| P4a BT propio | Probado y descartado: el BT por defecto anduvo mejor. |
| P4b destrabe lateral | Hecho (con el scan, en la misión). |
| P4c encadenar objetivos | Hecho: a 0.3 m de un punto de vista se pasa al próximo (`distancia_encadenar`). |
| P4d costmap local fino | Hecho: local a 0.025 m, solo lidar (sin static_layer: con el error de amcl el mapa quedaba encima del robot), padding 0.01 local y 0.03 global. |
| P4e progress checker | Hecho: `PoseProgressChecker`, `failure_tolerance` 0.5. Además DWB sin trayectorias quietas (`min_speed_xy` 0.08, `min_speed_theta` 0.3). |
| P4f inflación | Hecho (global 0.5 / 4.0). P4g-h sin cambios; P4i MPPI probado y descartado; P4j no hizo falta. |
| P5a-c, P5g-i | Hechos (12 detecciones, tf al stamp, filtro de giro, dispersión, reportes sobrantes al final, sin nodo `detector`). |
| P5d-f | Hechos: imagen más cercana al scan (`desfase_maximo_imagen`, negativo lo apaga), rojo a más de 3.5 m no cuenta, erosión de la máscara. |
| Nuevo: arranque de Nav2 | Si a los 20 s Nav2 no está activo (el `lifecycle_manager` de Humble se puede colgar esperando sin timeout la respuesta de un nodo: pasó en maze_4 y la corrida dio 0), la misión configura y activa cada nodo con `get_state`/`change_state`. |
| Nuevo: oclusión en `detector_scan` | De cerca, puntos del lidar detrás de la caja caían sobre pixeles rojos (paralaje cámara-lidar) y armaban una víctima falsa a 0.85 m. Un rayo rojo con otro más corto en la misma zona de la imagen se descarta. |
| Nuevo: rojo pegado a paredes | `margen_pared` 0.08 en `detector_obstaculos`: rayos de una pared que caían sobre pixeles de una caja cercana (desfase imagen-scan al girar) armaban una víctima falsa en maze_1. |
| Revisión del investigador (2/11) | Hechos: `use_sim_time` de Nav2 ya no lo pisa `nav2_params.yaml` (en el robot real quedaba en true); `umbral_probabilidad`, `racha_minima` y `distancia_minima_valida` del detector rojo pasan a `params.yaml`; tf con `spin_thread`; una excepción en el ciclo no mata la misión; sin pose no se explora; el objetivo encadenado se cancela al inspeccionar o volver; `/done` sin espera si ya están los N reportes; si Nav2 no responde en la vuelta, `/done` igual a los 10 s. El rojo visto de pasada también crea o suma a un candidato (la misión va a mirarlo de cerca). Giro de inspección parametrizado (`despeje_giro`, `giro_minimo_inspeccion`, ...); si se saltean más de 20 scans seguidos por desfase, se usa la imagen más reciente. Segunda pasada con alcance 1.8 m si faltan víctimas y sobran 120 s (laberinto_1: 200 → 300). Límite de la corrida en tiempo de pared (como el juez) y tope de 15 inspecciones. Para la arena (solo `params/`): `mitad_caja: 0.09`, HSV, menos muestras de DWB si falta CPU. |
| Segunda revisión (2/11) | Hechos: Nav2 cuenta como listo solo con bt_navigator activo (GetState); tope de 15 s por nodo en el arranque directo; `rechazos` se reinicia; freno después del giro de espera; los watchdogs cortan el ciclo (no pisan el destrabe); id por objetivo (respuestas viejas se ignoran o se cancelan); tope de tiempo por objetivo (`tiempo_objetivo_base_s` + `tiempo_objetivo_por_metro_s`·distancia); conteo de inspecciones corregido; cola de 15 imágenes; `bond_timeout` en `nav2_params.yaml`. Hoja de ajuste para la arena: ver más abajo. |
| Tercera revisión: tiempos (2/11) | Hechos: se corta el objetivo al 2.º recovery de Nav2 (`recoveries_para_cortar`; con 1 maze_3 empeoraba) y la misión hace su destrabe; `intentos_vuelta` 6; el watchdog de desplazamiento ignora los giros (12 s); inspección: `tiempo_mirar_s` 1.2 (2 s si hubo 1 detección roja), giro a 1 rad/s hasta 0.25 rad, candidato que pide menos giro; `distancia_encadenar` 0.45 con línea de vista. maze_2: 199 → 130-139 s; maze_3 101-125 s; el resto igual, todos con puntaje máximo. |
| Prueba con poca CPU (equipo en 2 núcleos) | Hallazgos y arreglos: en la vuelta no se cortan los recoveries de Nav2 y no se rinde hasta 20 s antes del final (maze_1 publicaba /done a 3.16 m); /done solo si amcl confirma ≤ 0.2 m aunque Nav2 diga que llegó; `juntar_victimas` mide la dispersión con los últimos 30 puntos (`puntos_recientes`; una víctima con 203 detecciones no se reportaba); no se inspecciona a menos de 0.5 m (`distancia_minima_inspeccion`: de muy cerca daba "no es rojo"); el punto de inspección prefiere lugares despejados. Con 2 núcleos maze_1 500 en 5 de 5 corridas; tanda normal: máximo en los 4 mazes, laberinto_1 200. |
| Cuarta revisión (2/11) | Hechos, todos en la vuelta: /done si el robot ya está a ≤ 0.2 m aunque Nav2 aborte; /done sí o sí 10 s antes del final; /done si pasan 40 s sin trasladarse (el juez corta a los 60 s); el watchdog de desplazamiento espera 25 s en la vuelta. `recoveries_para_cortar` sigue en 2 (también cuenta borrados de costmap; probado mejor que 1). Tanda: máximo en los 4 mazes, laberinto_1 200; maze_1 con 2 núcleos 500 (x2). |
| Repetibilidad (ac7cac8, 2 tandas, 12 corridas) | maze_1 500 ×4 (154-235 s), maze_2 600 ×2 (104/132 s), maze_3 600 ×2 (94/97 s), maze_4 500 ×2 (46/55 s), laberinto_1 300 ×2 (203/294 s). Sin fallas. |
| Noche del 2 al 3/11 | Inspección inmediata de candidatos claros (≥ 0.2 m de pared), una vez por candidato; paddings local y global iguales (0.03); corte por quietud en la vuelta según /odom; **subobjetivos en las puntas de pared** (si no avanza 8 cm en 4 s, un punto 0.5 m adelante sobre el camino global: maze_2 dejó de variar entre 94 y 402 s); **giro de inspección seguro con el lidar** (las cajas no están en el mapa; antes la esquina del chasis las tocaba); nodo `registro` + `analizar_corrida.py` + `LOGS_CLASIFICACION.md`. Probado y descartado: inflado local 0.45, `ObstacleFootprint` 0.04-0.05, padding local 0.05, `forward_prune_distance` 0.4 (resuelve puntas pero frena todo), `sim_time` 0.9, `xy_goal_tolerance` 0.15, `detecciones_minimas` 10, `distancia_encadenar` 0.6, amcl más fino. Validación de 4cc56d5: largada desviada 500/600/600/500/300; con 2 núcleos maze_1 500, maze_2 600. Después (madrugada): destrabe hacia la dirección más libre (el empuje promedio podía mandar el robot contra una punta de pared); `juntar_victimas` ignora rojo a menos de 0.35 m (un reporte salió a 0.39 m con el robot pegado a la caja); los reenvíos de objetivo ya no reinician los watchdogs (los subobjetivos los anulaban); un giro de inspección cortado no descarta el candidato (quedaba mirando para otro lado y perdía la víctima). Pendiente: roce de 1-2 cm en una punta de pared con una caja encima (maze_1, en la vuelta). |
| P9 | Hecho: los 5 mundos con cada cambio, y maze_1/2/3 con la largada desviada (±0.2 m, ±0.6 rad): máximo en los tres. Con 0ff1b50, los 5 mundos con la largada desviada: 500/600/600/500/300. En laberinto_1, V1 y V2 están en bolsillos sin entrada (también en el `.world`): solo se ven por rendijas. |

## Resumen: plan recomendado hasta el cierre (martes 3/11, 9:00)

1. **Hoy no se puede sumar ningún punto.** `CoberturaMapa` es un `Node` que nadie hace spin, así que su `ActionClient` nunca recibe
   respuestas. Además `next_goal()` se llama cada 10 s sin mirar si el objetivo anterior terminó, y no hay objetivos.
   Lo primero (P0, unas 5 h) es una **máquina de estados en `nodo_equipo`** con un `ActionClient` de `NavigateToPose`
   y callbacks, que corte cuando se reportan N víctimas, vuelva a la largada y publique `/done`.
2. **Exploración: ruta fija para cubrir el mapa con el LIDAR** (360°, ve las cajas a varios metros), no con la cámara.
   El script offline da **8 a 11 puntos de vista y 20 a 28 m de recorrido** por mundo de práctica, contra 20 a 33 vistas
   con rumbo si se cubre con la cámara (FOV de 61°). Se calcula en 2 s al recibir `/map` y se ordena con vecino más
   cercano + 2-opt.
3. **El lidar avisa y la cámara confirma**: todo lo que el lidar ve y no está en `/map` es candidato. Hay 4 o 5 por
   mundo de práctica. La misión se desvía para **apuntar la cámara** a cada candidato pendiente (giro dirigido, no
   barridos de 360°). Así la cámara solo mira donde hace falta.
4. **Detección**: `detecciones_minimas` de 30 a ~10. Hoy son ~4.2 s mirando cada caja a 7.2 Hz, y pasando de largo no
   llega nunca. Transformar con la tf **del instante del scan**, ignorar detecciones mientras gira rápido y descartar
   puntos pegados a paredes del mapa. Al final, **gastar los reportes que sobren** en los mejores grupos sin confirmar:
   no restan puntos.
5. **Pose inicial**: con `set_initial_pose: true`, amcl arranca con **covarianza cero** (lo verifiqué en el fuente de
   Humble): todas las partículas en (0, 0, 0). El juez coloca el robot a mano, y en el workshop se probó con
   desplazamientos de 30 cm y giros de 45°. Hay que publicar una pose inicial con covarianza y llamar a
   `request_nomotion_update` antes de moverse.
6. **Nav2**: árbol de comportamiento (BT) propio con BackUp a 0.15 m/s primero y sin `Wait 5 s`, `PoseProgressChecker`,
   costmap local a 0.025 m con `footprint_padding` 0.01, y encadenar objetivos sin frenar. Destrabe lateral propio
   (mecanum) cuando DWB dice "No valid trajectories". MPPI solo si sobra tiempo.
7. **Todo umbral va a `params/`**: es lo único que se puede tocar después del cierre, y la corrida es con el **robot
   real**, con cajas de ~18 cm, otra iluminación y patinamiento.
8. **Cronograma**: 1/11 a la noche P0 y la ruta. 2/11 a la mañana candidatos por lidar y ajustes de detección.
   2/11 a la tarde Nav2 y pose inicial, más pruebas en los 5 mundos con `spawn_x/spawn_y/spawn_yaw` desviados.
   2/11 a la noche CI en verde y PR. **No hace falta pedir dependencias nuevas**: scipy y networkx ya están permitidas.

**Cuenta de tiempo esperado** (mundos de práctica): ~25 m a ~0.3 m/s efectivos son ~85 s, más ~10 s de objetivos
encadenados, ~25 s de desvíos a candidatos y ~30 s de vuelta. Da **2.5 a 3 min**, con margen para el bono de 5 min.
Con N = 3 son 300 + 200 + 100 = **600 puntos**.

---

## Datos que sostienen las decisiones

| Dato | Valor | Fuente |
| --- | --- | --- |
| Cámara | 320×240, f = 271.8 px, **HFOV = 2·atan(160/271.8) = 60.9°**, 30 Hz | `rosmaster_ws/src/yahboom_rosmaster/yahboom_rosmaster_description/urdf/sensors/rgbd_camera.urdf.xacro` |
| Lidar | 1080 rayos en 360° (0.33°/rayo), 0.05 a 12 m, **7.17 Hz** en el sim, ~7.6 Hz en el real | `.../robots/rosmaster_x3.urdf.xacro`, workshop semana 10 |
| Latencia | el lidar tarda ~0.13 s por vuelta; la imagen llega ~0.11 s después del scan; "a 50°/s, en 0.13 s gira 6.5°; a 2 m, 5° corren el punto ~20 cm"; "los puntos rojos del lidar se corren por la pared durante los giros" | https://airclub-udesa.github.io/jar_site/workshops/semana-10-sim-a-real/ |
| Corrida | **con el robot físico**; cajas reales de ~18 cm (`mitad_caja` 0.09) | https://airclub-udesa.github.io/jar_site/reglamento/ y semana 10 |
| Mundos de práctica | 2 o 3 cajas rojas y 2 azules por mundo (laberinto_1: 3 y 1), puestas "en pasajes angostos... para que no queden a la vista desde el spawn" | `yahboom_rosmaster_gazebo/worlds/*_victimas.world` |
| Simulador | admite `spawn_x`, `spawn_y`, `spawn_yaw`; "/odom still starts at zero" | `yahboom_rosmaster_gazebo/launch/rosmaster_gazebo_fortress.launch.py` |
| Juez | inmovilidad medida con la velocidad de `/odom`; los choques los marca una persona (en el sim, con el servicio `/juez/colision`) | `tools/juez.py` |

### Simulación offline de cobertura (script en el scratchpad, no en el repo)

Pasos: celdas libres a ≥0.25 m de pared, candidatos en máximos locales de la distancia a la pared (centro del
pasillo), visibilidad por raycasting de 180 rayos, *set cover* greedy hasta el 98 % de las celdas donde puede haber
una caja, y orden con vecino más cercano + 2-opt sobre distancias geodésicas (Dijkstra en la grilla).

| Mundo | Lidar 360°, 4 m | Lidar 360°, 2 m | Cámara 58°, 2.5 m (vistas celda+rumbo) |
| --- | --- | --- | --- |
| maze_1_6x5 | 8 puntos, 27.4 m + 11.3 m de vuelta | 9 puntos, 27.5 m | 20 vistas en 18 lugares |
| maze_2_6x5 | 8 puntos, 20.2 m + 3.5 m | 9 puntos, 21.7 m | — |
| maze_3_6x6 | 9 puntos, 28.4 m + 4.9 m | 10 puntos, 33.0 m | 23 vistas en 23 lugares |
| maze_4_metal_6x6 | 9 puntos, 27.1 m + 5.0 m | 11 puntos, 32.1 m | — |
| laberinto_1 | 11 puntos, 22.8 m + 5.0 m | 12 puntos, 29.3 m | 33 vistas en 28 lugares |

Tiempo de cálculo: 1.5 a 2.5 s en Python puro, muy por debajo de la regla de 60 s quieto.
**Conclusión**: con la cámara hacen falta 2 a 3 veces más paradas, y cada parada cuesta un giro (~2 a 5 s, con el
giro rindiendo al 65 %). Cubrir con el lidar y verificar con la cámara solo los 4 o 5 candidatos es claramente más
rápido.

---

## Mejoras priorizadas

### P0. Misión mínima que funcione (bloqueante)

**Qué es.** Reescribir `nodo_equipo.py` y `cobertura_mapa.py` como **un solo nodo** con una máquina de estados:
`ESPERANDO_MAPA → LOCALIZANDO → EXPLORANDO ⇄ INSPECCIONANDO → VOLVIENDO → TERMINADO`.

**Por qué.** Hoy da 0 puntos. Problemas concretos del código:
- `CoberturaMapa(Node)` se crea dentro de `NodoEquipo` pero nunca se agrega a un executor. `send_goal_async` no
  resuelve nunca y tampoco se llama a `wait_for_server()`.
- `ciclo_de_control` corre a `frecuencia_control_hz: 0.1` (`params.yaml`) y manda un objetivo nuevo cada 10 s sin
  mirar el estado. Cada objetivo nuevo **reemplaza** al anterior en `bt_navigator`.
- No hay vuelta ni `/done`, que valen +100 y además **cortan el reloj** para el desempate.

**Esfuerzo.** 4 a 5 h. **Riesgo.** Bajo.

**Cómo.** Recomiendo un `ActionClient` a mano, con callbacks y un timer de 5 Hz en el mismo nodo, en vez de
`BasicNavigator`:
- `BasicNavigator` (Humble) es bloqueante: `goToPose()` hace `spin_until_future_complete` y `isTaskComplete()` hace
  spin del propio nodo. Se puede usar en un `while`, pero mezclarlo con suscripciones y timers propios complica todo.
- **Ojo**: `waitUntilNav2Active()` con `localizer='amcl'` **publica `/initialpose` con la pose por defecto (0, 0) y
  covarianza cero** hasta recibir `amcl_pose` (`_waitForInitialPose` → `_setInitialPose` en
  `/opt/ros/humble/lib/python3.10/site-packages/nav2_simple_commander/robot_navigator.py`, líneas 552-576). Eso pisa
  cualquier localización. Si lo usan, llamen a `waitUntilNav2Active(localizer='robot_localization')` o directamente
  a `_waitForNodeToActivate('bt_navigator')`.

Esqueleto sugerido para `nodo_equipo.py`:

```python
class NodoEquipo(Node):
    def __init__(self):
        ...
        self.nav = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.goal_handle = None; self.navegando = False; self.resultado = None
        self.create_timer(0.2, self.ciclo)                      # 5 Hz (frecuencia_control_hz: 5.0)
        self.create_subscription(PointStamped, '/victimas', self.contar_reporte, 10)  # cuenta lo que publica juntar_victimas

    def ir_a(self, x, y, yaw, cuando_termine):
        if not self.nav.server_is_ready():                       # no bloquea
            return False
        meta = NavigateToPose.Goal(); meta.pose = pose_stamped(x, y, yaw, self.get_clock().now())
        fut = self.nav.send_goal_async(meta, feedback_callback=self.al_feedback)
        fut.add_done_callback(lambda f: self._aceptado(f, cuando_termine))
        self.navegando = True
        return True

    def _aceptado(self, fut, cb):
        gh = fut.result()
        if not gh.accepted: self.navegando = False; cb(False); return
        self.goal_handle = gh
        gh.get_result_async().add_done_callback(
            lambda r: self._terminado(gh, r.result().status, cb))

    def _terminado(self, gh, status, cb):
        if gh is not self.goal_handle:
            return              # resultado de un objetivo reemplazado: se ignora
        self.navegando = False
        cb(status == GoalStatus.STATUS_SUCCEEDED)

    def al_feedback(self, fb):
        self.distancia_restante = fb.feedback.distance_remaining   # para encadenar (P4c)

    def ciclo(self):
        if self.mapa is None: return
        t = self.segundos_desde_mapa()
        if self.estado == 'EXPLORANDO':
            if self.reportadas >= self.N or t > self.t_limite - self.t_vuelta_estimada() - 20:
                self.cancelar(); self.estado = 'VOLVIENDO'; self.ir_a(*self.largada, ...)
            elif not self.navegando:
                self.ir_a(*self.siguiente_objetivo(), self.al_llegar)
        elif self.estado == 'VOLVIENDO' and self.llegue_a_largada():
            self.pub_done.publish(Empty()); self.estado = 'TERMINADO'
```

Detalles importantes:
- **Contar reportes**: suscribirse a `/victimas` desde el propio nodo (es una suscripción, no publica) o que
  `juntar_victimas` publique un contador en un tópico interno.
- **Largada en `map`**: no usen (0, 0). Al recibir `/map`, guardar la pose de `odom → base_footprint`. Después de
  converger (ver P1), transformarla con `map → odom`: esa es la largada real en `map`, aunque el juez haya puesto el
  robot a 20 cm del origen. Volver con tolerancia fina (`xy_goal_tolerance` 0.05, ver P4) y publicar `/done` recién
  cuando la distancia por tf sea < 0.12 m. Si no llega en un tiempo razonable, publicar igual: el `/done` además corta
  el reloj.
- **Tiempo límite**: `t_limite` = 600 s menos la vuelta estimada (distancia geodésica / 0.25 m/s) menos un margen de
  30 s.
- **Regla de 60 s quieto**: un *watchdog* que, si `/odom` no se mueve en 25 s (por ejemplo, Nav2 rechazando
  objetivos), descarte el objetivo actual y pase al siguiente, o gire 90°.
- Nombres: el chequeo estático avisa (*warning*, no error) por variables como `ruta = [`, `waypoints = [` o
  `victimas_x = [` (`RE_NOMBRES` en `tools/verificar_estatico.py`). Usen `puntos_de_vista`, `pendientes`, etc.
- Las funciones de geometría (raycast, set cover, 2-opt) van en un módulo **sin ROS** (`exploracion.py`), con tests en
  `test/` que corren en el CI.

### P1. Pose inicial robusta (crítico en el robot real)

**Qué es.** No usar `set_initial_pose: true` con covarianza cero. Publicar una pose inicial con covarianza y forzar
actualizaciones sin moverse.

**Por qué.** En `amcl_node.cpp` (Humble), `set_initial_pose` arma un `PoseWithCovarianceStamped` **sin covarianza**
y llama a `initialPoseReceived` → `handleInitialPose` → `pf_init(..., cov = 0)`. Todas las partículas quedan
exactamente en (0, 0, 0). Como `recovery_alpha_*` = 0, no se inyectan partículas aleatorias: la nube solo se abre por
el ruido de movimiento (alpha 0.2). El juez coloca el robot **cerca** del origen, y en la semana 10 se probaron
desplazamientos de **30 cm y 45°**. Si amcl arranca mal, falla todo lo demás: reportes a más de 0.25 m, choques,
vuelta. El ~3 cm medido es con el robot en el origen exacto.

**Esfuerzo.** 1 h, más pruebas con `spawn_x:=0.2 spawn_y:=-0.15 spawn_yaw:=0.6`. **Riesgo.** Bajo.

**Cómo.**
- `nav2_params.yaml`: `set_initial_pose: false`. En el launch, **remapear el tópico de amcl**:
  `remappings=[('initialpose', 'equipo/pose_inicial')]`. Así no se cruza con un `/initialpose` de la plataforma (el
  auditor marca tópicos no oficiales que usa la plataforma).
- En `nodo_equipo`, al recibir `/map`: publicar en `equipo/pose_inicial` (0, 0, 0) con covarianza
  `[0]=[7]=0.3²` y `[35]=0.6²` (σ de 30 cm y 35°). Después llamar **10 veces** al servicio de amcl
  `request_nomotion_update` (`std_srvs/Empty`, existe en Humble: `amcl_node.cpp` línea 1556), una cada 0.2 s. Esperar
  a que la covarianza de `amcl_pose` baje (σxy < 0.05, σyaw < 0.1) o 3 s. Todo eso es antes de mover el robot.
- Opcional: subir `recovery_alpha_slow/fast` a 0.001/0.1, para recuperar si se pierde. Pruébenlo: puede meter saltos.
- Por qué no usar la localización global: con 2000 partículas en ~30 m² y pasillos simétricos, no converge en pocos
  segundos.

### P2. Exploración: ruta de cobertura con el lidar, calculada al inicio

**Qué es.** Al recibir `/map`, calcular un conjunto mínimo de **puntos de vista desde donde el lidar ve todas las
celdas libres donde puede haber una caja**, ordenarlos como un viajante (TSP) y recorrerlos.

**Por qué esta estrategia y no otra:**

| Estrategia | Veredicto |
| --- | --- |
| *Frontier-based* (Yamauchi 1997; `m-explore-ros2`) | Está pensada para mapas **desconocidos**: frontera = libre junto a desconocido. Acá el mapa llega completo, así que no hay fronteras. Solo sirve como fallback con "fronteras de lo que vio la cámara". |
| Cobertura con la cámara (FOV 61°, ~2.5 m) | 20 a 33 vistas con rumbo, cada una con frenada y giro. Es lo que hace el fork más avanzado (ver P8). Más lento, y el giro patina en el real. |
| Girar 360° en cada punto | 360° a ~0.65 rad/s reales son **~9.7 s por punto**; con 10 puntos son ~100 s perdidos. Descartado. |
| **Lidar 360° + verificación dirigida con la cámara** | 8 a 11 puntos, 20 a 28 m. El lidar ve los objetos no mapeados y la cámara los mira solo a ellos (P3). **Recomendada.** |

**Ganancia estimada.** Frente a la cobertura con cámara: ~10 a 20 paradas menos, a 3 a 5 s cada una, son **40 a
80 s**. Frente a girar en cada punto: ~100 s.

**Esfuerzo.** 3 h, con tests. **Riesgo.** Bajo a medio. Lo principal: en el real la arena puede ser distinta; el
algoritmo no depende del tamaño, pero hay que limitar la cantidad de puntos y el tiempo de cálculo.

**Cómo** (módulo nuevo `equipo_jar/exploracion.py`, solo numpy y scipy):

```python
from scipy import ndimage

def preparar(mapa):                       # OccupancyGrid -> arrays
    d = np.array(mapa.data).reshape(H, W)
    ocupado = (d >= 50) | (d < 0)         # lo desconocido se trata como pared
    libre = d == 0
    dist = ndimage.distance_transform_edt(~ocupado) * res   # m a la pared más cercana
    seguro = libre & (dist >= despeje_min)                  # despeje_min ~0.25-0.28 (params)
    seguro = componente_conexa_de(seguro, celda_robot)
    objetivos = libre & (dist >= 0.10)    # donde puede estar el centro de una caja
    return ...

def puntos_de_vista(seguro, ocupado, dist, objetivos, alcance=3.5, n_rayos=180):
    # candidatos: centros de pasillo (máximos locales de dist), submuestreados a <=400
    cand = np.argwhere((dist == ndimage.maximum_filter(dist, 5)) & seguro)
    vis = [raycast(ocupado, c, alcance) & objetivos for c in cand]   # vectorizado: n_rayos x pasos
    cubierto = 0; elegidos = []
    while cubierto.sum() < 0.98 * alcanzables.sum():                # set cover greedy
        i = argmax(|vis[i] & ~cubierto|); elegidos.append(i); cubierto |= vis[i]
    return cand[elegidos]

def ordenar(inicio, puntos, transitable):
    D = matriz geodésica (un Dijkstra por punto, o scipy.sparse.csgraph.dijkstra)
    orden = vecino_mas_cercano(D); orden = dos_opt(orden, D)   # <=12 puntos: instantáneo
    return orden
```

- `alcance` del lidar para cobertura: **3.5 m**. Una caja de 0.18 m a 3.5 m son ~9 rayos, de sobra para
  `racha_minima` 3. Más alcance casi no reduce puntos (tabla de arriba) y empeora la precisión.
- **Yaw del objetivo**: el de la dirección hacia el siguiente punto. Igual se ignora mientras `yaw_goal_tolerance` sea
  3.15.
- **Saltear puntos ya cubiertos**: mantener una grilla "vista por el lidar", que se marca con el scan real como en
  `grilla_cobertura.py` del workshop 8. Si al llegar a un punto su visibilidad prevista ya está cubierta al ≥90 %,
  saltearlo. Es opcional, pero ahorra puntos cuando el robot ve lejos.
- **Fallback si se termina la ruta sin N víctimas**:
  1. revisar candidatos descartados con pocas observaciones;
  2. ruta de cámara solo sobre las celdas que el `detector_scan` todavía no tuvo en el FOV, con la misma función de
     set cover usando un sector de 58° en vez de 360°;
  3. si se acaba el tiempo, volver.
- Usen `networkx` (`nx.approximation.traveling_salesman_problem`) solo si son más de ~20 puntos. Para 8 a 12, el
  2-opt casero es exacto en la práctica y no agrega dependencias.

### P3. Candidatos por lidar (objetos no mapeados) e inspección dirigida con la cámara

**Qué es.** Todo grupo de puntos del `/scan` que cae en celdas libres del mapa (lejos de paredes) es un candidato:
víctima, obstáculo azul o distractor. La misión los persigue y **apunta la cámara** a cada uno hasta clasificarlo.

**Por qué.** El lidar ve los 360° y hasta 12 m. La cámara ve 61° y, con precisión, ~3 m. Las cajas están puestas
"para que no queden a la vista desde el spawn", pero el lidar las ve por los pasillos mucho antes que la cámara. Hay
**4 o 5 objetos no mapeados por mundo**, así que verificar cada uno cuesta ~3 a 6 s y como mucho ~25 s en total. Y
cuando se encontraron N rojos se corta (P0), muchas veces antes de terminar la ruta.

**Esfuerzo.** 3 h. **Riesgo.** Medio: en el real, las imperfecciones del mapa dan candidatos falsos (ver filtros).

**Cómo:**
- **Detección sin código nuevo**: una segunda instancia de `obstaculos_no_mapeados` en el launch, con
  `name='detector_no_mapeados'`, `umbral_probabilidad: 50.0` y
  `remappings=[('scan_rojo', '/scan'), ('objetos_rojos', 'objetos_no_mapeados'), ('objetos_rojos_markers', 'no_mapeados_markers'), ('scan_no_mapeado', 'scan_no_mapeado_todo')]`.
- **Cambio chico en `detector_obstaculos.py`** (sirve también para P5): parámetro `margen_pared` (0.08 a 0.10 m).
  En `recibir_mapa`, calcular `self.dist = ndimage.distance_transform_edt(campo < 50) * res`. En `recibir_scan`, tratar
  como "mapeado" todo punto con `dist[fila, col] < margen_pared`. Sin esto, con 3 cm de error de amcl y celdas de 5 cm,
  los rayos que pegan en las paredes caen en celdas libres y aparecen rachas falsas a lo largo de las paredes.
- **Seguimiento en la misión**: suscribirse a `objetos_no_mapeados` (`PoseArray` en `map`) y asociar cada detección al
  candidato a menos de 0.35 m, o crear uno nuevo. El candidato es válido con ≥3 detecciones.
  Estados: `PENDIENTE → ROJO | DESCARTADO`.
- **Clasificación**: en cada ciclo, para cada candidato pendiente, ver si está **en el FOV**:
  `|ángulo(cámara→candidato) − yaw_cámara| < 25°`, distancia < 2.5 m y línea de vista libre en el mapa.
  - Si junta ≥7 ciclos en el FOV (~1 s) sin puntos de `objetos_rojos` a menos de 0.3 m → `DESCARTADO`.
  - Si hay ≥3 puntos rojos cerca → `ROJO`, y se queda mirando hasta que `juntar_victimas` reporte (o ~3 s más).
- **`siguiente_objetivo()`**:

```python
def siguiente_objetivo(self):
    pendientes = [c for c in self.candidatos if c.estado == 'PENDIENTE' and c.intentos < 2]
    if pendientes:
        c = min(pendientes, key=lambda c: self.costo_geodesico(robot, c))
        if self.distancia(robot, c) < 2.5 and self.linea_de_vista(robot, c):
            return ('GIRAR', yaw_hacia(c))          # no hace falta moverse: solo girar
        p = punto_de_inspeccion(c)  # celda segura a 0.7-1.5 m con línea de vista, mínimo costo geodésico
        return ('IR', p.x, p.y, yaw_hacia(c))
    while self.puntos and self.ya_cubierto(self.puntos[0]):
        self.puntos.pop(0)
    return ('IR', *self.puntos[0]) if self.puntos else self.fallback()
```

- **Girar hacia el candidato**: el `yaw` de la meta de Nav2 se **ignora** (`yaw_goal_tolerance: 3.15` y sin el crítico
  `RotateToGoal`). Opciones:
  - (a) Recomendada: con el robot detenido y **sin objetivo de Nav2 activo**, publicar en `/cmd_vel` un control
    proporcional sobre el yaw de la tf `map → base_footprint`: `wz = clip(1.5·err, ±1.0)` hasta |err| < 0.2 rad.
    Lazo cerrado con amcl, así que no le afecta que el giro rinda 65 %.
  - (b) La acción `spin` de `behavior_server`. Usa la odometría, que con patinamiento **gira de menos**, así que
    probablemente haya que repetirla.
- **Prioridad por forma** (opcional, después): las víctimas son cajas cuadradas de 0.18 a 0.25 m; los distractores
  tienen "otras formas". El ancho de la racha da un indicio para ordenar, no para descartar.

### P4. Navegación más rápida y sin trabas (Nav2/DWB)

**Contexto.** Holonómico ya se compara bien con girar (−25 % de tiempo), así que me quedo con DWB y ajusto lo que hoy
pierde tiempo. Todo esto es `params/` o launch.

**(a) BT propio con recuperaciones útiles.** Esfuerzo 1 h, riesgo bajo, ahorro de **5 a 15 s por cada falla**.
El BT por defecto de Humble (`navigate_to_pose_w_replanning_and_recovery.xml`) hace, en ronda:
`ClearCostmaps → Spin 1.57 → Wait 5 s → BackUp 0.30 m a 0.05 m/s` (6 s), con 6 reintentos.
`Wait 5` es tiempo muerto, y `Spin` en un pasillo con el footprint tocando la pared puede chocar o fallar.
Además, según un comentario del fork más avanzado, el robot real "no se mueve por debajo de ~0.15 m/s" desde parado
(pruebas de los organizadores). Es un dato secundario, pero vale verificarlo: a 0.05 m/s el BackUp quizá ni se mueve.

```xml
<RecoveryNode number_of_retries="3" name="NavigateRecovery">
  <PipelineSequence name="NavigateWithReplanning">
    <RateController hz="2.0"> ... ComputePathToPose ... </RateController>
    <RecoveryNode number_of_retries="1" name="FollowPath">
      <FollowPath path="{path}" controller_id="FollowPath" goal_checker_id="{goal_checker}"/>  <!-- o fijo -->
      <ClearEntireCostmap service_name="local_costmap/clear_entirely_local_costmap"/>
    </RecoveryNode>
  </PipelineSequence>
  <ReactiveFallback name="RecoveryFallback">
    <GoalUpdated/>
    <RoundRobin name="RecoveryActions">
      <Sequence>
        <ClearEntireCostmap service_name="local_costmap/clear_entirely_local_costmap"/>
        <BackUp backup_dist="0.15" backup_speed="0.15"/>
      </Sequence>
      <Sequence>
        <ClearEntireCostmap service_name="global_costmap/clear_entirely_global_costmap"/>
        <Spin spin_dist="0.5"/>
      </Sequence>
    </RoundRobin>
  </ReactiveFallback>
</RecoveryNode>
```

Guárdenlo en `src/equipo_jar/behavior_trees/navegar.xml`, agreguen
`('share/' + package_name + '/behavior_trees', glob('behavior_trees/*.xml'))` a `setup.py` y pásenlo desde el launch:
`{'default_nav_to_pose_bt_xml': os.path.join(pkg_equipo, 'behavior_trees', 'navegar.xml')}`. Tiene que ser una ruta
absoluta, así que no puede ir en `params/`. Con pocos reintentos, Nav2 aborta antes y la **misión** decide: destrabe
lateral y reintento, o saltar al siguiente punto.

**(b) Destrabe lateral propio (aprovecha el mecanum).** Esfuerzo 1 h, riesgo bajo.
"No valid trajectories" aparece cuando el footprint **ya** toca una celda letal: `ObstacleFootprint` invalida toda
trayectoria cuyo footprint toque una celda letal, **incluida la pose inicial**. `BackUp` solo retrocede en x y no sirve
si la pared está al costado. Cuando un objetivo aborta: tomar el `/scan`, buscar el punto más cercano a menos de
0.35 m y publicar en `/cmd_vel` la velocidad **opuesta**, 0.15 m/s durante 0.7 s (~10 cm, también de costado). Después,
limpiar el costmap local (servicio `local_costmap/clear_entirely_local_costmap`) y reintentar.
**Ojo**: `laser_link` está montado con `rpy = 0 0 π` (`rosmaster_x3.urdf.xacro`), así que el ángulo del scan en
`base_link` es `ángulo + π`. Mejor transformar con tf. Publicar en `/cmd_vel` **solo** cuando no hay meta de Nav2
activa.

**(c) Encadenar objetivos sin frenar.** Esfuerzo 0.5 h, ahorra **~1 a 1.5 s por punto de vista** (desacelerar, 0.1 m de
tolerancia, arrancar el BT y planificar): unos 10 a 15 s por corrida. En `al_feedback`, cuando
`distance_remaining < 0.5` y el objetivo es un punto de cobertura (no una inspección ni la largada), mandar ya el
siguiente objetivo. `bt_navigator` hace *preemption* y el robot sigue de largo. La alternativa,
`NavigateThroughPoses`, replanifica a 0.333 Hz y usa `RemovePassedGoals` con radio de 0.7 m; sirve, pero no se puede
interrumpir tan fácil para inspeccionar. Prefiero `NavigateToPose` encadenado.

**(d) Costmap local más fino y menos padding.** Esfuerzo 15 min, riesgo bajo.
Hueco de 0.4 m: cada borde se marca en la celda que contiene el punto, así que se pierde hasta una celda de 0.05 m por
lado y quedan ~0.30 m. El robot mide 0.20 de ancho + 2·0.02 de padding = 0.24 m: quedan **3 cm por lado** yendo de
frente, y **no entra de costado** (0.30 + 0.04 = 0.34 m). Con `local_costmap.resolution: 0.025` y
`footprint_padding: 0.01` (en los dos costmaps) se recuperan ~5 cm. Una ventana de 3×3 m a 0.025 son 120×120 celdas,
nada para la CPU. El global puede quedar en 0.05.

**(e) Progress checker y goal checkers.** Esfuerzo 30 min.
- `progress_checker.plugin: nav2_controller::PoseProgressChecker` (está en el Humble instalado, en
  `nav2_controller/plugins.xml`) con `required_movement_radius: 0.3`, `required_movement_angle: 0.5` y
  `movement_time_allowance: 6.0`. Así un giro cuenta como progreso.
- Dos goal checkers: `general_goal_checker` con `xy_goal_tolerance` 0.20 para cobertura (no hace falta precisión) y
  `preciso` con 0.05 para la vuelta. En el BT se elige con `GoalCheckerSelector` (existe en Humble), o con dos XML,
  porque `NavigateToPose.Goal.behavior_tree` acepta una ruta de BT por objetivo.
- Si usan dos goal checkers, `FollowPath` **tiene que** nombrar `goal_checker_id`; si no, falla.
- `controller_server.failure_tolerance`: de 0.3 a 0.5, para tolerar instantes sin trayectoria válida.

**(f) Costos de inflación para centrar el camino.** Esfuerzo 30 min de prueba.
Hoy: `inflation_radius` 0.35 y `cost_scaling_factor` 5. En un pasillo de 0.8 m el centro está a 0.4 de cada pared,
fuera del radio: costo 0 en una franja de ±5 cm, así que el plan de NavFn puede pegarse a los bordes de esa franja.
La guía de ajuste de Nav2 recomienda un radio grande con una caída suave. Prueben `inflation_radius` 0.45 y
`cost_scaling_factor` 6 a 8 en el **global**, para que el camino vaya centrado y DWB no tenga que esquivar. En el local
pueden dejarlo como está.

**(g) Críticos de DWB.** `ObstacleFootprint.scale: 0.02` casi no pesa frente a `PathDist`/`GoalDist` (24-32). Si el
camino global va centrado (f), no hace falta tocarlo. Si siguen rozando, prueben 0.1 a 0.3 de a poco: con mucho más, el
robot no se anima a pasar por los huecos de 0.4 m.

**(h) Diagonal.** En el sim la diagonal rinde **~0.6 m/s** contra ~0.4 de frente. `max_speed_xy: 0.707` ya lo permite,
pero cruzado el footprint mide ~0.35 m y no pasa por los huecos. No lo fuercen; solo para pasillos rectos y libres,
y si sobra tiempo.

**(i) MPPI (`nav2_mppi_controller`, en Humble y en la lista permitida).** Con `motion_model: Omni` y `CostCritic`
`consider_footprint: true` pasa mejor por huecos angostos, y el fork más avanzado lo usa. Pero es un cambio grande que
hay que ajustar desde cero. **Solo si a la tarde del 2/11 DWB sigue trabándose en huecos.** RPP no sirve (no es
holonómico) y RotationShim no aporta con mecanum.

**(j) Collision Monitor (`nav2_collision_monitor`).** Esfuerzo 2 h, riesgo medio (remapear `cmd_vel` del controller y
del behavior_server). Polígono `slowdown` frente al robot. Evita algún −50 por inercia, pero no es prioritario si
(b) y (d) andan.

### P5. Detección de víctimas más rápida y robusta

**(a) Bajar `detecciones_minimas` de 30 a 10.** Esfuerzo 5 min, en `params.yaml`.
Cada scan aporta como mucho **una** detección por caja: 30 a 7.2 Hz son **4.2 s con la caja a la vista**. Pasando por
un pasillo, una caja queda a la vista ~2 s, así que hoy **no se reporta** salvo que el robot se pare. Con 10 son ~1.4 s.
El riesgo (reportar algo malo) se cubre con (b) a (e), y la mediana es robusta a los atípicos.
Ganancia: ~3 s por víctima, y sobre todo, no perder víctimas al pasar.

**(b) tf en el instante del scan.** Esfuerzo 20 min.
`detector_obstaculos.obtener_transform_a_map` y `juntar_victimas.posicion_robot` usan `Time()`, la última tf
disponible. A 1 rad/s, 0.13 s de desfase son 7.5°, y a 2 m eso **corre el punto ~26 cm**: el límite de los 100 puntos.
Usar
`lookup_transform('map', msg.header.frame_id, msg.header.stamp, timeout=Duration(seconds=0.1))` y, si falla, descartar
ese scan.

**(c) No contar detecciones mientras gira rápido.** Esfuerzo 20 min.
Suscribirse a `/odom` y descartar el scan rojo si `|wz| > 0.4 rad/s`. El workshop 10 dice textual que los puntos rojos
"se corren por la pared durante los giros". Ese corrimiento forma grupos falsos sobre las paredes, que con
`detecciones_minimas` bajo podrían llegar a reportarse.

**(d) Sincronizar imagen y scan.** Esfuerzo 30 min. `detector_scan` usa `self.ultima_imagen` sin mirar su stamp.
Guarden un `deque` de 5 imágenes con su stamp y usen la más cercana a `scan.stamp + scan_time/2`. Si la diferencia es
de más de 0.1 s, salteen el scan.

**(e) Filtro de paredes y de rango.** El `margen_pared` de P3, aplicado al camino rojo: descartar puntos rojos a menos de
0.08 a 0.10 m de una pared del mapa. Hoy, con `umbral_probabilidad` 101, pasa todo. Descartar también los de más de
3.5 m: a esa distancia 1° de error de amcl son 6 cm, y con la sincronización imperfecta, más. Una víctima apoyada en
una pared igual tiene su cara frontal a ≥0.18 m de la pared, así que no se pierde.

**(f) Erosionar la máscara roja.** Esfuerzo 5 min. `cv2.erode(mascara, np.ones((3, 3)))` en `detector_scan`. Los
rayos que rozan el borde de la caja (y siguen a la pared de atrás) dejan de contar como rojos.

**(g) Chequeo de dispersión antes de reportar.** En `quizas_publicar`, exigir además que la mediana de las distancias a
la mediana del grupo sea < 0.06 m. Un grupo "corrido" por una pared es alargado y no pasa.

**(h) Gastar los reportes que sobran.** Esfuerzo 30 min, ganancia potencial de +50 o +100.
Un reporte errado da 0 y no resta: el único costo es el lugar. Antes de `/done`, o a los ~9:30 si no se volvió, la
misión pide a `juntar_victimas` (por un tópico interno, por ejemplo `equipo/vaciar_reportes` con `std_msgs/Empty`) que
publique los **N − publicados** grupos no reportados con más puntos, aunque tengan menos de `detecciones_minimas`
(mínimo 3). Si una víctima se vio de pasada, suma.

**(i) Sacar el nodo `detector` del launch.** Esfuerzo 2 min. Publica `/rojo_detectado` y `/mascara_rojo`, que nadie
usa, y en `procesar_imagen` llama **dos veces** a `hay_cuadrado_rojo()` en cada ciclo: convierte a HSV y publica la
máscara al doble de lo necesario. Es CPU que les falta a amcl, DWB y `detector_scan`, sobre todo en la computadora de
la organización.

**(j) Centro de la caja.** Hoy es el promedio de la cara más `mitad_caja` en la dirección del robot: error de unos
4 cm si se ve en esquina. Alcanza (el umbral es 0.25 m); no lo toquen. `mitad_caja` **tiene que** cambiar a 0.09 en
el real (cajas de ~18 cm), y por eso está bien que esté en `params/`.

**Nube RGB-D (`/cam_1/depth/color/points`)**: no la usen. El reglamento advierte latencia, el lidar ya da la distancia
con 5 mm de ruido y el perfil `physical` del sim mete NaN por debajo de 0.6 m y error de escala.

### P6. Todo lo ajustable, a `params/`

**Qué es.** Que cada número que pueda cambiar en el real sea un parámetro en `params/params.yaml` o
`params/nav2_params.yaml`, no una constante en el código.

**Por qué.** `params/` "es la única carpeta que se puede modificar después del cierre de entregas". La corrida es con el
**robot físico**: cajas de otro tamaño, otra iluminación (HSV) y patinamiento.

Lista: rangos HSV de `detector_scan` (hoy son `declare_parameter` pero el launch no les pasa `params.yaml`, así que
**hay que agregar `params` a `parameters=[...]` de `detector`/`detector_scan`/`detector_obstaculos`**), `mitad_caja`,
`detecciones_minimas`, `margen_pared`, alcance del lidar para cobertura, despeje mínimo, umbrales de inspección,
tiempo límite y margen de vuelta, velocidades de destrabe.

`verificar_params.py` no admite listas anidadas ni listas de 4 números o más, y avisa por nombres con `victim`,
`coord`, `waypoint` o `posicion_`. Usen nombres como `distancia_inspeccion_max`.

### P7. Riesgos de reglamento y del CI

- **Dependencias**: si usan scipy (`ndimage`, `csgraph`), agreguen `<exec_depend>python3-scipy</exec_depend>` a
  `package.xml` (está en `dependencias_permitidas.yaml`). Igual con `python3-numpy`, `python3-opencv`, `cv_bridge`,
  `tf2_ros`, `nav2_msgs`, `action_msgs` y `std_srvs`, si los importan. Nada de pip.
- **Tópicos**: los internos están permitidos, salvo que la plataforma los use (el auditor compara contra el grafo base).
  Por eso conviene remapear el `initialpose` de amcl a un nombre propio (P1).
- **QoS**: `/victimas`, `/done` y `/cmd_vel` **reliable** (lo es el `create_publisher(..., 10)` por defecto). Sensores
  con `qos_profile_sensor_data`. Un suscriptor nuevo a `/odom` (P5c) también va best effort, por las dudas: un
  suscriptor best effort se conecta tanto a un publicador reliable como a uno best effort.
- **Espera del mapa**: el CI corre el launch sin simulador y verifica que no se mueva ni reporte antes de `/map`. La
  misión no debe publicar `/cmd_vel` ni la pose inicial hasta recibir `/map`. `juntar_victimas` ya respeta eso, porque
  sin tf no hace nada.
- **60 s quieto**: el juez lo mide con la velocidad de `/odom`. Los riesgos son el cálculo inicial (~2 s, OK), los
  rechazos repetidos de Nav2 y quedar trabado. Solución: el watchdog de P0. **Después de `/done` no muevan más el
  robot**: cancelen la meta y publiquen un `Twist()` cero.
- **Hardcodeo**: el footprint `[[0.12, 0.10], ...]` de `nav2_params.yaml` dispara la advertencia de "3+ pares de
  coordenadas" en `verificar_estatico.py`. Es solo un aviso para el juez técnico; un comentario aclarando que es el
  chasis del URDF ayuda.
- **`/done` sin víctimas**: el +100 exige al menos una víctima puntuada. Igual conviene publicar `/done` al volver,
  porque corta el reloj del desempate.
- **Fecha**: las dependencias nuevas se piden hasta el **lunes 2/11 a las 23:59**. Con este plan no hace falta ninguna.

### P8. Qué hacen otros equipos (solo como inspiración)

`gh` no está instalado en esta máquina; usé la API pública de GitHub por `curl`.
El leaderboard https://airclub-udesa.github.io/jar_challenge/ todavía dice "Todavía no hay corridas en verde", así que
no hay puntajes para comparar. Forks públicos de `AIRclub-UdeSA/jar_challenge`: 13. La mayoría sigue con el template
(fierro, TomasD1az, Daniel-Espinn, changuitron, etc.).

- **Emiliano-Aguilera/jar_challenge, rama `development`**: el más trabajado.
  https://github.com/Emiliano-Aguilera/jar_challenge/tree/development
  - Cobertura **con la cámara**: candidatos cada 0.45 m, 8 rumbos, FOV 61°, alcance 2 m, *set cover* y 2-opt. Es
    nuestra alternativa B; los números de arriba sugieren que la de lidar es más rápida.
  - Fusión de candidatas con covarianza, que es un filtro de Kalman estático (`candidate_fusion.py`), e "inspección"
    de candidatas dudosas.
  - **MPPI Omni** con `CostCritic.consider_footprint`, `PathAngleCritic` con `max_angle_to_furthest: 0.35`, para
    "llegar alineado a los huecos".
  - `PoseProgressChecker`, dos goal checkers (uno preciso de 0.05 para la vuelta) y BackUp a 0.15 m/s.
  - Corrección de rumbo con el IMU por el patinamiento al girar en el real (`rumbo_imu.py`, `odometria_imu.py`).
  - Lo que confirma: el giro en el real es el punto débil, de ahí que convenga girar poco (P2, P3) y con lazo cerrado
    sobre amcl.
- **bautipeirone/jar_challenge**: enfoque reactivo propio (evasor, localizador y `gestor_victimas`), sin Nav2. No
  aporta ideas nuevas para nosotros.

### P9. Pruebas que conviene correr (cuando el sim esté libre)

1. Los 5 mundos con `victimas` correcto, con `spawn_x/spawn_y/spawn_yaw` desviados (±0.2 m, ±0.6 rad): mide P1.
2. Registrar en cada corrida: tiempo hasta cada reporte, error de cada reporte (lo da `resultado.json` del juez),
   tiempo total hasta `/done` y distancia final a la largada.
3. **Choques**: en el sim no se detectan solos. Graben la distancia mínima entre el scan y el footprint; por debajo de
   ~1 cm cuenten un probable choque. Además se puede revisar la trayectoria con `tools/graficar_corrida.py`.
4. `tools/verificar.sh` antes de cada push.

---

## Tabla resumen

| # | Mejora | Ganancia estimada | Esfuerzo | Riesgo |
| --- | --- | --- | --- | --- |
| P0 | Misión con ActionClient, corte en N, vuelta y `/done` | de 0 a hasta 100·N + 300 | 4-5 h | bajo |
| P1 | Pose inicial con covarianza + `nomotion_update` | evita perderse con el robot mal puesto: todo o nada | 1 h | bajo |
| P2 | Ruta de cobertura con lidar (set cover + 2-opt) | 40-100 s frente a cámara o giros | 3 h | bajo-medio |
| P3 | Candidatos por lidar + giro dirigido de la cámara | encuentra cajas antes; ≤25 s de desvíos | 3 h | medio |
| P4a-e | BT propio, destrabe lateral, encadenar objetivos, costmap fino, progress checker | 10-30 s y menos trabas | 3 h | bajo |
| P5a-i | Detección: 10 detecciones, tf al stamp, filtro de giro y paredes, reportes sobrantes | ~3 s por víctima, sin pérdidas al pasar; +50/+100 posibles | 2.5 h | bajo |
| P6 | Umbrales a `params/` | ajuste en el real después del cierre | 1 h | bajo |
| P4f-j | Inflación, críticos, MPPI, collision monitor | marginal o incierto | 1-6 h | medio-alto |

## Fuentes

- Reglamento oficial: https://airclub-udesa.github.io/jar_site/reglamento/
- Workshops: https://airclub-udesa.github.io/jar_site/workshops/,
  https://airclub-udesa.github.io/jar_site/workshops/semana-09-nav2/,
  https://airclub-udesa.github.io/jar_site/workshops/semana-10-sim-a-real/
- Leaderboard: https://airclub-udesa.github.io/jar_challenge/
- Forks: https://api.github.com/repos/AIRclub-UdeSA/jar_challenge/forks ;
  https://github.com/Emiliano-Aguilera/jar_challenge/tree/development
- AMCL de Humble (pose inicial con covarianza cero, `request_nomotion_update`):
  https://github.com/ros-navigation/navigation2/blob/humble/nav2_amcl/src/amcl_node.cpp ;
  https://api.nav2.org/nav2-humble/html/classnav2__amcl_1_1AmclNode.html
- Simple Commander de Humble (`waitUntilNav2Active` publica `initialpose`):
  https://github.com/ros-navigation/navigation2/blob/humble/nav2_simple_commander/nav2_simple_commander/robot_navigator.py ;
  https://docs.nav2.org/commander_api/index.html
- Nav2: guía de ajuste https://docs.nav2.org/tuning/index.html ; DWB
  https://docs.nav2.org/configuration/packages/configuring-dwb-controller.html ; MPPI
  https://docs.nav2.org/configuration/packages/configuring-mppic.html ; BT Navigator
  https://docs.nav2.org/configuration/packages/configuring-bt-navigator.html ; Collision Monitor
  https://docs.nav2.org/configuration/packages/configuring-collision-monitor.html ; nodos del BT
  https://docs.nav2.org/configuration/packages/configuring-bt-xml.html
- Exploración por fronteras: B. Yamauchi, "A frontier-based approach for autonomous exploration", CIRA 1997;
  https://github.com/robo-friends/m-explore-ros2
- Archivos locales: `/opt/ros/humble/share/nav2_bt_navigator/behavior_trees/*.xml`,
  `/opt/ros/humble/share/nav2_controller/plugins.xml`, `rosmaster_ws/src/yahboom_rosmaster/...` (URDF, mundos y mapas),
  `rosmaster_ws/src/jar_workshops/semana-08-cobertura-mapa/` (raycast con Bresenham, A*).

## Hoja de ajuste en la arena (solo `src/equipo_jar/params/`)

| Qué medir | Parámetro (archivo) | Valor actual | Alternativa |
|---|---|---|---|
| Lado de la caja | `mitad_caja` (params) | 0.125 | lado/2 (cajas de 18 cm: **0.09**) |
| ¿El lidar ve la caja? | — | — | `ros2 topic echo /scan` frente a una caja |
| Rojo con la luz del lugar | `saturacion_min`, `valor_min`, `hue_rojo_alto_1`, `hue_rojo_bajo_2` | 120 / 80 / 10 / 170 | 90 / 50 / 12 / 165 (sin ensanchar el hue si hay distractores naranjas o rosas) |
| Desfase entre stamps de imagen y scan | `desfase_maximo_imagen` | 0.2 | 0.3–0.5; −1 si los relojes no coinciden |
| Bordes rojos falsos o caja lejana sin detectar | `erosion_mascara` | 1 | 2 / 0 |
| amcl ruidoso de lejos | `rango_maximo`, `dispersion_maxima` | 3.5 / 0.08 | 2.5 / 0.12 |
| Reportes lentos | `detecciones_minimas` | 12 | 8 |
| Zona muerta de motores | `min_speed_xy`, `min_speed_theta` (nav2), `giro_minimo_inspeccion` | 0.08 / 0.3 / 0.0 | 0.10 / 0.45 / 0.4 |
| Rebotes del propio robot en el lidar | `obstacle_min_range` (costmaps), `laser_min_range` (amcl) | 0.0 / −1 | 0.12 / 0.15 |
| Choques (−50 cada uno) | `max_vel_x` / `max_vel_y` / `max_speed_xy` | 0.5 / 0.5 / 0.707 | 0.35 / 0.35 / 0.5 |
| Margen de la vuelta | `velocidad_vuelta_estimada`, `margen_vuelta_s` | 0.2 / 30 | 0.15 / 45 |
| Muchos candidatos falsos en paredes | `margen_pared` (detector_no_mapeados), `inspecciones_maximas` | 0.15 / 15 | 0.20 / 10 (0.10 probado: 40 falsos en laberinto_1) |
| CPU ("Control loop missed its desired rate") | `vx/vy/vtheta_samples` | 15 / 11 / 9 | 9 / 7 / 7 |
| Localización lenta | `tiempo_localizacion_s` | 5 | 8 |

Si `install/` no se compiló con `--symlink-install`, cualquier cambio en `params/` necesita `colcon build --packages-select equipo_jar`.
