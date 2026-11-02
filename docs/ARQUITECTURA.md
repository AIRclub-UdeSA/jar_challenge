# Arquitectura

Este documento propone cómo partir el problema para trabajar en paralelo. Lo que ya está hecho
figura en la sección 0; el resto sigue siendo propuesta.

## 0. Estado actual

| Módulo | Estado | Cómo |
| --- | --- | --- |
| **Localización** | ✅ **Hecho** (PR #2) | AMCL (`nav2_amcl`) + `nodo_pose_inicial` |
| **Percepción** | ✅ Filtros de distractores y bordes, respaldo con odometría | `nodo_percepcion` (ver abajo) |
| **Navegación** | ✅ **Nav2 + RPP** | Smac + RPP, `nodo_arranque` y el escape de `escape.py` |
| **Misión** | ✅ Recorre, reporta, vuelve a la largada y publica `/done` | `nodo_mision`, `cobertura.py` |

`nodo_equipo` (seguir la pared) queda en el repo pero ya no se lanza: lo reemplaza `nodo_mision`.

### Localización (hecho)

- **AMCL** compara el `/scan` con `/map` y publica TF `map → odom`. Lo activa `nodo_arranque`
  (AMCL es un nodo con ciclo de vida de Nav2).
- **`nodo_pose_inicial`**, al recibir `/map`, manda por `/initialpose` (topic interno) una
  gaussiana centrada en el origen (σ en `params.yaml`). Se usa en lugar de `set_initial_pose` de
  AMCL, que arranca todas las partículas en el mismo punto (covarianza cero).
- **`nodo_mision`** lee la pose de TF `map → base_footprint`.
- **Medido en simulación** (`maze_1_6x5`, arranque en (0,2, −0,15) girado 0,3 rad): error de AMCL
  de 0,02 a 0,16 m y menos de 2°; `/odom` acumulaba 0,9 m y 17°. Reporte de V1 a 0,15 m del
  centro de la caja (100 puntos).
- **Para ajustar entre corridas** (son hiperparámetros): `alpha1` y `alpha2` (cuánto se desconfía
  del giro de las ruedas: subirlos si en el robot real la pose salta al girar), las σ de
  `nodo_pose_inicial` y `z_rand` (tolerancia a ventanas y a obstáculos que no están en el mapa).
- **Robot real (bag `minilaberinto` del 30/10, ya con la odometría corregida):**
  - Publica TF `odom → base_footprint` a ~10 Hz y los mismos frames que el simulador
    (`base_link`, `laser_link`, `imu_link`, `cam_1_*`). `map → odom` no está: lo pone AMCL.
  - El lidar está montado girado 180° (`laser_link`, yaw 180°), igual que en el simulador:
    `angulo_frente_deg: 180` vale también en el robot. Scan de 360° en 1080 rayos (0,33°).
  - El IMU está montado dado vuelta (eje z para abajo): su `angular_velocity.z` tiene el signo
    opuesto al giro del robot. Si alguien lo fusiona (robot_localization), usar TF, no el valor crudo.
  - Giro de `/odom` contra el giroscopio: 705° contra 725° en total (~3% menos); en giros
    sueltos, entre 0,81 y 1,08 veces lo real. Con `alpha1: 0.4`, AMCL supone un desvío de ~0,63
    veces el giro (`sqrt(alpha1)`, ver `omni_motion_model.cpp`): cubre de sobra ese error.

### Navegación (hecho)

- **Cadena del comando:** `controller_server` (RPP) → `velocity_smoother` → `/equipo/cmd_vel_nav` →
  `nodo_mision` → `/cmd_vel`. Las recuperaciones de Nav2 también van a `/equipo/cmd_vel_nav`.
  `nodo_mision` es el único que publica en `/cmd_vel`: corta a Nav2 cuando mira un objeto, escapa
  de una traba o terminó.
- **`nodo_arranque`** reemplaza al `lifecycle_manager`. Ese gestor espera sin límite cada respuesta
  y, si una se pierde, el robot no se mueve en toda la corrida. Pasaba en 3 de cada 30 corridas.
- **Planificador:** `SmacPlanner2D`. NavFn fallaba en pasos justos con *failed to create a plan from
  potential*: encontraba que había camino pero no lo podía trazar.
- **RPP sin `use_collision_detection`:** con la detección prendida, la huella real toca el costmap
  en huecos por los que el robot entra, y RPP no avanza (quedó en 0 puntos en `maze_4_metal_6x6`).
- **Trabas** (`escape.py`): tras 6 s quieto, o 15 s sin alejarse 0,15 m aunque gire, se cancela la
  meta y el robot sale despacio hacia donde el lidar ve más lugar, también de costado. Reintenta la
  meta una vez; a la segunda traba la abandona.
- **Medido en simulación** (corridas de 300 s): `maze_1_6x5` con esta configuración dio 400, 500,
  500 y 500. En los cuatro mapas, solo apagar la detección de RPP subió la media de 200 a 350.
- **Pendiente:** validar los cuatro mapas (la tanda de 16 corridas se cortó en la 2), bajar los
  contactos cerca de las paredes (hasta 8 por corrida contra el umbral de 2,5 cm del monitor) y
  probar en el robot real.
- **Depende de AMCL:** los resultados de arriba usaron AMCL con `alpha4: 0.02`, `sigma_hit: 0.1`,
  `z_hit: 0.8`, `z_rand: 0.2`, `max_beams: 180` y `update_min_d/a: 0.05`, que ya son los de
  `params.yaml`. Con los anteriores (`alpha4: 0.2`, `z_hit/z_rand: 0.5`, `max_beams: 60`),
  `maze_1_6x5` dio 0 puntos (error típico de AMCL: 0,74 m).

### Percepción

`nodo_percepcion` publica cada caja una sola vez en `/equipo/cajas_confirmadas`; `nodo_mision` la
reporta (tras `/done` no reporta más). Percepción mira también con el robot
quieto (`velocidad_lineal_min_deteccion: 0.0`), que es cuando `nodo_mision` frena frente a un
objeto para inspeccionarlo; no mira girando a más de `giro_max_deteccion`. Ubica cada mancha
con forma de caja de la imagen, no solo la más grande (si no, una caja ya confirmada tapaba a
las demás). Hacia dónde mira el lidar sale de TF, como en `nodo_mision` (`angulo_frente_deg`
queda solo de respaldo). Un error en un ciclo se loguea con su traceback y el nodo sigue. La
lógica está en módulos puros con tests:

- **`deteccion.py`**: color (HSV con límites de H para no tomar naranjas ni magentas, y limpieza
  de bordes finos), forma (relleno y proporción de caja, sin tocar ningún borde de la imagen),
  lidar (mediana del tramo central y rayos parejos: si no, es un borde), oclusión (algo más cerca
  justo al costado de la mancha) y cámara (profundidad, no rango, contra el alto; y la mancha donde
  estaría una caja apoyada en el piso).
- **`mapa.py`**: descarta un centro adentro de una pared o afuera del mapa, y una caja que en el
  `/map` queda detrás de una pared (eso pasa cuando la pose está mal).
- **`agrupador.py`**: junta las vistas por cercanía y publica la mediana; una vista mala no borra
  las buenas.
- **`supervisor_localizacion.py`**: la pose sale de AMCL mientras coincida con `/odom` (contra lo
  que decía AMCL en los últimos 2 m; la tolerancia de posición crece con lo avanzado y la de
  ángulo, también con lo girado: girar en el lugar no corre la posición de `/odom`). Si AMCL salta,
  deriva o deja de publicar, la pose sale de la referencia más vieja de la ventana + `/odom`, y
  vuelve a AMCL cuando coincide y está quieto 2 s. Probado con los logs de `pruebas/` (salto de
  3,4 m, deriva de 2 m en `maze_1_6x5`, deriva de 0,9 m girando en `maze_4_metal_6x6`). Antes de
  ubicar cajas espera el aviso de `nodo_mision` (ver abajo).

**Convergencia al arrancar: decide solo `nodo_mision`.** AMCL convergió cuando el robot se alejó
0,6 m de la largada y pasaron 8 s más (`distancia_convergencia_m`, `espera_convergencia_s`).
Lo avisa una vez en `/equipo/amcl_convergido` (`std_msgs/Empty`, interno, reliable + transient
local, depth 1, como `/map`), y el supervisor no toma su primera referencia hasta recibirlo.
Antes, el supervisor le creía a AMCL apenas su covarianza bajaba de 0,12 m y 6°, con el robot
quieto. En `maze_4_metal_6x6` (2026-11-01, pose real del simulador): AMCL arrancó con 18° de
error, las actualizaciones sin movimiento lo bajaron a 5,5°, la covarianza ya daba < 6° y se
quedó así hasta que el robot se movió (en 0,5 m bajó a < 1°). Con eso, una caja a 2,7 m quedó
ubicada 0,29–0,32 m mal (50 puntos en vez de 100): la covarianza mide cuánto se dispersan las
partículas, no cuánto se equivocan, y las 5 vistas desde el mismo lugar comparten el mismo error.
Las cajas en espera de `nodo_mision` quedan como red de seguridad.

**Inspecciones: no ir a mirar lo que la cámara ya vio sin rojo.** `nodo_mision` va a mirar de cerca
(inspección, 4 s quieto) cada objeto que el lidar ve fuera del `/map`. En 21 corridas de
`pruebas/` (2026-11-01), 82 de 120 inspecciones fueron a objetos que no son víctimas: 41 a cajas
azules distractoras y 41 a "fantasmas", pedazos de pared a 21–35 cm de una pared del `/map` que en
el mapa faltan o están corridos (siempre en los mismos lugares). Las de fantasmas tardaron 15 s de
media y causaron 11 de las 18 trabas. En `maze_3_6x6`, las corridas con 4 a 7 fantasmas pasaron
150–180 s inspeccionando, lo visto quedó estancado entre 54 % y 59 % de 60 s a 200 s y la última
caja llegó después de los 5 min (400 puntos); la que tuvo 2 fantasmas inspeccionó 47 s (600).

Ahora percepción anota qué miró la cámara (`memoria_camara.py`): con cada imagen, la celda del
`/map` donde pega cada rayo del lidar que cae adentro de la imagen, y si en la imagen había rojo
donde estaría una caja apoyada ahí (un solo píxel alcanza, y una celda que vio rojo queda así para
siempre). Lo publica en `/equipo/vistas_camara` (`nav_msgs/OccupancyGrid` con la geometría de
`/map`: 0 sin ver, 1–99 imágenes sin rojo, 100 rojo; reliable + transient local, depth 1, cada
0,5 s si cambió; se puede ver en RViz). Misión no inspecciona un objeto cuyas celdas (las que el
lidar vio fuera del mapa) la cámara ya vio sin rojo (`veces_sin_rojo` imágenes en
`celdas_sin_rojo` celdas). Si se entera en camino corta el viaje, y si se entera mirándolo deja de
mirar (también si el objeto ya es una caja confirmada). Si después aparece rojo, el objeto vuelve
a quedar para inspeccionar. Con `descartar_con_camara: false` se inspecciona todo, como antes. El
margen contra las paredes del mapa quedó en 0,20 m (`margen_pared_objetos_m`): agrandarlo sacaría
fantasmas, pero también cajas contra una pared, y el reglamento no dice que no pueda haberlas.

**Pendiente:** cuando AMCL se pierde, se lo podría reinicializar por `/initialpose` con la pose
del respaldo (es tema de localización).

## 1. Módulos

La misión (§1 del reglamento) se parte en cuatro problemas casi independientes:

| Módulo | Qué resuelve | Entra | Sale | Base de los workshops |
| --- | --- | --- | --- | --- |
| **Localización** ✅ | Dónde estoy en `/map`, arrancando *cerca* del origen | `/map`, `/scan`, `/odom` | TF `map → odom` | AMCL (09), ver sección 0 |
| **Percepción** | Ver cajas rojas y estimar su centro en `map` | cámara, `/scan`, TF | candidatos de víctima (topic interno) | 04 (`detector`, `detector_scan`) |
| **Navegación** | Ir a un punto sin chocar, incluidos obstáculos que no están en el mapa | `/map`, `/scan`, pose | `/cmd_vel` | 07 + 08 (A*) o Nav2 (09) |
| **Misión** | Qué recorrer, cuándo reportar, cuándo volver y publicar `/done` | todo lo anterior | `/victimas`, `/done`, metas | 08 (`explorador`, cobertura) |

Contratos a respetar entre módulos:

- **Un solo nodo publica en `/victimas`** (el de misión) y lleva la cuenta de los N reportes.
  Percepción publica *candidatos* en un topic interno (por ejemplo `/equipo/candidatos`); misión
  los agrupa y decide cuándo una víctima está confirmada.
- **Un solo dueño de `/cmd_vel`.** Si dos nodos lo publican, el robot tiembla.
- Todos los nodos se levantan desde `competencia.launch.py`, con `parameters=[params, ...]` y
  `use_sim_time`.

## 2. La decisión grande: ¿Nav2 o navegación propia?

| | **A. Nav2** (workshop 09) | **B. Propia** (workshops 06–08) |
| --- | --- | --- |
| Localización | AMCL, probado y robusto | **ya resuelto con AMCL** en las dos opciones (sección 0) |
| Obstáculos no mapeados | los ve el costmap local solo | hay que detectarlos y replanificar a mano (07) |
| Esfuerzo hasta "anda" | bajo: configurar YAML | alto: planificador + control + recuperación |
| Entender qué falla | difícil: muchas piezas y parámetros | fácil: es nuestro código |
| Imagen Docker | +`nav2_bringup` en `package.xml`, imagen más pesada | sin dependencias nuevas |
| Riesgo en el robot real | pasillos angostos: hay que afinar el radio del robot y los costmaps | depende de cuánto lo probemos |

Con ~10 días hasta la clasificación, **A** suele ser lo más seguro para "moverse sin chocar", y el
trabajo propio se concentra en percepción y misión, que es donde están los puntos. **B** da más
control, pero se come el tiempo en reinventar el planificador y las recuperaciones.

**Decisión: A, Nav2 con RPP** (estado en la sección 0). Sus dependencias están en `package.xml` y
sus parámetros en `params/params.yaml`.

## 3. Puntos de diseño que valen puntos

- **Precisión del reporte:** ≤ 0,25 m da 100, hasta 0,50 m da 50. Se mide contra el **centro** de la
  caja (0,25 × 0,25 m), no contra la cara que ve el LiDAR: sumar medio lado en la dirección del rayo.
- **No gastar reportes:** filtrar falsos positivos (bordes, paralaje) acumulando varias detecciones
  antes de confirmar. Distractores azules, amarillos y verdes: el umbral de rojo tiene que ser
  estricto.
- **Bono de velocidad (+200):** las N víctimas antes de los 5 minutos.
- **Bono de regreso (+100):** volver a ≤ 0,25 m de la largada y publicar `/done`. Guardar la pose
  inicial estimada al recibir `/map`.
- **Colisiones (−50 cada una):** márgenes de seguridad generosos antes que velocidad.

## 4. Reparto sugerido

Una persona (o pareja) por módulo, cada uno en su branch y su nodo, integrando en `main` por PR.
Lo primero en llegar a `main` debería ser el esqueleto de los cuatro nodos conectados por los topics
internos acordados, aunque no hagan nada: así cada uno puede probar el suyo dentro del launch
completo.
