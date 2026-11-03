# Qué loguear en la 1.ª corrida de la clasificación para ajustar la 2.ª

Clasificación del miércoles 4/11: dos corridas en la misma arena y cuenta la mejor. La organización
permite loguear datos. La idea es que la primera corrida sirva además como medición del robot real
en la arena real, y que entre las dos corridas se ajuste `params/`.

## 1. Restricciones que mandan

- **Después del cierre (martes 3/11, 9:00) solo se puede tocar `params/`.** El código, el launch y
  los nodos quedan fijos. Cualquier registro que necesite código nuevo tiene que estar en el PR
  antes de las 9:00. Lo que no esté, no se puede medir. Después solo se pueden cambiar valores.
- Entre corridas no sabemos cuánto tiempo hay. Hay que planear un análisis de **5 a 10 minutos**,
  con un script que lea los logs y tire un resumen, no mirar a mano.
- El robot y la arena son los mismos en las dos corridas. Las víctimas también, salvo que la
  organización las mueva. Todo lo que se mida (iluminación, tamaño de cajas, zona muerta de
  motores, CPU, errores de localización) vale para la segunda corrida.
- Cada cambio de parámetro que no se probó es un riesgo. Hay que cambiar **pocos parámetros y solo
  si el dato lo justifica**. Si la primera corrida salió bien (todas las víctimas, bono y vuelta),
  conviene **no tocar nada**, salvo que el tiempo sea malo y la causa esté clara.

## 2. Qué se loguea hoy sin hacer nada

`ros2 launch` guarda la salida de todos los nodos en `~/.ros/log/<fecha>-<host>-<pid>/launch.log`.
Copiar esa carpeta (o el archivo) después de la corrida ya da mucho:

| Fuente | Qué dice | Ejemplo de línea |
| --- | --- | --- |
| `nodo_equipo` | Línea de tiempo: estados, localización, puntos de vista, inspecciones, cortes, destrabes, vuelta, `/done` | `[ 90.6 s] EXPLORANDO -> VOLVIENDO` |
| `nodo_equipo` | Convergencia de amcl al arrancar y su dispersión durante la corrida (cada ~15 s) | `Localizado (convergió): sigma xy 0.047 m` |
| `nodo_equipo` | Cada inspección: desde dónde, a qué candidato, error de rumbo al terminar de girar y resultado | `Candidato (2.02, -0.92): no es rojo (0 detecciones rojas)` |
| `nodo_equipo` | Atascos: `Atascado en ...`, `sin desplazarse`, `Nav2 entró en recovery`, `No se llegó a ...` | |
| `juntar_victimas` | Cada reporte (posición publicada) y grupos frenados por dispersión | `Publicada víctima 1/2 en (-0.15, -1.98)` |
| `controller_server` | `Control loop missed its desired rate` (CPU), `No valid trajectories` | |
| `planner_server` | `Starting point in lethal space`, `no valid path found` | |
| `detector_scan` | Aviso si imagen y scan están desfasados | `Imagen y scan desfasados 0.35 s` |

Con eso solo ya se puede contar cuántos candidatos falsos hubo, dónde se trabó el robot, si amcl
tardó en converger y si la CPU no da abasto. **Hay que confirmar con la organización que nos dejan
copiar `~/.ros/log` de su computadora después de la corrida.**

## 3. Qué falta medir y para qué parámetro sirve

Ordenado por cuánto puede mover el puntaje en la segunda corrida.

### 3.1 Detección de rojo (lo que más puede costar: una víctima sin ver son 100 + 200 puntos)

- **Qué medir:** el color real de las cajas rojas con la luz de la arena. Histograma de H, S y V de
  los pixeles rojizos, unas pocas imágenes (1 cada 5-10 s, achicadas) y, por cada scan, cuántos
  rayos dan rojo.
- **Para qué:** `hue_rojo_*`, `saturacion_min`, `valor_min` y `erosion_mascara` (`detector_scan`).
  Si las cajas reales salen con S ≈ 90 y nuestro mínimo es 120, no se detecta nada. Es el riesgo
  número uno del robot real, y en el sim no lo podemos ver.
- **También:** que no haya distractores (naranja, rosa) que entren en el rango.
- **Cómo decidir:** el percentil 5 de S y V de los pixeles de la caja pasa a ser el mínimo (con un
  margen de −10). El hue se ensancha solo si el de la caja queda afuera, y no tanto como para
  tomar el naranja.

### 3.2 Tamaño de caja y sesgo de los reportes

- **Qué medir:** la posición de cada reporte (ya se loguea) y el ancho de la cara de la caja que ve
  el lidar (los puntos de `scan_rojo` de un mismo grupo). Si la organización publica el error de
  cada reporte, mejor todavía.
- **Para qué:** `mitad_caja` (`juntar_victimas`): hoy es 0.125, para las cajas del sim (25 cm); las
  reales son de ~18 cm → **0.09**. Con el ancho medido se confirma.
- Si el reporte puntuó +50 (entre 0.25 y 0.50 m) en vez de +100, el error es sistemático: revisar
  `mitad_caja` y la localización en ese momento.

### 3.3 Localización

- **Qué medir:** la pose y la covarianza de amcl (`/amcl_pose`) durante toda la corrida, cuánto
  tardó en converger al arrancar y la distancia final a la largada medida por amcl.
- **Para qué:** `dispersion_inicial_*`, `tiempo_localizacion_s`, `sigma_localizado_*`
  (`params.yaml`); `alpha1-4`, `sigma_hit`, `max_beams` (amcl en `nav2_params.yaml`). Si amcl salta
  o tiene σ > 0.15 m seguido, hay candidatos falsos sobre las paredes y reportes corridos.
- **Señal clara:** la vuelta a la largada. Si amcl dice que el robot está a 0.10 m y la
  organización dice que no, el error de amcl en la largada es la diferencia.

### 3.4 Movimiento real del robot (zona muerta, patinamiento, velocidad)

- **Qué medir:** `/cmd_vel` contra `/odom`, cada 0.2 s. Cuánto se mueve el robot cuando se le pide
  0.05-0.15 m/s y 0.2-0.4 rad/s. La velocidad real de crucero. El giro real contra el pedido.
- **Para qué:**
  - `min_speed_xy` / `min_speed_theta` (DWB) y `giro_minimo_inspeccion`: si el robot no se mueve
    por debajo de cierta velocidad, hay que subirlos.
  - `velocidad_destrabe`: misma lógica.
  - `velocidad_vuelta_estimada` (para decidir cuándo volver).
  - `max_vel_*` si hubo choques.

### 3.5 Roces y choques (−50 cada uno)

- **Qué medir:** la distancia mínima entre el scan y el rectángulo del robot, por scan, con la pose
  y el estado de la misión. Cada episodio a menos de 2 cm es un choque probable. Ya lo usamos en
  el sim (`roces.py`).
- **Para qué:** `footprint_padding` (los dos costmaps), `inflation_radius` y `cost_scaling_factor`
  del costmap local, velocidades máximas. Si los choques son siempre en la misma situación (por
  ejemplo, al rodear puntas de pared), ajustar `tiempo_atasco_s` / `distancia_subobjetivo`.

### 3.6 CPU de la computadora de la organización

- **Qué medir:** cuántas veces aparece `Control loop missed its desired rate` (ya está en el log de
  launch), la frecuencia real de `/cmd_vel` y la carga de CPU (`/proc/loadavg` cada 5 s).
- **Para qué:** `vx_samples`, `vy_samples`, `vtheta_samples`, `controller_frequency` (DWB),
  `max_beams` (amcl). En el sim, con el equipo limitado a 2 núcleos, aparecían 200-470 avisos por
  corrida y una vez se perdió la vuelta.

### 3.7 Candidatos falsos y tiempos de inspección

- **Qué medir:** la posición de cada candidato, su distancia a la pared más cercana del mapa, si
  era rojo o no, y cuánto tardó cada inspección (ya se loguea casi todo).
- **Para qué:** `margen_pared` y `racha_minima` (`detector_no_mapeados`), `detecciones_candidato`,
  `despeje_candidato_claro`, `inspecciones_maximas`, `tiempo_mirar_s`. En el sim los falsos estaban
  a 0.10-0.20 m de una pared y las cajas reales a más de 0.45 m. Si en la arena es parecido, subir
  `despeje_candidato_claro` o `margen_pared` ahorra tiempo.

### 3.8 Tiempos de la corrida

- **Qué medir:** el tiempo de cada fase (localizar, explorar, inspeccionar, volver) y en qué segundo
  sale cada reporte. Ya está en el log.
- **Para qué:** si se pierde el bono de 5 minutos por poco, mirar dónde se fue el tiempo
  (inspecciones falsas, atascos) antes de tocar velocidades.

## 4. Cómo loguearlo

Hay dos opciones y conviene tener las dos.

**A. Nodo `registro` propio, dentro de nuestro launch (hay que hacerlo antes de las 9:00).**
Se suscribe a `/amcl_pose`, `/cmd_vel`, `/odom`, `/scan`, `scan_rojo` y la imagen de la cámara y
escribe archivos chicos en `~/.ros/polibots/<fecha-hora>/`:

| Archivo | Contenido | Frecuencia | Tamaño en 10 min |
| --- | --- | --- | --- |
| `pose.csv` | t, x, y, yaw, σxy, σyaw (amcl) | 2 Hz | ~60 KB |
| `movimiento.csv` | t, cmd vx/vy/wz, odom vx/vy/wz | 5 Hz | ~200 KB |
| `roces.csv` | t, distancia mínima scan-robot, rayos rojos del scan | cada scan (~7 Hz) | ~200 KB |
| `rojo.csv` | t, percentiles 5/50/95 de H, S y V de los pixeles rojizos, cantidad de pixeles de la máscara | 1 Hz | ~40 KB |
| `img/*.jpg` | imagen de la cámara achicada a 320×240 | 1 cada 5 s | ~2 MB |
| `cpu.csv` | t, loadavg, frecuencia de `/cmd_vel` | cada 5 s | ~5 KB |

Se activa con un parámetro (`registrar: true` en `params.yaml`) para poder apagarlo en la final si
molesta. Usa poca CPU: guarda solo agregados y una imagen cada 5 s.

**B. `ros2 bag record`, si la organización lo deja correr en su computadora.** Sin imágenes en
crudo: una imagen de 640×480 a 15 Hz son ~8 GB en 10 minutos. Comando sugerido:

```bash
ros2 bag record -o corrida1 /amcl_pose /cmd_vel /odom /scan /scan_rojo /objetos_rojos \
  /objetos_no_mapeados /victimas /done /plan /tf /tf_static /rosout
```

Con el bag se puede reproducir la corrida en nuestra máquina y probar parámetros (sobre todo de
detección) con los datos reales. Pero hace falta tiempo entre corridas que quizás no haya.

## 5. Procedimiento entre corridas (5 a 10 minutos)

1. Copiar `~/.ros/log/<último>/launch.log` y `~/.ros/polibots/<último>/` a la notebook.
2. Correr `plan_de_trabajo/analizar_corrida.py <carpeta>`, que imprime el resumen y las sugerencias de la
   tabla de abajo.
3. Mirar el puntaje que dio la organización: víctimas encontradas, error de cada reporte, si hubo
   choques, vuelta y tiempo.
4. Cambiar **como mucho 2 o 3 parámetros**, los que el dato justifique. Anotar el cambio.
5. `colcon build --packages-select equipo_jar` si `install/` no tiene symlinks y verificar que el
   launch arranca (sin simulador queda esperando `/map`).

| Síntoma en la 1.ª corrida | Dato que lo confirma | Ajuste |
| --- | --- | --- |
| No se reportó una víctima que se vio | `rojo.csv`: S o V de la caja por debajo de `saturacion_min`/`valor_min`; candidato inspeccionado con "no es rojo" | Bajar `saturacion_min`/`valor_min` al percentil 5 − 10 |
| Reporte a 0.25-0.50 m (+50) | error de la organización; ancho de la cara medido | `mitad_caja: 0.09` (cajas de 18 cm) |
| Muchos candidatos falsos ("no es rojo") pegados a paredes | distancia a la pared de los candidatos en el log | Subir `despeje_candidato_claro` (0.25) o `margen_pared` de `detector_no_mapeados` (0.20) |
| El robot no arranca o no gira cuando la velocidad pedida es baja | `movimiento.csv`: odom ≈ 0 con cmd de 0.05-0.15 | Subir `min_speed_xy`/`min_speed_theta`, `giro_minimo_inspeccion`, `velocidad_destrabe` |
| Choques | `roces.csv` < 2 cm, la pose y el estado de la misión | Subir `footprint_padding` local a 0.04-0.05; bajar `max_vel_x/y` a 0.35 |
| Muchos "Control loop missed" | log de launch, `cpu.csv` | `vx/vy/vtheta_samples` a 9/7/7; `controller_frequency` a 15 |
| amcl tardó o se perdió | `pose.csv` con σ > 0.15 o saltos | `tiempo_localizacion_s` a 8; `dispersion_inicial_xy` si el robot quedó lejos del origen |
| Se trabó en puntas de pared | `Atascado en` y `sin desplazarse` en el log | `tiempo_atasco_s` a 3, `distancia_subobjetivo` a 0.6 |
| Se perdió el bono por poco tiempo | tiempos por fase | Bajar `tiempo_mirar_s` e `inspecciones_maximas`, según dónde se fue el tiempo |
| No volvió a la largada | el log de la vuelta | `velocidad_vuelta_estimada` a 0.15 o `margen_vuelta_s` a 45 (vuelve antes) |

## 6. Recomendación

1. **Antes de las 9:00:** sumar el nodo `registro` (opción A) y el script
   `plan_de_trabajo/analizar_corrida.py`, probados en el sim y con el CI en verde. Es código chico, que solo
   lee tópicos y escribe archivos, y no toca la misión. Si algo falla, el nodo se cae solo y la
   corrida sigue.
2. **Confirmar con la organización:** que se puede copiar `~/.ros/log` y `~/.ros/polibots` después
   de la primera corrida, cuánto tiempo hay entre corridas y si se puede correr `ros2 bag record`.
3. **En la primera corrida no cambiar nada** respecto de lo entregado. Salvo `mitad_caja: 0.09`,
   que es un dato conocido de las cajas reales, y el rango de color, si se calibró antes con una
   foto de la arena.
4. **Entre corridas:** seguir la tabla de la sección 5 y cambiar poco. Si la primera corrida ya
   sacó el máximo, dejar todo igual para la segunda: cuenta la mejor y el desempate es por tiempo.
