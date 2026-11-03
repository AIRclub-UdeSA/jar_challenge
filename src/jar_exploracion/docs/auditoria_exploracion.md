# Auditoría de exploración y pasos estrechos

Se revisaron planificación, elección de objetivos, seguimiento, scan, TF, cobertura,
reintentos y supervisión de deslizamiento. Esta revisión no certifica ausencia de
cualquier error ni reemplaza la prueba física de Gazebo.

## Hallazgos y cambios

| Hallazgo | Cambio |
| --- | --- |
| El círculo de 18 cm no representa el chasis ni su offset y desperdicia ancho lateral | Espacio de configuraciones con huella rectangular orientada y píxeles completos |
| Ruedas esféricas sobresalen lateralmente más que el ancho visual nominal | Envolvente conservadora de 37 × 24 cm respecto de `base_footprint` |
| Fuerza repulsiva proporcional al número de rayos, capaz de cancelar avance o empujar contra un lateral | Retirada la suma repulsiva; se valida el barrido del comando de seguimiento |
| El suavizado podía conservar una dirección anterior insegura | Comprobación de huella y mapa después del suavizado |
| Comando sin comprobación de frenado | Barrido con horizonte, reacción y desaceleración configurables |
| Scans exclusivamente NaN o vacíos se interpretaban como ausencia de obstáculos | Frenado; solo impactos válidos o infinito positivo aportan evidencia |
| Uso de impactos viejos en la base actual | Transformación temporal completa del scan al marco actual mediante `map` |
| Transformación de pose potencialmente vencida o no finita | Rechazo con timeout y validación de finitud |
| A* podía monopolizar un callback y se repetía cada dos segundos | Grafo cacheado, cálculo compilado fuera del executor ROS y conservación del camino válido |
| Reinicialización de objetivos ante retransmitir el mismo mapa | Se conservan camino y objetivo cuando no cambian contenido ni metadatos |
| Cambio de mapa o pose podía conservar un camino anterior | Frenado e invalidación inmediata |
| Objetivos alternantes por recalcular el más cercano | Objetivo persistente hasta cobertura, rechazo o suspensión |
| Zonas difíciles cercanas bloqueaban zonas alcanzables | Prioridad de zonas conectadas y rotación de las búsquedas difíciles |
| Oscilaciones contaban como progreso | Se exige reducción del recorrido restante |
| Salto de waypoints podía cortar esquinas | Comprobación del segmento en el espacio de configuraciones |
| Cobertura podía atravesar el contacto diagonal entre paredes | Comprobación de las celdas laterales del rayo |
| Cobertura continuaba después de detectar deslizamiento | Se invalida y espera nueva confirmación de pose |

## Dimensiones y orientación

Fuentes locales: `yahboom_rosmaster_description/urdf/mech/rosmaster_x3_base.urdf.xacro`
y `mecanum_wheel.urdf.xacro`. El chasis mide 0.300 m, con centro x=-0.031 m;
la esfera de cada rueda tiene radio 0.0325 m y centros y=±0.0845 m.
La caja simétrica de 0.37 × 0.24 m contiene estas colisiones respecto de la base.
No se redujo `margen_seguridad` de 0.05 m.
La planificación subdivide cada píxel en 2 × 2 subceldas sin cambiar el área
ocupada. Con mapas de 5 cm, los centros candidatos quedan separados por 2.5 cm.

El mapa reserva además el soporte del píxel ocupado. Se añade una tolerancia de
6 mm para variaciones de orientación menores a 0.03 rad; cambios mayores invalidan
la geometría y el camino. Unknown y el exterior del mapa están bloqueados.

El controlador sigue siendo holonómico con orientación conservada. No planifica
rotaciones autónomas SE(2): un paso que exige girar primero necesita un planificador
de orientación. Confirmar una pose inicial diferente no gira físicamente el robot.
No se presupone que pueda rotar dentro de un pasillo demasiado angosto.

## Pruebas

- Suite de `jar_exploracion`: 60 pruebas aprobadas, incluida integración de callbacks
  con mensajes ROS reales y transporte interceptado, flake8 y PEP257. Copyright
  sigue omitido por el decorador que ya tenía el repositorio.
- Pasillos longitudinales sintéticos, resolución 1 cm y margen 5 cm: 28 y 32 cm
  rechazados; 38 y 50 cm conectados. A 38 cm, integración cinemática de 2 m sin
  atravesar las paredes. Esto no mide el contacto ni la tracción de Gazebo.
- Pasillo de 40 cm en mapa de 5 cm: la subdivisión recupera centros seguros que
  no existían en la grilla original, sin reducir el área de ninguna pared.
- Geometría rotada: distingue entre una orientación que cabe y otra que no cabe.
- Obstáculo frontal, reducción de velocidad, barrido entre muestras, NaN, mapas
  inválidos, TF vencida, corrección de pose, objetivos persistentes y oscilación.
- Auditoría de 7 mapas del simulador, dos orientaciones por mapa y tres rutas por
  orientación: 42 rutas validadas sin cruzar obstáculos ni cortar esquinas.
  Resultados: `auditoria_mapas.json`.

Reproducir desde la raíz del repositorio:

```bash
source /opt/ros/humble/setup.bash
PYTHONPATH="$PWD/jar_exploracion:$PYTHONPATH" python3 -m pytest jar_exploracion/test -q
PYTHONPATH="$PWD/jar_exploracion:$PYTHONPATH" python3 jar_exploracion/tools/auditar_mapas.py \
  ../yahboom_rosmaster/yahboom_rosmaster_gazebo/maps
```

El auditor de mapas requiere Pillow y PyYAML; no son dependencias del controlador.

## Parámetros y límites pendientes de validar en simulación

`largo_robot=0.37`, `ancho_robot=0.24`, `margen_seguridad=0.05`,
`velocidad_pasillo=0.10`, `horizonte_colision_s=0.8`,
`desaceleracion_segura=0.4`, `timeout_pose_s=0.5`, `subdivisiones_mapa=2`.

La desaceleración es una hipótesis configurable del comprobador, no una medición
de la dinámica del X3. Antes de aumentar velocidad se debe medir frenado y latencia.
La posición y el mapa deben corresponder al mundo cargado. La comprobación es 2D:
un obstáculo fuera del plano del láser y ausente del mapa no está modelado.
No se ha ejecutado otra instancia de Gazebo durante esta auditoría para evitar
interferir con la simulación del usuario. Queda pendiente repetir el pasillo real
con contactos y perfil de fricción del simulador. AMCL puede perder localización:
el supervisor de deslizamiento detiene la misión, no corrige automáticamente AMCL.

## Referencias consultadas

- [Nav2: footprint vs. radius e inflación](https://docs.nav2.org/rolling/configuration_and_development/tuning_guide/).
- [Nav2 Humble: seguimiento regulado y comprobación predictiva de colisión](https://api.nav2.org/nav2-humble/html/md_nav2_regulated_pure_pursuit_controller_README.html).

Se aplican principios de huella y barrido al controlador holonómico propio; no se
ha sustituido por Regulated Pure Pursuit, que no aprovecha la traslación lateral.

## Corrección de pausas y parpadeo del destino

La planificación por etapas dejaba al robot quieto mientras recorría muchas
expansiones y publicaba ADD/DELETE sobre el mismo marcador en cada tick. Ahora
se conserva el marcador durante una planificación pendiente y se borra solamente
cuando el objetivo se descarta o cubre. La construcción del grafo y la búsqueda
con SciPy corren en un trabajador único, sin usar ROS desde ese trabajador.
El grafo conserva los ocho vecinos, los costos de destino y la prohibición de
cortar esquinas. Solo se construye de nuevo si cambia mapa, costo u orientación.
Las rutas pendientes se cancelan o descartan si ya no corresponden al objetivo.

Comparación aislada en café, grilla interna 898 × 380, desde (-1.21, 3.99) hasta
(-3.36, 7.84), tres repeticiones alternadas por método en el entorno x86_64:

| Método | Mediana | Rango |
| --- | --- | --- |
| A* de referencia, cómputo sin pausas | 3.413 s | 3.162–3.900 s |
| Compilado, grafo reutilizado | 0.184 s | 0.174–0.193 s |

Construcción inicial del grafo: 0.094 s. Ambos métodos produjeron rutas de costo
1074.0572826370953. Estos tiempos son de cálculo aislado; no son una medición del
factor de tiempo real de Gazebo ni garantizan los mismos tiempos en otro equipo.
Datos: `comparacion_planificadores.json`.

Reproducir con `tools/comparar_planificadores.py` y los argumentos
`MAPA_CAFE.yaml --inicio -1.21 3.99 --objetivo -3.36 7.84`.
La suite incluye un futuro mantenido pendiente durante diez ticks: no se borra el
marcador, no se reinicia el cálculo y el robot comienza a moverse al recibir la ruta.
Veinte mapas aleatorios adicionales comprueban equivalencia de costos con A*.
