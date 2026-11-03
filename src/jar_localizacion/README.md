# jar_localizacion

Localización en el mapa con un filtro de partículas propio (Monte Carlo
Localization). Es la solución de la semana 06 de
[`jar_workshops`](https://github.com/AIRclub-UdeSA/jar_workshops/tree/main/semana-06-localizacion),
con la pose inicial por `initialpose` de la semana 08 y la corrección de escala
de odometría de la semana 10. La teoría está en el README de ese workshop.

## Nodos

| Ejecutable | Entrada | Salida |
| --- | --- | --- |
| `publicar_mapa` | `yaml_filename` (mapa de map_server) | `/map` (transient local), sin Nav2 |
| `campo_verosimilitud` | `/map` | `/likelihood_map` (transient local) |
| `localizador` | `/likelihood_map`, `/scan`, `/odom`, `initialpose` | TF `map -> odom`, `/particlecloud`, `/camino_odom`, `/camino_corregido` |
| `medir_odometria` | `/odom`, `/imu/data`, `/scan` | Log con giro odom/IMU y avance odom/lidar trasero |
| `odom_corregida` | `/odom` | `/camino_odom`, `/camino_corregido` con las escalas aplicadas |

`odom_corregida` publica en los mismos caminos que `localizador`: no correrlos juntos.
Las cuentas del filtro y de la odometría están en `filtro.py` y `odometria.py`,
sin ROS, y se prueban en `test/`.

## Uso

```bash
ros2 launch jar_localizacion localizacion.launch.py            # mapa + filtro, sin Nav2
ros2 launch jar_bringup navegacion.launch.py localizador:=jar  # en lugar de AMCL
ros2 launch jar_bringup robot_real.launch.py                   # bag o robot real
```

**2D Pose Estimate** de RViz vuelve a sembrar las partículas alrededor de la pose marcada.

## Escalas de odometría

`escala_avance` y `escala_giro` multiplican el paso de `/odom` antes de aplicarle
ruido. En el simulador valen 1.0. En el bag del robot real de la semana 10:

| | `/odom` | Referencia | Escala |
| --- | --- | --- | --- |
| Avance | 0.949 m | 0.744 m (lidar hacia atrás) | **0.784** |
| Giro | 1339.0° | 1076.5° (IMU) | **0.804** |

El IMU del bag integra el giro con el signo invertido: el scan matching del lidar
coincide en signo con `/odom` en el 95 % de los intervalos con giro, y con el IMU
en ninguno. Por eso la escala es positiva.

Con esas escalas, el filtro corrido sobre todo el bag deja el scan a 0–5 cm de
las paredes. Con 1.0 se pierde después de los primeros giros. Los valores están en
`jar_bringup/config/robot_real.yaml`.

## Parámetros del localizador

| Parámetro | Default | Qué es |
| --- | --- | --- |
| `num_particulas` | 300 | Partículas del filtro |
| `pose_inicial_x/y/theta` | 0.0 | Centro de la nube inicial |
| `dispersion_inicial_xy/theta` | 0.3 | Desvío de la nube inicial (también al re-sembrar) |
| `gamma1`..`gamma6` | 0.0025 | Varianzas del modelo de movimiento mecanum |
| `escala_avance`, `escala_giro` | 1.0 | Corrección de escala de `/odom` |
| `submuestreo_scan` | 15 | Se usa uno de cada N rayos para pesar |
| `ruido_remuestreo_xy`, `ruido_remuestreo_theta` | 0.01 | Ruido (m, rad) después de remuestrear; 0 lo desactiva |
| `respaldo_odometria` | true | Seguir con odometría cuando el filtro no es confiable |
| `dispersion_maxima` | 0.15 | Desvío xy (m) de la nube a partir del cual el filtro no se usa |
| `tolerancia_calidad` | 5.0 | Cuánto peor (0-100) puede calzar el scan desde el filtro que desde la odometría |
| `scans_para_resembrar` | 10 | Scans seguidos con odometría antes de re-sembrar alrededor de ella |
| `dispersion_resembrado` | 0.05 | Desvío (m y rad) de esa nube |

### Respaldo con odometría

En cada scan se compara la pose del filtro con la que da la odometría desde la
última corrección aceptada (`map -> odom` sin cambiar). Para cada una se mide la
calidad: la probabilidad media del campo donde caen los puntos del scan. Si la
nube está abierta (más de `dispersion_maxima`) o el filtro calza más de
`tolerancia_calidad` peor, `map -> odom` queda fija y la pose sigue solo a la
odometría. Si eso dura `scans_para_resembrar` scans y la odometría calza
claramente mejor, el filtro se perdió: las partículas se re-siembran alrededor
de la pose de odometría. Los cambios de modo quedan en el log del localizador.

### Por qué el ruido después de remuestrear

Con el robot quieto, `mover_particulas` no agrega ruido (el desvío es
proporcional al movimiento) y cada scan remuestrea: en pocos segundos las 300
partículas son copias de una sola, con dispersión 0. El filtro queda
sobreconfiado en la pose donde colapsó. En el bag de la semana 10, 10 corridas con
semillas distintas reportan las víctimas con error medio de 6.9 cm (peor caso 18.1 cm)
sin el ruido, y de 2.4 cm (peor 3.5 cm) con 1 cm / 0.01 rad. Corregir solo cuando
el robot se movió, como AMCL, dio 6.2 cm de error medio (peor 14.8 cm).
