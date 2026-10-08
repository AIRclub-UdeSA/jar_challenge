# Guía de QoS

La QoS (calidad de servicio) es la forma en que un publicador y un suscriptor se ponen de acuerdo
para intercambiar mensajes. Si no son compatibles, **no se conectan y ROS no avisa**: el suscriptor
simplemente no recibe nada. La regla que más problemas da:

| Publicador | Suscriptor | ¿Se conectan? |
| --- | --- | --- |
| reliable | reliable | sí |
| reliable | best effort | sí |
| best effort | best effort | sí |
| **best effort** | **reliable** | **no** |

La QoS de cada topic de la competencia es fija y está en el reglamento (§7.1). **Es responsabilidad
del equipo usar la que corresponde.** Un equipo que use otra puede no recibir datos o no ser
escuchado por el robot o por el juez, y eso no es una falla de la organización.

## Lo que tenés que usar

### Topics que publica el robot o el juez (vos te suscribís)

| Topic | Publica | Reliability | Durability | Depth | Cómo suscribirte |
| --- | --- | --- | --- | --- | --- |
| `/scan` | robot | **best effort** | volatile | 5 | `qos_profile_sensor_data` |
| `/cam_1/color/image_raw` | robot | **best effort** | volatile | 5 | `qos_profile_sensor_data` |
| `/cam_1/color/camera_info` | robot | **best effort** | volatile | 5 | `qos_profile_sensor_data` |
| `/cam_1/depth/color/points` | robot | **best effort** | volatile | 5 | `qos_profile_sensor_data` |
| `/odom` | robot | reliable | volatile | 50 | cualquiera (por defecto sirve) |
| `/joint_states` | robot | reliable | volatile | 100 | cualquiera |
| `/imu/data` | robot | reliable | volatile | 5 | cualquiera |
| `/tf` | robot | reliable | volatile | 100 | cualquiera (`tf2_ros` lo hace solo) |
| `/tf_static` | robot | reliable | **transient local** | 1 | `tf2_ros` lo hace solo; a mano, con *transient local* |
| `/map` | juez | **reliable** | **transient local** | 1 | *reliable* + *transient local*, depth 1 |

Lo importante: **`/scan` y las cámaras son best effort.** Si te suscribís con la QoS por defecto
(reliable), no llega ni un mensaje. `/map` es *transient local*: te suscribís con esa QoS para
recibir el mapa aunque el juez lo haya publicado un instante antes.

### Topics que publicás vos

| Topic | Va a | Reliability | Durability | Depth | Cómo publicar |
| --- | --- | --- | --- | --- | --- |
| `/victimas` | juez | **reliable** | volatile (o transient local) | 10 | `create_publisher(msg, topic, 10)`, la QoS por defecto |
| `/done` | juez | **reliable** | volatile (o transient local) | 10 | la QoS por defecto |
| `/cmd_vel` | robot | **reliable** | volatile | 10 | la QoS por defecto |

Si publicás con la QoS por defecto de ROS 2 (`create_publisher(msg, topic, 10)`) estás bien. Lo que
rompe es copiarles a estos topics el perfil de los sensores (best effort): el juez o el robot no te
escuchan, sin ningún error.

**Watchdog de `/cmd_vel`:** el robot se frena solo si pasan 0,5 s sin recibir un `/cmd_vel`. Mientras
quieras que se mueva, tenés que seguir publicando. Cuando publicás `/done` tu software tiene que
detener el robot y no volver a moverlo.

## Ejemplos

```python
from rclpy.qos import (qos_profile_sensor_data, QoSProfile, ReliabilityPolicy,
                       DurabilityPolicy)

# Sensores: best effort
self.create_subscription(LaserScan, '/scan', self.al_scan, qos_profile_sensor_data)

# Mapa: reliable + transient local
qos_mapa = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                      durability=DurabilityPolicy.TRANSIENT_LOCAL)
self.create_subscription(OccupancyGrid, '/map', self.al_mapa, qos_mapa)

# Lo que publicás: la QoS por defecto (reliable)
self.pub_victimas = self.create_publisher(PointStamped, '/victimas', 10)
```

Para comprobar una conexión: `ros2 topic info -v /scan` muestra la QoS de cada publicador y de cada
suscriptor.

## De dónde salen estos valores

- **Robot** (`AIRclub-UdeSA/physical_rosmaster`, `origin/main` al 2026-09-25, commit `b144d31`):
  - `/scan`: `SensorDataQoS()` en `sllidar_node.cpp`.
  - Cámara y nube de puntos: `qos_profile_sensor_data` en `sensor_adapter.py`.
  - `/odom` (depth 50) y `/joint_states` (depth 100): en `base_node_X3.cpp` y `Mcnamu_driver_X3.py`,
    con la QoS por defecto (reliable, volatile). Coincide con la QoS ofrecida que registraron los
    rosbags de `robot_artifacts/`, que también muestran `/tf` reliable y volatile.
  - Suscriptor de `/cmd_vel`: `create_subscription(Twist, "cmd_vel", ..., 1)` en
    `Mcnamu_driver_X3.py`, con la QoS por defecto (reliable, volatile, depth 1). `cmd_vel_timeout`
    es 0,5 s.
- **`/imu/data` y `/tf_static`** no aparecen en esas capturas. Los valores de la tabla se midieron
  corriendo los nodos estándar con sus valores por defecto (`imu_filter_madgwick` y el publicador
  estático de `tf2_ros`); la configuración del robot no los cambia. Falta confirmarlos en el robot.
- Los dos PR abiertos de `physical_rosmaster` a esa fecha (#44 y #45) solo agregan mediciones y
  documentación; no cambian la QoS.
- **`/map`, `/victimas` y `/done`** los fija la organización: `/map` como ya decía el reglamento
  (reliable, transient local); los otros dos, reliable, que es lo que el juez escucha.
