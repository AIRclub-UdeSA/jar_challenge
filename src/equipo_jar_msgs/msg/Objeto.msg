# An object the map doesn't show, from the world model (docs/WORLD_MODEL.md §6).
uint32 id                        # stable while the object exists
float32 p_objetivo               # P(victim): color, box shape, arena, N in total
float32 p_rojo                   # the color evidence alone
geometry_msgs/Point centro       # map frame, box-fit center; z = 0
float32[3] cov_centro            # center covariance xx, xy, yy (m^2), pose error included
float32 lado_a                   # m, fitted box sides
float32 lado_b
float32 yaw                      # rad, fitted box heading
float32 forma_caja               # 0..1, how well a box of the allowed sizes fits
uint32 scans                     # distinct scans that hit it
float32 valor_mirada             # how much one more camera look could change p_objetivo
geometry_msgs/Point[] miradores  # sensor positions the lidar saw it from: clear lines of sight
uint32[] celdas                  # its cells on /equipo/ocupacion (row * width + col): a line of
                                 # sight reaches it where the first occupied cell is one of them
