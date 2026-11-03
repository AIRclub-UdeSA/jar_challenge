# What the motion node is doing with the last path it got on /equipo/ruta.
std_msgs/Header header
builtin_interfaces/Time ruta     # header.stamp of the path this refers to
uint8 QUIETO = 0                 # no path (or an empty one): stopped
uint8 SIGUIENDO = 1
uint8 LLEGO = 2                  # at the path's last pose, position and heading
uint8 BLOQUEADO = 3              # every command it could give would touch something
uint8 estado
float32 restante                 # m of path left
