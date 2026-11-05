# Every object the world model knows of: a full snapshot, latched.
std_msgs/Header header
Objeto[] objetos
float32 busqueda_restante        # 0..1, search belief left: how incomplete the list is
