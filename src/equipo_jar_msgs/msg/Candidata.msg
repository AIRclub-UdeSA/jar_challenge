# Una caja que percepción cree haber visto (docs/team/INTERFACES.md).
uint32 id                     # estable mientras exista; si dos se fusionan, queda una
geometry_msgs/Point centro    # centro estimado de la caja, en map
float64[4] covarianza_xy      # matriz 2x2 por filas, en m²
uint32 observaciones          # cuántas detecciones la respaldan
builtin_interfaces/Time ultima_vez
