#!/usr/bin/env bash
# Prueba rápida del software del equipo en el simulador, con un solo comando.
#
#   scripts/probar.sh [opciones]
#
#   --mapa NOMBRE      mapa de práctica (por defecto maze_1_6x5; también maze_3_6x6)
#   --tiempo SEG       segundos de corrida desde que se publica /map (por defecto 300)
#   --spawn "X Y YAW"  dónde arranca el robot (por defecto "0.2 -0.15 0.3": cerca del
#                      origen pero no en él, como en la competencia)
#   --origen           arrancar exacto en (0, 0, 0)
#   --ventana          abrir Gazebo y RViz para mirar la corrida
#   --nota "texto"     qué cambiaste; queda en el historial
#   --sin-compilar     no recompilar equipo_jar
#
# Hace lo mismo que las cuatro terminales a mano: simulador, juez con puntaje, nuestro launch
# y la medición de la localización contra la pose real. Al final grafica la corrida y suma una
# línea a pruebas/historial.csv para ver cómo evoluciona el puntaje entre cambios.
# Todo queda en pruebas/<fecha>_<mapa>/ (carpeta ignorada por git).
#
# Corre dentro del contenedor (./container.sh enter). No reemplaza a tools/verificar.sh, que
# es el control del CI: este es para iterar rápido.
set -o pipefail  # sin -u: el setup.bash de ROS usa variables sin definir

MAPA=maze_1_6x5
TIEMPO=300
SPAWN="0.2 -0.15 0.3"
VENTANA=false
NOTA=""
COMPILAR=true
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mapa) MAPA="$2"; shift ;;
    --tiempo) TIEMPO="$2"; shift ;;
    --spawn) SPAWN="$2"; shift ;;
    --origen) SPAWN="0 0 0" ;;
    --ventana) VENTANA=true ;;
    --nota) NOTA="$2"; shift ;;
    --sin-compilar) COMPILAR=false ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "opción desconocida: $1 (ver --help)"; exit 1 ;;
  esac
  shift
done
read -r SPAWN_X SPAWN_Y SPAWN_YAW <<<"$SPAWN"

RAIZ="$(cd "$(dirname "$0")/.." && pwd)"
cd "$RAIZ"
export TZ='<-03>3'  # hora de Argentina para las carpetas y el historial (el contenedor usa UTC)
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
# shellcheck disable=SC1091
[[ -f "$HOME/rosmaster_ws/install/setup.bash" ]] && source "$HOME/rosmaster_ws/install/setup.bash"

# Red: dominio propio (no se cruza con verificar.sh, que usa el 77) y Gazebo solo por
# localhost (con una VPN como Cloudflare WARP activa, el simulador se cuelga sin esto).
export ROS_DOMAIN_ID="${JAR_ROS_DOMAIN_ID:-78}"
export ROS_LOCALHOST_ONLY=1
export IGN_IP=127.0.0.1 GZ_IP=127.0.0.1
export IGN_PARTITION="jar_prueba_$$" GZ_PARTITION="jar_prueba_$$"

if $COMPILAR; then
  echo "[1/6] Compilando equipo_jar"
  colcon build --base-paths src --packages-select equipo_jar > /dev/null ||
    { echo "ERROR: falló colcon build (correlo a mano para ver el error)"; exit 1; }
fi
# shellcheck disable=SC1091
source install/setup.bash

SIM="$(ros2 pkg prefix --share yahboom_rosmaster_gazebo)" ||
  { echo "ERROR: no encuentro yahboom_rosmaster_gazebo"; exit 1; }
MUNDO="$SIM/worlds/${MAPA}_victimas.world"
YAML="$SIM/maps/${MAPA}.yaml"
[[ -f "$MUNDO" ]] || { echo "ERROR: no existe el mapa '$MAPA'"; exit 1; }
N="$(python3 tools/mundos.py "$MUNDO")"
COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo '?')"
[[ -n "$(git status --porcelain src 2>/dev/null)" ]] && COMMIT="$COMMIT*"  # * = sin commitear

SALIDA="$RAIZ/pruebas/$(date +%Y-%m-%d_%H%M%S)_$MAPA"
mkdir -p "$SALIDA"
echo "      mapa $MAPA, N=$N víctimas, spawn ($SPAWN_X, $SPAWN_Y, $SPAWN_YAW), $TIEMPO s, commit $COMMIT"
echo "      logs en $SALIDA"

PIDS=()
limpiar() {
  for pid in "${PIDS[@]:-}"; do [[ -n "$pid" ]] && kill -INT -- "-$pid" 2>/dev/null; done
  sleep 2
  for pid in "${PIDS[@]:-}"; do [[ -n "$pid" ]] && kill -KILL -- "-$pid" 2>/dev/null; done
  wait 2>/dev/null  # recoger los procesos sin que bash avise "Killed" por cada uno
}
trap limpiar EXIT

echo "[2/6] Simulador"
if $VENTANA; then VISTA=(headless:=false gui:=true rviz:=true); else VISTA=(headless:=true gui:=false rviz:=false); fi
setsid ros2 launch yahboom_rosmaster_gazebo rosmaster_gazebo_fortress.launch.py \
  world:="$MUNDO" "${VISTA[@]}" spawn_x:="$SPAWN_X" spawn_y:="$SPAWN_Y" spawn_yaw:="$SPAWN_YAW" \
  > "$SALIDA/simulador.log" 2>&1 &
PIDS+=("$!")
for _ in $(seq 1 90); do
  ros2 topic list 2>/dev/null | grep -qx /scan && break
  sleep 2
done
ros2 topic list 2>/dev/null | grep -qx /scan ||
  { echo "ERROR: el simulador no arrancó (mirá $SALIDA/simulador.log)"; exit 1; }

echo "[3/6] Juez con puntaje y mediciones"
setsid python3 tools/juez.py --mundo "$MAPA" --mapa "$YAML" --esperar-topic /scan --retardo 20 \
  --duracion "$TIEMPO" --ground-truth --salida "$SALIDA/resultado.json" \
  --registro "$SALIDA/eventos.jsonl" > "$SALIDA/juez.log" 2>&1 &
JUEZ=$!; PIDS+=("$JUEZ")
setsid python3 tools/grabar_trayectoria.py --salida "$SALIDA/trayectoria.jsonl" \
  > /dev/null 2>&1 &
PIDS+=("$!")
setsid python3 scripts/medir_localizacion.py --salida "$SALIDA/localizacion.jsonl" \
  > "$SALIDA/medidor.log" 2>&1 &
PIDS+=("$!")

echo "[4/6] Nuestro software (victimas:=$N)"
setsid ros2 launch equipo_jar competencia.launch.py use_sim_time:=true victimas:="$N" \
  > "$SALIDA/equipo.log" 2>&1 &
EQUIPO=$!; PIDS+=("$EQUIPO")

# Mientras corre, mostrar solo los mensajes importantes del equipo.
setsid bash -c "tail -F '$SALIDA/equipo.log' 2>/dev/null | grep --line-buffered -E \
  'Mapa recibido|localizado|odometría|Víctima|/done|ERROR|Traceback|WARN.*nodo_'" &
PIDS+=("$!")

echo "[5/6] Corriendo: hasta $TIEMPO s después del mapa (Ctrl+C corta todo)"
wait "$JUEZ"
kill -0 "$EQUIPO" 2>/dev/null || echo "ATENCIÓN: nuestro launch terminó antes de tiempo (mirá equipo.log)"

echo "[6/6] Resultado"
if [[ -f "$SALIDA/resultado.json" ]]; then
  python3 tools/graficar_corrida.py --mapa "$YAML" --resultado "$SALIDA/resultado.json" \
    --trayectoria "$SALIDA/trayectoria.jsonl" --salida "$SALIDA/grafico.png" > /dev/null 2>&1 &&
    echo "      gráfico: $SALIDA/grafico.png"
else
  echo "ATENCIÓN: el juez no dejó resultado.json (mirá juez.log)"
fi
limpiar  # cerrar el simulador antes de escribir el historial: el medidor guarda al cerrarse
trap - EXIT
python3 scripts/historial.py --carpeta "$SALIDA" --historial "$RAIZ/pruebas/historial.csv" \
  --commit "$COMMIT" --mapa "$MAPA" --spawn "$SPAWN_X $SPAWN_Y $SPAWN_YAW" --nota "$NOTA"
