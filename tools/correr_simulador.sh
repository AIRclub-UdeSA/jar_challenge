#!/usr/bin/env bash
# Una corrida corta del software del equipo en el simulador, con el juez con puntaje.
#
#   tools/correr_simulador.sh <mapa> [segundos_de_corrida]
#
# <mapa> es el nombre de un mapa de práctica del simulador, por ejemplo
# maze_1_6x5 (usa worlds/<mapa>_victimas.world y maps/<mapa>.yaml).
#
# Levanta Gazebo sin ventana, el juez con puntaje y el launch del equipo; y al final
# audita qué interfaces usó el software (§7.1). Requiere el workspace con
# yahboom_rosmaster compilado y "sourceado", más `colcon build` de este repo.
set -o pipefail  # sin -u: el setup.bash de ROS usa variables sin definir

MAPA="${1:?uso: correr_simulador.sh <mapa> [segundos]}"
DURACION="${2:-90}"

RAIZ="$(cd "$(dirname "$0")/.." && pwd)"
cd "$RAIZ"
# shellcheck disable=SC1091
[[ -f /opt/ros/humble/setup.bash ]] && source /opt/ros/humble/setup.bash
[[ -f install/setup.bash ]] && source install/setup.bash
export ROS_DOMAIN_ID="${JAR_ROS_DOMAIN_ID:-77}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-1}"
# Gazebo comparte su transporte entre todos los procesos de la máquina, sin importar el
# dominio de ROS: si hay otro simulador abierto, mezcla su ground truth y sus sensores.
export IGN_PARTITION="${IGN_PARTITION:-jar_verificacion_$$}" GZ_PARTITION="${GZ_PARTITION:-jar_verificacion_$$}"

SIM="$(ros2 pkg prefix --share yahboom_rosmaster_gazebo 2>/dev/null)" ||
  { echo "::error::no encuentro yahboom_rosmaster_gazebo: compilá y hacé source del simulador"; exit 1; }
MUNDO="$SIM/worlds/${MAPA}_victimas.world"
YAML="$SIM/maps/${MAPA}.yaml"
[[ -f "$MUNDO" && -f "$YAML" ]] || { echo "::error::no existe el mapa de práctica '$MAPA'"; exit 1; }
N="$(python3 tools/mundos.py "$MUNDO")" ||
  { echo "::error::no pude contar las víctimas de $MUNDO"; exit 1; }
echo "Este mundo tiene N = $N víctimas: el equipo corre con victimas:=$N"

SALIDA="$RAIZ/build/verificacion/$MAPA"
mkdir -p "$SALIDA"
PIDS=()
limpiar() {
  for pid in "${PIDS[@]:-}"; do [[ -n "$pid" ]] && kill -INT -- "-$pid" 2>/dev/null; done
  sleep 2
  for pid in "${PIDS[@]:-}"; do [[ -n "$pid" ]] && kill -KILL -- "-$pid" 2>/dev/null; done
}
trap limpiar EXIT

echo "[1/6] Simulador: $MAPA (sin ventana)"
setsid ros2 launch yahboom_rosmaster_gazebo rosmaster_gazebo_fortress.launch.py \
  world:="$MUNDO" headless:=true gui:=false rviz:=false > "$SALIDA/simulador.log" 2>&1 &
PIDS+=("$!")

echo "[2/6] Juez con puntaje (publica /map 40 s después del primer /scan)"
setsid python3 tools/juez.py --mundo "$MAPA" --mapa "$YAML" --esperar-topic /scan --retardo 40 \
  --duracion "$DURACION" --ground-truth --salida "$SALIDA/resultado.json" --registro "$SALIDA/eventos.jsonl" \
  > "$SALIDA/juez.log" 2>&1 &
JUEZ=$!; PIDS+=("$JUEZ")

# Antes de la línea base: si arrancara después, el auditor lo contaría como nodo del equipo.
setsid python3 tools/grabar_trayectoria.py --salida "$SALIDA/trayectoria.jsonl" \
  > "$SALIDA/trayectoria.log" 2>&1 &
PIDS+=("$!")

# Esperar a que el simulador esté completo: los topics del robot presentes y
# el grafo de nodos estable. Si la línea base se toma antes, nodos del simulador
# que arrancan tarde se confundirían con nodos del equipo.
echo "     esperando a que el simulador termine de arrancar..."
listos=0
previos=-1
for _ in $(seq 1 180); do
  topics="$(ros2 topic list 2>/dev/null)"
  nodos="$(ros2 node list 2>/dev/null | wc -l)"
  if grep -qx /scan <<<"$topics" && grep -qx /odom <<<"$topics" &&
     grep -qx /joint_states <<<"$topics" && [[ "$nodos" -eq "$previos" ]]; then
    listos=$((listos + 1))
  else
    listos=0
  fi
  previos="$nodos"
  [[ "$listos" -ge 4 ]] && break   # 4 chequeos seguidos (~8 s) sin cambios
  sleep 2
done
[[ "$listos" -ge 4 ]] ||
  { echo "::error::el simulador no terminó de arrancar"; tail -30 "$SALIDA/simulador.log"; exit 1; }

echo "[3/6] Línea base del grafo (simulador + juez + grabador de trayectoria)"
python3 tools/auditor_interfaces.py --capturar "$SALIDA/base.txt" || exit 1

echo "[4/6] Software del equipo (use_sim_time:=true, victimas:=$N)"
setsid ros2 launch equipo_jar competencia.launch.py use_sim_time:=true victimas:="$N" \
  > "$SALIDA/equipo.log" 2>&1 &
EQUIPO=$!; PIDS+=("$EQUIPO")
sleep 12
RC=0
python3 tools/auditor_interfaces.py --auditar "$SALIDA/base.txt" || RC=1

echo "[5/6] Esperando el fin de la corrida (hasta ${DURACION}s tras el mapa)"
wait "$JUEZ" || RC=1
cat "$SALIDA/juez.log"
kill -0 "$EQUIPO" 2>/dev/null || { echo "::error::el launch del equipo terminó antes de tiempo"; tail -20 "$SALIDA/equipo.log"; RC=1; }

echo "[6/6] Gráfico y resumen"
if [[ -f "$SALIDA/resultado.json" ]]; then
  python3 tools/validar_resultado.py "$SALIDA/resultado.json" || RC=1
  TRAZA=""
  [[ -f "$SALIDA/trayectoria.jsonl" ]] && TRAZA="--trayectoria $SALIDA/trayectoria.jsonl"
  # shellcheck disable=SC2086
  python3 tools/graficar_corrida.py --mapa "$YAML" --resultado "$SALIDA/resultado.json" \
    $TRAZA --salida "$SALIDA/grafico.png" || echo "::warning::no se pudo graficar la corrida"
  python3 tools/resumen_corrida.py --resultado "$SALIDA/resultado.json" \
    --salida "$SALIDA/resumen.md" && cat "$SALIDA/resumen.md"
else
  echo "::warning::sin resultado.json, no hay gráfico ni resumen"
fi
exit $RC
