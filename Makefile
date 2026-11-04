# Atajos para probar el equipo en el simulador (Docker de yahboom_rosmaster).
# Todo corre dentro del contenedor, salvo instalar, arrancar y parar.
#
#   make instalar      una sola vez: imagen, contenedor, Nav2 y compilar el simulador
#   make arrancar      después de reiniciar la compu (parar: make parar)
#   make build         compilar equipo_jar (antes de probar un cambio)
#   make test          pruebas del paquete, sin simulador
#   make verificar     los controles rápidos del CI
#   make corrida       corrida sin ventana con juez y puntaje -> build/verificacion/<MAPA>/
#   make ci            exactamente el job de simulador del CI (maze_1 y maze_3, 90 s)
#   make sim           Gazebo con ventana     } las tres en terminales distintas,
#   make equipo        nuestro launch         } en este orden: el juez último
#   make juez          juez de práctica       }
#   make teleop        mover el robot con el teclado
#   make shell         una terminal dentro del contenedor
#   make limpiar       cortar lo que quedó corriendo (si cerraste una terminal sin Ctrl+C)
#
# Variables: MAPA (maze_1_6x5, maze_2_6x5, maze_3_6x6, maze_4_metal_6x6, ...)
#            SEG  segundos de corrida desde que llega el mapa (CI: 90, competencia: 600)
#   ej.: make corrida MAPA=maze_3_6x6 SEG=300

MAPA ?= maze_1_6x5
SEG ?= 90
SIM_DIR ?= $(HOME)/rosmaster_ws/src/yahboom_rosmaster/dockerfiles
CONT := yahboom_rosmaster_sim
IMAGEN := yahboom_rosmaster:$(shell id -un)
X11 := /tmp/rosmaster-x11-$(shell id -u)-$(CONT)
TTY := $(shell [ -t 0 ] && echo -t)

# $(call EN,comando): corre en el contenedor, en el fork, con ROS, el simulador y
# nuestro install/ cargados (este último, para que gane sobre la copia del workspace).
# El comando no puede llevar comas ni comillas simples.
EN = docker exec -i $(TTY) -u alumno -e DISPLAY -e XAUTHORITY=/tmp/.x11host/Xauthority \
	-w /home/alumno/rosmaster_ws/src/jar_challenge_fierro $(CONT) bash -c '\
	source /opt/ros/humble/setup.bash; source ~/rosmaster_ws/install/setup.bash; \
	[ -f install/setup.bash ] && source install/setup.bash; \
	SIM=$$(ros2 pkg prefix --share yahboom_rosmaster_gazebo); $(1)'

.PHONY: ayuda instalar arrancar parar build test verificar corrida ci sim equipo juez teleop shell limpiar x11

ayuda:
	@sed -n '1,20s/^# \{0,1\}//p' $(firstword $(MAKEFILE_LIST))

# --network=host: con el DNS de Tailscale, el build de container.sh se corta a mitad.
instalar:
	docker image inspect $(IMAGEN) >/dev/null 2>&1 || docker build --network=host \
		--build-arg USER_UID=$$(id -u) --build-arg USER_GID=$$(id -g) -t $(IMAGEN) $(SIM_DIR)
	$(SIM_DIR)/container.sh start
	$(SIM_DIR)/container.sh build
	$(MAKE) build

arrancar:
	$(SIM_DIR)/container.sh start

parar:
	$(SIM_DIR)/container.sh stop

# Sin --symlink-install, igual que verificar.sh: mezclar los dos modos rompe build/.
build:
	$(call EN,colcon build --base-paths src)

test: build
	$(call EN,python3 -m pytest -q src/equipo_jar/test)

verificar:
	$(call EN,tools/verificar.sh)

corrida: build
	$(call EN,tools/correr_simulador.sh $(MAPA) $(SEG))

ci:
	$(call EN,tools/verificar.sh --sim maze_1_6x5 --sim maze_3_6x6)

sim: x11
	$(call EN,ros2 launch yahboom_rosmaster_gazebo rosmaster_gazebo_fortress.launch.py \
		world:=$$SIM/worlds/$(MAPA)_victimas.world)

equipo:
	$(call EN,ros2 launch equipo_jar competencia.launch.py use_sim_time:=true \
		victimas:=$$(python3 tools/mundos.py $$SIM/worlds/$(MAPA)_victimas.world))

juez:
	$(call EN,python3 tools/juez.py --mundo $(MAPA) --mapa $$SIM/maps/$(MAPA).yaml \
		--ground-truth --esperar-topic /scan --retardo 5 --duracion $(SEG))

teleop:
	$(call EN,ros2 run teleop_twist_keyboard teleop_twist_keyboard)

shell:
	$(call EN,exec bash)

# Un Gazebo olvidado se mezcla con el siguiente: mismos topics, otro robot.
limpiar:
	@docker exec $(CONT) pkill -INT -f 'ros2 launch|ros2 run|ign gazebo|tools/juez.py' || true
	@sleep 3
	@docker exec $(CONT) pkill -KILL -f 'ros2 launch|ros2 run|ign gazebo|tools/juez.py|_bridge|robot_state_pub' || true

# Permiso para abrir ventanas en tu pantalla (lo mismo que hace container.sh).
# Sin cookie (Xwayland sin -auth) no hace falta: el contenedor entra por el socket.
x11:
	@xauth nlist $$DISPLAY | grep -q . || exit 0; \
	mkdir -p $(X11) && xauth nlist $$DISPLAY | sed -e 's/^..../ffff/' | \
		xauth -f $(X11)/Xauthority nmerge - && chmod 644 $(X11)/Xauthority || \
		{ echo "No puedo escribir $(X11) (la creó Docker como root): sudo chown $$USER $(X11)"; exit 1; }
