# Imagen del equipo (reglamento §7.4 y §9.2).
#
# La organización construye esta imagen a partir de tu commit y la corre SIN
# acceso a internet: todo lo que tu software necesite (paquetes, librerías,
# pesos de modelos) tiene que quedar adentro al construirla. Descargar cosas al
# ejecutar no funciona.
#
# Cómo agregar dependencias:
#   - paquetes de ROS o del sistema: declaralos en src/equipo_jar/package.xml
#     (<exec_depend>) y rosdep los instala acá;
#   - paquetes de Python: requirements.txt, con versiones fijas;
#   - pesos de modelos: se pueden descargar en un RUN de esta imagen (hay
#     internet al construir, no al correr), o incluirlos en el repo si pesan poco.
FROM ros:humble-ros-base-jammy@sha256:860e53793ccf575e5369b5e9a19939de2bfcfb46a299954d06f54f2af54fed19

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3-colcon-common-extensions python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /jar_challenge

COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

COPY src/ src/
RUN apt-get update \
    && rosdep update --rosdistro humble \
    && rosdep install --from-paths src --ignore-src -y --rosdistro humble \
    && rm -rf /var/lib/apt/lists/*

RUN source /opt/ros/humble/setup.bash \
    && colcon build --base-paths src

# La organización informa la cantidad de víctimas (N) antes de cada corrida y la pasa con
#   docker run -e VICTIMAS=<N> ...
# El robot real no publica /clock: use_sim_time queda en false por defecto.
ENV VICTIMAS=5
CMD ["bash", "-c", "source /opt/ros/humble/setup.bash && source install/setup.bash && exec ros2 launch equipo_jar competencia.launch.py victimas:=${VICTIMAS}"]
