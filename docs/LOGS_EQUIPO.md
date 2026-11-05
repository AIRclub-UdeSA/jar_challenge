# Registro de cada corrida

El comando oficial sigue siendo:

```bash
ros2 launch equipo_jar competencia.launch.py victimas:=N
```

El software crea un único `log.txt` en `logs_jar/`, en la raíz del workspace
donde se compiló la entrega. Identifica esa raíz a partir del prefijo instalado
de `equipo_jar`, aunque el comando se ejecute desde otra carpeta. Acepta tanto
instalación aislada como `--merge-install`.

Si se define `JAR_LOG_DIR` con un directorio, se usa ese destino:

```bash
JAR_LOG_DIR=/ruta/indicada/por/el/evento ros2 launch equipo_jar competencia.launch.py victimas:=N
```

La carpeta se crea si hace falta. Al comenzar un launch se reemplaza el
contenido anterior de `log.txt`: cada ejecución corresponde a una corrida.
La organización guarda el archivo de la corrida previa.

El archivo es texto UTF-8. Contiene los mensajes del launch, la exploración,
la detección y Nav2, con los nombres de los procesos y sus marcas de tiempo.
Incluye localización, recuperación de atascos, objetos observados, reportes y
cierre de la misión. Es información pública: no se vuelca el entorno completo,
credenciales, imágenes ni datos personales deliberadamente.

El escritor limita el tamaño a **25.000.000 bytes**, contando los caracteres
UTF-8 por su tamaño real. Al llegar al límite deja de añadir mensajes. No usa
rotación ni crea archivos `.1`, copias o archivos temporales de log. Las
escrituras son sin buffering de Python para que los mensajes ya recibidos
permanezcan en el archivo si la organización detiene el proceso.

Si falla la creación, apertura o escritura, se deshabilita el registro y la
corrida continúa. El aviso sale por consola; no se intenta escribir un log en
otra carpeta.

## Integración con ROS 2 Humble

ROS 2 normalmente crea `launch.log` y archivos de log por proceso. La extensión
`ros2launch.option` del paquete configura el escritor único en `prestart`,
antes de que el servicio de launch cree su primer archivo. Solo actúa sobre
`ros2 launch equipo_jar competencia.launch.py`; los launch de otros paquetes
conservan su comportamiento.

Todos los nodos de la entrega reciben `--disable-external-lib-logs`, que
desactiva sus archivos de log de ROS sin desactivar la consola. El launch reúne
esas salidas en el mismo escritor acotado. Los parámetros dinámicos se pasan
como argumentos ROS: no se generan YAML temporales en disco durante la corrida.
Los nodos, sus remapeos y los tiempos de arranque son los mismos del stack
anterior; los algoritmos y los valores de los YAML no se modificaron.

Es necesario recompilar y cargar `install/setup.bash` para registrar la
extensión. No hay comandos adicionales para activar el registro.

`logs_jar/` está en `.gitignore`. El archivo generado no forma parte del PR.
