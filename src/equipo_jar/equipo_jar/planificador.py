from collections import defaultdict, deque
from heapq import heappop, heappush

import numpy as np



def bcd(mapa):
    """
    Boustrophedon Cellular Decomposition.

    Entrada:
        mapa[y][x]
        0  = espacio libre
        != 0 = obstáculo o desconocido

    Retorna:
        regiones: matriz con el ID de cada región.
                  -1 representa obstáculos.

        grafo: diccionario de adyacencia entre regiones.
    """

    if len(mapa) == 0 or len(mapa[0]) == 0:
        raise ValueError("Mapa vacío")
    alto = len(mapa)
    ancho = len(mapa[0])

    if any(len(fila) != ancho for fila in mapa):
        raise ValueError("Todas las filas deben tener el mismo ancho")

    regiones = [[-1] * ancho for _ in range(alto)]
    grafo = {}

    anteriores = []
    siguiente_id = 0

    for x in range(ancho):

        # 1. Encontrar intervalos libres de la columna
        actuales = []
        y = 0

        while y < alto:
            if mapa[y][x] != 0:
                y += 1
                continue

            inicio = y

            while y < alto and mapa[y][x] == 0:
                y += 1

            actuales.append((inicio, y - 1))

        # 2. Detectar solapamientos entre columnas
        anteriores_a_actuales = [set() for _ in anteriores]
        actuales_a_anteriores = [set() for _ in actuales]

        i = j = 0

        while i < len(anteriores) and j < len(actuales):

            a_ini, a_fin, _ = anteriores[i]
            b_ini, b_fin = actuales[j]

            if max(a_ini, b_ini) <= min(a_fin, b_fin):
                anteriores_a_actuales[i].add(j)
                actuales_a_anteriores[j].add(i)

            if a_fin <= b_fin:
                i += 1

            if b_fin <= a_fin:
                j += 1

        # 3. Asignar regiones
        nuevos = []

        for j, (y_inicio, y_fin) in enumerate(actuales):

            predecesores = actuales_a_anteriores[j]

            # Continuación: una región anterior conecta
            # exclusivamente con este intervalo.
            if len(predecesores) == 1:
                i = next(iter(predecesores))

                if len(anteriores_a_actuales[i]) == 1:
                    id_region = anteriores[i][2]
                else:
                    id_region = siguiente_id
                    siguiente_id += 1
                    grafo[id_region] = set()

            # Inicio, división o unión: nueva región
            else:
                id_region = siguiente_id
                siguiente_id += 1
                grafo[id_region] = set()

            # 4. Registrar conexiones entre regiones
            for i in predecesores:

                id_anterior = anteriores[i][2]

                if id_anterior != id_region:
                    grafo[id_region].add(id_anterior)
                    grafo[id_anterior].add(id_region)

            # 5. Etiquetar las celdas
            for y in range(y_inicio, y_fin + 1):
                regiones[y][x] = id_region

            nuevos.append((y_inicio, y_fin, id_region))

        anteriores = nuevos

    return regiones, grafo



def enriquecer_grafo(regiones, resolucion=0.05):
    """
    Recibe la matriz de regiones generada por BCD.

    Devuelve:
        nodos: información geométrica de cada región.
        conexiones: regiones vecinas y puntos de conexión.

    Las coordenadas se representan como (x, y).
    """

    if not regiones or not regiones[0]:
        raise ValueError("Matriz vacía")

    alto = len(regiones)
    ancho = len(regiones[0])

    if any(len(fila) != ancho for fila in regiones):
        raise ValueError("Matriz irregular")

    celdas = defaultdict(list)
    conexiones = defaultdict(dict)

    for y in range(alto):
        for x in range(ancho):

            region = regiones[y][x]

            if region == -1:
                continue

            celdas[region].append((x, y))

            # Buscar fronteras con otras regiones
            for dx, dy in [(1, 0), (0, 1)]:

                nx = x + dx
                ny = y + dy

                if nx >= ancho or ny >= alto:
                    continue

                vecino = regiones[ny][nx]

                if vecino == -1 or vecino == region:
                    continue

                # Guardar la conexión en ambos sentidos
                conexiones[region].setdefault(
                    vecino, {"portales": []}
                )["portales"].append(
                    ((x, y), (nx, ny))
                )

                conexiones[vecino].setdefault(
                    region, {"portales": []}
                )["portales"].append(
                    ((nx, ny), (x, y))
                )

    # Calcular propiedades de cada región
    nodos = {}

    for region, puntos in celdas.items():

        cantidad = len(puntos)

        cx = sum(x for x, y in puntos) / cantidad
        cy = sum(y for x, y in puntos) / cantidad

        # Elegir una celda libre cercana al centro geométrico
        centro = min(
            puntos,
            key=lambda p: (p[0] - cx)**2 + (p[1] - cy)**2
        )

        nodos[region] = {
            "celdas": cantidad,
            "area_m2": cantidad * resolucion**2,
            "centro_celda": centro,
            "limites": (
                min(x for x, y in puntos),
                min(y for x, y in puntos),
                max(x for x, y in puntos),
                max(y for x, y in puntos),
            )
        }

    return nodos, dict(conexiones)


def ordenar_regiones(regiones, seguro, inicio):
    """Orden voraz por distancia transitable; devuelve orden e inaccesibles.

    Las distancias se calculan en la grilla segura, no por adyacencia BCD:
    un portal demasiado estrecho no permite atravesar una pared inflada.
    inicio y los puntos de referencia usan (x, y).
    """
    regiones, seguro = np.asarray(regiones), np.asarray(seguro, dtype=bool)
    if regiones.shape != seguro.shape or regiones.ndim != 2:
        raise ValueError('Matrices incompatibles')
    alto, ancho = regiones.shape
    x, y = inicio
    if not (0 <= x < ancho and 0 <= y < alto and seguro[y, x]):
        raise ValueError('El robot está fuera del espacio transitable seguro')
    pendientes = set(int(r) for r in np.unique(regiones) if r >= 0)
    orden = []
    while pendientes:
        cola = deque([(x, y)])
        visitados = {(x, y)}
        elegido = None
        while cola:
            cx, cy = cola.popleft()
            region = int(regiones[cy, cx])
            if region in pendientes:
                elegido = (region, cx, cy)
                break
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = cx + dx, cy + dy
                if (0 <= nx < ancho and 0 <= ny < alto and seguro[ny, nx]
                        and (nx, ny) not in visitados):
                    visitados.add((nx, ny))
                    cola.append((nx, ny))
        if elegido is None:
            break
        region, x, y = elegido
        orden.append(region)
        pendientes.remove(region)
    return orden, sorted(pendientes)


def depurar_regiones(regiones, seguro, inicio, resolucion,
                     area_minima_m2=0.35, ancho_minimo_m=0.45):
    """Fusiona celdas BCD pequeñas con vecinos unidos por un portal seguro.

    Excluye componentes seguros inaccesibles desde inicio. Las celdas de margen
    se asignan al componente seguro más cercano dentro de su región original
    """
    import math
    etiquetas = np.asarray(regiones, dtype=int).copy()
    seguro = np.asarray(seguro, dtype=bool)
    if etiquetas.ndim != 2 or etiquetas.shape != seguro.shape or not etiquetas.size:
        raise ValueError('Matrices incompatibles')
    if not math.isfinite(resolucion) or resolucion <= 0:
        raise ValueError('Resolución inválida')
    if any(not math.isfinite(v) or v < 0 for v in (area_minima_m2, ancho_minimo_m)):
        raise ValueError('Los umbrales de sectores deben ser finitos y no negativos')
    alto, ancho = etiquetas.shape
    x, y = inicio
    seguro = seguro & (etiquetas >= 0)
    if not (0 <= x < ancho and 0 <= y < alto and seguro[y, x]):
        raise ValueError('El robot está fuera del espacio transitable seguro')

    def vecinos(x, y):
        for nx, ny in ((x+1, y), (x-1, y), (x, y+1), (x, y-1)):
            if 0 <= nx < ancho and 0 <= ny < alto:
                yield nx, ny

    alcanzable = np.zeros_like(seguro)
    alcanzable[y, x] = True
    cola = deque([(x, y)])
    while cola:
        x, y = cola.popleft()
        for nx, ny in vecinos(x, y):
            if seguro[ny, nx] and not alcanzable[ny, nx]:
                alcanzable[ny, nx] = True
                cola.append((nx, ny))

    # Propagar pertenencia desde todos los componentes, no sólo el alcanzable.
    asignado = seguro.copy()
    conservar = alcanzable.copy()
    ys, xs = np.where(seguro)
    cola = deque(zip(xs.tolist(), ys.tolist()))
    while cola:
        x, y = cola.popleft()
        for nx, ny in vecinos(x, y):
            if not asignado[ny, nx] and etiquetas[ny, nx] == etiquetas[y, x]:
                asignado[ny, nx] = True
                conservar[ny, nx] = conservar[y, x]
                cola.append((nx, ny))
    excluidas = (etiquetas >= 0) & ~conservar
    etiquetas[excluidas] = -1
    fusiones = []
    while True:
        ids, cantidades = np.unique(etiquetas[etiquetas >= 0], return_counts=True)
        areas = dict(zip(ids.tolist(), cantidades.tolist()))
        candidatos = []
        for region, cantidad in areas.items():
            ys, xs = np.where(etiquetas == region)
            extension = max(int(xs.max()-xs.min()+1), int(ys.max()-ys.min()+1))
            ancho_efectivo = cantidad * resolucion / extension
            if cantidad * resolucion**2 < area_minima_m2 or ancho_efectivo < ancho_minimo_m:
                candidatos.append(region)
        fusionado = False
        for region in sorted(candidatos, key=lambda r: (areas[r], r)):
            portales = defaultdict(int)
            ys, xs = np.where((etiquetas == region) & alcanzable)
            for x, y in zip(xs, ys):
                for nx, ny in vecinos(x, y):
                    vecino = int(etiquetas[ny, nx])
                    if alcanzable[ny, nx] and vecino >= 0 and vecino != region:
                        portales[vecino] += 1
            if not portales:
                continue  # Sector alcanzable aislado: conservar su cobertura.
            destino = max(portales, key=lambda r: (portales[r], areas[r], -r))
            etiquetas[etiquetas == region] = destino
            fusiones.append((region, destino))
            fusionado = True
            break
        if not fusionado:
            break
    # IDs compactos para RViz; el informe de fusiones conserva los IDs originales.
    finales = np.full_like(etiquetas, -1)
    for nuevo, viejo in enumerate(sorted(np.unique(etiquetas[etiquetas >= 0]))):
        finales[etiquetas == viejo] = nuevo
    return finales.tolist(), {'fusiones_bcd': fusiones,
                              'celdas_excluidas': int(excluidas.sum()),
                              'area_excluida_m2': float(excluidas.sum()*resolucion**2)}


def _camino_seguro(seguro, inicio, destino, regiones=None, region=None):
    """Camino mínimo de cuatro vecinos, con menos cruces y giros al empatar."""
    alto, ancho = seguro.shape
    inicio, destino = tuple(inicio), tuple(destino)
    inicial = (inicio[0], inicio[1], 0, 0)
    padres = {inicial: None}
    costes = {inicial: (0, 0, 0)}
    pendientes = [(abs(destino[0]-inicio[0]) + abs(destino[1]-inicio[1]),
                   0, 0, 0, inicial)]
    while pendientes:
        _, cruces, giros, pasos, estado = heappop(pendientes)
        if costes.get(estado) != (pasos, cruces, giros):
            continue
        x, y, previo_x, previo_y = estado
        if (x, y) == destino:
            camino = []
            while estado is not None:
                camino.append(estado[:2])
                estado = padres[estado]
            return camino[::-1]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x+dx, y+dy
            if (not (0 <= nx < ancho and 0 <= ny < alto) or not seguro[ny, nx]
                    or (region is not None and regiones[ny, nx] != region)):
                continue
            cruce = regiones is not None and regiones[ny, nx] != regiones[y, x]
            giro = (previo_x, previo_y) not in ((0, 0), (dx, dy))
            coste = (pasos+1, cruces+int(cruce), giros+int(giro))
            siguiente = (nx, ny, dx, dy)
            if siguiente in costes and costes[siguiente] <= coste:
                continue
            costes[siguiente] = coste
            padres[siguiente] = estado
            estimacion = coste[0] + abs(destino[0]-nx) + abs(destino[1]-ny)
            heappush(pendientes, (estimacion, coste[1], coste[2], coste[0], siguiente))
    return []


def segmento_transitable(seguro, inicio, destino):
    """Comprueba todas las celdas del segmento, incluidas ambas de cada esquina."""
    alto, ancho = seguro.shape

    def libre(x, y):
        return 0 <= x < ancho and 0 <= y < alto and seguro[y, x]

    x, y = inicio
    dx, dy = destino[0] - x, destino[1] - y
    nx, ny = abs(dx), abs(dy)
    sx, sy = (1 if dx > 0 else -1), (1 if dy > 0 else -1)
    ix = iy = 0
    if not libre(x, y):
        return False
    while ix < nx or iy < ny:
        cruce_x, cruce_y = (1 + 2 * ix) * ny, (1 + 2 * iy) * nx
        if cruce_x == cruce_y:
            if not libre(x + sx, y) or not libre(x, y + sy):
                return False
            x, y, ix, iy = x + sx, y + sy, ix + 1, iy + 1
        elif cruce_x < cruce_y:
            x, ix = x + sx, ix + 1
        else:
            y, iy = y + sy, iy + 1
        if not libre(x, y):
            return False
    return True


def simplificar_ruta(seguro, inicio, ruta, seguro_inicio=None):
    """Retira guías sólo con línea libre; conserva cada cruce de portal.

    La primera esquina también se evalúa desde la pose real del robot. Si el
    inicio fue ajustado a una máscara más holgada, conserva esa primera guía.
    """
    if not ruta:
        return []
    guias = []
    ancla = inicio
    indice = 0
    if not segmento_transitable(seguro, inicio, inicio):
        primera = ruta[0][0]
        if seguro_inicio is None or not segmento_transitable(seguro_inicio, inicio, primera):
            return []
        guias.append(ruta[0])
        ancla = primera
        indice = 1
    while indice < len(ruta):
        limite = next((i for i in range(indice, len(ruta)) if ruta[i][1]), len(ruta) - 1)
        ultimo_visible = None
        for i in range(indice, limite + 1):
            if not segmento_transitable(seguro, ancla, ruta[i][0]):
                break
            ultimo_visible = i
        if ultimo_visible is None:
            return []
        guia = ruta[ultimo_visible]
        if guia[0] != ancla or not guias:
            guias.append(guia)
        ancla = guia[0]
        indice = ultimo_visible + 1
    return guias


def _distancia_al_borde(seguro):
    """Distancia en celdas al obstáculo o borde más cercano (ocho vecinos)."""
    ampliado = np.pad(seguro, 1, constant_values=False)
    distancia = np.full(ampliado.shape, ampliado.size, dtype=np.int32)
    ys, xs = np.nonzero(~ampliado)
    cola = deque(zip(xs.tolist(), ys.tolist()))
    distancia[ys, xs] = 0
    alto, ancho = ampliado.shape
    while cola:
        x, y = cola.popleft()
        siguiente = distancia[y, x] + 1
        for dx, dy in ((-1, -1), (0, -1), (1, -1),
                       (-1, 0), (1, 0),
                       (-1, 1), (0, 1), (1, 1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < ancho and 0 <= ny < alto and siguiente < distancia[ny, nx]:
                distancia[ny, nx] = siguiente
                cola.append((nx, ny))
    return distancia[1:-1, 1:-1]


def guias_por_zona(seguro, regiones, inicio, ruta, seguro_inicio=None):
    """Conserva entrada, medio y salida de cada zona recorrida.

    Recibe la ruta sin simplificar de ``ruta_por_portales`` y devuelve guías
    ``(celda_xy, es_portal)``, incluyendo la pose inicial. Los dos lados de
    cada cambio de región son obligatorios. También lo es un punto intermedio
    de cada tramo, elegido cerca de su mitad y con separación de las paredes.
    Las esquinas se simplifican únicamente entre esas anclas y dentro de la
    misma región; ninguna simplificación posterior debe eliminar las anclas.
    """
    seguro, regiones = np.asarray(seguro, bool), np.asarray(regiones)
    if seguro.ndim != 2 or regiones.shape != seguro.shape:
        raise ValueError('Matrices incompatibles')
    if not ruta:
        return []
    if seguro_inicio is not None:
        seguro_inicio = np.asarray(seguro_inicio, bool)
        if seguro_inicio.shape != seguro.shape:
            raise ValueError('Matrices incompatibles')
    alto, ancho = seguro.shape
    inicio = tuple(int(c) for c in inicio)

    def dentro(p):
        return 0 <= p[0] < ancho and 0 <= p[1] < alto

    if not dentro(inicio):
        return []
    camino = [inicio]
    portales = set()
    ajuste_inicial = None
    for i, (objetivo, portal) in enumerate(ruta):
        objetivo = tuple(int(c) for c in objetivo)
        if not dentro(objetivo) or not seguro[objetivo[1], objetivo[0]]:
            return []
        origen = camino[-1]
        mascara = seguro
        if i == 0 and not seguro[origen[1], origen[0]]:
            mascara = seguro_inicio
            ajuste_inicial = objetivo
        if mascara is None or not mascara[origen[1], origen[0]]:
            return []
        if origen[0] == objetivo[0] or origen[1] == objetivo[1]:
            if not segmento_transitable(mascara, origen, objetivo):
                return []
            dx = (objetivo[0] > origen[0]) - (objetivo[0] < origen[0])
            dy = (objetivo[1] > origen[1]) - (objetivo[1] < origen[1])
            largo = abs(objetivo[0] - origen[0]) + abs(objetivo[1] - origen[1])
            tramo = [(origen[0] + j*dx, origen[1] + j*dy)
                     for j in range(largo + 1)]
        else:
            # Sólo el enlace desde la pose real al inicio ajustado suele ser
            # diagonal. Reconstruirlo evita inventar cruces a través de muros.
            tramo = _camino_seguro(mascara, origen, objetivo)
            if not tramo:
                return []
        camino.extend(tramo[1:])
        if portal:
            portales.add(objetivo)

    holgura = _distancia_al_borde(seguro)
    obligatorios = portales | ({ajuste_inicial} if ajuste_inicial is not None else set())

    def distancias_desde(mascara, origen):
        distancias = {origen: 0}
        cola = deque([origen])
        while cola:
            x, y = cola.popleft()
            for vecino in ((x+1, y), (x-1, y), (x, y+1), (x, y-1)):
                nx, ny = vecino
                if (dentro(vecino) and mascara[ny, nx]
                        and vecino not in distancias):
                    distancias[vecino] = distancias[(x, y)] + 1
                    cola.append(vecino)
        return distancias

    def medio_con_holgura(tramo, mascara):
        interiores = [i for i in range(1, len(tramo)-1)
                      if seguro[tramo[i][1], tramo[i][0]]]
        if not interiores:
            return tramo, None
        mitad = (len(tramo)-1) / 2
        ventana = max(1, len(tramo)//6)
        cercanos = [i for i in interiores if abs(i-mitad) <= ventana]
        elegido = max(cercanos or interiores,
                      key=lambda i: (holgura[tramo[i][1], tramo[i][0]],
                                     -abs(i-mitad)))
        # El comienzo excepcional fuera de la máscara estricta conserva su
        # recorrido original. Lo mismo ocurre con otros portales obligatorios.
        if (any(not seguro[y, x] for x, y in tramo)
                or any(p in obligatorios for p in tramo[1:-1])):
            return tramo, elegido
        centro = tramo[int(mitad)]
        dist_a = distancias_desde(mascara, tramo[0])
        dist_b = distancias_desde(mascara, tramo[-1])
        # Mejorar la holgura del medio no debe convertirlo en una excursión
        # lateral: como máximo dos celdas y el 10% del camino más corto.
        largo_minimo = dist_a.get(tramo[-1], len(tramo)-1)
        desvio_maximo = min(2, largo_minimo//10)
        candidatos = [p for p in dist_a
                      if p in dist_b and p not in (tramo[0], tramo[-1])
                      and (p[0]-centro[0])**2 + (p[1]-centro[1])**2 <= 25
                      and abs(dist_a[p]-dist_b[p]) <= max(2, largo_minimo//3)
                      and dist_a[p] + dist_b[p] <= largo_minimo + desvio_maximo]
        candidatos.sort(key=lambda p: (
            -int(holgura[p[1], p[0]]),
            dist_a[p] + dist_b[p],
            abs(dist_a[p]-dist_b[p]),
            (p[0]-centro[0])**2 + (p[1]-centro[1])**2))
        for candidato in candidatos:
            entrada = _camino_seguro(mascara, tramo[0], candidato)
            salida = _camino_seguro(mascara, candidato, tramo[-1])
            # Una rama lateral implicaría alcanzar el medio y retroceder por
            # el mismo pasillo. Se descarta aunque tenga más holgura.
            if set(entrada[:-1]).isdisjoint(salida[1:]):
                return entrada + salida[1:], len(entrada)-1
        return tramo, elegido

    guias = []

    def agregar(p, portal):
        if guias and guias[-1][0] == p:
            guias[-1] = (p, guias[-1][1] or portal)
        else:
            guias.append((p, bool(portal)))

    primero = 0
    while primero < len(camino):
        ultimo = primero
        region = regiones[camino[primero][1], camino[primero][0]]
        while (ultimo + 1 < len(camino)
               and regiones[camino[ultimo+1][1], camino[ultimo+1][0]] == region):
            ultimo += 1
        tramo = camino[primero:ultimo+1]
        mascara = seguro & (regiones == region)
        tramo, medio = medio_con_holgura(tramo, mascara)
        anclas = {0, len(tramo)-1}
        if medio is not None:
            anclas.add(medio)
        anclas.update(i for i, p in enumerate(tramo) if p in obligatorios)
        anclas = sorted(anclas)
        agregar(tramo[0], primero > 0 or tramo[0] in portales)
        for a, b in zip(anclas, anclas[1:]):
            es_portal = tramo[b] in portales or (
                b == len(tramo)-1 and ultimo < len(camino)-1)
            fragmento = [(p, False) for p in tramo[a+1:b+1]]
            fragmento[-1] = (fragmento[-1][0], bool(es_portal))
            mascara_tramo = mascara
            if any(not seguro[y, x] for x, y in tramo[a:b+1]):
                if seguro_inicio is None:
                    return []
                mascara_tramo = seguro_inicio & (regiones == region)
            simplificado = simplificar_ruta(mascara_tramo, tramo[a], fragmento)
            if not simplificado:
                return []
            for p, portal in simplificado:
                agregar(p, portal)
        # Una zona de una única celda puede ser a la vez entrada y salida.
        if ultimo < len(camino)-1:
            agregar(tramo[-1], True)
        primero = ultimo + 1
    return guias


def ruta_por_portales(seguro, regiones, conexiones, inicio, destino, resolucion,
                      paso_maximo_m=0.50):
    """Guías seguras por portales, esquinas y tramos cortos.

    Cada cruce se desplaza localmente hacia la parte del portal con más
    separación de paredes. La guía queda del lado del sector de entrada.
    """
    import math
    if not math.isfinite(paso_maximo_m) or paso_maximo_m <= 0:
        raise ValueError('paso_maximo_m debe ser positivo')
    seguro, regiones = np.asarray(seguro, bool), np.asarray(regiones)
    alto, ancho = seguro.shape
    for x, y in (inicio, destino):
        if not (0 <= x < ancho and 0 <= y < alto and seguro[y, x]):
            return []
    camino = _camino_seguro(seguro, inicio, destino, regiones)
    if not camino:
        return []
    cruces = [(camino[i-1], camino[i]) for i in range(1, len(camino))
              if regiones[camino[i-1][1], camino[i-1][0]]
              != regiones[camino[i][1], camino[i][0]]]
    if cruces:
        distancia = _distancia_al_borde(seguro)
        radio = max(1, int(0.40 / resolucion))
        # Presupuesto para toda la ruta, no medio metro por cada portal.
        desvio_maximo = min(int(0.20 / resolucion), (len(camino)-1)//10)
        desvio_restante = desvio_maximo
        ajustado = [inicio]
        actual = inicio
        for i, (salida, entrada) in enumerate(cruces):
            origen_id = int(regiones[salida[1], salida[0]])
            destino_id = int(regiones[entrada[1], entrada[0]])
            proxima = cruces[i+1][0] if i+1 < len(cruces) else destino
            base_entrada = _camino_seguro(seguro, actual, salida, regiones, origen_id)
            base_salida = _camino_seguro(seguro, entrada, proxima, regiones, destino_id)
            if not base_entrada or not base_salida:
                ajustado = []
                break
            pares = conexiones.get(origen_id, {}).get(destino_id, {}).get('portales', [])
            candidatos = [(a, b) for a, b in pares
                          if seguro[a[1], a[0]] and seguro[b[1], b[0]]
                          and (b[0]-entrada[0])**2 + (b[1]-entrada[1])**2 <= radio**2]
            if (salida, entrada) not in candidatos:
                candidatos.append((salida, entrada))
            candidatos.sort(key=lambda par: (
                -min(distancia[par[0][1], par[0][0]],
                     distancia[par[1][1], par[1][0]]),
                (par[1][0]-entrada[0])**2 + (par[1][1]-entrada[1])**2))
            for a, b in candidatos:
                tramo_entrada = _camino_seguro(seguro, actual, a, regiones, origen_id)
                if not tramo_entrada:
                    continue
                tramo_salida = _camino_seguro(seguro, b, proxima, regiones, destino_id)
                if (tramo_salida and len(tramo_entrada) + len(tramo_salida)
                        <= len(base_entrada) + len(base_salida) + desvio_restante):
                    desvio_restante -= max(0, len(tramo_entrada) + len(tramo_salida)
                                            - len(base_entrada) - len(base_salida))
                    ajustado.extend(tramo_entrada[1:])
                    ajustado.append(b)
                    actual = b
                    break
            else:
                ajustado = []
                break
        if ajustado:
            final = _camino_seguro(seguro, actual, destino, regiones,
                                   int(regiones[destino[1], destino[0]]))
            if final:
                ajustado.extend(final[1:])
                if len(ajustado) <= len(camino) + desvio_maximo:
                    camino = ajustado
    portales = {tuple(b) for vecinos in conexiones.values() for datos in vecinos.values()
                for a, b in datos['portales']}
    salto = max(1, int(paso_maximo_m / resolucion))
    paradas = []
    ultimo = 0
    for i in range(1, len(camino)):
        x, y = camino[i]
        px, py = camino[i-1]
        portal = regiones[y, x] != regiones[py, px] and (x, y) in portales
        esquina = (i+1 < len(camino) and
                   (x-px, y-py) != (camino[i+1][0]-x, camino[i+1][1]-y))
        if portal or esquina or i-ultimo >= salto or i == len(camino)-1:
            paradas.append(((x, y), bool(portal)))
            ultimo = i
    return paradas or [(destino, False)]
