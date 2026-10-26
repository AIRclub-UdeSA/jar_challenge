import math

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

# --- Mapa -----------------------------------------------------------------

def normalize(a):
    """
    Lleva un ángulo (rad) a [-π, π): la misma dirección, siempre con el mismo número.

    IMPORTANTE: Siempre que se midan distancias angulares hay que pasar por esta función porque sino Fierro va a dar vueltas como un bobolón antes
    de avanzar (179° − (−179°) da −2°, no 358°). 
    
    Acepta un número o un array de numpy. 
    """
    return (a + np.pi) % (2 * np.pi) - np.pi

class Board:
    """
    El mapa del laberinto que manda el juez en /map, convertido a tablas de numpy.

    Al arrancar, el juez publica un OccupancyGrid: una lista plana de números (0 libre, 100 pared), uno por celda, más el tamaño, la resolución y el origen.
    Board loconvierte una sola vez y después responde rápido las preguntas que hace todo el software: ¿es pared?, ¿a qué distancia está la pared más cercana?, ¿cae dentro?

    Convenciones:
        - Celda (i, j): i es la fila y crece con y; j es la columna y crece con x. La fila 0 es la de abajo (y mínima), al revés que en la imagen .pgm.
        - (ox, oy) es la esquina de la celda (0, 0) en metros; res, los metros por celda.

    Atributos:
        res, ox, oy: resolución (m/celda) y origen (m).
        occupied: (height, width) bool, True en las paredes.
        free: (height, width) bool, todo lo que no es pared.
        dist: (height, width) float, metros hasta la pared más cercana (0 en las paredes). Se precalcula porque el filtro la consulta unas 45.000 veces por scan.
    """

    def __init__(self, data, width, height, res, ox, oy):
        g = np.asarray(data, dtype=np.int16).reshape(height, width)
        self.res, self.ox, self.oy = res, ox, oy
        self.occupied = g > 0
        self.free = ~self.occupied
        self.dist = ndimage.distance_transform_edt(~self.occupied) * res # Distancia (m) de cada celda a la pared más cercana.

    @classmethod
    def from_msg(cls, msg):
        i = msg.info
        return cls(msg.data, i.width, i.height, i.resolution, i.origin.position.x, i.origin.position.y)

    def to_cell(self, x, y): # Metros -> Celdas
        j = np.floor((np.asarray(x) - self.ox) / self.res).astype(int)
        i = np.floor((np.asarray(y) - self.oy) / self.res).astype(int)
        
        return i, j

    def to_world(self, i, j): # Celdas -> Metros
        return self.ox + (np.asarray(j) + 0.5) * self.res, self.oy + (np.asarray(i) + 0.5) * self.res

    def inside(self, i, j):
        return (i >= 0) & (i < self.occupied.shape[0]) & (j >= 0) & (j < self.occupied.shape[1])

    def distance(self, x, y, out=1.0):
        """Distancia a la pared más cercana en (x, y). Valor `out` si cae afuera del mapa."""
        i, j = self.to_cell(x, y)
        ok = self.inside(i, j)
        d = np.full(np.shape(i), out, dtype=float)
        d[ok] = self.dist[i[ok], j[ok]]
        
        return d

# --- Localización (localizador) -------------------------------------------

def log_likelihood(grid, poses, pts, sigma, z_rand):
    """
    Puntaje de cada pose candidata: qué tan bien explica el scan del lidar.

    Para cada pose, lleva los puntos del scan al mapa y mira a qué distancia d de una pared cae cada uno. Si la pose es la correcta, los puntos caen sobre las
    paredes (d ≈ 0). Es el modelo de campo de verosimilitud.
    """
    c, s = np.cos(poses[:, 2:3]), np.sin(poses[:, 2:3])
    x = poses[:, 0:1] + c * pts[:, 0] - s * pts[:, 1]
    y = poses[:, 1:2] + s * pts[:, 0] + c * pts[:, 1]
    d = grid.distance(x, y)
    
    return np.log(np.exp(-0.5 * (d / sigma) ** 2) + z_rand).sum(axis=1)

def initial_search(grid, pts, center=(0.0, 0.0, 0.0), radius=0.4, sigma=0.08, z_rand=0.05, sigma_yaw=0.6):
    """
    Pose (x, y, θ) que mejor explica el scan cerca de `center`, por fuerza bruta.

    Dos pasadas: una gruesa (±radius y la vuelta entera) y una fina alrededor de la ganadora. Un prior suave hacia `center` solo desempata poses que el lidar no distingue.
    Sirve para la largada y para relocalizarse (con `center` = última pose buena).

    Si el robot está fuera de ±radius devuelve igual una pose equivocada: verificar con match_fraction.
    """
    x0, y0, th0 = center

    def sweep(cx, cy, cth, r, step, r_ang, step_ang):
        xs = np.arange(-r, r + 1e-9, step)
        xx, yy = [a.ravel() for a in np.meshgrid(cx + xs, cy + xs)]
        best = (-np.inf, None)
        
        for th in cth + np.arange(-r_ang, r_ang, step_ang):
            poses = np.column_stack([xx, yy, np.full_like(xx, th)])
            ll = log_likelihood(grid, poses, pts, sigma, z_rand)
            ll -= 0.5 * (normalize(th - th0) / sigma_yaw) ** 2 \
                + 0.5 * ((xx - x0) ** 2 + (yy - y0) ** 2) / radius ** 2
            k = int(np.argmax(ll))
            
            if ll[k] > best[0]: best = (ll[k], poses[k])
        
        return best[1]

    p = sweep(x0, y0, th0, radius, 0.05, np.pi, math.radians(3))
    p = sweep(p[0], p[1], p[2], 0.05, 0.01, math.radians(4), math.radians(0.5))
    
    return np.array([p[0], p[1], normalize(p[2])])

def match_fraction(grid, pose, pts, tolerance=0.10):
    """
    Confianza de una pose: fracción (0 a 1) de rayos del scan que caen a menos de "tolerance" de una pared.

    A diferencia de log_likelihood, que sirve para comparar muchas poses, esta da un número que se lee solo 
    y sirve para decidir si una pose es confiable (después de initial_search o durante la corrida). 
    
    Las cajas y señuelos no están en el mapa: con la pose correcta da ~90 %, no 100 %. `tolerance` tiene que ser mayor a una celda (dist se mide entre centros).
    """
    c, s = math.cos(pose[2]), math.sin(pose[2])
    x = pose[0] + c * pts[:, 0] - s * pts[:, 1]
    y = pose[1] + s * pts[:, 0] + c * pts[:, 1]
    
    return float(np.mean(grid.distance(x, y) < tolerance))

class ParticleFilter:
    """
    Sigue la pose del robot durante la corrida con n hipótesis pesadas.

    Cada partícula es una pose (x, y, θ) con un peso. El ciclo es:
        predict: todas se mueven según la odometría, cada una con su propio ruido.
        correct: las que explican el scan del lidar ganan peso. Si el peso se concentra en pocas, se remuestrea (se clonan las buenas, se tiran las malas).
        estimate: el promedio ponderado es la pose del robot.
    
    La odometría dice cuánto se movió (y se equivoca), utilizamos el lidar para corregir dónde está.

    Atributos:
        p: (n, 3) poses de las partículas.
        w: (n,) pesos, suman 1.
    """
    def __init__(self, grid, n=500, seed=None):
        self.grid, self.n = grid, n # n: cantidad de partículas
        self.rng = np.random.default_rng(seed)
        self.p = np.zeros((n, 3))
        self.w = np.full(n, 1.0 / n)

    def initialize(self, pose, sigma_xy=0.05, sigma_th=0.05):
        """
        Reparte las partículas en una nube alrededor de `pose` (normalmente, la de initial_search).

        La nube (sigma_xy en m, sigma_th en rad) le deja margen al filtro para corregir el error de la pose inicial. Los pesos quedan todos iguales.
        """
        self.p = np.asarray(pose, float) + self.rng.normal(0, [sigma_xy, sigma_xy, sigma_th], (self.n, 3))
        self.w[:] = 1.0 / self.n

    def predict(self, dx, dy, dth, trans_noise=0.1, rot_noise=0.1):
        """
        Mueve las partículas según la odometría desde la última llamada.
          dx, dy: avance en el frame del robot.
          dth: giro (rad), de la IMU y pasado por normalize.
          
        Cada partícula recibe el movimiento más un ruido proporcional a lo recorrido (trans_noise) y a lo girado (rot_noise), así la nube cubre los errores de la
        odometría.
        """
        d = math.hypot(dx, dy)
        s_t = trans_noise * d + 0.002
        s_r = rot_noise * abs(dth) + 0.5 * trans_noise * d + 0.002
        r = self.rng.normal(0, 1, (self.n, 3)) * [s_t, s_t, s_r]
        ex, ey, eth = dx + r[:, 0], dy + r[:, 1], dth + r[:, 2]
        c, s = np.cos(self.p[:, 2]), np.sin(self.p[:, 2])
        self.p[:, 0] += c * ex - s * ey
        self.p[:, 1] += s * ex + c * ey
        self.p[:, 2] = normalize(self.p[:, 2] + eth)

    def correct(self, pts, sigma=0.08, z_rand=0.2, temperature=0.3):
        """
        Repesa las partículas con un scan.

        Peso nuevo ∝ peso viejo × verosimilitud^temperature. 
            -> Bayes de manual, la única diferncia es elevar a la temperatura, basicamente disminuimos las diferencias para mantener el grupo variado
        
        0.3 sale de la teoría, falta calibrarlo con datos reales. Remuestrea solo si las partículas efectivas (1/Σw²) bajan de n/2: remuestrear siempre empobrece la nube.
        """
        lw = temperature * log_likelihood(self.grid, self.p, pts, sigma, z_rand) + np.log(self.w + 1e-300)
        w = np.exp(lw - lw.max())
        self.w = w / w.sum()
        
        if 1.0 / np.sum(self.w ** 2) < self.n / 2: self._resample()

    def _resample(self):
        """Remuestreo de baja varianza. Usa un solo número al azar, recorre los pesos en orden."""
        pos = (self.rng.random() + np.arange(self.n)) / self.n
        idx = np.minimum(np.searchsorted(np.cumsum(self.w), pos), self.n - 1)
        self.p = self.p[idx]
        self.w[:] = 1.0 / self.n

    def estimate(self):
        """Pose estimada y qué tan segura es: devuelve (np.array([x, y, θ]), spread)."""
        x, y = np.average(self.p[:, 0], weights=self.w), np.average(self.p[:, 1], weights=self.w)
        th = math.atan2(np.average(np.sin(self.p[:, 2]), weights=self.w), np.average(np.cos(self.p[:, 2]), weights=self.w))
        spread = math.sqrt(np.average((self.p[:, 0] - x) ** 2 + (self.p[:, 1] - y) ** 2, weights=self.w))
        
        return np.array([x, y, th]), spread

# --- Plan de búsqueda (misión) --------------------------------------------

def visibility(grid, x, y, max_range=3.0, n_rays=360):
    """
    Máscara (height, width) de las celdas que se verían desde (x, y): cada rayo hasta la primera pared. Es para planear, sobre el mapa:
    viewpoint_tour la usa para elegir puntos de vista que entre todos vean el laberinto.
    
    Lanza n_rays rayos en 360° y los recorre cada media celda; maximum.accumulate bloquea todo lo que queda detrás de la primera pared o fuera del mapa.
        max_range: 3 m es hasta donde la cámara reconoce una caja (~23 px a esa distancia), no el alcance del lidar. 
        n_rays: con 360, a 3 m los rayos quedan a una celda de distancia.
    """
    ang = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    r = np.arange(0, max_range, grid.res * 0.5)
    i, j = grid.to_cell(x + np.outer(np.cos(ang), r), y + np.outer(np.sin(ang), r))
    ok = grid.inside(i, j)
    i, j = np.where(ok, i, 0), np.where(ok, j, 0)
    blocked = np.maximum.accumulate(grid.occupied[i, j] | ~ok, axis=1)
    
    vis = np.zeros_like(grid.occupied)
    vis[i[~blocked], j[~blocked]] = True
    return vis

def line_of_sight(grid, a, b):
    """
    True si el segmento recto de a a b (puntos (x, y) en m) no cruza ninguna pared del mapa.

    Revisa puntos cada media celda (así no se saltea una pared de una celda de ancho). Si algún punto cae fuera del mapa da False. Solo mira el mapa: cajas y señuelos no tapan.
    La misión la usa para decidir si gira a mirar un objeto que vio el lidar.
    """
    n = max(2, int(math.hypot(b[0] - a[0], b[1] - a[1]) / (grid.res / 2)))
    i, j = grid.to_cell(np.linspace(a[0], b[0], n), np.linspace(a[1], b[1], n))
    return bool(np.all(grid.inside(i, j)) and not grid.occupied[i, j].any())

def _graph(mask, res):
    """
    Grafo de los pasillos para Dijkstra: un nodo por celda True de `mask`, aristas a sus 8 vecinas.

    Cada arista mide la distancia entre centros (res en recto, res·√2 en diagonal). Basta mirar 4 direcciones: las otras 4 son las mismas aristas al revés
    y Dijkstra se llama con directed=False. Los vecinos se encuentran comparando dos recortes de `idx` corridos una celda, sin recorrer celda por celda.

    Returns:
        idx: (height, width) número de nodo de cada celda (-1 si no es nodo).
        grafo: matriz dispersa (n, n) en formato CSR. Una densa ocuparía ~300 MB en un mapa real.
    """
    idx = -np.ones(mask.shape, int)
    idx[mask] = np.arange(mask.sum())
    rows, cols, weights = [], [], []
    height, width = idx.shape
    
    for di, dj in ((0, 1), (1, 0), (1, 1), (1, -1)):
        a = idx[0:height - di, max(0, -dj):width - max(0, dj)]
        b = idx[di:height, max(0, dj):width + min(0, dj)]
        ok = (a >= 0) & (b >= 0)
        rows.append(a[ok]); cols.append(b[ok])
        weights.append(np.full(ok.sum(), res * math.hypot(di, dj)))
    n = int(mask.sum())
    
    return idx, coo_matrix((np.concatenate(weights), (np.concatenate(rows), np.concatenate(cols))), (n, n)).tocsr()

def path_dist(grid, drivable, points):
    """
    Mapas de distancia caminando por los pasillos: uno (height, width) por cada punto (x, y).

    Cada punto se engancha a su celda transitable más cercana, y Dijkstra (sobre el grafo de _graph) calcula desde ahí la distancia a todas las celdas en una sola corrida.
    Las celdas no transitables o inalcanzables quedan en inf.

    viewpoint_tour lee estos mapas en las celdas de los otros puntos para armar la tabla de distancias entre puntos de vista. El mapa desde la largada sirve también para saber
    cuánto falta para volver.

    Returns:
        (S, height, width) con S = len(points), en metros.
    """
    idx, g = _graph(drivable, grid.res)
    sources = [idx[tuple(nearest_drivable(grid, drivable, px, py))] for px, py in points]
    d = dijkstra(g, directed=False, indices=sources)
    out = np.full((len(points),) + drivable.shape, np.inf)
    out[:, drivable] = d
    
    return out

def nearest_drivable(grid, drivable, x, y):
    """
    Celda (i, j) "navegable" más cercana a (x, y): engancha un punto que no cae sobre una.

    Dijkstra y el recorte por conectividad solo arrancan desde celdas drivable. 
    Un punto a menos de robot_radius de una pared se mueve a la más cercana (Si ya está sobre una, devuelve la suya).
    """
    i, j = grid.to_cell(x, y)
    ii, jj = np.nonzero(drivable)
    k = np.argmin((ii - i) ** 2 + (jj - j) ** 2) # La más cercana

    return ii[k], jj[k]

def _two_opt(order, D):
    """
    Visitar todos los puntos de vista y volver a la largada por el camino más corto es el problema del viajante (TSP). Encontrar el óptimo exacto es demasiado caro, así que 
    viewpoint_tour arma primero una solución greedy: desde cada punto, ir al más cercano que falte. Es rápida, pero deja malas decisiones, sobre todo al final, cuando los
    puntos que quedan están lejos y el recorrido termina cruzándose.

    2-opt corrige eso: por cada par de tramos i->j y k->l prueba reconectarlos como i->k y j->l, invirtiendo el pedazo del medio (es "descruzarlos"). 
    Si suma menos, lo deja y repite hasta que ninguna inversión mejore. Incluye el tramo de vuelta a la largada, así optimiza el circuito completo.
    IMPORTANTE: El resultado no tiene cruces que convenga deshacer, pero no es necesariamente el óptimo.

    Supone D simétrica: ir de a a b cuesta lo mismo que de b a a (SI QUEREMOS PROBAR PESAR LOS NODOS POR VELOCIDAD HAY QUE CAMBIAR EL ENFOQUE).
    """
    improved = True
    while improved:
        improved = False

        for a in range(1, len(order) - 1):
            for b in range(a + 1, len(order)):
                i, j, k, l = order[a - 1], order[a], order[b], order[(b + 1) % len(order)]
                
                if D[i, k] + D[j, l] < D[i, j] + D[k, l] - 1e-9:
                    order[a:b + 1] = order[a:b + 1][::-1]
                    improved = True
    
    return order

def viewpoint_tour(grid, start, robot_radius=0.20, max_range=3.0, step=0.5, coverage=0.98):
    """
    Como el mapa se conoce, no hace falta explorar: hay que elegir desde dónde mirar y en qué orden. Se calcula una sola vez, al llegar /map (~0,5 s):

        1. Dónde puede ir el robot: celdas a más de robot_radius de una pared, conectadas con la largada. Lo que hay que ver: el libre a ~35 cm de esa zona.
        2. Candidatos: el centro del pasillo (la celda más lejos de las paredes) de cada bloque de step m.
        3. Selección golosa (set cover): agrega el candidato que ve más celdas nuevas, hasta cubrir "coverage" de lo que hay que ver. Con 1.0, hasta que ninguno vea nada nuevo.
        4. Orden: vecino más cercano + 2-opt con distancias por los pasillos, como circuito cerrado (al final hay que volver a la largada).

    Los puntos no traen orientación: visibility supone 360°, así que la misión tiene que girar en cada uno (la cámara ve 61°). La vuelta a la largada y el /done también son de la misión.

    Returns:
        (points, visibilities, drivable): los puntos en orden, lo que ve cada uno y la máscara de dónde puede ir el robot.
    """
    drivable = grid.free & (grid.dist > robot_radius)
    labels, _ = ndimage.label(drivable, structure=np.ones((3, 3)))
    drivable = labels == labels[nearest_drivable(grid, drivable, *start)]

    target = grid.free & ndimage.binary_dilation(drivable, iterations=int(robot_radius / grid.res) + 3)

    k = max(1, int(round(step / grid.res)))
    candidates = [] # Son los más alejados de las paredes -> Centro del pasillo
    for bi in range(0, drivable.shape[0], k):
        for bj in range(0, drivable.shape[1], k):
            block = np.where(drivable[bi:bi + k, bj:bj + k], grid.dist[bi:bi + k, bj:bj + k], -1)
            
            if block.max() > 0:
                di, dj = np.unravel_index(np.argmax(block), block.shape)
                candidates.append(grid.to_world(bi + di, bj + dj))
    
    vis = [visibility(grid, cx, cy, max_range) & target for cx, cy in candidates]

    covered = np.zeros_like(target)
    chosen, total = [], target.sum()
    while covered.sum() < coverage * total:
        gain = [np.count_nonzero(v & ~covered) for v in vis]
        best = int(np.argmax(gain))
        if gain[best] == 0: break
        
        chosen.append(best)
        covered |= vis[best]

    points = [tuple(start)] + [candidates[e] for e in chosen]
    geo = path_dist(grid, drivable, points)
    cells = [nearest_drivable(grid, drivable, px, py) for px, py in points]
    D = np.array([[geo[a][cells[b]] for b in range(len(points))] for a in range(len(points))])
    order, remaining = [0], set(range(1, len(points)))

    while remaining:
        nearest = min(remaining, key=lambda b: D[order[-1], b])
        order.append(nearest)
        remaining.remove(nearest)
    order = _two_opt(order, D)[1:]

    return [points[o] for o in order], [vis[chosen[o - 1]] for o in order], drivable

# --- Detecciones (percepción y misión) ------------------------------------

def segments(ranges, max_jump, min_len):
    """
    Parte un scan en tramos de rayos consecutivos: cada tramo es la superficie de un objeto.

    Recorre los rayos en orden de ángulo y corta donde la distancia salta más de "max_jump" entre vecinos (el borde entre un objeto y lo que está detrás) o donde hay un rayo no finito.
    Descarta los tramos de menos de "min_len" rayos. La percepción la usa con los rayos rojos (cajas; cortar por color separa una caja apoyada contra la pared) y con los rayos que caen
    lejos del mapa (objetos que el mapa no tiene: víctimas sin mirar o señuelos).

    Returns:
        Lista de tramos; cada tramo, una lista de índices de rayos.

    IMPORTANTE: no une el último rayo con el primero y los dos apuntan adelante: una caja justo enfrente sale partida en dos tramos.
    """
    found, current = [], []

    for k, r in enumerate(ranges):
        if np.isfinite(r) and current and abs(r - ranges[current[-1]]) < max_jump:
            current.append(k)
            continue

        if len(current) >= min_len: found.append(current)
        current = [k] if np.isfinite(r) else []
    
    if len(current) >= min_len: found.append(current)
    
    return found

class DetectionClusterer:
    """
    Junta las detecciones de cada caja roja y decide cuándo hay evidencia para reportarla.

    La percepción manda decenas de detecciones (x, y) por segundo por caja, cada una con unos centímetros de error y algunas muy malas. Llegan ya como centros 
    de caja: la percepción corrige media caja (mitad_caja) desde la cara que ve el sensor, que es lo que mide el juez. Esta clase las agrupa por caja, estima
    el centro con la mediana y solo da por lista una caja vista muchas veces, desde lugares distintos y con poca dispersión: cada reporte equivocado gasta uno de los N.
    
    Atributos:
        - groups: lista de {'obs': [(x, y, rx, ry), ...], 'reported': bool}. (x, y) es la detección y (rx, ry) dónde estaba el robot. 'reported' lo marca la misión al
        publicar en /victimas; la clase no lo toca.
        - radius: a menos de esto del centro de un grupo, la detección es de esa caja (las cajas miden 0,25 m). max_obs: cuántas observaciones guarda por grupo (las últimas).
    """
    def __init__(self, radius=0.35, max_obs=300):
        self.radius, self.max_obs = radius, max_obs
        self.groups = []  # cada uno: {'obs': [(x, y, rx, ry)], 'reported': bool}

    def add(self, x, y, rx, ry):
        for g in self.groups:
            cx, cy = self.center(g)

            if math.hypot(x - cx, y - cy) < self.radius:
                g['obs'] = (g['obs'] + [(x, y, rx, ry)])[-self.max_obs:]
                return g

        g = {'obs': [(x, y, rx, ry)], 'reported': False}
        self.groups.append(g)

        return g

    @staticmethod
    def center(g):
        """Centro (x, y) del grupo: la mediana de las detecciones, robusta a las malas."""
        o = np.asarray(g['obs'])
        return float(np.median(o[:, 0])), float(np.median(o[:, 1]))

    def ready(self, g, n_min=15, max_spread=0.08, min_baseline=0.20):
        """n_min observaciones, poco dispersas, y vistas desde dos lugares distintos
        (o el doble de observaciones si el robot no se movió)."""
        o = np.asarray(g['obs'])
        if len(o) < n_min: return False

        cx, cy = np.median(o[:, 0]), np.median(o[:, 1])
        mad = 1.4826 * np.median(np.hypot(o[:, 0] - cx, o[:, 1] - cy))
        baseline = np.ptp(o[:, 2]) + np.ptp(o[:, 3])

        return mad < max_spread and (baseline > min_baseline or len(o) >= 2 * n_min)
