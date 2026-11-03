"""Decide de dónde sale la pose del robot en el mapa: AMCL o el respaldo con odometría. Sin ROS.

Las dos fuentes se equivocan distinto:
- **/odom** casi no se equivoca en un tramo corto, pero el error se acumula sin corregirse.
- **AMCL** no acumula error, pero a veces salta o se va a otro lado: medido en nuestras corridas
  (pruebas/), un salto de 3,4 m y 123° en un segundo, una deriva de 2 m en 15 s, y otra de 0,9 m
  mientras el robot giraba en el lugar.

AMCL publica la transformación map -> odom (la "corrección" de la odometría). Si AMCL anda bien,
esa corrección cambia poco y despacio: lo que haya derivado la odometría. Si cambia mucho más
de lo que la odometría puede haber derivado con lo que se movió el robot, el que se equivocó es
AMCL. Avanzar hace derivar la posición y el ángulo de /odom; girar en el lugar, solo el ángulo:
por eso se llevan las dos cuentas por separado (metros avanzados y radianes girados).

Este supervisor guarda las correcciones en las que confía (las *referencias*: una cada
`paso_referencia_m` avanzados, las de los últimos `ventana_referencias_m`) y, en cada ciclo,
compara la pose que da AMCL con la que da cada referencia + /odom:
- **coinciden con todas** (dentro de una tolerancia que crece con lo que se movió el robot desde
  cada una): se usa AMCL. Comparar contra varias, y no solo contra la última, es lo que agarra una
  deriva lenta: de a pasitos cada uno parece chico, pero contra la referencia de hace 2 m ya no.
- **no coinciden**, o AMCL dejó de publicar: **respaldo**, la pose sale de la referencia *más
  vieja* de la ventana + /odom. La más vieja, porque una deriva empieza de a poco, por debajo de
  la tolerancia, y las últimas referencias ya la tienen adentro. Mientras el robot se mueva poco
  en respaldo, la pose es buena; después, no confiable.
- AMCL vuelve a ser aceptado si coincide de nuevo y además se queda **quieto** (su corrección no
  cambia) durante `confirmar_regreso_s`: cuando vuelve de perderse, AMCL pasa por el lugar
  correcto sin estar todavía ahí. También, si se quedó estable en otro lugar mientras el robot
  anduvo un buen tramo, con covarianza chica (se relocalizó de verdad y el que derivó fue /odom).

Al arrancar, AMCL todavía no convergió (en las corridas, 18° de error al recibir la pose inicial)
y quieto no termina de converger: la primera referencia se toma recién cuando nodo_mision avisa
que convergió (`avisar_convergencia`), después de que el robot se movió. La covarianza de AMCL
no alcanza para eso: mide cuánto se dispersan las partículas, no cuánto se equivocan. En
simulación, con el robot quieto, llegó a σ < 6° con el ángulo 5,5° corrido, y una caja a
2,7 m quedó ubicada 0,3 m mal. Si AMCL nunca publica, se usa /odom como si fuera el mapa (el
robot arranca cerca del origen).
"""
import math
from collections import deque
from dataclasses import dataclass

from equipo_jar.geometria import componer, diferencia

IDENTIDAD = (0.0, 0.0, 0.0)

# De dónde sale la pose. Son textos para que el log se entienda solo.
AMCL = 'AMCL'
CONVERGIENDO = 'AMCL convergiendo'
RESPALDO = 'respaldo con odometría'
SOLO_ODOM = 'solo odometría (AMCL nunca publicó)'
SIN_POSE = 'sin pose'


@dataclass(frozen=True)
class ConfigSupervisor:
    max_antiguedad_s: float = 1.5            # map -> odom más viejo que esto: AMCL no publica
    margen_m: float = 0.10                   # diferencia que se tolera siempre (ruido de AMCL)
    margen_deg: float = 8.0
    tolerancia_por_metro_m: float = 0.08     # cuánto puede derivar /odom por metro avanzado...
    tolerancia_por_metro_deg: float = 10.0
    tolerancia_por_radian_deg: float = 4.0   # ...y en ángulo por radián girado (no corre x, y)
    tolerancia_max_m: float = 0.60           # tope: más que esto, nunca es deriva de /odom
    tolerancia_max_deg: float = 30.0
    paso_referencia_m: float = 0.50          # cada cuánto avance se guarda una referencia
    ventana_referencias_m: float = 2.0       # contra las referencias de los últimos X m
    confirmar_regreso_s: float = 2.0         # AMCL coincide y quieto esto: se vuelve a usar
    sigma_xy_convergida_m: float = 0.12      # covarianza chica, para aceptar una relocalización
    sigma_yaw_convergida_deg: float = 6.0
    espera_sin_amcl_s: float = 15.0          # sin ningún map -> odom tras esto: solo odometría
    avance_max_respaldo_m: float = 6.0       # desde la referencia (hasta 2 m antes del respaldo)
    giro_max_respaldo_deg: float = 1800.0    # ...o más giro que esto (cinco vueltas)
    estable_para_reaceptar_s: float = 10.0   # AMCL quieto en otro lugar este tiempo...
    avance_para_reaceptar_m: float = 3.0     # ...mientras el robot avanzó esto: se acepta


@dataclass(frozen=True)
class EstadoPose:
    correccion: tuple       # map -> odom que hay que usar ahora, o None si no hay pose
    fuente: str             # AMCL, RESPALDO, ...
    confiable: bool         # ¿sirve para ubicar una caja en el mapa?
    avance_respaldo_m: float
    motivo: str             # por qué, para el log


class Movimiento:
    """Cuánto se movió el robot según /odom: metros avanzados y radianes girados."""

    def __init__(self, avance=0.0, giro=0.0):
        self.avance = avance
        self.giro = giro

    def desde(self, otro):
        return Movimiento(self.avance - otro.avance, self.giro - otro.giro)


class SupervisorLocalizacion:
    def __init__(self, config):
        self.c = config
        self.t_inicio = None
        self.convergencia_avisada = False  # nodo_mision dijo que AMCL ya convergió
        self.amcl_publico = False     # AMCL publicó map -> odom al menos una vez
        self.odom_anterior = None
        self.mov = Movimiento()       # acumulado desde el arranque
        self.referencias = deque()    # (corrección confiable, Movimiento cuando se tomó)
        self.t_coincide = None        # desde cuándo AMCL coincide de nuevo (estando en respaldo)
        self.corr_coincide = None     # su corrección en ese momento: tiene que quedarse quieta
        self.en_respaldo = False
        self.respaldos = 0            # cuántas veces se entró en respaldo en la corrida
        self.t_respaldo = None        # cuándo se entró en el respaldo actual
        self.candidata = None         # corrección de AMCL que no coincide, para ver si es estable
        self.t_candidata = None
        self.mov_candidata = None

    # --- Tolerancias -------------------------------------------------------------------------

    def tolerancias(self, mov):
        """(m, rad) que puede haber derivado /odom con el movimiento `mov`."""
        tol_m = min(self.c.margen_m + self.c.tolerancia_por_metro_m * mov.avance,
                    self.c.tolerancia_max_m)
        tol_deg = min(self.c.margen_deg + self.c.tolerancia_por_metro_deg * mov.avance
                      + self.c.tolerancia_por_radian_deg * mov.giro, self.c.tolerancia_max_deg)
        return tol_m, math.radians(tol_deg)

    def coinciden(self, corr_a, corr_b, odom, mov):
        """¿Las dos correcciones ponen al robot en (casi) el mismo lugar del mapa?

        Se compara la pose del robot que da cada una, no las correcciones en sí: una diferencia
        chica de ángulo en map -> odom mueve mucho al robot si está lejos del origen de odom.
        """
        d_m, d_rad = diferencia(componer(corr_a, odom), componer(corr_b, odom))
        tol_m, tol_rad = self.tolerancias(mov)
        return d_m <= tol_m and d_rad <= tol_rad

    def convergida(self, sigmas):
        if sigmas is None:
            return False
        sigma_xy, sigma_yaw = sigmas
        return (sigma_xy <= self.c.sigma_xy_convergida_m
                and sigma_yaw <= math.radians(self.c.sigma_yaw_convergida_deg))

    def avisar_convergencia(self):
        """nodo_mision considera que AMCL convergió: desde ahora se le puede creer."""
        self.convergencia_avisada = True

    # --- Ciclo -------------------------------------------------------------------------------

    def actualizar(self, t, odom, correccion_amcl, t_correccion, sigmas):
        """Un ciclo. Devuelve un EstadoPose.

        t: ahora (s). odom: pose (x, y, yaw) de base_footprint en `odom`, o None.
        correccion_amcl: map -> odom (x, y, yaw) que publica AMCL, o None. t_correccion: su stamp.
        sigmas: (sigma_xy en m, sigma_yaw en rad) de la covarianza de AMCL, o None si no hay.
        """
        if self.t_inicio is None:
            self.t_inicio = t
        if odom is None:
            return EstadoPose(None, SIN_POSE, False, 0.0, 'no hay odometría')
        self.sumar_movimiento(odom)
        fresca = (correccion_amcl is not None and t_correccion is not None
                  and t - t_correccion <= self.c.max_antiguedad_s)
        if fresca:
            self.amcl_publico = True

        if not self.referencias:
            return self.sin_referencia(t, correccion_amcl if fresca else None, sigmas)
        if not fresca:
            self.t_coincide = None
            return self.respaldo(t, 'AMCL dejó de publicar map -> odom')

        if self.coincide_con_referencias(correccion_amcl, odom):
            self.candidata = None
            if self.en_respaldo:
                # Volver de golpe es peligroso: AMCL puede estar pasando por acá mientras vuelve
                # de otro lado. Tiene que coincidir un rato seguido y sin moverse.
                if self.t_coincide is None or not self.coinciden(
                        correccion_amcl, self.corr_coincide, odom, Movimiento()):
                    self.t_coincide, self.corr_coincide = t, correccion_amcl
                if t - self.t_coincide < self.c.confirmar_regreso_s:
                    return self.respaldo(t, 'AMCL vuelve a coincidir: espero a confirmarlo')
                motivo = 'AMCL volvió a coincidir' + self.salir_del_respaldo(t)
                self.tomar_referencia(correccion_amcl)
                return EstadoPose(correccion_amcl, AMCL, True, 0.0, motivo)
            if self.mov.avance - self.referencias[-1][1].avance >= self.c.paso_referencia_m:
                self.agregar_referencia(correccion_amcl)
            return EstadoPose(correccion_amcl, AMCL, True, 0.0, 'AMCL coincide con /odom')

        self.t_coincide = None
        if self.amcl_se_relocalizo(t, odom, correccion_amcl, sigmas):
            motivo = ('AMCL estable en otro lugar con covarianza chica: lo acepto'
                      + self.salir_del_respaldo(t))
            self.tomar_referencia(correccion_amcl)
            return EstadoPose(correccion_amcl, AMCL, True, 0.0, motivo)
        return self.respaldo(t, self.texto_peor_referencia(correccion_amcl, odom))

    def ultima(self):
        return self.referencias[-1][0]

    def coincide_con_referencias(self, correccion, odom):
        return all(self.coinciden(correccion, ref, odom, self.mov.desde(mov_ref))
                   for ref, mov_ref in self.referencias)

    def peor_referencia(self, correccion, odom):
        """La referencia con la que AMCL coincide peor: la que más se pasa de su tolerancia.

        Devuelve (d_m, d_rad, tol_m, tol_rad, hace_m): la diferencia de pose, la tolerancia de esa
        referencia y cuántos metros avanzó el robot desde que se tomó. Es la que dispara el
        respaldo: coincide_con_referencias() pide coincidir con todas, y la que falla suele ser una
        vieja (una deriva lenta ya está adentro de las nuevas). Por eso no sirve medir contra la
        última: daba "se apartó 0,00 m y 0°" justo cuando el respaldo lo disparaba otra.
        """
        peor, exceso_peor = None, -math.inf
        for ref, mov_ref in self.referencias:
            mov = self.mov.desde(mov_ref)
            d_m, d_rad = diferencia(componer(correccion, odom), componer(ref, odom))
            tol_m, tol_rad = self.tolerancias(mov)
            exceso = max(d_m / max(tol_m, 1e-9), d_rad / max(tol_rad, 1e-9))
            if exceso > exceso_peor:
                peor, exceso_peor = (d_m, d_rad, tol_m, tol_rad, mov.avance), exceso
        return peor

    def texto_peor_referencia(self, correccion, odom):
        d_m, d_rad, tol_m, tol_rad, hace_m = self.peor_referencia(correccion, odom)
        return (f'AMCL se apartó {d_m:.2f} m y {math.degrees(d_rad):.0f}° de la referencia de '
                f'hace {hace_m:.1f} m (tolera {tol_m:.2f} m y {math.degrees(tol_rad):.0f}°)')

    def salir_del_respaldo(self, t):
        """Sale del respaldo, si estaba. Devuelve el texto con cuánto duró, para el log."""
        if not self.en_respaldo:
            return ''
        self.en_respaldo = False
        return f' tras {t - self.t_respaldo:.1f} s en respaldo (n.º {self.respaldos})'

    def sin_referencia(self, t, correccion, sigmas):
        """Antes de confiar en AMCL por primera vez."""
        if correccion is not None:
            if self.convergencia_avisada:
                self.tomar_referencia(correccion)
                return EstadoPose(correccion, AMCL, True, 0.0,
                                  f'nodo_mision avisó que AMCL convergió ({texto_sigmas(sigmas)})')
            return EstadoPose(correccion, CONVERGIENDO, False, 0.0,
                              f'esperando el aviso de nodo_mision ({texto_sigmas(sigmas)})')
        if not self.amcl_publico and t - self.t_inicio >= self.c.espera_sin_amcl_s:
            return EstadoPose(IDENTIDAD, SOLO_ODOM, True, self.mov.avance,
                              'AMCL nunca publicó: uso /odom como mapa')
        return EstadoPose(None, SIN_POSE, False, 0.0, 'esperando a AMCL')

    def respaldo(self, t, motivo):
        if not self.en_respaldo:
            # Al entrar, se descartan las referencias nuevas y queda la más vieja de la ventana.
            while len(self.referencias) > 1:
                self.referencias.pop()
            self.respaldos += 1
            self.t_respaldo = t
            motivo += f' (respaldo n.º {self.respaldos})'
        self.en_respaldo = True
        mov = self.mov.desde(self.referencias[-1][1])
        confiable = (mov.avance <= self.c.avance_max_respaldo_m
                     and math.degrees(mov.giro) <= self.c.giro_max_respaldo_deg)
        if not confiable:
            motivo += (f'; en respaldo avanzó {mov.avance:.1f} m y giró '
                       f'{math.degrees(mov.giro):.0f}°: ya no confío en la pose')
        return EstadoPose(self.ultima(), RESPALDO, confiable, mov.avance, motivo)

    def amcl_se_relocalizo(self, t, odom, correccion, sigmas):
        """¿AMCL lleva un rato estable en otro lugar, mientras el robot anduvo, y convergido?"""
        if self.candidata is None or not self.coinciden(
                correccion, self.candidata, odom, self.mov.desde(self.mov_candidata)):
            self.candidata, self.t_candidata = correccion, t
            self.mov_candidata = Movimiento(self.mov.avance, self.mov.giro)
            return False
        return (t - self.t_candidata >= self.c.estable_para_reaceptar_s
                and self.mov.avance - self.mov_candidata.avance >= self.c.avance_para_reaceptar_m
                and self.convergida(sigmas))

    def tomar_referencia(self, correccion):
        """Empieza de cero con esta corrección como única referencia."""
        self.referencias.clear()
        self.agregar_referencia(correccion)
        self.candidata = None
        self.t_coincide = None

    def agregar_referencia(self, correccion):
        self.referencias.append((correccion, Movimiento(self.mov.avance, self.mov.giro)))
        # Se olvidan las de hace más de `ventana_referencias_m`, pero siempre queda una vieja que
        # cubra la ventana entera (si no, la ventana real sería más corta).
        while (len(self.referencias) > 1 and self.mov.avance - self.referencias[1][1].avance
               >= self.c.ventana_referencias_m):
            self.referencias.popleft()

    def sumar_movimiento(self, odom):
        if self.odom_anterior is not None:
            d_m, d_rad = diferencia(odom, self.odom_anterior)
            self.mov.avance += d_m
            self.mov.giro += d_rad
        self.odom_anterior = odom


def texto_sigmas(sigmas):
    if sigmas is None:
        return 'sin /amcl_pose'
    return f'σ {sigmas[0]:.2f} m, {math.degrees(sigmas[1]):.1f}°'
