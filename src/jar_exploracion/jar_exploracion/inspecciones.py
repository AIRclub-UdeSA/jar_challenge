"""Cuántas veces se mandó a mirar cada candidato, y cuáles quedaron dudosos (sin ROS).

Cada candidato se inspecciona como mucho `intentos_max` veces: si no, uno inalcanzable o que
desde ningún lado se ve bien se volvía a proponer sin parar. Un intento se cuenta al llegar
(visitado), cuando la misión no pudo llegar (descartado) o cuando no hay pose desde donde
mirarlo (se agota).

Con `seguir_dudosos`:
- una visita sin ningún voto nuevo (el robot llegó girando, quedó fuera de la imagen) no gasta
  el intento, hasta `visitas_sin_voto_max` seguidas;
- una visita con votos pero sin decisión sí lo gasta, y su punto de vista no se repite: el
  próximo intento lo mira desde otro lado;
- un candidato que agotó los intentos sin decidirse es dudoso: cuando la cobertura termina,
  se lo manda a mirar una vez más (`revisar_dudosos`). Si sigue sin decidirse, la percepción lo
  ofrece para los cupos que sobren al cierre;
- un goal de inspección que la misión descartó con el robot lejos del candidato (más de
  `descarte_lejos_m`) no es una visita: el camino falló, no la mirada. No gasta el intento hasta
  `descartes_lejos_max` (laberinto_puertas: trabado en la puerta de E, V3 gastó sus dos intentos
  sin que el robot la viera nunca);
- un dudoso que queda a menos de `dudoso_de_paso_m` del robot recibe un intento más, una vez
  (`de_paso`): se lo mira ahora, en vez de irse y volver al final (V3, a 0.5 m, no se miró).
"""
import math


def _votos(cand):
    return int(cand.get('rojo', 0)) + int(cand.get('no_rojo', 0))


class Inspecciones:
    def __init__(self, intentos_max=2, seguir_dudosos=False, visitas_sin_voto_max=2,
                 descarte_lejos_m=1.0, descartes_lejos_max=3, dudoso_de_paso_m=1.0):
        self.intentos_max = int(intentos_max)
        self.seguir_dudosos = bool(seguir_dudosos)
        self.visitas_sin_voto_max = int(visitas_sin_voto_max)
        self.descarte_lejos_m = float(descarte_lejos_m)
        self.descartes_lejos_max = int(descartes_lejos_max)
        self.dudoso_de_paso_m = float(dudoso_de_paso_m)
        self.descartes_lejos = {}     # id -> goals descartados lejos desde el último intento
        self.de_paso_dados = set()    # dudosos que ya recibieron el intento de paso
        self.intentos = {}            # id -> intentos gastados
        self.sin_voto = {}            # id -> visitas seguidas sin votos nuevos
        self.votos_vistos = {}        # id -> votos (rojo + no_rojo) en la última visita
        self.puntos_sin_decidir = []  # (x, y) de visitas que votaron sin decidir
        self.revisados = set()        # dudosos que ya se mandaron a mirar al final

    def pendientes(self, candidatos):
        """Los sin inspeccionar que todavía tienen intentos: id -> (x, y)."""
        return {c['id']: (c['x'], c['y']) for c in candidatos
                if c.get('estado') == 'sin_inspeccionar'
                and self.intentos.get(c['id'], 0) < self.intentos_max}

    def contar(self, ident):
        self.intentos[ident] = self.intentos.get(ident, 0) + 1

    def agotar(self, ident):
        self.intentos[ident] = self.intentos_max

    def uno_mas(self, ident):
        """Le devuelve un intento (si los gastó todos)."""
        self.intentos[ident] = min(self.intentos.get(ident, 0), self.intentos_max - 1)

    def descartado(self, ident, candidatos, pose):
        """La misión no pudo llegar al punto de inspección de `ident`; `pose`: (x, y) del robot."""
        cand = next((c for c in candidatos if c['id'] == ident), None)
        if (not self.seguir_dudosos or cand is None or pose is None
                or math.hypot(cand['x'] - pose[0], cand['y'] - pose[1]) <= self.descarte_lejos_m):
            return self.contar(ident)
        self.descartes_lejos[ident] = self.descartes_lejos.get(ident, 0) + 1
        if self.descartes_lejos[ident] >= self.descartes_lejos_max:
            self.descartes_lejos[ident] = 0
            self.contar(ident)

    def de_paso(self, candidatos, pose):
        """Un intento más a los dudosos cerca del robot, una vez a cada uno. Devuelve sus ids."""
        if not self.seguir_dudosos or pose is None:
            return []
        cerca = [c['id'] for c in candidatos if c['id'] in self.dudosos([c])
                 and c['id'] not in self.de_paso_dados
                 and math.hypot(c['x'] - pose[0], c['y'] - pose[1]) < self.dudoso_de_paso_m]
        for k in cerca:
            self.de_paso_dados.add(k)
            self.uno_mas(k)
            self.sin_voto[k] = 0
        return cerca

    def visitado(self, ident, candidatos, punto):
        """La misión llegó al punto de inspección de `ident` y terminó de mirar."""
        if not self.seguir_dudosos:
            return self.contar(ident)
        cand = next((c for c in candidatos if c['id'] == ident), None)
        if cand is None or cand.get('estado') != 'sin_inspeccionar':
            return self.contar(ident)             # ya se decidió (o se borró)
        votos = _votos(cand)
        if votos > self.votos_vistos.get(ident, 0):
            self.votos_vistos[ident] = votos
            self.sin_voto[ident] = 0
            self.puntos_sin_decidir.append((float(punto[0]), float(punto[1])))
            return self.contar(ident)
        self.sin_voto[ident] = self.sin_voto.get(ident, 0) + 1
        if self.sin_voto[ident] >= self.visitas_sin_voto_max:
            self.sin_voto[ident] = 0
            self.contar(ident)

    def dudosos(self, candidatos):
        """Sin inspeccionar y sin intentos: no se pudo decidir."""
        return [c['id'] for c in candidatos if c.get('estado') == 'sin_inspeccionar'
                and self.intentos.get(c['id'], 0) >= self.intentos_max]

    def revisar_dudosos(self, candidatos):
        """Con la cobertura terminada: un intento más a cada dudoso, una sola vez. Devuelve los
        ids que vuelven a quedar pendientes."""
        if not self.seguir_dudosos:
            return []
        nuevos = [k for k in self.dudosos(candidatos) if k not in self.revisados]
        for k in nuevos:
            self.revisados.add(k)
            self.uno_mas(k)
            self.sin_voto[k] = 0
        return nuevos
