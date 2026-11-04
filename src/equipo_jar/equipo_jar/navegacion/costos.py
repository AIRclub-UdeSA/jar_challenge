"""Inflado por huella, costo blando y obstáculos vistos que no están en el mapa."""
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class ParametrosCosto:
    radio_inscripto_m: float
    radio_circunscripto_m: float
    margen_m: float
    alcance_blando_m: float
    peso: float

    def __post_init__(self):
        valores = (self.radio_inscripto_m, self.radio_circunscripto_m, self.margen_m,
                   self.alcance_blando_m, self.peso)
        if not all(math.isfinite(v) and v > 0 for v in valores):
            raise ValueError('Parámetros de costo deben ser positivos y finitos')
        if self.radio_inscripto_m > self.radio_circunscripto_m:
            raise ValueError('El radio inscripto supera al circunscripto')


def distancia_a_ocupado(ocupada, resolucion, maximo_m=1.0):
    """Distancia euclídea entre centros de celda, saturada en maximo_m.

    Afuera del mapa cuenta como ocupado. Recorre desplazamientos en orden de
    distancia, así cada celda recibe la del ocupado más cercano.
    """
    alto, ancho = ocupada.shape
    r = max(1, math.ceil(maximo_m / resolucion))
    borde = np.pad(ocupada, r, constant_values=True)
    desplazamientos = sorted(
        (math.hypot(dx, dy), dy, dx)
        for dy in range(-r, r + 1) for dx in range(-r, r + 1)
        if math.hypot(dx, dy) * resolucion <= maximo_m)
    distancia = np.full((alto, ancho), float(maximo_m))
    pendiente = np.ones((alto, ancho), dtype=bool)
    for d, dy, dx in desplazamientos:
        vecino = borde[r + dy:r + dy + alto, r + dx:r + dx + ancho]
        nuevo = pendiente & vecino
        if nuevo.any():
            distancia[nuevo] = d * resolucion
            pendiente &= ~nuevo
            if not pendiente.any():
                break
    return distancia


def mapa_costos(grilla, p, extra_ocupadas=None):
    """Devuelve (costos, distancia): inf: cuerpo completo y margen; >= 0: costo blando."""
    ocupada = grilla.ocupada if extra_ocupadas is None else grilla.ocupada | extra_ocupadas
    distancia = distancia_a_ocupado(ocupada, grilla.resolucion)
    duro = p.radio_circunscripto_m + p.margen_m + grilla.resolucion / 2
    costos = p.peso * np.clip(1.0 - (distancia - duro) / p.alcance_blando_m, 0.0, None)
    costos[distancia <= duro] = np.inf
    return costos, distancia


def _minimos_intervalos(values,lo,hi):
    """Mínimos exactos por intervalos solapados; tabla dispersa de NumPy."""
    lengths=hi-lo
    levels=np.frexp(lengths)[1]-1
    table=[values]
    for level in range(1,int(levels.max())+1):
        half=1<<(level-1)
        previous=table[-1]
        table.append(np.minimum(previous[:-half],previous[half:]))
    result=np.empty(len(lo),dtype=float)
    for level in np.unique(levels):
        mask=levels==level
        result[mask]=np.minimum(table[level][lo[mask]],
                               table[level][hi[mask]-(1<<int(level))])
    return result


class CapaViva:
    """Celdas con puntos del LiDAR que el mapa no explica; vencen solas."""

    def __init__(self, grilla, distancia_mapa, vida_s, tolerancia_mapa_m):
        if not (math.isfinite(vida_s) and vida_s > 0 and math.isfinite(tolerancia_mapa_m)):
            raise ValueError('Parámetros de capa viva inválidos')
        self.grilla = grilla
        self.distancia_mapa = distancia_mapa
        self.vida_s = vida_s
        self.tolerancia = tolerancia_mapa_m
        self.marca = np.full((grilla.alto, grilla.ancho), -np.inf)

    def actualizar(self, ahora, puntos_mapa, origen=None, incremento_angular=None):
        puntos=np.asarray(puntos_mapa,dtype=float).reshape((-1,2))
        puntos=puntos[np.isfinite(puntos).all(axis=1)]
        if not len(puntos):return
        if origen is not None and incremento_angular is not None:
            self._liberar_observadas(ahora,puntos,origen,incremento_angular)
        g=self.grilla;c,s=math.cos(g.origen_yaw),math.sin(g.origen_yaw)
        delta=puntos-np.array([g.origen_x,g.origen_y])
        col=np.floor((c*delta[:,0]+s*delta[:,1])/g.resolucion).astype(int)
        fila=np.floor((-s*delta[:,0]+c*delta[:,1])/g.resolucion).astype(int)
        valid=(col>=0)&(col<g.ancho)&(fila>=0)&(fila<g.alto)
        fila,col=fila[valid],col[valid]
        unknown=self.distancia_mapa[fila,col]>self.tolerancia
        f,k=fila[unknown],col[unknown]
        self.marca[f,k]=np.maximum(self.marca[f,k],ahora)

    def _liberar_observadas(self,ahora,puntos,origen,incremento):
        # Mismo criterio de celda completa: ninguna oclusión ni hueco.
        if not math.isfinite(incremento) or incremento<=0:return
        origen=np.asarray(origen,dtype=float)
        if origen.shape!=(2,) or not np.isfinite(origen).all():return
        cells=np.argwhere(ahora-self.marca<self.vida_s)
        if not len(cells):return
        g=self.grilla;cell_radius=g.resolucion/math.sqrt(2)
        c,s=math.cos(g.origen_yaw),math.sin(g.origen_yaw)
        u=(cells[:,1]+.5)*g.resolucion;v=(cells[:,0]+.5)*g.resolucion
        centers=np.column_stack((g.origen_x+c*u-s*v,g.origen_y+s*u+c*v))-origen
        distances=np.hypot(centers[:,0],centers[:,1])
        valid=distances>cell_radius
        cells,centers,distances=cells[valid],centers[valid],distances[valid]
        if not len(cells):return
        angles=np.arctan2(centers[:,1],centers[:,0])
        widths=np.arcsin(np.minimum(1.,cell_radius/distances))
        rel=puntos-origen;ranges=np.hypot(rel[:,0],rel[:,1])
        beams=np.arctan2(rel[:,1],rel[:,0]);order=np.argsort(beams)
        beams,ranges=beams[order],ranges[order]
        beams=np.concatenate((beams-2*math.pi,beams,beams+2*math.pi))
        ranges=np.tile(ranges,3)
        lo=np.searchsorted(beams,angles-widths-incremento)
        hi=np.searchsorted(beams,angles+widths+incremento,side='right')
        valid=hi-lo>=2
        cells,distances,angles,widths,lo,hi=[a[valid] for a in (cells,distances,angles,widths,lo,hi)]
        if not len(cells):return
        # Prefijo de huecos: equivale a comprobar todos los saltos del sector.
        gaps=np.r_[0,np.cumsum(np.diff(beams)>1.5*incremento)]
        valid=(beams[lo]<=angles-widths)&(beams[hi-1]>=angles+widths)&(gaps[hi-1]==gaps[lo])
        cells,distances,lo,hi=[a[valid] for a in (cells,distances,lo,hi)]
        if not len(cells):return
        clear=_minimos_intervalos(ranges,lo,hi)>distances+cell_radius+1e-9
        self.marca[cells[clear,0],cells[clear,1]]=-np.inf

    def registrar_bloqueo(self, ahora, puntos_mapa):
        """Un retorno que detuvo el barrido prevalece sobre la tolerancia al mapa.

        No cambia el mapa estático. La misma evidencia de espacio libre usada
        para las otras marcas puede retirar estas observaciones después.
        """
        for x,y in puntos_mapa:
            if not (math.isfinite(x) and math.isfinite(y)):continue
            cell=self.grilla.a_celda(x,y)
            if cell is not None and not self.grilla.ocupada[cell]:
                self.marca[cell]=max(self.marca[cell],ahora)

    def registrar_fija(self, puntos_mapa):
        """Obstáculo conocido (p. ej. caja ya reportada): no vence por tiempo.

        Sólo la evidencia de espacio libre del LiDAR puede retirarlo, igual que
        las demás marcas; así una posición reportada con error no cierra un paso.
        """
        for x,y in puntos_mapa:
            if not (math.isfinite(x) and math.isfinite(y)):continue
            cell=self.grilla.a_celda(x,y)
            if cell is not None and not self.grilla.ocupada[cell]:
                self.marca[cell]=np.inf

    def ocupadas(self, ahora):
        return ahora - self.marca < self.vida_s
