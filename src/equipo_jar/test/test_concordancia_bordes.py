"""Rayos junto a esquinas: tolerancia espacial sin ocultar paredes atravesadas."""
from types import SimpleNamespace as S

from equipo_jar.localizacion_activa import concordancia_mapa


def escena(alta=False):
    width, height, resolution = 100, 60, .05
    cells = [0] * (width * height)
    for row in range(height):
        cells[row * width + 80] = 100  # pared del fondo, x=4 m
        if row < (40 if alta else 20):
            cells[row * width + 40] = 100  # esquina o pared continua, x=2 m
    return S(info=S(width=width, height=height, resolution=resolution,
                    origin=S(position=S(x=0., y=0.),
                             orientation=S(x=0., y=0., z=0., w=1.))), data=cells)


def test_error_de_35_mm_junto_a_esquina_no_inventa_pared_atravesada():
    # Pose verdadera y=1.025: el rayo pasa por encima de la esquina y llega
    # al fondo. Una estimación y=.99 difiere 35 mm, dentro de tolerancia=.10.
    assert concordancia_mapa(escena(), [(0, 3.)], 0., 0., (1., .99, 0.))[0] == 1.


def test_pared_continua_antes_del_fondo_sigue_siendo_contradiccion():
    assert concordancia_mapa(escena(True), [(0, 3.)], 0., 0., (1., .99, 0.))[0] == 0.
