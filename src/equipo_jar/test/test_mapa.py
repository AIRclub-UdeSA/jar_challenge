import math
from types import SimpleNamespace

from equipo_jar.mapa import cell_value, is_free, line_is_clear

RES = 0.05  # lado de cada celda, como en los mapas de práctica


def make_grid(width, height, origin_x=0.0, origin_y=0.0):
    """OccupancyGrid falso, todo libre: mismos campos que nav_msgs/OccupancyGrid."""
    info = SimpleNamespace(
        resolution=RES, width=width, height=height,
        origin=SimpleNamespace(position=SimpleNamespace(x=origin_x, y=origin_y)))
    return SimpleNamespace(info=info, data=[0] * (width * height))


def set_cell(grid, i, j, value):
    grid.data[j * grid.info.width + i] = value


def center(i, j):
    """Centro, en metros, de la celda (i, j) de un mapa con origen (0, 0)."""
    return ((i + 0.5) * RES, (j + 0.5) * RES)


def wall_map():
    """2 x 2 m. Pared en x de 1.00 a 1.05 (columna 20) desde y = 0.5 hacia arriba:
    abajo queda un hueco. Una celda desconocida en (0.3, 1.5)."""
    grid = make_grid(40, 40)
    for j in range(10, 40):
        set_cell(grid, 20, j, 100)
    set_cell(grid, 6, 30, -1)
    return grid


# --- cell_value ---

def test_cell_value_inside():
    grid = make_grid(10, 10)
    set_cell(grid, 3, 7, 100)
    assert cell_value(grid, *center(3, 7)) == 100


def test_cell_value_goes_row_by_row():
    # La celda de la columna 3 y la fila 1 no es la de la columna 1 y la fila 3.
    grid = make_grid(10, 10)
    set_cell(grid, 3, 1, 100)
    assert cell_value(grid, *center(3, 1)) == 100
    assert cell_value(grid, *center(1, 3)) == 0


def test_cell_value_uses_the_origin():
    grid = make_grid(10, 10, origin_x=-2.0, origin_y=-2.0)
    set_cell(grid, 0, 0, 100)
    assert cell_value(grid, -2.0 + RES / 2, -2.0 + RES / 2) == 100


def test_cell_value_outside_is_none():
    grid = make_grid(10, 10)   # 0.5 x 0.5 m
    assert cell_value(grid, -0.01, 0.2) is None
    assert cell_value(grid, 0.2, -0.01) is None
    assert cell_value(grid, 0.51, 0.2) is None
    assert cell_value(grid, 0.2, 0.51) is None


def test_cell_value_with_missing_data_is_none():
    # Un /map mal armado (menos datos que width x height) no tiene que tirar abajo el nodo.
    grid = make_grid(10, 10)
    grid.data = []
    assert cell_value(grid, 0.2, 0.2) is None


# --- is_free ---

def test_free_far_from_everything():
    assert is_free(wall_map(), 0.5, 1.0, 0.25)


def test_not_free_with_a_wall_inside_clearance():
    assert not is_free(wall_map(), 0.8, 1.0, 0.25)      # pared a 0.20 m


def test_free_with_the_wall_just_outside_clearance():
    assert is_free(wall_map(), 0.74, 1.0, 0.25)         # pared a 0.26 m


def test_not_free_near_the_map_edge():
    assert not is_free(wall_map(), 0.1, 1.0, 0.25)      # borde a 0.10 m


def test_not_free_near_an_unknown_cell():
    assert not is_free(wall_map(), 0.3, 1.4, 0.25)


def test_free_on_the_other_side_of_the_wall():
    assert is_free(wall_map(), 1.5, 1.0, 0.25)


def test_zero_clearance_checks_only_that_cell():
    grid = wall_map()
    assert is_free(grid, 0.5, 1.0, 0.0)
    assert not is_free(grid, 1.02, 1.0, 0.0)


# --- line_is_clear ---

def test_line_through_the_wall_is_blocked():
    assert not line_is_clear(wall_map(), 0.5, 1.0, 1.5, 1.0)


def test_line_that_stops_before_the_wall_is_clear():
    assert line_is_clear(wall_map(), 0.5, 1.0, 0.9, 1.0)


def test_line_through_the_gap_is_clear():
    assert line_is_clear(wall_map(), 0.5, 0.2, 1.5, 0.2)


def test_diagonal_through_the_wall_is_blocked():
    assert not line_is_clear(wall_map(), 0.5, 1.8, 1.5, 0.2)


def test_unknown_cells_do_not_block_the_line():
    # Lo desconocido no tapa la vista; que el robot quepa lo revisa is_free.
    assert line_is_clear(wall_map(), 0.1, 1.5, 0.5, 1.5)


def test_line_of_one_point_does_not_divide_by_zero():
    grid = wall_map()
    assert line_is_clear(grid, 0.5, 1.0, 0.5, 1.0)
    assert not line_is_clear(grid, 1.02, 1.0, 1.02, 1.0)


def test_line_starting_outside_the_map():
    assert line_is_clear(wall_map(), -0.5, 1.0, 0.5, 1.0)
