"""Pruebas de la lectura de mapas sin ejecutar ROS."""

import numpy as np

from jar_localizacion.mapa import cargar_mapa


def test_trinario_y_fila_cero_abajo(tmp_path):
    """Negro ocupado, blanco libre, gris desconocido; la imagen se invierte."""
    pixeles = np.array([[0, 254], [128, 254]], dtype=np.uint8)
    (tmp_path / 'm.pgm').write_bytes(b'P5\n# comentario\n2 2\n255\n' + pixeles.tobytes())
    (tmp_path / 'm.yaml').write_text(
        'image: m.pgm\nresolution: 0.05\norigin: [-1.0, 2.0, 0.0]\nnegate: 0\n'
        'occupied_thresh: 0.65\nfree_thresh: 0.25\n')
    grilla, resolucion, origen = cargar_mapa(tmp_path / 'm.yaml')
    assert grilla.tolist() == [[-1, 0], [100, 0]]
    assert resolucion == 0.05 and origen == [-1.0, 2.0]
