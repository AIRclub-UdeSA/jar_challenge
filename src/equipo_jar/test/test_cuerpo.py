"""The motion node's self-filter drops only returns inside the real body: the robot's rear is
0.156 m behind base_link, the URDF's 0.181. A wall between the two, right behind the start pose,
is a wall: kept, safe() lets the robot drive away from it and not back into it."""
import numpy as np
import pytest

from equipo_jar.motion import DEFAULTS, safe

motion_node = pytest.importorskip('equipo_jar.motion_node')  # needs equipo_jar_msgs

BODY = motion_node.PARAMS['cuerpo_x']
HALF_Y = DEFAULTS['huella_y']
# A wall 0.165 m behind base_link, across the chassis' width: past the real rear, inside the
# URDF's.
WALL = np.column_stack([np.full(21, -0.165), np.linspace(-0.12, 0.12, 21)])


def test_a_wall_between_the_real_and_the_urdf_rear_is_kept():
    assert outside(WALL).all()


def test_the_cable_and_the_body_are_dropped():
    # The cable, 0.13 m from the lidar head (x 0.043) behind and to the side; and the chassis.
    pts = np.array([[0.043 - 0.12, -0.05], [0.043 - 0.10, 0.08], [-0.15, 0.0], [0.11, 0.09]])
    assert not outside(pts).any()


def test_outside_the_body_is_kept():
    pts = np.array([[0.13, 0.0], [0.0, 0.11], [-0.2, 0.0]])
    assert outside(pts).all()


def test_with_the_wall_kept_backing_up_is_unsafe_and_driving_on_is_safe():
    p = dict(DEFAULTS)
    still = (0.0, 0.0, 0.0)
    assert not safe(WALL, still, (-0.1, 0.0, 0.0), p)
    assert safe(WALL, still, (0.2, 0.0, 0.0), p)


def outside(pts):
    return motion_node.outside_body(pts[:, 0], pts[:, 1], BODY, HALF_Y)
