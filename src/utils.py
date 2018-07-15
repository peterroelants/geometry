import numpy as np
from numpy import sin, cos


# 3d rotations
def get_rx(rad):
    """
    A basic rotation matrix around the x-axis
    Rotate theta radians around the clockwise rotation.
    """
    return np.asarray((
        (1., 0.,       0.),
        (0., cos(rad), -sin(rad)),
        (0., sin(rad), cos(rad))))


def get_ry(rad):
    """
    A basic rotation matrix around the y-axis
    Rotate theta radians around the clockwise rotation.
    """
    return np.asarray((
        (cos(rad),   0.,  sin(rad)),
        (0.,         1.,  0.),
        (-sin(rad),  0.,  cos(rad))))


def get_rz(rad):
    """
    A basic rotation matrix around the z-axis
    Rotate theta radians around the clockwise rotation.
    """
    return np.asarray((
        (cos(rad), -sin(rad),  0.),
        (sin(rad), cos(rad),   0.),
        (0.,       0.,         1.)))
