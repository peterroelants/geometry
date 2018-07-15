import numpy as np
from numbers import Number

from .utils import get_rx, get_ry, get_rz


class Position:
    def __init__(self, m):
        """
        m is column vector
        """
        assert len(m.shape) == 2, '`m` must have 2 dimensions: {}'.format(
            m.shape)
        assert m.shape[0] == 4, '`m` must have 4 rows'
        assert (m[3, :] == 1.).all(), 'last row of `m` must be 1s'
        self.m = m

    @staticmethod
    def from_xyz(xs, ys, zs):
        assert len(xs) == len(ys) == len(zs), \
            'length of xs, ys, zs must be the same'
        m = np.asarray([
            xs, ys, zs, [1. for _ in xs]
        ])
        return Position(m)

    @staticmethod
    def from_vectors(vecs):
        m = np.vstack(vecs)
        m = np.hstack((m, np.ones((len(vecs), 1))))
        return Position(m.T)

    @property
    def xyz(self):
        return self.m[0, :], self.m[1, :], self.m[2, :]

    @property
    def vectors(self):
        return [
            self.m[0:3, i] for i in range(self.m.shape[1])
        ]

    def __add__(self, other):
        if isinstance(other, Position):
            new_m = self.m + other.m
            new_m[3, :] = 1.
            return Position(new_m)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, Number):
            new_m = self.m * other
            new_m[3, :] = 1.
            return Position(new_m)
        return NotImplemented

    def __sub__(self, other):
        return self + (-1. * other)

    def __neg__(self):
        return self * -1


class Pose:
    def __init__(self, m):
        assert len(m.shape) == 2, \
            '`m` must have 2 dimensions: {}'.format(m.shape)
        assert m.shape == (4, 4), '`m` must have 4 rows and 4 columns'
        assert (m[3, 0:3] == 0.).all(), \
            'last 3 elements of last row of `m` must be 0s'
        assert m[3, 3] == 1., 'last element must by 1.'
        self.m = m

    @staticmethod
    def from_rot_vec(r, v):
        m = np.zeros((4, 4))
        m[0:3, 0:3] = r
        m[0:2, 3] = v[0:2]
        m[3, 3] = 1
        return Pose(m)

    @property
    def xyz(self):
        return self.m[0:3, 3]

    def __mul__(self, other):
        if isinstance(other, Pose):
            new_m = self.m.dot(other.m)
            return Pose(new_m)
        if isinstance(other, Position):
            return Position(self.m.dot(other.m))
        return NotImplemented

    def __add__(self, other):
        if isinstance(other, Position) and (other.m.shape[1] == 1):
            new_m = self.m.copy()
            new_m[0:3, 3] += other.m[0:3, 0]
            return Pose(new_m)
        return NotImplemented

    def __neg__(self):
        return Pose(np.linalg.inv(self.m))

    def rot_x(self, rad):
        r = get_rx(rad)
        m = self.m.copy()
        m[0:3, 0:3] = m[0:3, 0:3].dot(r)
        return Pose(m)

    def rot_y(self, rad):
        r = get_ry(rad)
        m = self.m.copy()
        m[0:3, 0:3] = m[0:3, 0:3].dot(r)
        return Pose(m)

    def rot_z(self, rad):
        r = get_rz(rad)
        m = self.m.copy()
        m[0:3, 0:3] = m[0:3, 0:3].dot(r)
        return Pose(m)
