import numpy as np

from .geometry import Position, Pose


# Create unit vectors at origin
origin_units = Position.from_vectors([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])

origin_pose = Pose(np.asarray([
    [1., 0., 0., 0.],
    [0., 1., 0., 0.],
    [0., 0., 1., 0.],
    [0., 0., 0., 1.]
]))
