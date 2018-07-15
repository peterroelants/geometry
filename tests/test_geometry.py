import numpy as np
from numpy import pi

from src.geometry import Pose, Position
from src.origin import origin_pose, origin_units
from src.utils import get_rx


def test_position_create():
    # Test Postion
    assert np.allclose(
        Position.from_vectors([[1, 0, 0]]).m,
        np.asarray([[1, 0, 0, 1]]).T)
    assert np.allclose(
        origin_units.m,
        Position.from_xyz(xs=[0, 1, 0, 0], ys=[0, 0, 1, 0], zs=[0, 0, 0, 1]).m)
    assert np.allclose(
        origin_units.m,
        Position.from_xyz(*origin_units.xyz).m)
    assert np.allclose(
        origin_units.m,
        Position.from_vectors(origin_units.vectors).m)


def test_position_add():
    assert np.allclose(
        (origin_units + origin_units).m,
        np.asarray([
            [0, 2, 0, 0],
            [0, 0, 2, 0],
            [0, 0, 0, 2],
            [1, 1, 1, 1]]))
    assert np.allclose(
        (origin_units + Position.from_vectors([[1, 2, 3]])).m,
        np.asarray([
            [1, 2, 1, 1],
            [2, 2, 3, 2],
            [3, 3, 3, 4],
            [1, 1, 1, 1]]))


def test_position_mul_scalar():
    assert np.allclose(
        (origin_units * 2).m,
        np.asarray([
            [0, 2, 0, 0],
            [0, 0, 2, 0],
            [0, 0, 0, 2],
            [1, 1, 1, 1]]))


def test_postion_neg():
    assert np.allclose(
        (-origin_units).m,
        np.asarray([
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, -1],
            [1, 1, 1, 1]]))


def test_pose_create():
    print(origin_units.vectors[0])
    p = Pose.from_rot_vec(
        get_rx(0), origin_units.vectors[0]
    )
    assert np.allclose(p.m, origin_pose.m)


def test_pose_neg():
    assert np.allclose(origin_pose.m, (-origin_pose).m)


def test_pose_mul():
    assert np.allclose(origin_pose.m, (origin_pose * origin_pose).m)


def test_pose_rot_x():
    assert np.allclose(
        origin_pose.rot_x(pi).m,
        np.asarray([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]]))


def test_pose_position_mul():
    assert np.allclose(
        (origin_pose * Position.from_vectors([[1, 0, 0]])).m,
        np.asarray([[1, 0, 0, 1]]).T)
    assert np.allclose(
        (origin_pose.rot_x(pi) * Position.from_vectors([[1, 0, 0]])).m,
        np.asarray([[1, 0, 0, 1]]).T)
    assert np.allclose(
        (origin_pose.rot_x(pi / 2) * Position.from_vectors([[0, 1, 0]])).m,
        np.asarray([[0, 0, 1, 1]]).T)
    assert np.allclose(
        (origin_pose.rot_y(pi) * Position.from_vectors([[1, 0, 0]])).m,
        np.asarray([[-1, 0, 0, 1]]).T)
    assert np.allclose(
        (origin_pose.rot_z(-pi / 2) * Position.from_vectors([[1, 0, 0]])).m,
        np.asarray([[0, -1, 0, 1]]).T)
    assert np.allclose(
        (origin_pose + Position.from_vectors([[1, 2, 3]])).m,
        np.asarray([
            [1, 0, 0, 1],
            [0, 1, 0, 2],
            [0, 0, 1, 3],
        [0, 0, 0, 1]]))
