import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from src.origin import origin_pose, origin_units


def fig_3d(figsize=(6, 6)):
    fig = plt.figure(figsize=figsize)
    ax = fig.gca(projection='3d')
    setup_ax(ax)
    return fig, ax


def setup_ax(ax):
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_aspect('equal')


def plot_pose(pose, ax, linewidth=2, linestyle='solid'):
    units = pose * origin_units
    o, x, y, z = units.vectors
    ax.plot(*zip(o, x), color='red',
            linewidth=linewidth, linestyle=linestyle)
    ax.plot(*zip(o, y), color='green',
            linewidth=linewidth, linestyle=linestyle)
    ax.plot(*zip(o, z), color='blue',
            linewidth=linewidth, linestyle=linestyle)
    ax.plot(*map(lambda x: [x], o),
            color='black', marker='o')


def plot_origin(ax):
    plot_pose(origin_pose, ax, linewidth=1, linestyle='dashed')


def plot_connect_poses(pose_a, pose_b, ax, color='black', linewidth=2, linestyle='solid'):
    xs, ys, zs = zip(pose_a.xyz, pose_b.xyz)
    ax.plot(xs=xs, ys=ys, zs=zs,
            color=color, linewidth=linewidth, linestyle=linestyle)


def set_apsect_equal(ax):
    """
    Set axes of 3D plot to equal aspect ratios (which is current not done properly in Matplotlib).
    Call after plotting all data.
    """
    x_lim = ax.get_xlim3d()
    y_lim = ax.get_ylim3d()
    z_lim = ax.get_zlim3d()
    # Get ranges and centers
    x_range = abs(x_lim[1] - x_lim[0])
    x_center = (x_lim[0] + x_lim[1]) / 2
    y_range = abs(y_lim[1] - y_lim[0])
    y_center = (y_lim[0] + y_lim[1]) / 2
    z_range = abs(z_lim[1] - z_lim[0])
    z_center = (z_lim[0] + z_lim[1]) / 2
    # Get maximum range and take half as radius from center
    radius = 0.5 * max((x_range, y_range, z_range))
    ax.set_xlim3d((x_center - radius, x_center + radius))
    ax.set_ylim3d((y_center - radius, y_center + radius))
    ax.set_zlim3d((z_center - radius, z_center + radius))
