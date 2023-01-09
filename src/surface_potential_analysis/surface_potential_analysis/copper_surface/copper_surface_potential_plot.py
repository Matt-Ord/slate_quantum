import numpy as np

from ..energy_data.energy_data import normalize_energy
from ..energy_data.plot_energy_data import (
    plot_xz_plane_energy,
    plot_z_direction_energy_comparison,
    plot_z_direction_energy_data,
)
from .copper_surface_data import save_figure
from .copper_surface_potential import (
    load_9h_copper_data,
    load_interpolated_copper_data,
    load_nc_raw_copper_data,
    load_raw_copper_data,
)


def plot_copper_raw_data():
    data = load_raw_copper_data()
    data = normalize_energy(data)

    fig, ax = plot_z_direction_energy_data(data)
    ax.set_ylim(bottom=0, top=1e-18)
    fig.show()
    save_figure(fig, "copper_raw_data_z_direction.png")

    plot_xz_plane_energy(data)


def plot_copper_nc_data():
    data = normalize_energy(load_nc_raw_copper_data())

    fig, ax = plot_z_direction_energy_data(data)
    ax.set_ylim(bottom=-0.1e-18, top=1e-18)
    fig.show()
    input()
    save_figure(fig, "copper_raw_data_z_direction_nc.png")


def plot_copper_9h_data():
    data = normalize_energy(load_9h_copper_data())
    # data_s = normalize_energy(load_simple_copper_data())
    data_7h = normalize_energy(load_raw_copper_data())

    fig, ax = plot_z_direction_energy_data(data)
    plot_z_direction_energy_data(data_7h, ax)
    ax.set_ylim(bottom=-0.1e-18, top=1e-18)

    fig.show()
    input()
    save_figure(fig, "copper_raw_data_z_direction_9h.png")


def plot_copper_interpolated_data():
    data = load_interpolated_copper_data()
    raw_data = normalize_energy(load_raw_copper_data())

    fig, ax = plot_z_direction_energy_comparison(data, raw_data)
    ax.set_ylim(bottom=0, top=1e-18)
    fig.show()
    save_figure(fig, "copper_interpolated_data_comparison.png")

    fig = plot_xz_plane_energy(data)
    fig.show()
    input()
    save_figure(fig, "copper_interpolated_data_xy.png")


def compare_bridge_hollow_energy():
    data = load_interpolated_copper_data()
    points = np.array(data["points"])
    print(points.shape)

    print(np.min(points[points.shape[0] // 2, 0, :]))
    print(np.min(points[points.shape[0] // 2, points.shape[1] // 2, :]))
    print(np.min(points[0, 0, :]))
    print(np.max(points[:, :, -1]))
    print(
        np.max(
            np.abs(points[:, :, -1] - np.max(points[:, :, -1]))
            / np.max(points[:, :, -1])
        )
    )

    data = normalize_energy(load_raw_copper_data())
    points = np.array(data["points"])
    print(points.shape)

    print(np.min(points[points.shape[0] // 2, 0, :]))
    print(np.min(points[points.shape[0] // 2, points.shape[1] // 2, :]))
    print(np.min(points[0, 0, :]))
    print(np.max(points[:, :, -1]))
    print(
        np.max(
            np.abs(points[:, :, -1] - np.max(points[:, :, -1]))
            / np.max(points[:, :, -1])
        )
    )
