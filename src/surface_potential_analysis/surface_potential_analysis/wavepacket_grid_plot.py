from typing import Literal

import matplotlib.animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
from matplotlib.figure import Figure

from surface_potential_analysis.surface_config import SurfaceConfigUtil
from surface_potential_analysis.surface_config_plot import (
    plot_ft_points_on_surface_xy,
    plot_points_on_surface_x0z,
    plot_points_on_surface_xy,
)

from .wavepacket_grid import WavepacketGrid


def plot_wavepacket_grid_xy(
    grid: WavepacketGrid,
    z_ind=0,
    *,
    ax: Axes | None = None,
    measure: Literal["real", "imag", "abs"] = "abs",
    norm: Literal["symlog", "linear"] = "symlog",
) -> tuple[Figure, Axes, QuadMesh]:

    fig, ax, mesh = plot_points_on_surface_xy(
        grid, grid["points"], z_ind, ax=ax, measure=measure
    )
    ax.set_xlabel("x direction")
    ax.set_ylabel("y direction")
    mesh.set_norm(norm)  # type: ignore
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(mesh, ax=ax, format="%4.1e")

    return fig, ax, mesh


def plot_ft_wavepacket_grid_xy(
    grid: WavepacketGrid,
    z_ind=0,
    *,
    ax: Axes | None = None,
    measure: Literal["real", "imag", "abs"] = "abs",
    norm: Literal["symlog", "linear"] = "linear",
) -> tuple[Figure, Axes, QuadMesh]:

    fig, ax, mesh = plot_ft_points_on_surface_xy(
        grid, grid["points"], z_ind, ax=ax, measure=measure
    )
    ax.set_xlabel("kx direction")
    ax.set_ylabel("ky direction")
    mesh.set_norm(norm)  # type: ignore
    fig.colorbar(mesh, ax=ax, format="%4.1e")

    return fig, ax, mesh


def animate_wavepacket_grid_3D_in_xy(
    grid: WavepacketGrid,
    *,
    ax: Axes | None = None,
    measure: Literal["real", "imag", "abs"] = "abs",
    norm: Literal["symlog", "linear"] = "symlog",
) -> tuple[Figure, Axes, matplotlib.animation.ArtistAnimation]:
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    _, _, mesh0 = plot_points_on_surface_xy(
        grid, grid["points"], 0, ax=ax, measure=measure
    )

    frames: list[list[QuadMesh]] = []
    for z_ind in range(np.shape(grid["points"])[2]):

        _, _, mesh = plot_points_on_surface_xy(
            grid, grid["points"], z_ind, ax=ax, measure=measure
        )
        frames.append([mesh])

    max_clim = np.max([i[0].get_clim()[1] for i in frames])
    min_clim = 0 if measure == "abs" else np.min([i[0].get_clim()[0] for i in frames])
    for (mesh,) in frames:
        mesh.set_norm(norm)  # type: ignore
        mesh.set_clim(min_clim, max_clim)
    mesh0.set_norm(norm)  # type: ignore
    mesh0.set_clim(min_clim, max_clim)

    ani = matplotlib.animation.ArtistAnimation(fig, frames)

    ax.set_xlabel("X direction")
    ax.set_ylabel("Y direction")
    ax.set_aspect("equal", adjustable="box")

    fig.colorbar(mesh0, ax=ax, format="%4.1e")

    return (fig, ax, ani)


def animate_ft_wavepacket_grid_3D_in_xy(
    grid: WavepacketGrid,
    *,
    ax: Axes | None = None,
    measure: Literal["real", "imag", "abs"] = "abs",
    norm: Literal["symlog", "linear"] = "symlog",
) -> tuple[Figure, Axes, matplotlib.animation.ArtistAnimation]:

    util = SurfaceConfigUtil(grid)
    ft_points = ft_points = np.fft.ifft2(grid["points"], axes=(0, 1))
    ft_grid: WavepacketGrid = {
        "delta_x0": util.dkx0,
        "delta_x1": util.dkx1,
        "z_points": grid["z_points"],
        "points": ft_points.tolist(),
    }
    (fig, ax, ani) = animate_wavepacket_grid_3D_in_xy(
        ft_grid, ax=ax, measure=measure, norm=norm
    )

    ax.set_xlabel("kx Direction")
    ax.set_ylabel("ky Direction")

    return fig, ax, ani


def plot_wavepacket_grid_x0z(
    grid: WavepacketGrid,
    x1_ind: int,
    *,
    measure: Literal["real", "imag", "abs"] = "abs",
    norm: Literal["symlog", "linear"] = "symlog",
    ax: Axes | None = None,
) -> tuple[Figure, Axes, QuadMesh]:

    fig, ax, mesh = plot_points_on_surface_x0z(
        grid, grid["points"], grid["z_points"], x1_ind=x1_ind, ax=ax, measure=measure
    )

    ax.set_xlabel("x direction")
    ax.set_ylabel("z direction")
    mesh.set_norm(norm)  # type: ignore
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(mesh, ax=ax, format="%4.1e")
    return fig, ax, mesh


def animate_wavepacket_grid_3D_in_x0z(
    grid: WavepacketGrid,
    *,
    ax: Axes | None = None,
    measure: Literal["real", "imag", "abs"] = "abs",
    norm: Literal["symlog", "linear"] = "symlog",
) -> tuple[Figure, Axes, matplotlib.animation.ArtistAnimation]:
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    _, _, mesh0 = plot_wavepacket_grid_x0z(grid, 0, ax=ax, measure=measure)

    frames: list[list[QuadMesh]] = []
    for x1_ind in range(np.shape(grid["points"])[1]):

        _, _, mesh = plot_wavepacket_grid_x0z(grid, x1_ind, ax=ax, measure=measure)
        frames.append([mesh])

    max_clim = np.max([i[0].get_clim()[1] for i in frames])
    min_clim = 0 if measure == "abs" else np.min([i[0].get_clim()[0] for i in frames])
    for (mesh,) in frames:
        mesh.set_norm(norm)  # type: ignore
        mesh.set_clim(min_clim, max_clim)
    mesh0.set_norm(norm)  # type: ignore
    mesh0.set_clim(min_clim, max_clim)

    ani = matplotlib.animation.ArtistAnimation(fig, frames)

    ax.set_xlabel("X direction")
    ax.set_ylabel("Z direction")

    fig.colorbar(mesh0, ax=ax, format="%4.1e")

    return (fig, ax, ani)


def plot_wavepacket_grid_x1(
    grid: WavepacketGrid,
    x2_ind=0,
    z_ind=0,
    ax: Axes | None = None,
    measure: Literal["real", "imag", "abs"] = "abs",
):
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()
    points = np.array(grid["points"])[:, x2_ind, z_ind]

    if measure == "real":
        data = np.real(points)
    elif measure == "imag":
        data = np.imag(points)
    else:
        data = np.abs(points)

    x1_points = np.linspace(0, np.linalg.norm(grid["delta_x1"]), data.shape[0])
    (line,) = ax.plot(x1_points, data)
    return fig, ax, line
