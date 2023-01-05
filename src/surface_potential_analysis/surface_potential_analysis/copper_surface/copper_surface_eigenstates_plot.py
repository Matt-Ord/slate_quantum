import matplotlib.pyplot as plt
import numpy as np

from ..energy_data.energy_eigenstates import (
    get_eigenstate_list,
    load_energy_eigenstates,
)
from ..energy_data.plot_energy_eigenstates import plot_lowest_band_in_kx
from ..plot_surface_hamiltonian import (
    plot_eigenvector_through_bridge,
    plot_eigenvector_z,
)
from .copper_surface_data import get_data_path, save_figure
from .copper_surface_hamiltonian import generate_hamiltonian


def analyze_eigenvalue_convergence():

    fig, ax = plt.subplots()

    path = get_data_path("copper_eigenstates_12_12_10.json")
    eigenstates = load_energy_eigenstates(path)
    _, _, ln = plot_lowest_band_in_kx(eigenstates, ax=ax)
    ln.set_label("(12,12,10)")

    path = get_data_path("copper_eigenstates_12_12_12.json")
    eigenstates = load_energy_eigenstates(path)
    _, _, ln = plot_lowest_band_in_kx(eigenstates, ax=ax)
    ln.set_label("(12,12,12)")

    path = get_data_path("copper_eigenstates_12_12_14.json")
    eigenstates = load_energy_eigenstates(path)
    _, _, ln = plot_lowest_band_in_kx(eigenstates, ax=ax)
    ln.set_label("(12,12,14)")

    path = get_data_path("copper_eigenstates_12_12_15.json")
    eigenstates = load_energy_eigenstates(path)
    _, _, ln = plot_lowest_band_in_kx(eigenstates, ax=ax)
    ln.set_label("(12,12,15)")

    path = get_data_path("copper_eigenstates_10_10_15.json")
    eigenstates = load_energy_eigenstates(path)
    _, _, ln = plot_lowest_band_in_kx(eigenstates, ax=ax)
    ln.set_label("(10,10,15)")

    ax.set_title(
        "Plot of energy against k for the lowest band of Copper for $K_y=0$\n"
        "showing convergence to about 2x$10^{-30}$J "
    )
    ax.set_xlabel("K /$m^{-1}$")
    ax.set_ylabel("energy / J")
    ax.legend()

    fig.tight_layout()
    fig.show()
    save_figure(fig, "copper_lowest_band_convergence.png")


def analyze_eigenvector_convergence_z():

    fig, ax = plt.subplots()

    path = get_data_path("copper_eigenstates_10_10_15.json")
    eigenstates = load_energy_eigenstates(path)
    h = generate_hamiltonian(resolution=eigenstates["resolution"])
    _, _, ln = plot_eigenvector_z(h, get_eigenstate_list(eigenstates)[0], ax=ax)
    ln.set_label("(10,10,15) kx=G/2")

    path = get_data_path("copper_eigenstates_12_12_14.json")
    eigenstates = load_energy_eigenstates(path)
    h2 = generate_hamiltonian(resolution=eigenstates["resolution"])
    _, _, l2 = plot_eigenvector_z(h2, get_eigenstate_list(eigenstates)[0], ax=ax)
    l2.set_label("(12,12,14) kx=G/2")

    path = get_data_path("copper_eigenstates_12_12_15.json")
    eigenstates = load_energy_eigenstates(path)
    h = generate_hamiltonian(resolution=eigenstates["resolution"])
    _, _, ln = plot_eigenvector_z(h, get_eigenstate_list(eigenstates)[0], ax=ax)
    ln.set_label("(12,12,15) kx=G/2")

    ax.set_title(
        "Plot of energy against k for the lowest band of Copper for $K_y=0$\n"
        "showing convergence to about 2x$10^{-30}$J "
    )
    ax.legend()
    fig.show()
    save_figure(fig, "copper_wfn_convergence.png")


def analyze_eigenvector_convergence_through_bridge():

    path = get_data_path("copper_eigenstates_12_12_15.json")
    eigenstates = load_energy_eigenstates(path)

    fig, ax = plt.subplots()
    ax2 = ax.twinx()

    path = get_data_path("copper_eigenstates_12_12_15.json")
    eigenstates = load_energy_eigenstates(path)
    h = generate_hamiltonian(resolution=eigenstates["resolution"])
    _, _, ln = plot_eigenvector_through_bridge(
        h, get_eigenstate_list(eigenstates)[5], ax=ax
    )
    _, _, _ = plot_eigenvector_through_bridge(
        h, get_eigenstate_list(eigenstates)[5], ax=ax2, view="angle"
    )
    ln.set_label("(12,12,15)")

    path = get_data_path("copper_eigenstates_12_12_14.json")
    eigenstates = load_energy_eigenstates(path)
    h = generate_hamiltonian(resolution=eigenstates["resolution"])
    _, _, ln = plot_eigenvector_through_bridge(
        h, get_eigenstate_list(eigenstates)[5], ax=ax
    )
    _, _, _ = plot_eigenvector_through_bridge(
        h, get_eigenstate_list(eigenstates)[5], ax=ax2, view="angle"
    )
    ln.set_label("(12,12,14)")

    path = get_data_path("copper_eigenstates_10_10_15.json")
    eigenstates = load_energy_eigenstates(path)
    h = generate_hamiltonian(resolution=eigenstates["resolution"])
    _, _, ln = plot_eigenvector_through_bridge(
        h, get_eigenstate_list(eigenstates)[5], ax=ax
    )
    _, _, _ = plot_eigenvector_through_bridge(
        h, get_eigenstate_list(eigenstates)[5], ax=ax2, view="angle"
    )
    ln.set_label("(10,10,15)")

    ax2.set_ylim(-np.pi, np.pi)
    ax.set_title(
        "Plot of energy against k for the lowest band of Copper for $K_y=0$\n"
        "showing convergence to about 2x$10^{-30}$J "
    )
    ax.legend()
    fig.show()
    save_figure(fig, "copper_wfn_convergence_through_bridge.png")
