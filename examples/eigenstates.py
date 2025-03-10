from __future__ import annotations

import numpy as np
from scipy.constants import hbar  # type: ignore stubs
from slate import plot
from slate.metadata.volume import spaced_volume_metadata_from_stacked_delta_x
from slate.plot import get_figure

from slate_quantum import operator, state

if __name__ == "__main__":
    # Metadata for a 1D volume with 60 points in the x direction
    metadata = spaced_volume_metadata_from_stacked_delta_x(
        (np.array([2 * np.pi]),), (60,)
    )

    potential = operator.build.cos_potential(metadata, 0)
    hamiltonian = operator.build.kinetic_hamiltonian(potential, hbar**2)
    eigenstates = operator.get_eigenstates_hermitian(hamiltonian)

    # The eigenstates of a free particle are plane waves
    fig, ax = get_figure()
    for eigenstate in list(eigenstates)[:3]:
        plot.array_against_axes_1d(eigenstate, ax=ax, measure="abs")
    ax.set_title("Free Particle Eigenstates")
    ax.set_xlabel("x")
    ax.set_ylabel(r"| \\psi(k) |")
    fig.show()

    fig, ax = get_figure()
    for eigenstate in list(eigenstates)[:3]:
        plot.array_against_axes_1d_k(eigenstate, ax=ax, measure="abs")
    ax.set_title("Free Particle Eigenstates in k-space")
    ax.set_xlabel("k")
    ax.set_ylabel(r"| \\psi(k) |")
    fig.show()

    # Now we modify the potential to have a barrier
    potential = operator.build.cos_potential(metadata, 1)
    hamiltonian = operator.build.kinetic_hamiltonian(potential, hbar**2)
    eigenstates = operator.get_eigenstates_hermitian(hamiltonian)

    # The eigenstates are now localized around the potential minima
    fig, ax = get_figure()
    for eigenstate in list(eigenstates)[:3]:
        plot.array_against_axes_1d(eigenstate, ax=ax, measure="abs")
    ax.set_title("Eigenstates in a Cos Potential")
    ax.set_xlabel("x")
    ax.set_ylabel(r"| \\psi(k) |")
    fig.show()

    # The eigenstates of a harmonic particle are given by the Hermite polynomials
    metadata = spaced_volume_metadata_from_stacked_delta_x(
        (np.array([4 * 2 * np.pi]),), (60,)
    )
    potential = operator.build.harmonic_potential(metadata, np.sqrt(hbar))
    hamiltonian = operator.build.kinetic_hamiltonian(potential, hbar)
    eigenstates = operator.get_eigenstates_hermitian(hamiltonian)

    fig, ax = get_figure()
    for n, eigenstate in enumerate(list(eigenstates)[:3]):
        _, _, line = plot.array_against_axes_1d(eigenstate, ax=ax, measure="abs")
        if n == 0:
            line.set_label("actual state")
        line.set_color("C0")

    # They should closely match the theoretical states
    for n in range(3):
        theoretical = state.build.bosonic(metadata, n=n, wrapped_x=True)
        _, _, line = plot.array_against_axes_1d(theoretical, ax=ax, measure="abs")
        if n == 0:
            line.set_label("theoretical")
        line.set_color("C1")
        line.set_linestyle("--")

    ax.set_title("Eigenstates in a Harmonic Potential")
    ax.set_xlabel("x")
    ax.set_ylabel(r"| \\psi(k) |")
    ax.legend()
    fig.show()

    plot.wait_for_close()
