from __future__ import annotations

import numpy as np
from scipy.constants import hbar  # type: ignore stubs
from slate.metadata.volume import spaced_volume_metadata_from_stacked_delta_x
from slate.plot import get_figure, plot_data_1d_k, plot_data_1d_x

from slate_quantum.model.operator import (
    build_cos_potential,
    build_kinetic_hamiltonian,
    into_diagonal_hermitian,
)

if __name__ == "__main__":
    # Metadata for a 1D volume with 100 points and a width of 14
    metadata = spaced_volume_metadata_from_stacked_delta_x(
        (np.array([2 * np.pi]),), (5,)
    )

    potential = build_cos_potential(metadata, 0)
    hamiltonian = build_kinetic_hamiltonian(potential, hbar**2)
    # TODO: we want to make this more natural ...  # noqa: FIX002
    diagonal_hamiltonian = into_diagonal_hermitian(hamiltonian)
    eigenstates = diagonal_hamiltonian.basis.inner[1].eigenvectors

    fig, ax = get_figure()
    for state in list(eigenstates)[:3]:
        plot_data_1d_x(state, ax=ax, measure="abs")
    ax.set_ylim(-5, 5)
    fig.show()

    fig, ax = get_figure()
    fig1, ax1 = get_figure()
    for state in list(eigenstates)[:3]:
        plot_data_1d_k(state, ax=ax, measure="imag")
        plot_data_1d_k(state, ax=ax1, measure="real")
    fig.show()
    fig1.show()

    input()
