from __future__ import annotations

import numpy as np
from scipy.constants import hbar  # type: ignore stubs
from slate.metadata.volume import spaced_volume_metadata_from_stacked_delta_x
from slate.plot import get_figure, plot_data_1d_k, plot_data_1d_x

from slate_quantum.model.operator import (
    build_cos_potential,
    build_kinetic_energy_operator,
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
    # TODO: we want to make this more natural ...
    diagonal_hamiltonian = into_diagonal_hermitian(hamiltonian)
    eigenstates = diagonal_hamiltonian.basis.inner[0].states

    np.testing.assert_allclose(diagonal_hamiltonian.as_array(), hamiltonian.as_array())
    op = build_kinetic_energy_operator(metadata, hbar**2)
    eigenstates = into_diagonal_hermitian(op).basis.inner[0].states
    np.testing.assert_allclose(op.as_array(), hamiltonian.as_array())
    fig, ax = get_figure()
    for state in list(eigenstates)[:3]:
        plot_data_1d_x(state, ax=ax, measure="abs")
    ax.set_ylim(0, None)
    fig.show()

    fig, ax = get_figure()
    for state in list(eigenstates)[:3]:
        plot_data_1d_k(state, ax=ax, measure="abs")
    fig.show()

    input()
