from __future__ import annotations

import numpy as np
from scipy.constants import hbar  # type: ignore stubs
from slate import plot
from slate.metadata import spaced_volume_metadata_from_stacked_delta_x
from slate.plot import get_figure

from slate_quantum import bloch, operator

if __name__ == "__main__":
    # For a periodic potential, we can make use of bloch's theorem
    # to find the eigenstates of the hamiltonian with a reduced computational cost.
    metadata = spaced_volume_metadata_from_stacked_delta_x(
        (np.array([2 * np.pi]),), (60,)
    )

    potential = operator.build.cos_potential(metadata, 0)
    hamiltonian = bloch.build.kinetic_hamiltonian(potential, hbar**2, (3,))
    eigenstates = operator.get_eigenstates_hermitian(hamiltonian)

    # The eigenstates of a free particle are plane waves
    fig, ax = get_figure()
    for state in list(eigenstates)[:3]:
        plot.basis_against_array_1d_x(state, ax=ax, measure="abs")
    fig.show()

    fig, ax = get_figure()
    for state in list(eigenstates)[:3]:
        plot.basis_against_array_1d_k(state, ax=ax, measure="abs")
    fig.show()

    plot.wait_for_close()
