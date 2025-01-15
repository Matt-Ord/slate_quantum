from __future__ import annotations

import numpy as np
from scipy.constants import hbar  # type: ignore stubs
from slate import plot
from slate.metadata.volume import spaced_volume_metadata_from_stacked_delta_x
from slate.plot import get_figure

from slate_quantum import operator

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
    for state in list(eigenstates)[:3]:
        plot.array_against_axes_1d(state, ax=ax, measure="abs")
    fig.show()

    fig, ax = get_figure()
    for state in list(eigenstates)[:3]:
        plot.array_against_axes_1d_k(state, ax=ax, measure="abs")
    fig.show()

    # Now we modify the potential to have a barrier
    potential = operator.build.cos_potential(metadata, 1)
    hamiltonian = operator.build.kinetic_hamiltonian(potential, hbar**2)
    eigenstates = operator.get_eigenstates_hermitian(hamiltonian)

    # The eigenstates are now localized around the potential minima
    fig, ax = get_figure()
    for state in list(eigenstates)[:3]:
        plot.array_against_axes_1d(state, ax=ax, measure="abs")
    fig.show()

    plot.wait_for_close()
