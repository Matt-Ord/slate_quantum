from __future__ import annotations

import numpy as np
from scipy.constants import hbar  # type: ignore stubs
from slate_core import Array, plot
from slate_core.metadata import spaced_volume_metadata_from_stacked_delta_x
from slate_core.plot import get_figure

from slate_quantum import bloch, operator

if __name__ == "__main__":
    # For a periodic potential, we can make use of bloch's theorem
    # to find the eigenstates of the hamiltonian with a reduced computational cost.
    metadata = spaced_volume_metadata_from_stacked_delta_x(
        (np.array([2 * np.pi]),), (60,)
    )

    rng = np.random.default_rng()
    potential = operator.build.potential_from_function(
        metadata, lambda x: 400 * rng.normal(size=x[0].size).astype(np.complex128)
    )
    hamiltonian = bloch.build.kinetic_hamiltonian(potential, hbar**2, (3,))

    # The hamiltonian is block diagonal, when we use the BlochTransposedBasis
    diag_data = Array.from_array(
        hamiltonian.with_basis(hamiltonian.basis.inner.inner.upcast())
        .assert_ok()
        .raw_data.reshape(hamiltonian.basis.inner.inner.shape)
    )
    fig, ax, _ = plot.array_against_axes_2d(diag_data)
    fig.show()

    # Into-diag is aware the initial matrix is block diagonal - so
    # this 'just works' and is much faster than the naive diag.
    eigenstates = operator.get_eigenstates_hermitian(hamiltonian)
    fig, ax = get_figure()
    for state in list(eigenstates)[:3]:
        plot.array_against_axes_1d(state, ax=ax, measure="abs")
    fig.show()

    fig, ax = get_figure()
    for state in list(eigenstates)[:3]:
        plot.array_against_axes_1d_k(state, ax=ax, measure="abs")
    fig.show()

    plot.wait_for_close()
