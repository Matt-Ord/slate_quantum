from __future__ import annotations

import numpy as np
from slate import metadata

from slate_quantum import bloch, operator


def test_build_bloch_operator() -> None:
    meta = metadata.volume.spaced_volume_metadata_from_stacked_delta_x(
        (np.array([2 * np.pi]),), (5,)
    )
    repeat = (5,)
    potential = operator.build.cos_potential(meta, 1)
    mass = 1

    sparse = bloch.build.kinetic_hamiltonian(potential, mass, repeat)
    repeat_potential = operator.build.repeat_potential(potential, repeat)
    full_potential = operator.build.kinetic_hamiltonian(repeat_potential, mass)

    assert sparse.basis.metadata() == full_potential.basis.metadata()
    np.testing.assert_allclose(sparse.as_array(), full_potential.as_array())
