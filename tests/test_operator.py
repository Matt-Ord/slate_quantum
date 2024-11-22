from __future__ import annotations

import numpy as np
from scipy.constants import hbar  # type: ignore stubs
from slate.basis import (
    fundamental_transformed_tuple_basis_from_metadata,
    tuple_basis,
)
from slate.metadata.volume import spaced_volume_metadata_from_stacked_delta_x

from slate_quantum.model.operator.build import (
    build_kinetic_energy_operator,
    build_kinetic_hamiltonian,
)
from slate_quantum.model.operator.linalg import into_diagonal_hermitian
from slate_quantum.model.operator.potential.build import build_cos_potential


def test_build_kinetic_operator() -> None:
    metadata = spaced_volume_metadata_from_stacked_delta_x(
        (np.array([2 * np.pi]),), (5,)
    )

    operator = build_kinetic_energy_operator(metadata, hbar**2)
    np.testing.assert_allclose(operator.raw_data, [0.0, 0.5, 2.0, 2.0, 0.5])


def test_build_hamiltonain() -> None:
    metadata = spaced_volume_metadata_from_stacked_delta_x(
        (np.array([2 * np.pi]),), (5,)
    )
    potential = build_cos_potential(metadata, 0)
    hamiltonian = build_kinetic_hamiltonian(potential, hbar**2)
    np.testing.assert_allclose(
        hamiltonian.raw_data, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 2.0, 2.0, 0.5]
    )
    transformed_basis = fundamental_transformed_tuple_basis_from_metadata(metadata)
    transformed_operator = hamiltonian.with_basis(
        tuple_basis((transformed_basis, transformed_basis))
    )

    expected = [
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.5, 0.0, 0.0, 0.0],
        [0.0, 0.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 2.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.5],
    ]
    np.testing.assert_allclose(
        transformed_operator.raw_data, np.ravel(expected), atol=1e-15
    )

    kinetic_operator = build_kinetic_energy_operator(metadata, hbar**2)
    np.testing.assert_allclose(kinetic_operator.as_array(), hamiltonian.as_array())


def test_hamiltonain_eigenstates() -> None:
    metadata = spaced_volume_metadata_from_stacked_delta_x(
        (np.array([2 * np.pi]),), (5,)
    )
    potential = build_cos_potential(metadata, 0)
    hamiltonian = build_kinetic_hamiltonian(potential, hbar**2)

    kinetic_operator = build_kinetic_energy_operator(metadata, hbar**2)

    np.testing.assert_allclose(
        into_diagonal_hermitian(kinetic_operator).raw_data,
        into_diagonal_hermitian(hamiltonian).raw_data,
    )
