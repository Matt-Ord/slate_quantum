from __future__ import annotations

import numpy as np
from scipy.constants import hbar  # type: ignore stubs
from slate import array
from slate.basis import (
    fundamental_transformed_tuple_basis_from_metadata,
    tuple_basis,
)
from slate.metadata import size_from_nested_shape
from slate.metadata.volume import spaced_volume_metadata_from_stacked_delta_x

from slate_quantum import operator
from slate_quantum.operator.linalg import into_diagonal_hermitian


def test_build_kinetic_operator() -> None:
    metadata = spaced_volume_metadata_from_stacked_delta_x(
        (np.array([2 * np.pi]),), (5,)
    )

    kinetic_operator = operator.build.kinetic_energy_operator(metadata, hbar**2)
    np.testing.assert_allclose(kinetic_operator.raw_data, [0.0, 0.5, 2.0, 2.0, 0.5])


def test_build_hamiltonain() -> None:
    metadata = spaced_volume_metadata_from_stacked_delta_x(
        (np.array([2 * np.pi]),), (5,)
    )
    potential = operator.build.cos_potential(metadata, 0)
    hamiltonian = operator.build.kinetic_hamiltonian(potential, hbar**2)
    np.testing.assert_allclose(
        hamiltonian.raw_data, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 2.0, 2.0, 0.5]
    )
    transformed_basis = fundamental_transformed_tuple_basis_from_metadata(metadata)
    transformed_operator = hamiltonian.with_basis(
        tuple_basis((transformed_basis, transformed_basis.dual_basis()))
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

    kinetic_operator = operator.build.kinetic_energy_operator(metadata, hbar**2)
    np.testing.assert_allclose(kinetic_operator.as_array(), hamiltonian.as_array())


def test_hamiltonain_eigenstates() -> None:
    metadata = spaced_volume_metadata_from_stacked_delta_x(
        (np.array([2 * np.pi]),), (5,)
    )
    potential = operator.build.cos_potential(metadata, 0)
    hamiltonian = operator.build.kinetic_hamiltonian(potential, hbar**2)

    kinetic_operator = operator.build.kinetic_energy_operator(metadata, hbar**2)

    np.testing.assert_allclose(
        np.sort(into_diagonal_hermitian(kinetic_operator).raw_data),
        np.sort(into_diagonal_hermitian(hamiltonian).raw_data),
        atol=1e-15,
    )


def test_build_axis_scattering_operator() -> None:
    metadata = spaced_volume_metadata_from_stacked_delta_x(
        (np.array([2 * np.pi]),), (5,)
    )[0]

    scatter_operator = operator.build.axis_scattering_operator(metadata, n_k=0)
    np.testing.assert_allclose(
        array.as_outer_array(scatter_operator).as_array(),
        np.ones(5) / np.sqrt(metadata.fundamental_size),
        atol=1e-15,
    )

    scatter_operator = operator.build.axis_scattering_operator(metadata, n_k=1)
    np.testing.assert_allclose(
        array.as_outer_array(scatter_operator).as_array(),
        np.exp(1j * 2 * np.pi * np.arange(5) / 5) / np.sqrt(metadata.fundamental_size),
        atol=1e-15,
    )

    scatter_operator = operator.build.axis_scattering_operator(metadata, n_k=-1)
    np.testing.assert_allclose(
        array.as_outer_array(scatter_operator).as_array(),
        np.exp(-1j * 2 * np.pi * np.arange(5) / 5) / np.sqrt(metadata.fundamental_size),
        atol=1e-15,
    )

    scatter_operator = operator.build.axis_scattering_operator(metadata, n_k=4)
    np.testing.assert_allclose(
        array.as_outer_array(scatter_operator).as_array(),
        np.exp(-1j * 2 * np.pi * np.arange(5) / 5) / np.sqrt(metadata.fundamental_size),
        atol=1e-15,
    )


def test_build_scattering_operator() -> None:
    metadata = spaced_volume_metadata_from_stacked_delta_x(
        (np.array([2 * np.pi]),), (5,)
    )
    size = size_from_nested_shape(metadata.fundamental_shape)

    scatter_operator = operator.build.scattering_operator(metadata, n_k=(0,))
    np.testing.assert_allclose(
        array.as_outer_array(scatter_operator).as_array(),
        np.ones(5) / np.sqrt(size),
        atol=1e-15,
    )

    scatter_operator = operator.build.scattering_operator(metadata, n_k=(1,))
    np.testing.assert_allclose(
        array.as_outer_array(scatter_operator).as_array(),
        np.exp(1j * 2 * np.pi * np.arange(5) / 5) / np.sqrt(size),
        atol=1e-15,
    )

    scatter_operator = operator.build.scattering_operator(metadata, n_k=(-1,))
    np.testing.assert_allclose(
        array.as_outer_array(scatter_operator).as_array(),
        np.exp(-1j * 2 * np.pi * np.arange(5) / 5) / np.sqrt(size),
        atol=1e-15,
    )

    scatter_operator = operator.build.scattering_operator(metadata, n_k=(4,))
    np.testing.assert_allclose(
        array.as_outer_array(scatter_operator).as_array(),
        np.exp(-1j * 2 * np.pi * np.arange(5) / 5) / np.sqrt(size),
        atol=1e-15,
    )


def test_build_all_scattering_operators() -> None:
    metadata = spaced_volume_metadata_from_stacked_delta_x(
        (np.array([2 * np.pi]),), (5,)
    )

    operators = operator.build.all_scattering_operators(metadata)
    size = size_from_nested_shape(metadata.fundamental_shape)
    assert operators.fundamental_shape[0] == size

    for i, scatter_operator in enumerate(operators):
        expected = operator.build.scattering_operator(metadata, n_k=(i,))
        np.testing.assert_allclose(
            expected.as_array(),
            scatter_operator.as_array(),
            atol=1e-15,
        )
