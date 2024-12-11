from __future__ import annotations

import numpy as np
from scipy.constants import hbar  # type: ignore stubs
from slate import array, basis
from slate.basis import (
    fundamental_transformed_tuple_basis_from_metadata,
    tuple_basis,
)
from slate.metadata import size_from_nested_shape
from slate.metadata.volume import spaced_volume_metadata_from_stacked_delta_x

from slate_quantum import operator
from slate_quantum.operator._linalg import into_diagonal_hermitian
from slate_quantum.operator._operator import Operator, OperatorList


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
        transformed_operator.raw_data,
        np.ravel(expected).astype(np.complex128),
        atol=1e-15,
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


def test_x_operator() -> None:
    metadata = spaced_volume_metadata_from_stacked_delta_x(
        (np.array([2 * np.pi]),), (5,)
    )
    position_operator = operator.build.x_operator(metadata, idx=0)

    np.testing.assert_array_equal(
        position_operator.as_array(),
        np.diag(np.linspace(0, 2 * np.pi, 5, endpoint=False)),
    )


def test_k_operator() -> None:
    metadata = spaced_volume_metadata_from_stacked_delta_x(
        (np.array([2 * np.pi]),), (5,)
    )
    momentum_operator = operator.build.k_operator(metadata, idx=0)

    basis_k = basis.fundamental_transformed_tuple_basis_from_metadata(metadata)
    np.testing.assert_array_equal(
        momentum_operator.with_basis(
            tuple_basis((basis_k, basis_k.dual_basis()))
        ).raw_data.reshape(5, 5),
        np.diag([0, 1, 2, -2, -1]),
    )
    np.testing.assert_array_equal(
        momentum_operator.as_array(),
        np.fft.fft(
            np.fft.ifft(np.diag([0, 1, 2, -2, -1]), norm="ortho", axis=0),
            norm="ortho",
            axis=1,
        ),
    )


def test_x_k_commutator() -> None:
    metadata = spaced_volume_metadata_from_stacked_delta_x(
        (np.array([2 * np.pi]),), (5,)
    )
    position_operator = operator.build.x_operator(metadata, idx=0)
    momentum_operator = operator.build.k_operator(metadata, idx=0)

    matmul_0 = operator.matmul(position_operator, momentum_operator)
    np.testing.assert_array_equal(
        matmul_0.as_array(),
        position_operator.as_array() @ momentum_operator.as_array(),
    )

    matmul_1 = operator.matmul(momentum_operator, position_operator)
    np.testing.assert_array_equal(
        matmul_1.as_array(),
        momentum_operator.as_array() @ position_operator.as_array(),
    )

    commutator_manual = matmul_0 - matmul_1
    np.testing.assert_array_equal(
        commutator_manual.as_array(),
        position_operator.as_array() @ momentum_operator.as_array()
        - momentum_operator.as_array() @ position_operator.as_array(),
    )

    commutator = operator.commute(position_operator, momentum_operator)
    np.testing.assert_array_equal(commutator.as_array(), commutator_manual.as_array())


def test_trivial_commutator() -> None:
    metadata = spaced_volume_metadata_from_stacked_delta_x(
        (np.array([2 * np.pi]),), (5,)
    )
    position_operator = operator.build.x_operator(metadata, idx=0)
    momentum_operator = operator.build.k_operator(metadata, idx=0)

    commutator = operator.commute(position_operator, position_operator)
    np.testing.assert_array_equal(commutator.as_array(), np.zeros((5, 5)))

    commutator = operator.commute(momentum_operator, momentum_operator)
    np.testing.assert_array_equal(commutator.as_array(), np.zeros((5, 5)))


def test_filter_scatter_operator() -> None:
    metadata = spaced_volume_metadata_from_stacked_delta_x(
        (np.array([2 * np.pi, 0]),), (5,)
    )
    basis_k = basis.fundamental_transformed_tuple_basis_from_metadata(metadata)
    test_operator = Operator(
        tuple_basis((basis_k, basis_k.dual_basis())),
        np.ones(25, dtype=np.complex128),
    )
    filtered = operator.build.filter_scatter_operator(test_operator)
    np.testing.assert_array_equal(
        filtered.raw_data,
        np.array(
            [
                [1, 1, 1, 1, 1],
                [1, 1, 1, 0, 1],
                [1, 1, 1, 0, 0],
                [1, 0, 0, 1, 1],
                [1, 1, 0, 1, 1],
            ]
        ).ravel(),
    )

    test_operators = OperatorList.from_operators([test_operator, test_operator])
    filtered = operator.build.filter_scatter_operators(test_operators)
    for op in filtered:
        np.testing.assert_array_equal(
            op.raw_data,
            np.array(
                [
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 0, 1],
                    [1, 1, 1, 0, 0],
                    [1, 0, 0, 1, 1],
                    [1, 1, 0, 1, 1],
                ]
            ).ravel(),
        )
