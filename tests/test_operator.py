from __future__ import annotations

from typing import Any

import numpy as np
from scipy.constants import hbar  # type: ignore stubs
from slate_core import array, basis
from slate_core.basis import (
    TupleBasis,
    transformed_from_metadata,
)
from slate_core.metadata import size_from_nested_shape
from slate_core.metadata.volume import spaced_volume_metadata_from_stacked_delta_x

from slate_quantum import noise, operator
from slate_quantum.operator._build._potential import (
    CorrugatedMorseParameters,
    corrugated_morse_potential_function,
)
from slate_quantum.operator._linalg import into_diagonal_hermitian
from slate_quantum.operator._operator import Operator, OperatorList


def test_build_kinetic_operator() -> None:
    metadata = spaced_volume_metadata_from_stacked_delta_x(
        (np.array([2 * np.pi]),), (5,)
    )

    kinetic_operator = operator.build.kinetic_energy(metadata, hbar**2)
    np.testing.assert_allclose(kinetic_operator.raw_data, [0.0, 0.5, 2.0, 2.0, 0.5])


def test_build_hamiltonian() -> None:
    metadata = spaced_volume_metadata_from_stacked_delta_x(
        (np.array([2 * np.pi]),), (5,)
    )
    potential = operator.build.cos_potential(metadata, 0)
    hamiltonian = operator.build.kinetic_hamiltonian(potential, hbar**2)
    np.testing.assert_allclose(
        hamiltonian.raw_data, [0.0, 0.0, 0.0, 0.0, 0.5, 2.0, 2.0, 0.5]
    )
    transformed_basis = transformed_from_metadata(metadata)
    transformed_operator = hamiltonian.with_basis(
        TupleBasis((transformed_basis, transformed_basis.dual_basis())).upcast()
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

    kinetic_operator = operator.build.kinetic_energy(metadata, hbar**2)
    np.testing.assert_allclose(kinetic_operator.as_array(), hamiltonian.as_array())


def test_hamiltonian_eigenstates() -> None:
    metadata = spaced_volume_metadata_from_stacked_delta_x(
        (np.array([2 * np.pi]),), (5,)
    )
    potential = operator.build.cos_potential(metadata, 0)
    hamiltonian = operator.build.kinetic_hamiltonian(potential, hbar**2)

    kinetic_operator = operator.build.kinetic_energy(metadata, hbar**2)

    np.testing.assert_allclose(
        np.sort(into_diagonal_hermitian(kinetic_operator).raw_data),
        np.sort(into_diagonal_hermitian(hamiltonian).raw_data),
        atol=1e-15,
    )


def test_build_axis_scattering_operator() -> None:
    metadata = spaced_volume_metadata_from_stacked_delta_x(
        (np.array([2 * np.pi]),), (5,)
    ).children[0]

    scatter_operator = operator.build.axis_scattering_operator(metadata, n_k=0)
    np.testing.assert_allclose(
        array.as_outer_basis(array.as_inner_basis(scatter_operator)).as_array(),
        np.ones(5) / np.sqrt(metadata.fundamental_size),
        atol=1e-15,
    )

    scatter_operator = operator.build.axis_scattering_operator(metadata, n_k=1)
    np.testing.assert_allclose(
        array.as_outer_basis(array.as_inner_basis(scatter_operator)).as_array(),
        np.exp(1j * 2 * np.pi * np.arange(5) / 5) / np.sqrt(metadata.fundamental_size),
        atol=1e-15,
    )

    scatter_operator = operator.build.axis_scattering_operator(metadata, n_k=-1)
    np.testing.assert_allclose(
        array.as_outer_basis(array.as_inner_basis(scatter_operator)).as_array(),
        np.exp(-1j * 2 * np.pi * np.arange(5) / 5) / np.sqrt(metadata.fundamental_size),
        atol=1e-15,
    )

    scatter_operator = operator.build.axis_scattering_operator(metadata, n_k=4)
    np.testing.assert_allclose(
        array.as_outer_basis(array.as_inner_basis(scatter_operator)).as_array(),
        np.exp(-1j * 2 * np.pi * np.arange(5) / 5) / np.sqrt(metadata.fundamental_size),
        atol=1e-15,
    )


def test_build_scattering_operator() -> None:
    metadata = spaced_volume_metadata_from_stacked_delta_x(
        (np.array([3 * np.pi]),), (5,)
    )
    size = size_from_nested_shape(metadata.fundamental_shape)

    scatter_operator = operator.build.scattering_operator(metadata, n_k=(0,))
    np.testing.assert_allclose(
        array.as_outer_basis(array.as_inner_basis(scatter_operator)).as_array(),
        np.ones(5) / np.sqrt(size),
        atol=1e-15,
    )

    scatter_operator = operator.build.scattering_operator(metadata, n_k=(1,))
    np.testing.assert_allclose(
        array.as_outer_basis(array.as_inner_basis(scatter_operator)).as_array(),
        np.exp(1j * 2 * np.pi * np.arange(5) / 5) / np.sqrt(size),
        atol=1e-15,
    )

    scatter_operator = operator.build.scattering_operator(metadata, n_k=(2,))
    np.testing.assert_allclose(
        array.as_outer_basis(array.as_inner_basis(scatter_operator)).as_array(),
        np.exp(2j * 2 * np.pi * np.arange(5) / 5) / np.sqrt(size),
        atol=1e-15,
    )

    scatter_operator = operator.build.scattering_operator(metadata, n_k=(-1,))
    np.testing.assert_allclose(
        array.as_outer_basis(array.as_inner_basis(scatter_operator)).as_array(),
        np.exp(-1j * 2 * np.pi * np.arange(5) / 5) / np.sqrt(size),
        atol=1e-15,
    )

    scatter_operator = operator.build.scattering_operator(metadata, n_k=(4,))
    np.testing.assert_allclose(
        array.as_outer_basis(array.as_inner_basis(scatter_operator)).as_array(),
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
    position_operator = operator.build.x(metadata, axis=0)

    np.testing.assert_array_equal(
        position_operator.as_array(),
        np.diag(np.linspace(0, 2 * np.pi, 5, endpoint=False)),
    )


def test_k_operator() -> None:
    metadata = spaced_volume_metadata_from_stacked_delta_x(
        (np.array([2 * np.pi]),), (5,)
    )
    momentum_operator = operator.build.k(metadata, axis=0)

    basis_k = basis.transformed_from_metadata(metadata)
    np.testing.assert_array_equal(
        momentum_operator.with_basis(
            TupleBasis((basis_k, basis_k.dual_basis())).upcast()
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

    np.testing.assert_allclose(
        (momentum_operator * hbar).as_array(),
        operator.build.p(metadata, axis=0).as_array(),
        atol=1e-15,
    )


def test_x_k_commutator() -> None:
    metadata = spaced_volume_metadata_from_stacked_delta_x(
        (np.array([2 * np.pi]),), (5,)
    )
    position_operator = operator.build.x(metadata, axis=0)
    momentum_operator = operator.build.k(metadata, axis=0)

    matmul_0 = operator.matmul(position_operator, momentum_operator)
    np.testing.assert_allclose(
        matmul_0.as_array(),
        position_operator.as_array() @ momentum_operator.as_array(),
        atol=1e-15,
    )

    matmul_1 = operator.matmul(momentum_operator, position_operator)
    np.testing.assert_allclose(
        matmul_1.as_array(),
        momentum_operator.as_array() @ position_operator.as_array(),
        atol=1e-15,
    )

    commutator_manual = matmul_0 - matmul_1
    np.testing.assert_allclose(
        commutator_manual.as_array(),
        position_operator.as_array() @ momentum_operator.as_array()
        - momentum_operator.as_array() @ position_operator.as_array(),
        atol=1e-15,
    )

    commutator = operator.commute(position_operator, momentum_operator)
    np.testing.assert_allclose(
        commutator.as_array(), commutator_manual.as_array(), atol=1e-15
    )


def test_trivial_commutator() -> None:
    metadata = spaced_volume_metadata_from_stacked_delta_x(
        (np.array([2 * np.pi]),), (5,)
    )
    position_operator = operator.build.x(metadata, axis=0)
    momentum_operator = operator.build.k(metadata, axis=0)

    commutator = operator.commute(position_operator, position_operator)
    np.testing.assert_array_equal(commutator.as_array(), np.zeros((5, 5)))

    commutator = operator.commute(momentum_operator, momentum_operator)
    np.testing.assert_array_equal(commutator.as_array(), np.zeros((5, 5)))


def test_filter_scatter_operator() -> None:
    metadata = spaced_volume_metadata_from_stacked_delta_x(
        (np.array([2 * np.pi, 0]),), (5,)
    )
    basis_k = basis.transformed_from_metadata(metadata).upcast()
    test_operator = Operator(
        TupleBasis((basis_k, basis_k.dual_basis())).upcast(),
        np.ones(25, dtype=np.complex128),
    )
    filtered = operator.build.filter_scatter(test_operator)
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
    test_operators = OperatorList(
        test_operators.basis.inner.upcast(), test_operators.raw_data
    )
    filtered = operator.build.all_filter_scatter(test_operators)
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


def test_build_cos_operator() -> None:
    metadata = spaced_volume_metadata_from_stacked_delta_x(
        (np.array([2 * np.pi, 0]),), (16,)
    )

    actual = operator.build.cos_potential(metadata, 2)
    expected = operator.build.potential_from_function(
        metadata, lambda x: 1 + np.cos(x[0])
    )
    np.testing.assert_allclose(actual.as_array(), expected.as_array())

    actual = operator.build.sin_potential(metadata, 2)
    expected = operator.build.potential_from_function(
        metadata, lambda x: 1 + np.sin(x[0])
    )
    np.testing.assert_allclose(actual.as_array(), expected.as_array())


def test_build_fcc_operator() -> None:
    metadata = spaced_volume_metadata_from_stacked_delta_x(
        (
            np.array([np.sin(np.pi / 3), np.cos(np.pi / 3)]),
            np.array([0, 1]),
        ),
        (16, 16),
    )

    actual = operator.build.fcc_potential(metadata, 1)

    # Functional form from david's thesis
    # page 106 eq. 4.9
    eta = 2 * np.pi * (2 / np.sqrt(3))
    expected = operator.build.potential_from_function(
        metadata,
        lambda x: (1 / 3)
        + (2 / 9)
        * (
            np.cos(eta * x[0] + 0 * x[1])
            + np.cos(eta * np.cos(np.pi / 3) * x[0] + eta * np.sin(np.pi / 3) * x[1])
            + np.cos(-eta * np.cos(np.pi / 3) * x[0] + eta * np.sin(np.pi / 3) * x[1])
        ),
    )
    np.testing.assert_allclose(actual.as_array(), expected.as_array())


def test_build_corrugated_morse_operator() -> None:
    metadata = spaced_volume_metadata_from_stacked_delta_x(
        (np.array([21, 0, 0]), np.array([0, 21, 0]), np.array([0, 0, 5])), (3, 3, 5)
    )
    params = CorrugatedMorseParameters(depth=1, height=0.5, offset=1.0, beta=0)

    actual = operator.build.morse_potential(metadata, params)
    expected = operator.build.potential_from_function(
        metadata,
        corrugated_morse_potential_function(metadata, params),
    )
    np.testing.assert_allclose(actual.as_array(), expected.as_array())

    params = params.with_beta(4)
    actual = operator.build.corrugated_morse_potential(metadata, params)
    expected = operator.build.potential_from_function(
        metadata,
        corrugated_morse_potential_function(metadata, params),
    )
    np.testing.assert_allclose(np.diag(actual.as_array()), np.diag(expected.as_array()))


def test_build_periodic_cl_operators() -> None:
    metadata = spaced_volume_metadata_from_stacked_delta_x(
        (np.array([2 * np.pi]),), (160,)
    )

    delta_x = metadata.children[0].spacing.delta
    expected = OperatorList.from_operators(
        [
            operator.build.potential_from_function(
                metadata,
                lambda x: np.cos(2 * np.pi * x[0] / delta_x)
                / np.sqrt(metadata.children[0].fundamental_size),
            ),
            operator.build.potential_from_function(
                metadata,
                lambda x: np.sin(2 * np.pi * x[0] / delta_x)
                / np.sqrt(metadata.children[0].fundamental_size),
            ),
        ]
    )
    operators = noise.build.real_periodic_caldeira_leggett_operators(metadata)
    np.testing.assert_allclose(operators.as_array(), expected.as_array(), atol=1e-15)

    expected = OperatorList.from_operators(
        [
            operator.build.potential_from_function(
                metadata,
                lambda x: np.exp(-1j * 2 * np.pi * x[0] / delta_x)
                / np.sqrt(metadata.children[0].fundamental_size),
            ),
            operator.build.potential_from_function(
                metadata,
                lambda x: np.exp(1j * 2 * np.pi * x[0] / delta_x)
                / np.sqrt(metadata.children[0].fundamental_size),
            ),
        ]
    )
    operators_complex = noise.build.periodic_caldeira_leggett_operators(metadata)
    np.testing.assert_allclose(
        operators_complex.as_array(), expected.as_array(), atol=1e-15
    )

    np.testing.assert_allclose(
        # Scaled by a factor of 4
        # NOTE: the operators are scales by sqrt(eigenvalue)
        operators_complex.basis.metadata().children[0].values * 4,
        operators.basis.metadata().children[0].values,
    )

    # The gradient of the non-periodic operator should match that of the periodic operator
    # At the origin the cos operator just provides a constant offset to the energy,
    # and the sin(x) operator is roughly linear
    operators_non_periodic = noise.build.caldeira_leggett_operators(metadata)[0, :]
    scaled_periodic = operators[1, :] * complex(
        np.sqrt(operators.basis.metadata().children[0].values[0])
    )
    scaled_periodic = scaled_periodic.with_basis(operators_non_periodic.basis)
    np.testing.assert_allclose(
        array.extract_diagonal(scaled_periodic)[0:2].as_array(),
        array.extract_diagonal(operators_non_periodic)[0:2].as_array(),
        atol=1e-4,
    )


def test_build_cl_operator() -> None:
    metadata = spaced_volume_metadata_from_stacked_delta_x(
        (np.array([2 * np.pi]),), (160,)
    )
    temperature = 1
    mass = 1

    operators = noise.build.caldeira_leggett_operators(metadata)
    # Should only have 1 operator for a 1D system
    assert operators.basis.metadata().fundamental_shape[0] == (1,)

    # The cl-operator should be the same as x
    x_operator = operator.build.x(metadata, axis=0)
    np.testing.assert_allclose(
        operators[0, :].as_array(), x_operator.as_array(), atol=1e-15
    )

    hamiltonian = operator.build.kinetic_energy(metadata, mass)
    corrected_operators = noise.build.temperature_corrected_operators(
        hamiltonian, operators, temperature
    )
    assert corrected_operators.basis.metadata().fundamental_shape[0] == (1,)
    # [x, p] = i hbar
    commutator = operator.commute(
        operator.build.x(metadata, axis=0), operator.build.p(metadata, axis=0)
    )
    np.testing.assert_allclose(commutator.as_array(), (1j * hbar), atol=1e-15)


def test_dagger() -> None:
    metadata = spaced_volume_metadata_from_stacked_delta_x(
        (np.array([2 * np.pi]),), (5,)
    )

    position_operator = operator.build.x(metadata, axis=0)
    np.testing.assert_allclose(
        operator.dagger(position_operator).as_array(),
        position_operator.as_array().T.conj(),
    )
    operator.dagger(position_operator)

    scatter_operator = operator.build.scattering_operator(metadata, n_k=(1,))
    np.testing.assert_allclose(
        operator.dagger(scatter_operator).as_array(),
        scatter_operator.as_array().T.conj(),
    )
    assert (
        operator.dagger(scatter_operator).basis.is_dual
        == scatter_operator.basis.is_dual
    )


def commutator_numpy(
    a: np.ndarray[Any, np.dtype[np.complexfloating]],
    b: np.ndarray[Any, np.dtype[np.complexfloating]],
) -> np.ndarray[Any, np.dtype[np.complexfloating]]:
    return a @ b - b @ a


def test_commute() -> None:
    metadata = spaced_volume_metadata_from_stacked_delta_x(
        (np.array([2 * np.pi * 10]),), (300,)
    )

    position_operator = operator.build.x(metadata, axis=0)
    k_operator = operator.build.k(metadata, axis=0)

    commutator = operator.commute(position_operator, k_operator)
    np.testing.assert_allclose(
        commutator.as_array(),
        commutator_numpy(position_operator.as_array(), k_operator.as_array()),
        atol=1e-13,
    )

    mass = hbar**2
    kinetic_operator = operator.build.kinetic_energy(metadata, mass)
    commutator = operator.commute(kinetic_operator, position_operator)
    np.testing.assert_allclose(
        commutator.as_array(),
        commutator_numpy(kinetic_operator.as_array(), position_operator.as_array()),
        atol=2e-12,
    )

    potential = operator.build.cos_potential(metadata, 0)
    hamiltonian = operator.build.kinetic_hamiltonian(potential, mass)
    commutator = operator.commute(hamiltonian, position_operator)
    np.testing.assert_allclose(
        commutator.as_array(),
        commutator_numpy(hamiltonian.as_array(), position_operator.as_array()),
        atol=2e-12,
    )
