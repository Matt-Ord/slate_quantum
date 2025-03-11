from __future__ import annotations

import itertools

import numpy as np
import pytest
from scipy.constants import hbar  # type: ignore module
from slate_core import TupleBasis, basis, metadata

from slate_quantum import bloch, operator
from slate_quantum.bloch.build import BlochFractionMetadata
from slate_quantum.operator._operator import Operator

TEST_LENGTHS = [1, 2, 3, 5]
SHAPES_1D = [(x,) for x in [1, 2, 3, 5]]
SHAPES_2D = [*itertools.product([1, 2, 3], [1, 2, 3]), (5, 5)]


@pytest.mark.parametrize(
    ("shape", "repeat"),
    [
        *itertools.product(SHAPES_1D, SHAPES_1D),
        *itertools.product(SHAPES_2D, SHAPES_2D),
    ],
)
def test_build_bloch_operator(shape: tuple[int, ...], repeat: tuple[int, ...]) -> None:
    meta = metadata.volume.spaced_volume_metadata_from_stacked_delta_x(
        tuple(2 * np.pi * np.eye(len(shape))), shape
    )
    potential = operator.build.cos_potential(meta, 1)
    mass = hbar**2

    sparse = bloch.build.kinetic_hamiltonian(potential, mass, repeat)
    repeat_potential = operator.build.repeat_potential(potential, repeat)
    full = operator.build.kinetic_hamiltonian(repeat_potential, mass)

    assert sparse.basis.metadata() == full.basis.metadata()

    transformed = basis.transformed_from_metadata(
        full.basis.metadata(), is_dual=full.basis.is_dual
    )

    np.testing.assert_allclose(
        sparse.with_basis(transformed).raw_data.reshape(transformed.shape),
        full.with_basis(transformed).raw_data.reshape(transformed.shape),
        atol=1e-14,
    )

    np.testing.assert_allclose(sparse.as_array(), full.as_array(), atol=1e-14)


@pytest.mark.parametrize(
    ("shape", "repeat"),
    [
        *itertools.product(SHAPES_1D, SHAPES_1D),
        *itertools.product(SHAPES_2D, SHAPES_2D),
    ],
)
def test_build_momentum_bloch_operator(
    shape: tuple[int, ...], repeat: tuple[int, ...]
) -> None:
    meta = metadata.volume.spaced_volume_metadata_from_stacked_delta_x(
        tuple(2 * np.pi * np.eye(len(shape))), shape
    )

    potential = operator.build.cos_potential(meta, 0)
    mass = hbar**2

    sparse = bloch.build.kinetic_hamiltonian(potential, mass, repeat)
    repeat_potential = operator.build.repeat_potential(potential, repeat)
    full = operator.build.kinetic_hamiltonian(repeat_potential, mass)

    assert sparse.basis.metadata() == full.basis.metadata()

    transformed = basis.transformed_from_metadata(
        full.basis.metadata(), is_dual=full.basis.is_dual
    )

    np.testing.assert_allclose(
        np.diag(full.with_basis(transformed).raw_data.reshape(transformed.shape)),
        np.diag(sparse.with_basis(transformed).raw_data.reshape(transformed.shape)),
        atol=1e-15,
    )

    np.testing.assert_allclose(sparse.as_array(), full.as_array(), atol=1e-15)


@pytest.mark.parametrize(
    ("shape", "repeat"),
    [
        *itertools.product(SHAPES_1D, SHAPES_1D),
        *itertools.product(SHAPES_2D, SHAPES_2D),
    ],
)
def test_build_potential_bloch_operator_1d(
    shape: tuple[int, ...], repeat: tuple[int, ...]
) -> None:
    meta = metadata.volume.spaced_volume_metadata_from_stacked_delta_x(
        tuple(2 * np.pi * np.eye(len(shape))), shape
    )

    potential = operator.build.cos_potential(meta, 1)
    fraction_basis = basis.from_metadata(BlochFractionMetadata.from_repeats(repeat))

    operator_list = operator.OperatorList(
        TupleBasis((fraction_basis, potential.basis)),
        np.tile(potential.raw_data, np.prod(repeat).item()),
    )

    sparse = bloch.build.bloch_operator_from_list(operator_list)
    full = operator.build.repeat_potential(potential, repeat)

    assert sparse.basis.metadata() == full.basis.metadata()
    transformed = basis.transformed_from_metadata(
        full.basis.metadata(), is_dual=full.basis.is_dual
    )

    np.testing.assert_allclose(
        sparse.with_basis(transformed).raw_data.reshape(transformed.shape),
        full.with_basis(transformed).raw_data.reshape(transformed.shape),
        atol=1e-15,
    )


def test_bloch_operator_from_list() -> None:
    meta = metadata.volume.spaced_volume_metadata_from_stacked_delta_x(
        (np.array([2 * np.pi]),), (3,)
    )

    operator_basis = basis.transformed_from_metadata(meta)
    fraction_basis = basis.from_metadata(BlochFractionMetadata.from_repeats((2,)))
    operator_basis_tuple = TupleBasis((operator_basis, operator_basis.dual_basis()))

    operator_0 = Operator(operator_basis_tuple, np.ones((3, 3)))
    operator_1 = Operator(operator_basis_tuple, 2 * np.ones((3, 3)))
    operator_list = operator.OperatorList(
        TupleBasis((fraction_basis, operator_basis_tuple)),
        np.array([operator_0.raw_data, operator_1.raw_data], dtype=complex),
    )

    sparse = bloch.build.bloch_operator_from_list(operator_list)
    basis.transformed_from_metadata(
        sparse.basis.inner.metadata(), is_dual=sparse.basis.is_dual
    )
