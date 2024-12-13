from __future__ import annotations

from typing import Any

import numpy as np
from slate import StackedMetadata, basis, tuple_basis
from slate.basis import CoordinateBasis, FundamentalBasis, TupleBasis2D
from slate.metadata import AxisDirections, SpacedLengthMetadata, size_from_nested_shape
from slate.metadata.length import fundamental_dk
from slate.metadata.volume import fundamental_stacked_dk

from slate_quantum.metadata._label import EigenvalueMetadata, eigenvalue_basis
from slate_quantum.operator import (
    OperatorList,
    RecastDiagonalOperatorBasis,
    build,
)


def build_periodic_caldeira_leggett_axis_operators[M: SpacedLengthMetadata](
    metadata: M,
) -> OperatorList[
    EigenvalueMetadata,
    M,
    np.complex128,
    TupleBasis2D[
        Any,
        FundamentalBasis[EigenvalueMetadata],
        RecastDiagonalOperatorBasis[M, Any],
        None,
    ],
]:
    k = fundamental_dk(metadata)
    n = metadata.fundamental_size
    eigenvalue = n / (4 * k**2)
    operators = build.all_axis_scattering_operators(metadata)
    operators = operators.with_basis(operators.basis.inner)

    list_basis = CoordinateBasis(
        [-1, 1], basis.from_metadata(operators.basis.metadata()[0])
    )
    converted = operators.with_list_basis(list_basis)
    return OperatorList(
        tuple_basis(
            (eigenvalue_basis(np.array([eigenvalue, eigenvalue])), converted.basis[1])
        ),
        converted.raw_data,
    )


def build_periodic_caldeira_leggett_operators[
    M: SpacedLengthMetadata,
    E: AxisDirections,
](
    metadata: StackedMetadata[M, E],
) -> OperatorList[
    EigenvalueMetadata,
    StackedMetadata[M, E],
    np.complex128,
    TupleBasis2D[
        Any,
        FundamentalBasis[EigenvalueMetadata],
        RecastDiagonalOperatorBasis[StackedMetadata[M, E], Any],
        None,
    ],
]:
    assert len(metadata.fundamental_shape) == 1
    k = fundamental_stacked_dk(metadata)[0][0]
    n = size_from_nested_shape(metadata.fundamental_shape)
    eigenvalue = n / (4 * k**2)
    operators = build.all_scattering_operators(metadata)
    operators = operators.with_basis(operators.basis.inner)

    list_basis = CoordinateBasis(
        [-1, 1], basis.from_metadata(operators.basis.metadata()[0])
    )
    converted = operators.with_list_basis(list_basis)
    return OperatorList(
        tuple_basis(
            (eigenvalue_basis(np.array([eigenvalue, eigenvalue])), converted.basis[1])
        ),
        converted.raw_data,
    )


def build_periodic_caldeira_leggett_real_operators[
    M: SpacedLengthMetadata,
    E: AxisDirections,
](
    metadata: StackedMetadata[M, E],
) -> OperatorList[
    EigenvalueMetadata,
    StackedMetadata[M, E],
    np.complex128,
    TupleBasis2D[
        Any,
        FundamentalBasis[EigenvalueMetadata],
        RecastDiagonalOperatorBasis[StackedMetadata[M, E], Any],
        None,
    ],
]:
    assert len(metadata.fundamental_shape) == 1
    k = fundamental_stacked_dk(metadata)[0][0]
    n = size_from_nested_shape(metadata.fundamental_shape)
    eigenvalue = n / (4 * k**2)
    delta_x = metadata[0].spacing.delta
    operators = OperatorList.from_operators(
        [
            build.potential_from_function(
                metadata, lambda x: np.cos(2 * np.pi * x[0] / delta_x) / np.sqrt(n)
            ),
            build.potential_from_function(
                metadata, lambda x: np.sin(2 * np.pi * x[0] / delta_x) / np.sqrt(n)
            ),
        ]
    )

    return OperatorList(
        tuple_basis(
            (eigenvalue_basis(np.array([eigenvalue, eigenvalue])), operators.basis[1])
        ),
        operators.raw_data,
    )
