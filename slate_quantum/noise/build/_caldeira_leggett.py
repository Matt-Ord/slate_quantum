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


def periodic_caldeira_leggett_axis_operators[M: SpacedLengthMetadata](
    metadata: M,
) -> OperatorList[
    EigenvalueMetadata,
    M,
    np.complexfloating,
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


def periodic_caldeira_leggett_operators[
    M: SpacedLengthMetadata,
    E: AxisDirections,
](
    metadata: StackedMetadata[M, E],
) -> OperatorList[
    EigenvalueMetadata,
    StackedMetadata[M, E],
    np.complexfloating,
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


def real_periodic_caldeira_leggett_operators[
    M: SpacedLengthMetadata,
    E: AxisDirections,
](
    metadata: StackedMetadata[M, E],
) -> OperatorList[
    EigenvalueMetadata,
    StackedMetadata[M, E],
    np.complexfloating,
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
    # The eigenvalue is scaled by sqrt(2),
    # When we convert from complex e^ikx operators
    # to real cos(kx) and sin(kx) operators, we actually
    # only have 1/(sqrt(2))(e^ikx +- e^-ikx)
    # This scaling therefore accounts for this difference
    eigenvalue *= np.sqrt(2)
    operators = OperatorList.from_operators(
        [
            build.potential_from_function(
                metadata, lambda x: np.cos(k * x[0]) / np.sqrt(n)
            ),
            build.potential_from_function(
                metadata, lambda x: np.sin(k * x[0]) / np.sqrt(n)
            ),
        ]
    )

    return OperatorList(
        tuple_basis(
            (eigenvalue_basis(np.array([eigenvalue, eigenvalue])), operators.basis[1])
        ),
        operators.raw_data,
    )
