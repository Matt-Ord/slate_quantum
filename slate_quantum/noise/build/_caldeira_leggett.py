from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from slate_core import StackedMetadata, basis, tuple_basis
from slate_core.basis import CoordinateBasis, FundamentalBasis, TupleBasis2D
from slate_core.metadata import (
    AxisDirections,
    SpacedLengthMetadata,
    size_from_nested_shape,
)
from slate_core.metadata.length import fundamental_dk
from slate_core.metadata.volume import fundamental_stacked_dk

from slate_quantum.metadata._label import EigenvalueMetadata, eigenvalue_basis
from slate_quantum.operator import (
    OperatorList,
    RecastDiagonalOperatorBasis,
    build,
)

if TYPE_CHECKING:
    from collections.abc import Callable


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
    # The eigenvalue is scaled wrt the e^ikx operators
    # When we convert from complex e^ikx operators
    # to real cos(kx) and sin(kx) operators, we actually
    # only have (e^ikx +- e^-ikx)
    # This scaling accounts for this difference
    eigenvalue = n / (k**2)
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


def caldeira_leggett_correlation_fn(
    a: float, lambda_: float
) -> Callable[
    [np.ndarray[Any, np.dtype[np.float64]]],
    np.ndarray[Any, np.dtype[np.complexfloating]],
]:
    r"""Get a correlation function for a lorentzian noise kernel.

    A lorentzian noise kernel is isotropic, and separable into individual
    axis kernels. The kernel is given by

    .. math::
        \beta(x, x') = a^2 - \frac{\lambda^2}{4} (x-x')^2
    """

    def fn(
        displacements: np.ndarray[Any, np.dtype[np.float64]],
    ) -> np.ndarray[Any, np.dtype[np.complexfloating]]:
        return (a**2 - (lambda_**2 / 4) * displacements**2).astype(np.complex128)

    return fn


def caldeira_leggett_operators[M: SpacedLengthMetadata, E: AxisDirections](
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
    operators = OperatorList.from_operators([build.x(metadata, axis=0)])

    return OperatorList(
        tuple_basis((eigenvalue_basis(np.array([1])), operators.basis[1])),
        operators.raw_data,
    )
