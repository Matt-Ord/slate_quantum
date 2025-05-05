from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from slate_core import FundamentalBasis, TupleBasis, TupleMetadata, basis
from slate_core.basis import AsUpcast, CoordinateBasis
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
    build,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from slate_quantum.operator._diagonal import PositionOperatorBasis
    from slate_quantum.operator._operator import (
        OperatorBasis,
        OperatorMetadata,
    )

type PositionNoiseOperatorList[M: SpacedLengthMetadata, E, DT: np.dtype[np.generic]] = (
    OperatorList[
        AsUpcast[
            TupleBasis[
                tuple[
                    FundamentalBasis[EigenvalueMetadata],
                    PositionOperatorBasis[M, E],
                ],
                None,
            ],
            TupleMetadata[
                tuple[
                    EigenvalueMetadata,
                    OperatorMetadata[TupleMetadata[tuple[M, ...], E]],
                ],
                None,
            ],
        ],
        DT,
    ]
)


def periodic_caldeira_leggett_axis_operators[M: SpacedLengthMetadata](
    metadata: M,
) -> OperatorList[
    AsUpcast[
        TupleBasis[
            tuple[FundamentalBasis[EigenvalueMetadata], OperatorBasis[M]],
            None,
        ],
        TupleMetadata[
            tuple[EigenvalueMetadata, OperatorMetadata[M]],
            None,
        ],
    ],
    np.dtype[np.complexfloating],
]:
    k = fundamental_dk(metadata)
    n = metadata.fundamental_size
    eigenvalue = n / (4 * k**2)
    operators = build.all_axis_scattering_operators(metadata)

    list_basis = CoordinateBasis(
        [-1, 1], basis.from_metadata(operators.basis.metadata().children[0])
    )
    converted = operators.with_list_basis(list_basis).assert_ok()
    return OperatorList.build(
        TupleBasis(
            (
                eigenvalue_basis(np.array([eigenvalue, eigenvalue])),
                converted.basis.inner.children[1],
            )
        ).upcast(),
        converted.raw_data,
    ).assert_ok()


def periodic_caldeira_leggett_operators[
    M: SpacedLengthMetadata,
    E: AxisDirections,
](
    metadata: TupleMetadata[tuple[M, ...], E],
) -> PositionNoiseOperatorList[M, E, np.dtype[np.complexfloating]]:
    assert len(metadata.fundamental_shape) == 1
    k = fundamental_stacked_dk(metadata)[0][0]
    n = size_from_nested_shape(metadata.fundamental_shape)
    eigenvalue = n / (4 * k**2)
    operators = build.all_scattering_operators(metadata)

    list_basis = CoordinateBasis(
        [-1, 1], basis.from_metadata(operators.basis.metadata().children[0])
    )
    converted = operators.with_list_basis(list_basis).assert_ok()
    return OperatorList.build(
        TupleBasis(
            (
                eigenvalue_basis(np.array([eigenvalue, eigenvalue])),
                converted.basis.inner.children[1],
            )
        ).upcast(),
        converted.raw_data,
    ).assert_ok()


def real_periodic_caldeira_leggett_operators[
    M: SpacedLengthMetadata,
    E: AxisDirections,
](
    metadata: TupleMetadata[tuple[M, ...], E],
) -> PositionNoiseOperatorList[M, E, np.dtype[np.complexfloating]]:
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

    return OperatorList.build(
        TupleBasis(
            (
                eigenvalue_basis(np.array([eigenvalue, eigenvalue])),
                operators.basis.inner.children[1],
            )
        ).upcast(),
        operators.raw_data,
    ).assert_ok()


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
    metadata: TupleMetadata[tuple[M, ...], E],
) -> PositionNoiseOperatorList[M, E, np.dtype[np.complexfloating]]:
    assert len(metadata.fundamental_shape) == 1
    operators = OperatorList.from_operators([build.x(metadata, axis=0)])

    return OperatorList.build(
        TupleBasis(
            (eigenvalue_basis(np.array([1])), operators.basis.inner.children[1])
        ).upcast(),
        operators.raw_data,
    ).assert_ok()
