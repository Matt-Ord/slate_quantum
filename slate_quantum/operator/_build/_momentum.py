from __future__ import annotations

from typing import TYPE_CHECKING, Any, Never

import numpy as np
from scipy.constants import hbar  # type: ignore stubs
from slate_core import BasisMetadata, Ctype, TupleMetadata, basis
from slate_core import metadata as _metadata
from slate_core.metadata import (
    AxisDirections,
    SpacedLengthMetadata,
    SpacedVolumeMetadata,
)

from slate_quantum.operator._diagonal import (
    MomentumOperator,
    MomentumOperatorBasis,
    momentum_operator_basis,
)
from slate_quantum.operator._operator import (
    Operator,
    OperatorBasis,
    OperatorList,
    OperatorListBasis,
    OperatorMetadata,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def momentum[M: BasisMetadata, E, CT: Ctype[Never], DT: np.dtype[np.generic]](
    outer_basis: basis.TupleBasisLike[tuple[M, ...], E, CT], data: np.ndarray[Any, DT]
) -> Operator[MomentumOperatorBasis[M, E], DT]:
    """Get the potential operator."""
    return Operator(momentum_operator_basis(outer_basis), data)


def k[M: SpacedLengthMetadata, E: AxisDirections](
    metadata: TupleMetadata[tuple[M, ...], E], *, axis: int
) -> MomentumOperator[M, E, Ctype[Never], np.dtype[np.complexfloating]]:
    """Get the k operator."""
    points = _metadata.volume.fundamental_stacked_k_points(metadata)[axis].astype(
        np.complex128
    )
    return momentum(basis.transformed_from_metadata(metadata).upcast(), points)


def p[M: SpacedLengthMetadata, E: AxisDirections](
    metadata: TupleMetadata[tuple[M, ...], E], *, axis: int
) -> MomentumOperator[M, E, Ctype[Never], np.dtype[np.complexfloating]]:
    """Get the p operator."""
    points = _metadata.volume.fundamental_stacked_k_points(metadata)[axis].astype(
        np.complex128
    )
    return momentum(
        basis.transformed_from_metadata(metadata).upcast(),
        (hbar * points).astype(np.complex128),
    )


def momentum_from_function[M: SpacedLengthMetadata, E: AxisDirections, DT: np.generic](
    metadata: TupleMetadata[tuple[M, ...], E],
    fn: Callable[
        [tuple[np.ndarray[Any, np.dtype[np.floating]], ...]],
        np.ndarray[Any, np.dtype[np.complexfloating]],
    ],
    *,
    wrapped: bool = False,
    offset: tuple[float, ...] | None = None,
) -> MomentumOperator[M, E, Ctype[Never], np.dtype[np.complexfloating]]:
    """Get the k operator from a function."""
    positions = _metadata.volume.fundamental_stacked_k_points(
        metadata, offset=offset, wrapped=wrapped
    )
    out_basis = basis.transformed_from_metadata(metadata).upcast()
    return momentum(out_basis, fn(positions))


def filter_scatter(
    operator: Operator[
        OperatorBasis[SpacedVolumeMetadata], np.dtype[np.complexfloating]
    ],
) -> Operator[OperatorBasis[SpacedVolumeMetadata], np.dtype[np.complexfloating]]:
    converted = operator.with_basis(
        basis.transformed_from_metadata(
            operator.basis.metadata(), is_dual=operator.basis.is_dual
        ).upcast()
    )
    data = converted.raw_data.reshape(converted.basis.inner.shape)
    nk_points = _metadata.fundamental_stacked_nk_points(
        operator.basis.metadata().children[0]
    )
    mask = np.all(
        [
            (np.abs(p[:, np.newaxis] - p[np.newaxis, :]) <= np.max(np.abs(p)))
            for p in nk_points
        ],
        axis=0,
    )
    return Operator(converted.basis, np.where(mask, data, 0))


def all_filter_scatter[M: BasisMetadata](
    operator: OperatorList[
        OperatorListBasis[M, OperatorMetadata[SpacedVolumeMetadata]],
        np.dtype[np.complexfloating],
    ],
) -> OperatorList[
    OperatorListBasis[M, OperatorMetadata[SpacedVolumeMetadata]],
    np.dtype[np.complexfloating],
]:
    is_dual = basis.as_tuple(operator.basis).is_dual[1]
    converted = operator.with_operator_basis(
        basis.transformed_from_metadata(
            operator.basis.metadata().children[1], is_dual=is_dual
        ).upcast()
    )
    data = converted.raw_data.reshape(
        -1, *converted.basis.inner.children[1].inner.shape
    )
    nk_points = _metadata.fundamental_stacked_nk_points(
        operator.basis.metadata().children[1].children[0].children[0]
    )
    mask = np.all(
        [
            (np.abs(p[:, np.newaxis] - p[np.newaxis, :]) <= np.max(np.abs(p)))
            for p in nk_points
        ],
        axis=0,
    )
    data[:, np.logical_not(mask)] = 0
    return OperatorList(converted.basis, data)
