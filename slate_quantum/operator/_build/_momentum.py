from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.constants import hbar  # type: ignore stubs
from slate import BasisMetadata, StackedMetadata, basis
from slate import metadata as _metadata
from slate.metadata import AxisDirections, SpacedLengthMetadata, SpacedVolumeMetadata

from slate_quantum.operator._diagonal import (
    MomentumOperator,
)
from slate_quantum.operator._operator import Operator, OperatorList

if TYPE_CHECKING:
    from collections.abc import Callable


def k[M: SpacedLengthMetadata, E: AxisDirections](
    metadata: StackedMetadata[M, E], *, axis: int
) -> MomentumOperator[M, E]:
    """Get the k operator."""
    points = _metadata.volume.fundamental_stacked_k_points(metadata)[axis].astype(
        np.complex128
    )
    return MomentumOperator(
        basis.fundamental_transformed_tuple_basis_from_metadata(metadata),
        points,
    )


def p[M: SpacedLengthMetadata, E: AxisDirections](
    metadata: StackedMetadata[M, E], *, axis: int
) -> MomentumOperator[M, E]:
    """Get the p operator."""
    points = _metadata.volume.fundamental_stacked_k_points(metadata)[axis].astype(
        np.complex128
    )
    return MomentumOperator(
        basis.fundamental_transformed_tuple_basis_from_metadata(metadata),
        (hbar * points).astype(np.complex128),
    )


def momentum_from_function[M: SpacedLengthMetadata, E: AxisDirections, DT: np.generic](
    metadata: StackedMetadata[M, E],
    fn: Callable[
        [tuple[np.ndarray[Any, np.dtype[np.floating]], ...]],
        np.ndarray[Any, np.dtype[np.complexfloating]],
    ],
    *,
    wrapped: bool = False,
    offset: tuple[float, ...] | None = None,
) -> MomentumOperator[M, E]:
    """Get the k operator from a function."""
    positions = _metadata.volume.fundamental_stacked_k_points(
        metadata, offset=offset, wrapped=wrapped
    )
    out_basis = basis.fundamental_transformed_tuple_basis_from_metadata(metadata)
    return MomentumOperator(out_basis, fn(positions))


def filter_scatter(
    operator: Operator[SpacedVolumeMetadata, np.complexfloating],
) -> Operator[SpacedVolumeMetadata, np.complexfloating]:
    converted = operator.with_basis(
        basis.fundamental_transformed_tuple_basis_from_metadata(
            operator.basis.metadata(), is_dual=operator.basis.is_dual
        )
    )
    data = converted.raw_data.reshape(converted.basis.shape)
    nk_points = _metadata.fundamental_stacked_nk_points(operator.basis.metadata()[0])
    mask = np.all(
        [
            (np.abs(p[:, np.newaxis] - p[np.newaxis, :]) <= np.max(np.abs(p)))
            for p in nk_points
        ],
        axis=0,
    )
    return Operator(converted.basis, np.where(mask, data, 0))


def all_filter_scatter[M: BasisMetadata](
    operator: OperatorList[M, SpacedVolumeMetadata, np.complexfloating],
) -> OperatorList[M, SpacedVolumeMetadata, np.complexfloating]:
    is_dual = basis.as_tuple_basis(operator.basis).is_dual[1]
    converted = operator.with_operator_basis(
        basis.fundamental_transformed_tuple_basis_from_metadata(
            operator.basis.metadata()[1], is_dual=is_dual
        )
    )
    data = converted.raw_data.reshape(-1, *converted.basis[1].shape)
    nk_points = _metadata.fundamental_stacked_nk_points(
        operator.basis.metadata()[1][0][0]
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
