from __future__ import annotations

import numpy as np
from scipy.constants import hbar  # type: ignore stubs
from slate import StackedMetadata, basis
from slate.metadata import AxisDirections, SpacedLengthMetadata
from slate.metadata.volume import fundamental_stacked_k_points

from slate_quantum.operator._diagonal import (
    MomentumOperator,
)


def k_operator[M: SpacedLengthMetadata, E: AxisDirections](
    metadata: StackedMetadata[M, E], *, idx: int
) -> MomentumOperator[M, E]:
    """Get the k operator."""
    points = fundamental_stacked_k_points(metadata)[idx].astype(np.complex128)
    return MomentumOperator(
        basis.fundamental_transformed_tuple_basis_from_metadata(metadata),
        points,
    )


def p_operator[M: SpacedLengthMetadata, E: AxisDirections](
    metadata: StackedMetadata[M, E], *, idx: int
) -> MomentumOperator[M, E]:
    """Get the p operator."""
    points = fundamental_stacked_k_points(metadata)[idx].astype(np.complex128)
    return MomentumOperator(
        basis.fundamental_transformed_tuple_basis_from_metadata(metadata),
        (hbar * points).astype(np.complex128),
    )
