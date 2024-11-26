from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
from slate import Basis
from slate.array import SlateArray
from slate.basis import (
    FundamentalBasis,
    TupleBasis,
    TupleBasis2D,
    fundamental_basis_from_metadata,
    tuple_basis,
)
from slate.metadata import (
    Metadata2D,
    SpacedLengthMetadata,
    fundamental_stacked_nx_points,
    shallow_shape_from_nested,
)
from slate.metadata.util import fundamental_size
from slate.metadata.volume import (
    AxisDirections,
    fundamental_stacked_delta_x,
    fundamental_stacked_x_points,
)

from slate_quantum.model.operator._operator import Operator, OperatorList
from slate_quantum.model.operator.potential._potential import Potential

if TYPE_CHECKING:
    from slate.metadata import (
        BasisMetadata,
        SimpleMetadata,
        SpacedVolumeMetadata,
        StackedMetadata,
    )
    from slate.metadata.length import LengthMetadata


def _wrap_displacements(
    displacements: np.ndarray[Any, np.dtype[np.float64]],
    max_displacement: float | np.float64,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    return (
        np.remainder((displacements + max_displacement), 2 * max_displacement)
        - max_displacement
    ).astype(np.float64)


def get_displacements_x[M: LengthMetadata](
    metadata: M, origin: float
) -> SlateArray[M, np.float64, FundamentalBasis[M]]:
    """Get the displacements from origin.

    Parameters
    ----------
    basis : BasisWithLengthLike
    origin : float

    Returns
    -------
    ValueList[FundamentalPositionBasis]
    """
    distances = np.array(list(metadata.values)) - origin
    max_distance = np.linalg.norm(metadata.delta) / 2
    data = _wrap_displacements(distances, max_distance)

    basis = FundamentalBasis(metadata)
    return SlateArray(basis, data)


def _get_displacements_x_along_axis(
    metadata: SpacedVolumeMetadata,
    origin: float,
    axis: int,
) -> SlateArray[
    SpacedVolumeMetadata,
    np.float64,
    TupleBasis[LengthMetadata, AxisDirections, np.generic],
]:
    distances = fundamental_stacked_x_points(metadata)[axis] - np.real(origin)
    delta_x = np.linalg.norm(fundamental_stacked_delta_x(metadata)[axis])
    max_distance = delta_x / 2
    data = _wrap_displacements(distances, max_distance)

    basis = fundamental_basis_from_metadata(metadata)
    return SlateArray(basis, data)


def get_displacements_x_stacked(
    metadata: SpacedVolumeMetadata, origin: tuple[float, ...]
) -> tuple[
    SlateArray[
        SpacedVolumeMetadata,
        np.float64,
        TupleBasis[LengthMetadata, AxisDirections, np.generic],
    ],
    ...,
]:
    """Get the displacements from origin."""
    return tuple(
        _get_displacements_x_along_axis(metadata, o, axis)
        for (axis, o) in enumerate(origin)
    )


def build_nx_displacement_operators_stacked[M: StackedMetadata[BasisMetadata, Any]](
    metadata: M,
) -> OperatorList[
    Metadata2D[SimpleMetadata, Metadata2D[M, M, None], None],
    np.int64,
    TupleBasis2D[
        np.generic,
        FundamentalBasis[SimpleMetadata],
        TupleBasis2D[np.generic, Basis[M, Any], Basis[M, Any], None],
        None,
    ],
]:
    """Get a matrix of displacements in nx, taken in a periodic fashion."""
    ax = cast(Basis[M, Any], fundamental_basis_from_metadata(metadata))
    basis = tuple_basis((ax, ax.dual_basis()))
    return OperatorList.from_operators(
        Operator(
            basis,
            (n_x_points[:, np.newaxis] - n_x_points[np.newaxis, :] + n // 2) % n
            - (n // 2),
        )
        for (n_x_points, n) in zip(
            fundamental_stacked_nx_points(metadata),
            shallow_shape_from_nested(metadata.fundamental_shape),
            strict=True,
        )
    )


def build_nx_displacement_operator[M: BasisMetadata](
    metadata: M,
) -> Operator[
    Metadata2D[M, M, None],
    np.int64,
    TupleBasis2D[np.generic, Basis[M, Any], Basis[M, Any], None],
]:
    """Get a matrix of displacements in nx, taken in a periodic fashion."""
    n_x_points = np.asarray(fundamental_stacked_nx_points(metadata))
    n = fundamental_size(metadata)
    data = (n_x_points[:, np.newaxis] - n_x_points[np.newaxis, :] + n // 2) % n - (
        n // 2
    )
    basis = cast(Basis[M, Any], fundamental_basis_from_metadata(metadata))
    return Operator(tuple_basis((basis, basis.dual_basis())), data)


def build_x_displacement_operator[M: LengthMetadata](
    metadata: M,
    origin: float = 0.0,
) -> Operator[
    Metadata2D[M, M, None],
    np.float64,
    TupleBasis2D[np.generic, FundamentalBasis[M], FundamentalBasis[M], None],
]:
    """Get the displacements from origin.

    Parameters
    ----------
    basis : BasisWithLengthLike
    origin : float

    Returns
    -------
    ValueList[FundamentalPositionBasis]
    """
    x_points = np.array(list(metadata.values))
    distances = x_points[:, np.newaxis] - x_points[np.newaxis, :] - origin
    max_distance = np.linalg.norm(metadata.delta) / 2
    data = np.remainder((distances + metadata.delta), max_distance) - max_distance

    basis = FundamentalBasis(metadata)
    return Operator(tuple_basis((basis, basis.dual_basis())), data)


def _get_displacements_matrix_x_along_axis[M: SpacedVolumeMetadata](
    metadata: M,
    origin: float = 0,
    *,
    axis: int,
) -> Operator[
    Metadata2D[M, M, None],
    np.float64,
    TupleBasis2D[np.generic, Basis[M, Any], Basis[M, Any], None],
]:
    x_points = fundamental_stacked_x_points(metadata)[axis]
    distances = x_points[:, np.newaxis] - x_points[np.newaxis, :] - origin
    delta_x = np.linalg.norm(fundamental_stacked_delta_x(metadata)[axis])
    max_distance = delta_x / 2
    data = _wrap_displacements(distances, max_distance)

    basis = cast(Basis[M, Any], fundamental_basis_from_metadata(metadata))
    return Operator(tuple_basis((basis, basis.dual_basis())), data)


def build_x_displacement_operators_stacked[M: SpacedVolumeMetadata](
    metadata: M, origin: tuple[float, ...] | None
) -> OperatorList[
    Metadata2D[SimpleMetadata, Metadata2D[Any, Any, None], None],
    np.float64,
    TupleBasis2D[
        np.generic,
        FundamentalBasis[SimpleMetadata],
        TupleBasis2D[np.generic, Basis[M, Any], Basis[M, Any], None],
        None,
    ],
]:
    """Get the displacements from origin."""
    shape = metadata.fundamental_shape
    origin = tuple(0.0 for _ in shape) if origin is None else origin

    return OperatorList.from_operators(
        _get_displacements_matrix_x_along_axis(metadata, o, axis=axis)
        for (axis, o) in enumerate(origin)
    )


def build_total_x_displacement_operator[M: SpacedVolumeMetadata](
    metadata: M,
    origin: tuple[float, ...] | None = None,
) -> Operator[
    Metadata2D[M, M, None],
    np.float64,
    TupleBasis2D[np.generic, Basis[M, Any], Basis[M, Any], None],
]:
    """Get a matrix of displacements in x, taken in a periodic fashion."""
    displacements = build_x_displacement_operators_stacked(metadata, origin)
    return Operator(
        displacements.basis[1],
        np.linalg.norm(
            displacements.raw_data.reshape(displacements.basis.shape), axis=0
        ),
    )


def build_x_operator[M: SpacedLengthMetadata, E: AxisDirections](
    metadata: StackedMetadata[M, E], *, idx: int
) -> Potential[StackedMetadata[M, E], np.complex128]:
    """Get the x operator."""
    basis = fundamental_basis_from_metadata(metadata)
    points = fundamental_stacked_x_points(metadata)[idx].astype(np.complex128)
    return Potential(basis, points)