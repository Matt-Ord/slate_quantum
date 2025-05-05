from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
from slate_core import Array, Basis, basis
from slate_core import metadata as _metadata
from slate_core.basis import (
    DiagonalBasis,
    FundamentalBasis,
    TransformedBasis,
    TruncatedBasis,
    Truncation,
    TupleBasis,
    diagonal_basis,
    tuple_basis,
)
from slate_core.metadata import (
    AxisDirections,
    SpacedLengthMetadata,
)
from slate_core.metadata.util import fundamental_size

from slate_quantum._util.legacy import StackedMetadata
from slate_quantum.operator._diagonal import (
    DiagonalOperator,
    PositionOperator,
    Potential,
    RecastDiagonalOperatorBasis,
    recast_diagonal_basis,
)
from slate_quantum.operator._operator import Operator, OperatorList
from slate_quantum.state._build import wrap_displacements

if TYPE_CHECKING:
    from slate_core.metadata import (
        BasisMetadata,
        SimpleMetadata,
    )
    from slate_core.metadata.length import LengthMetadata

    from slate_quantum._util.legacy import LegacyArray, LegacyTupleBasis2D


def get_displacements_x[M: LengthMetadata](
    metadata: M, origin: float
) -> LegacyArray[M, np.floating, FundamentalBasis[M]]:
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
    data = wrap_displacements(distances, max_distance)

    basis = FundamentalBasis(metadata)
    return Array(basis, data)


def nx_displacement_operators_stacked[M: StackedMetadata[BasisMetadata, Any]](
    metadata: M,
) -> OperatorList[
    SimpleMetadata,
    M,
    np.int64,
    LegacyTupleBasis2D[
        np.generic,
        FundamentalBasis[SimpleMetadata],
        LegacyTupleBasis2D[np.generic, Basis[M, Any], Basis[M, Any], None],
        None,
    ],
]:
    """Get a matrix of displacements in nx, taken in a periodic fashion."""
    ax = cast("Basis[M, Any]", basis.from_metadata(metadata))
    operator_basis = tuple_basis((ax, ax.dual_basis()))
    return OperatorList.from_operators(
        Operator(
            operator_basis,
            (n_x_points[:, np.newaxis] - n_x_points[np.newaxis, :] + n // 2) % n
            - (n // 2),
        )
        for (n_x_points, n) in zip(
            _metadata.fundamental_stacked_nx_points(metadata),
            _metadata.shallow_shape_from_nested(metadata.fundamental_shape),
            strict=True,
        )
    )


def nx_displacement_operator[M: BasisMetadata](
    metadata: M,
) -> Operator[
    M,
    np.int64,
    LegacyTupleBasis2D[np.generic, Basis[M, Any], Basis[M, Any], None],
]:
    """Get a matrix of displacements in nx, taken in a periodic fashion."""
    n_x_points = np.asarray(_metadata.fundamental_stacked_nx_points(metadata))
    n = fundamental_size(metadata)
    data = (n_x_points[:, np.newaxis] - n_x_points[np.newaxis, :] + n // 2) % n - (
        n // 2
    )
    ax = basis.from_metadata(metadata)
    return Operator(tuple_basis((ax, ax.dual_basis())), data)


def x_displacement_operator[M: LengthMetadata](
    metadata: M,
    origin: float = 0.0,
) -> Operator[
    M,
    np.floating,
    LegacyTupleBasis2D[np.generic, FundamentalBasis[M], FundamentalBasis[M], None],
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


def _get_displacements_matrix_x_along_axis[M: SpacedLengthMetadata, E: AxisDirections](
    metadata: StackedMetadata[M, E],
    origin: float = 0,
    *,
    axis: int,
) -> Operator[
    StackedMetadata[M, E],
    np.floating,
    LegacyTupleBasis2D[
        np.generic,
        Basis[StackedMetadata[M, E], Any],
        Basis[StackedMetadata[M, E], Any],
        None,
    ],
]:
    x_points = _metadata.volume.fundamental_stacked_x_points(metadata)[axis]
    distances = x_points[:, np.newaxis] - x_points[np.newaxis, :] - origin
    delta_x = np.linalg.norm(
        _metadata.volume.fundamental_stacked_delta_x(metadata)[axis]
    )
    max_distance = delta_x / 2
    data = wrap_displacements(distances, max_distance)

    ax = basis.from_metadata(metadata)
    return Operator(tuple_basis((ax, ax.dual_basis())), data)


def x_displacement_operators_stacked[M: SpacedLengthMetadata, E: AxisDirections](
    metadata: StackedMetadata[M, E], origin: tuple[float, ...] | None
) -> OperatorList[
    SimpleMetadata,
    StackedMetadata[M, E],
    np.floating,
    LegacyTupleBasis2D[
        np.generic,
        FundamentalBasis[SimpleMetadata],
        LegacyTupleBasis2D[
            np.generic,
            Basis[StackedMetadata[M, E], Any],
            Basis[StackedMetadata[M, E], Any],
            None,
        ],
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


def total_x_displacement_operator[M: SpacedLengthMetadata, E: AxisDirections](
    metadata: StackedMetadata[M, E],
    origin: tuple[float, ...] | None = None,
) -> Operator[
    StackedMetadata[M, E],
    np.float64,
    LegacyTupleBasis2D[
        np.generic,
        Basis[StackedMetadata[M, E], Any],
        Basis[StackedMetadata[M, E], Any],
        None,
    ],
]:
    """Get a matrix of displacements in x, taken in a periodic fashion."""
    displacements = x_displacement_operators_stacked(metadata, origin)
    return Operator(
        displacements.basis[1],
        np.linalg.norm(
            displacements.raw_data.reshape(displacements.basis.shape), axis=0
        ),
    )


def x[M: SpacedLengthMetadata, E: AxisDirections](
    metadata: StackedMetadata[M, E],
    *,
    axis: int,
    offset: float = 0,
    wrapped: bool = False,
) -> PositionOperator[M, E, np.complexfloating]:
    """Get the x operator."""
    inner_offset = tuple(offset if i == axis else 0 for i in range(metadata.n_dim))
    points = _metadata.volume.fundamental_stacked_x_points(
        metadata, offset=inner_offset, wrapped=wrapped
    )[axis]
    return Potential(basis.from_metadata(metadata), points.astype(np.complex128))


def axis_periodic_operator[M: BasisMetadata](
    inner_basis: Basis[M, Any], *, n_k: int
) -> DiagonalOperator[M, np.complexfloating]:
    """Get the generalized e^(ik.x) operator in some general basis.

    k is chosen such that k = 2 * np.pi n_k / N
    """
    transformed = TransformedBasis(inner_basis, direction="backward")
    outer_basis = TruncatedBasis(Truncation(1, 1, n_k), transformed)
    return DiagonalOperator(inner_basis, outer_basis, np.array([1 + 0j]))


def axis_scattering_operator[M: BasisMetadata](
    metadata: M, *, n_k: int
) -> DiagonalOperator[M, np.complexfloating]:
    """Get the e^(ik.x) operator.

    k is chosen such that k = 2 * np.pi n_k / N
    """
    inner_basis = basis.from_metadata(metadata)
    return axis_periodic_operator(inner_basis, n_k=n_k)


def all_axis_periodic_operators[M: BasisMetadata](
    inner_basis: Basis[M, Any],
) -> OperatorList[
    SimpleMetadata,
    M,
    np.complexfloating,
    DiagonalBasis[
        Any,
        FundamentalBasis[SimpleMetadata],
        RecastDiagonalOperatorBasis[M, Any],
        None,
    ],
]:
    """Get all generalized e^(ik.x) operator."""
    outer_basis = TransformedBasis(inner_basis)
    operator_basis = recast_diagonal_basis(inner_basis, outer_basis)

    list_basis = diagonal_basis(
        (FundamentalBasis.from_size(outer_basis.size), operator_basis)
    )
    return OperatorList(list_basis, np.ones(outer_basis.size, dtype=np.complex128))


def all_axis_scattering_operators[M: BasisMetadata](
    metadata: M,
) -> OperatorList[
    SimpleMetadata,
    M,
    np.complexfloating,
    DiagonalBasis[
        Any,
        FundamentalBasis[SimpleMetadata],
        RecastDiagonalOperatorBasis[M, Any],
        None,
    ],
]:
    """Get the e^(ik.x) operator."""
    inner_basis = basis.from_metadata(metadata)
    return all_axis_periodic_operators(inner_basis)


def periodic_operator[M: BasisMetadata, E](
    inner_basis: TupleBasis[M, E, Any], *, n_k: tuple[int, ...]
) -> DiagonalOperator[StackedMetadata[M, E], np.complexfloating]:
    """Get the generalized e^(ik.x) operator in some general basis.

    k is chosen such that k = 2 * np.pi * n_k / N
    """
    outer_basis = basis.with_modified_children(
        inner_basis,
        lambda i, inner: TruncatedBasis(
            Truncation(1, 1, n_k[i]), TransformedBasis(inner, direction="backward")
        ),
    )
    return DiagonalOperator(inner_basis, outer_basis, np.array([1 + 0j]))


def scattering_operator[M: SpacedLengthMetadata, E: AxisDirections](
    metadata: StackedMetadata[M, E], *, n_k: tuple[int, ...]
) -> PositionOperator[M, E, np.complexfloating]:
    """Get the e^(ik.x) operator.

    k is chosen such that k = 2 * np.pi * n_k / N
    """
    outer_basis = basis.with_modified_children(
        basis.from_metadata(metadata),
        lambda i, inner: TruncatedBasis(
            Truncation(1, 1, n_k[i]), TransformedBasis(inner, direction="backward")
        ),
    )
    return PositionOperator(outer_basis, np.array([1]))


def all_periodic_operators[M: BasisMetadata, E](
    inner_basis: TupleBasis[M, E, Any],
) -> OperatorList[
    SimpleMetadata,
    StackedMetadata[M, E],
    np.complexfloating,
    DiagonalBasis[
        Any,
        FundamentalBasis[SimpleMetadata],
        RecastDiagonalOperatorBasis[StackedMetadata[M, E], Any],
        None,
    ],
]:
    """Get all generalized e^(ik.x) operator."""
    outer_basis = basis.with_modified_children(
        inner_basis,
        lambda _i, inner: TransformedBasis(inner, direction="backward"),
    )
    operator_basis = recast_diagonal_basis(inner_basis, outer_basis)

    list_basis = diagonal_basis(
        (FundamentalBasis.from_size(outer_basis.size), operator_basis)
    )
    return OperatorList(list_basis, np.ones(outer_basis.size, dtype=np.complex128))


def all_scattering_operators[M: BasisMetadata, E](
    metadata: StackedMetadata[M, E],
) -> OperatorList[
    SimpleMetadata,
    StackedMetadata[M, E],
    np.complexfloating,
    DiagonalBasis[
        Any,
        FundamentalBasis[SimpleMetadata],
        RecastDiagonalOperatorBasis[StackedMetadata[M, E], Any],
        None,
    ],
]:
    """Get the e^(ik.x) operator."""
    inner_basis = basis.from_metadata(metadata)
    return all_periodic_operators(inner_basis)
