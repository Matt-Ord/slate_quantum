from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from slate_core import Array, Basis, TupleBasisLike, TupleMetadata, basis, ctype
from slate_core import metadata as _metadata
from slate_core.basis import (
    AsUpcast,
    DiagonalBasis,
    FundamentalBasis,
    TransformedBasis,
    TruncatedBasis,
    Truncation,
    TupleBasis,
    is_tuple_basis,
)
from slate_core.metadata import (
    AxisDirections,
    SpacedLengthMetadata,
)
from slate_core.metadata.util import fundamental_size

from slate_quantum.operator._build._potential import potential
from slate_quantum.operator._diagonal import (
    DiagonalOperator,
    DiagonalOperatorLike,
    PositionOperator,
    Potential,
    recast_diagonal_basis,
)
from slate_quantum.operator._operator import (
    Operator,
    OperatorList,
    OperatorListMetadata,
    OperatorMetadata,
    operator_basis,
)
from slate_quantum.state._build import wrap_displacements

if TYPE_CHECKING:
    from slate_core.metadata import (
        BasisMetadata,
        SimpleMetadata,
    )
    from slate_core.metadata.length import LengthMetadata

    from slate_quantum.operator._diagonal import DiagonalOperatorBasis


def get_displacements_x[M: LengthMetadata](
    metadata: M, origin: float
) -> Array[FundamentalBasis[M], np.dtype[np.floating]]:
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
    return Array.build(basis, data).ok()


def nx_displacement_operators_stacked[M: BasisMetadata](
    metadata: M,
) -> OperatorList[
    Basis[OperatorListMetadata[SimpleMetadata, OperatorMetadata[M]]],
    np.dtype[np.int64],
]:
    """Get a matrix of displacements in nx, taken in a periodic fashion."""
    out_basis = operator_basis(basis.from_metadata(metadata))
    return OperatorList.from_operators(
        Operator.build(
            out_basis,
            (n_x_points[:, np.newaxis] - n_x_points[np.newaxis, :] + n // 2) % n
            - (n // 2),
        ).ok()
        for (n_x_points, n) in zip(
            _metadata.fundamental_stacked_nx_points(metadata),
            _metadata.shallow_shape_from_nested(metadata.fundamental_shape),
            strict=True,
        )
    )


def nx_displacement_operator[M: BasisMetadata](
    metadata: M,
) -> Operator[Basis[OperatorMetadata[M]], np.dtype[np.int64]]:
    """Get a matrix of displacements in nx, taken in a periodic fashion."""
    n_x_points = np.asarray(_metadata.fundamental_stacked_nx_points(metadata))
    n = fundamental_size(metadata)
    data = (n_x_points[:, np.newaxis] - n_x_points[np.newaxis, :] + n // 2) % n - (
        n // 2
    )
    ax = basis.from_metadata(metadata)
    return Operator.build(
        TupleBasis((ax, ax.dual_basis())).resolve_ctype().upcast(), data
    ).ok()


def x_displacement_operator[M: LengthMetadata](
    metadata: M,
    origin: float = 0.0,
) -> Operator[
    Basis[OperatorMetadata[M], ctype[np.generic]],
    np.dtype[np.floating],
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
    return Operator.build(
        TupleBasis((basis, basis.dual_basis())).resolve_ctype().upcast(),
        data,
    ).ok()


def _get_displacements_matrix_x_along_axis[M: SpacedLengthMetadata, E: AxisDirections](
    metadata: TupleMetadata[tuple[M, ...], E],
    origin: float = 0,
    *,
    axis: int,
) -> Operator[
    Basis[OperatorMetadata[TupleMetadata[tuple[M, ...], E]], ctype[np.generic]],
    np.dtype[np.floating],
]:
    x_points = _metadata.volume.fundamental_stacked_x_points(metadata)[axis]
    distances = x_points[:, np.newaxis] - x_points[np.newaxis, :] - origin
    delta_x = np.linalg.norm(
        _metadata.volume.fundamental_stacked_delta_x(metadata)[axis]
    )
    max_distance = delta_x / 2
    data = wrap_displacements(distances, max_distance)
    out_basis = operator_basis(basis.from_metadata(metadata).upcast())
    return Operator.build(out_basis, data).ok()


def x_displacement_operators_stacked[M: SpacedLengthMetadata, E: AxisDirections](
    metadata: TupleMetadata[tuple[M, ...], E], origin: tuple[float, ...] | None
) -> OperatorList[
    Basis[
        OperatorListMetadata[
            SimpleMetadata, OperatorMetadata[TupleMetadata[tuple[M, ...], E]]
        ],
    ],
    np.dtype[np.floating],
]:
    """Get the displacements from origin."""
    shape = metadata.fundamental_shape
    origin = tuple(0.0 for _ in shape) if origin is None else origin

    return OperatorList.from_operators(
        _get_displacements_matrix_x_along_axis(metadata, o, axis=axis)
        for (axis, o) in enumerate(origin)
    )


def total_x_displacement_operator[M: SpacedLengthMetadata, E: AxisDirections](
    metadata: TupleMetadata[tuple[M, ...], E],
    origin: tuple[float, ...] | None = None,
) -> Operator[
    Basis[OperatorMetadata[TupleMetadata[tuple[M, ...], E]]],
    np.dtype[np.float64],
]:
    """Get a matrix of displacements in x, taken in a periodic fashion."""
    displacements = x_displacement_operators_stacked(metadata, origin)
    inner_basis = displacements.basis
    assert is_tuple_basis(inner_basis)
    return Operator.build(
        inner_basis.children[1],
        np.linalg.norm(displacements.raw_data.reshape(inner_basis.shape), axis=0),
    ).ok()


def x[M: SpacedLengthMetadata, E: AxisDirections](
    metadata: TupleMetadata[tuple[M, ...], E],
    *,
    axis: int,
    offset: float = 0,
    wrapped: bool = False,
) -> Potential[M, E, ctype[np.complexfloating], np.dtype[np.complexfloating]]:
    """Get the x operator."""
    inner_offset = tuple(offset if i == axis else 0 for i in range(metadata.n_dim))
    points = _metadata.volume.fundamental_stacked_x_points(
        metadata, offset=inner_offset, wrapped=wrapped
    )[axis]
    return potential(
        basis.from_metadata(metadata).upcast(), points.astype(np.complex128)
    ).ok()


def axis_periodic_operator[M: BasisMetadata](
    inner_basis: Basis[M, Any], *, n_k: int
) -> DiagonalOperatorLike[M, np.dtype[np.complexfloating]]:
    """Get the generalized e^(ik.x) operator in some general basis.

    k is chosen such that k = 2 * np.pi n_k / N
    """
    transformed = TransformedBasis(inner_basis, direction="backward")
    outer_basis = TruncatedBasis(Truncation(1, 1, n_k), transformed)
    return DiagonalOperator(inner_basis, outer_basis, np.array([1 + 0j]))


def axis_scattering_operator[M: BasisMetadata](
    metadata: M, *, n_k: int
) -> DiagonalOperatorLike[M, np.dtype[np.complexfloating]]:
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
    np.dtype[np.complexfloating],
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
    np.dtype[np.complexfloating],
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
    inner_basis: TupleBasis[tuple[Basis[M], ...], E, Any], *, n_k: tuple[int, ...]
) -> DiagonalOperatorLike[
    TupleMetadata[tuple[M, ...], E], np.dtype[np.complexfloating]
]:
    """Get the generalized e^(ik.x) operator in some general basis.

    k is chosen such that k = 2 * np.pi * n_k / N
    """
    outer_basis = basis.with_modified_children(
        inner_basis,
        lambda i, inner: TruncatedBasis(
            Truncation(1, 1, n_k[i]),
            TransformedBasis(inner, direction="backward").resolve_ctype().upcast(),
        )
        .resolve_ctype()
        .upcast(),
    )
    basis_recast = recast_diagonal_basis(inner_basis, outer_basis)
    return Operator.build(basis_recast, np.array([1 + 0j])).ok()


def scattering_operator[M: SpacedLengthMetadata, E: AxisDirections](
    metadata: TupleMetadata[tuple[M, ...], E], *, n_k: tuple[int, ...]
) -> Potential[M, E, ctype[np.complexfloating], np.dtype[np.complexfloating]]:
    """Get the e^(ik.x) operator.

    k is chosen such that k = 2 * np.pi * n_k / N
    """
    outer_basis = basis.with_modified_children(
        basis.from_metadata(metadata),
        lambda i, inner: TruncatedBasis(
            Truncation(1, 1, n_k[i]),
            TransformedBasis(inner, direction="backward").resolve_ctype().upcast(),
        )
        .resolve_ctype()
        .upcast(),
    )
    return PositionOperator(outer_basis, np.array([1]))


def all_periodic_operators[M: BasisMetadata, E](
    inner_basis: TupleBasis[tuple[Basis[M], ...], E, Any],
) -> OperatorList[
    AsUpcast[
        DiagonalBasis[
            TupleBasis[
                tuple[
                    FundamentalBasis[SimpleMetadata],
                    DiagonalOperatorBasis[
                        TupleBasisLike[tuple[M, ...], E],
                        TupleBasisLike[tuple[M, ...], E],
                    ],
                ],
                None,
            ],
        ],
        TupleMetadata[
            tuple[SimpleMetadata, OperatorMetadata[TupleMetadata[tuple[M, ...], None]]],
            E,
        ],
    ],
    np.dtype[np.complexfloating],
]:
    """Get all generalized e^(ik.x) operator."""
    outer_basis = basis.with_modified_children(
        inner_basis,
        lambda _i, inner: TransformedBasis(inner, direction="backward").upcast(),
    )
    operator_basis = recast_diagonal_basis(inner_basis, outer_basis)

    list_basis = DiagonalBasis(
        TupleBasis((FundamentalBasis.from_size(outer_basis.size), operator_basis))
    ).upcast()
    return OperatorList.build(
        list_basis, np.ones(outer_basis.size, dtype=np.complex128)
    ).ok()


def all_scattering_operators[M: BasisMetadata, E](
    metadata: TupleMetadata[tuple[M, ...], E],
) -> OperatorList[
    AsUpcast[
        DiagonalBasis[
            TupleBasis[
                tuple[
                    FundamentalBasis[SimpleMetadata],
                    DiagonalOperatorBasis[
                        TupleBasisLike[tuple[M, ...], E],
                        TupleBasisLike[tuple[M, ...], E],
                    ],
                ],
                None,
            ],
        ],
        TupleMetadata[
            tuple[SimpleMetadata, OperatorMetadata[TupleMetadata[tuple[M, ...], None]]],
            E,
        ],
    ],
    np.dtype[np.complexfloating],
]:
    """Get the e^(ik.x) operator."""
    inner_basis = basis.from_metadata(metadata)
    return all_periodic_operators(inner_basis)
