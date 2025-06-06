from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from slate_core import (
    Array,
    Basis,
    Ctype,
    TupleBasis,
    TupleBasisLike,
    TupleMetadata,
    basis,
)
from slate_core import metadata as _metadata
from slate_core.metadata import (
    AxisDirections,
    EvenlySpacedLengthMetadata,
    EvenlySpacedVolumeMetadata,
)

from slate_quantum.state._state import State
from slate_quantum.state._state import normalize as normalize_state

if TYPE_CHECKING:
    from collections.abc import Callable


def wrap_displacements(
    displacements: np.ndarray[Any, np.dtype[np.floating]],
    max_displacement: float | np.floating,
) -> np.ndarray[Any, np.dtype[np.floating]]:
    return (
        np.remainder((displacements + max_displacement), 2 * max_displacement)
        - max_displacement
    ).astype(np.float64)


def _get_displacements_x_along_axis(
    metadata: EvenlySpacedVolumeMetadata,
    origin: float,
    axis: int,
) -> Array[
    TupleBasis[
        tuple[Basis[EvenlySpacedLengthMetadata], ...], AxisDirections, Ctype[np.generic]
    ],
    np.dtype[np.floating],
]:
    distances = _metadata.volume.fundamental_stacked_x_points(metadata)[axis] - np.real(
        origin
    )
    delta_x = np.linalg.norm(
        _metadata.volume.fundamental_stacked_delta_x(metadata)[axis]
    )
    max_distance = delta_x / 2
    data = wrap_displacements(distances, max_distance)

    return Array(basis.from_metadata(metadata), data)


def get_displacements_x_stacked(
    metadata: EvenlySpacedVolumeMetadata, origin: tuple[float, ...]
) -> tuple[
    Array[
        TupleBasis[
            tuple[Basis[EvenlySpacedLengthMetadata], ...],
            AxisDirections,
            Ctype[np.generic],
        ],
        np.dtype[np.floating],
    ],
    ...,
]:
    """Get the displacements from origin."""
    return tuple(
        _get_displacements_x_along_axis(metadata, o, axis)
        for (axis, o) in enumerate(origin)
    )


def coherent[M: EvenlySpacedLengthMetadata, E: AxisDirections](
    metadata: TupleMetadata[tuple[M, ...], E],
    x_0: tuple[float, ...],
    k_0: tuple[float, ...],
    sigma_0: tuple[float, ...],
) -> State[TupleBasisLike[tuple[M, ...], E]]:
    displacements = get_displacements_x_stacked(metadata, origin=x_0)
    raw_displacements = np.array([d.as_array() for d in displacements])

    # stores distance from x0
    distance = np.linalg.norm(
        [d / s for d, s in zip(raw_displacements, sigma_0, strict=False)],
        axis=0,
    )

    # i k.(x - x')
    phi = np.einsum("ij,i->j", raw_displacements, k_0)  # type: ignore unknown lib type
    data = np.exp(1j * phi - np.square(distance) / 2)
    norm = np.sqrt(np.sum(np.square(np.abs(data))))

    return State(basis.from_metadata(metadata).upcast(), data / norm)


def position[M: EvenlySpacedLengthMetadata, E: AxisDirections](
    metadata: TupleMetadata[tuple[M, ...], E],
    idx: tuple[int, ...],
) -> State[TupleBasisLike[tuple[M, ...], E]]:
    """Get a position eigenstate."""
    position_basis = basis.from_metadata(metadata).upcast()
    data = np.zeros(metadata.shape, dtype=np.complex128)
    idx = tuple(i % n for (i, n) in zip(idx, metadata.shape, strict=True))
    data[idx] = 1.0
    return State(position_basis, data)


def momentum[M: EvenlySpacedLengthMetadata, E: AxisDirections](
    metadata: TupleMetadata[tuple[M, ...], E],
    idx: tuple[int, ...],
) -> State[TupleBasisLike[tuple[M, ...], E]]:
    """Get a momentum eigenstate."""
    momentum_basis = basis.transformed_from_metadata(metadata).upcast()
    data = np.zeros(metadata.shape, dtype=np.complex128)
    idx = tuple(i % n for (i, n) in zip(idx, metadata.shape, strict=True))
    data[idx] = 1.0
    return State(momentum_basis, data)


def from_function[M: EvenlySpacedLengthMetadata, E: AxisDirections](
    metadata: TupleMetadata[tuple[M, ...], E],
    fn: Callable[
        [tuple[np.ndarray[Any, np.dtype[np.floating]], ...]],
        np.ndarray[Any, np.dtype[np.complexfloating]],
    ],
    *,
    normalize: bool = True,
    offset: tuple[float, ...] | None = None,
    wrapped: bool = False,
) -> State[TupleBasisLike[tuple[M, ...], E]]:
    """Get the potential operator."""
    positions = _metadata.volume.fundamental_stacked_x_points(
        metadata, offset=offset, wrapped=wrapped
    )

    result = State(basis.from_metadata(metadata).upcast(), fn(positions))
    return normalize_state(result) if normalize else result
