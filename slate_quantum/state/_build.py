from __future__ import annotations

from typing import Any

import numpy as np
from slate import Array, StackedMetadata, TupleBasis, basis
from slate import metadata as _metadata
from slate.metadata import (
    AxisDirections,
    LengthMetadata,
    SpacedLengthMetadata,
    SpacedVolumeMetadata,
)

from slate_quantum.state._state import State


def wrap_displacements(
    displacements: np.ndarray[Any, np.dtype[np.floating]],
    max_displacement: float | np.floating,
) -> np.ndarray[Any, np.dtype[np.floating]]:
    return (
        np.remainder((displacements + max_displacement), 2 * max_displacement)
        - max_displacement
    ).astype(np.float64)


def _get_displacements_x_along_axis(
    metadata: SpacedVolumeMetadata,
    origin: float,
    axis: int,
) -> Array[
    SpacedVolumeMetadata,
    np.floating,
    TupleBasis[LengthMetadata, AxisDirections, np.generic],
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
    metadata: SpacedVolumeMetadata, origin: tuple[float, ...]
) -> tuple[
    Array[
        SpacedVolumeMetadata,
        np.floating,
        TupleBasis[LengthMetadata, AxisDirections, np.generic],
    ],
    ...,
]:
    """Get the displacements from origin."""
    return tuple(
        _get_displacements_x_along_axis(metadata, o, axis)
        for (axis, o) in enumerate(origin)
    )


def build_coherent_state[M: SpacedLengthMetadata, E: AxisDirections](
    metadata: StackedMetadata[M, E],
    x_0: tuple[float, ...],
    k_0: tuple[float, ...],
    sigma_0: tuple[float, ...],
) -> State[StackedMetadata[M, E]]:
    displacements = get_displacements_x_stacked(metadata, origin=x_0)
    raw_displacements = np.array([d.as_array() for d in displacements])

    # stores distance from x0
    distance = np.linalg.norm(
        [d / s for d, s in zip(raw_displacements, sigma_0)],
        axis=0,
    )

    # i k.(x - x')
    phi = np.einsum("ij,i->j", raw_displacements, k_0)  # type: ignore unknown lib type
    data = np.exp(1j * phi - np.square(distance) / 2)
    norm = np.sqrt(np.sum(np.square(np.abs(data))))

    return State(basis.from_metadata(metadata), data / norm)
