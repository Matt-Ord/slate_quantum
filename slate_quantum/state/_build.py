from __future__ import annotations

import numpy as np
from slate import StackedMetadata, basis
from slate.metadata import AxisDirections, SpacedLengthMetadata

from slate_quantum.operator.build._position import (
    get_displacements_x_stacked,
)
from slate_quantum.state._state import State


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
