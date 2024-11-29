from __future__ import annotations

import numpy as np
from slate import StackedMetadata, basis
from slate.metadata import AxisDirections, SpacedLengthMetadata

from slate_quantum.operator._operator import operator_basis
from slate_quantum.operator.build._position import (
    build_x_displacement_operators_stacked,
)
from slate_quantum.state._state import State


def build_coherent_state[M: SpacedLengthMetadata, E: AxisDirections](
    metadata: StackedMetadata[M, E],
    x_0: tuple[float, ...],
    k_0: tuple[float, ...],
    sigma_0: tuple[float, ...],
) -> State[M]:
    displacements = build_x_displacement_operators_stacked(metadata, origin=x_0)

    displacements = displacements.with_operator_basis(
        operator_basis(basis.from_metadata(metadata))
    )
    raw_displacements = displacements.raw_data.reshape(displacements.basis.shape[0], -1)

    # stores distance from x0
    distance = np.linalg.norm(
        [d / s for d, s in zip(raw_displacements, sigma_0)],
        axis=0,
    )

    # i k.(x - x')
    phi = np.einsum("ij,i->j", raw_displacements, k_0)  # type: ignore unknown lib type
    data = np.exp(1j * phi - np.square(distance) / 2)
    norm = np.sqrt(np.sum(np.square(np.abs(data))))

    return State(displacements.basis[1][0], data / norm)
