from __future__ import annotations

from typing import TYPE_CHECKING, Any, Never, cast

import numpy as np
from scipy.constants import hbar  # type: ignore lib
from slate_core import TupleMetadata
from slate_core.basis import (
    AsUpcast,
    SplitBasis,
    transformed_from_metadata,
)
from slate_core.metadata.volume import (
    fundamental_stacked_delta_k,
    fundamental_stacked_dk,
    fundamental_stacked_k_points,
)

from slate_quantum.operator._build._momentum import momentum
from slate_quantum.operator._operator import Operator, OperatorMetadata

if TYPE_CHECKING:
    from slate_core import Ctype
    from slate_core.metadata import (
        SpacedLengthMetadata,
    )
    from slate_core.metadata.volume import AxisDirections

    from slate_quantum.operator._diagonal import (
        MomentumOperator,
        MomentumOperatorBasis,
        PositionOperatorBasis,
        Potential,
    )


def _wrap_k_points(
    k_points: np.ndarray[Any, np.dtype[np.floating]], delta_k: tuple[float, ...]
) -> np.ndarray[Any, np.dtype[np.floating]]:
    """Wrap values into the range [-delta_k/2, delta_k/2]."""
    shift = np.array(delta_k).reshape(-1, 1) / 2
    return np.mod(k_points + shift, 2 * shift) - shift


def kinetic_energy[M: SpacedLengthMetadata, E: AxisDirections](
    metadata: TupleMetadata[tuple[M, ...], E],
    mass: float,
    bloch_fraction: np.ndarray[Any, np.dtype[np.floating]] | None = None,
) -> MomentumOperator[M, E, Ctype[Never], np.dtype[np.complexfloating]]:
    """Given a mass and a basis calculate the kinetic part of the Hamiltonian."""
    bloch_fraction = (
        np.zeros(len(metadata.children)) if bloch_fraction is None else bloch_fraction
    )

    bloch_phase = np.tensordot(
        fundamental_stacked_dk(metadata), bloch_fraction, axes=(0, 0)
    )
    k_points = fundamental_stacked_k_points(metadata) + bloch_phase[:, np.newaxis]
    k_points = _wrap_k_points(
        k_points, tuple(np.linalg.norm(fundamental_stacked_delta_k(metadata), axis=1))
    )
    energy = cast(
        "np.ndarray[Any, np.dtype[np.complexfloating]]",
        np.sum(np.square(hbar * k_points) / (2 * mass), axis=0, dtype=np.complex128),
    )
    momentum_basis = transformed_from_metadata(metadata).upcast()

    return momentum(momentum_basis, energy)


def kinetic_hamiltonian[M: SpacedLengthMetadata, E: AxisDirections](
    potential: Potential[M, E, Ctype[Never], np.dtype[np.complexfloating]],
    mass: float,
    bloch_fraction: np.ndarray[Any, np.dtype[np.floating]] | None = None,
) -> Operator[
    AsUpcast[
        SplitBasis[
            PositionOperatorBasis[M, E],
            MomentumOperatorBasis[M, E],
        ],
        OperatorMetadata[TupleMetadata[tuple[M, ...], E]],
    ],
    np.dtype[np.complexfloating],
]:
    """Calculate the total hamiltonian in momentum basis for a given potential and mass."""
    metadata = potential.basis.metadata().children[0]
    kinetic_hamiltonian = kinetic_energy(metadata, mass, bloch_fraction)
    basis = SplitBasis(potential.basis, kinetic_hamiltonian.basis)
    upcast = AsUpcast(basis, TupleMetadata((metadata, metadata)))

    return Operator(
        upcast,
        np.concatenate([potential.raw_data, kinetic_hamiltonian.raw_data], axis=None),
    )
